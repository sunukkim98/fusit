"""
V_cot quotes -> content-word units, and the null words a unit is compared against.

A unit is a content word inside a V_cot quote of one (item, attribute): a whitespace token with
its edge punctuation stripped, stopwords / bare punctuation skipped (V_att's rule, is_content),
deduplicated case-insensitively. Its positions are every occurrence in D (word_positions).
With --verify-att (2026-10-06) the attribute's V_att words become units by the same rule (att_units).
"""

import random
from typing import List, Optional

from fusit.trace.spans import WORD_RE, is_functional_word, merge_spans, occurrences  # noqa: F401
from fusit.utils import seed_for

#: punctuation stripped off the edges of a whitespace token to get the word itself
EDGE_PUNCT = ".,!?;:()[]{}\"'“”‘’«»-–—/"


def core(word_start: int, word: str):
    """Edge punctuation stripped off a whitespace token: (start, end, core string)."""
    left = len(word) - len(word.lstrip(EDGE_PUNCT))
    stripped = word.strip(EDGE_PUNCT)
    return word_start + left, word_start + left + len(stripped), stripped


def is_content(token: str, core_str: str) -> bool:
    """Not a stopword / bare punctuation, on the raw token and on its core."""
    return bool(core_str) and not is_functional_word(token) and not is_functional_word(core_str)


#: the order attributes are tagged and scored in (the attack harness's)
ATTRIBUTE_ORDER = ["income_level", "age", "sex", "education", "relationship_status", "occupation",
                   "city_country", "birth_city_country"]
#: random content words per (item, attribute), shared by its units
NULL_PER_PAIR = 20


def content_words(text: str, lo: int = 0, hi: Optional[int] = None):
    """[(start, end, core)] of the content words whose token lies in text[lo:hi]."""
    hi = len(text) if hi is None else hi
    out = []
    for m in WORD_RE.finditer(text[lo:hi]):
        s, e, w = core(lo + m.start(), m.group())
        if is_content(m.group(), w):
            out.append((s, e, w))
    return out


def word_positions(text: str, word: str) -> List[List[int]]:
    """Every place `word` occurs: tokens whose core equals it (case-insensitive), plus
    case-insensitive word-bounded matches -- the all-occurrences scope for one word."""
    low = word.lower()
    spans = [[s, e] for s, e, w in (core(m.start(), m.group()) for m in WORD_RE.finditer(text))
             if w.lower() == low]
    return merge_spans(spans + occurrences(text, word))


def word_units(text: str, cs) -> List[dict]:
    """[{u_id, text, spans, from_c}] from one attribute's V_cot candidates (the tagger's "c")."""
    units = {}
    for c in cs:
        for s, e in c["spans"]:
            for _, _, w in content_words(text, s, e):
                u = units.setdefault(w.lower(), {"text": w, "from_c": set()})
                u["from_c"].add(c["c_id"])
    return [{"u_id": i, "text": u["text"], "spans": word_positions(text, u["text"]),
             "from_c": sorted(u["from_c"]), "source": "cot"} for i, u in enumerate(units.values())]


def att_units(text: str, words, cot_units: List[dict]) -> List[dict]:
    """[{u_id, text, spans, source="att"}] from one attribute's V_att words (2026-10-06): each word's core
    (edge punctuation stripped), content words only, every occurrence in D -- the same unit rule as V_cot's.
    A word that is already a V_cot unit of the attribute is left to that unit (one unit, one decision).
    u_ids continue after `cot_units`."""
    seen = {u["text"].lower() for u in cot_units}
    picked = {}
    for w in words:
        _, _, c = core(0, w)
        if is_content(w, c) and c.lower() not in seen:
            picked.setdefault(c.lower(), c)
    start = len(cot_units)
    return [{"u_id": start + i, "text": t, "spans": word_positions(text, t), "source": "att"}
            for i, t in enumerate(picked.values())]


def null_words(text: str, dataset: str, item_id: str, attribute: str, n: int = NULL_PER_PAIR) -> List[str]:
    """`n` distinct content words of D drawn at random (seed crc32(dataset, item, attribute, "null"))."""
    pool = sorted({w.lower(): w for _, _, w in content_words(text)}.items())
    rng = random.Random(seed_for(dataset, item_id, attribute, "null"))
    pick = rng.sample(pool, min(n, len(pool)))
    return [w for _, w in pick]


__all__ = ["ATTRIBUTE_ORDER", "EDGE_PUNCT", "NULL_PER_PAIR", "att_units", "content_words", "core", "is_content",
           "null_words", "occurrences", "word_positions", "word_units"]
