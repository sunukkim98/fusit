"""
Span algebra, and the controls a cue-quality claim has to be measured against.

Every signal source in this package returns `[[start, end], ...]` char offsets into the raw
text, so the merge/coverage helpers here are what the whole package speaks in.

The controls exist because "the cue tagger lowered attack accuracy" is not by itself evidence
that the cue tagger picked the *right* words. In a pilot (2026-09-02, N=20) `ner_cot` deleted
49.7% of the document and landed at ASR@1 0.403, while plain `ner` deleted 1.0% and landed at
0.390 -- an ordering indistinguishable from "more text was deleted". A cue condition has to be
compared against a random selection of the SAME size, which is what `random_spans_matched`
builds, and against the removal paradigm's own ceiling, which is `oracle_all_content`.
"""

import random as _random
import re
from typing import List

from fusit.trace.prompts import FUNCTIONAL_WORDS

WORD_RE = re.compile(r"\S+")


def merge_spans(spans: List[List[int]]) -> List[List[int]]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: s[0])
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged

def is_functional_word(word: str) -> bool:
    """True for stopwords and bare punctuation, per TRACE's functional_words set."""
    stripped = word.strip(".,!?-;:()[]{}\"'").lower()
    return (not stripped) or stripped in FUNCTIONAL_WORDS


def coverage(text: str, spans: List[List[int]]) -> float:
    """Fraction of `text` the spans cover, after merging overlaps."""
    if not text:
        return 0.0
    return sum(e - s for s, e in merge_spans(spans)) / len(text)


def redact(text: str, spans: List[List[int]], placeholder: str = "_") -> str:
    """Replace every spanned character with `placeholder`, preserving length."""
    out, prev = [], 0
    for s, e in merge_spans(spans):
        out.append(text[prev:s])
        out.append(placeholder * (e - s))
        prev = e
    out.append(text[prev:])
    return "".join(out)


def is_functional_word(word: str) -> bool:
    stripped = word.strip(".,!?-;:()[]{}\"'").lower()
    return (not stripped) or stripped in FUNCTIONAL_WORDS


def content_word_spans(text: str) -> List[List[int]]:
    """Every non-functional whitespace-delimited word. This is the candidate pool that
    attention_spans() ranks and takes the top-K of, so it is also the right pool to draw
    the E2 random control from: drawing from ALL words instead would make the control
    weaker than it should be (it would be partly measuring "cues avoid stopwords")."""
    return [[m.start(), m.end()] for m in WORD_RE.finditer(text) if not is_functional_word(m.group())]


def random_spans_matched(text: str, target_spans: List[List[int]], seed: int = 0) -> List[List[int]]:
    """E2's coverage-matched random control for `target_spans`.

    Matches the target on THREE axes rather than just total coverage, so that any ASR gap
    between a cue condition and its control is attributable to *which* words were picked
    and not to an incidental difference in how the redaction is shaped:
      - span count            (one control span emitted per target span)
      - per-span char length  (each control span grows to >= its partner's length)
      - word-boundary alignment / content-word-only (same pool as content_word_spans)

    Control spans are placed at random non-overlapping content-word starts. If the text is
    too small to host them all without overlap, placement stops early and the caller sees
    lower coverage than the target -- reported honestly by repro/sweep_cue.py's
    `coverage` field rather than silently padded, since a forced overlap would break the
    length-distribution match that makes this a fair control.
    """
    if not target_spans:
        return []
    rng = _random.Random(seed)
    words = content_word_spans(text)
    if not words:
        return []

    placed: List[List[int]] = []

    def _overlaps(s: int, e: int) -> bool:
        return any(s < pe and e > ps for ps, pe in placed)

    for tstart, tend in target_spans:
        want = tend - tstart
        order = list(range(len(words)))
        rng.shuffle(order)
        for idx in order:
            s = words[idx][0]
            e = words[idx][1]
            j = idx
            while e - s < want and j + 1 < len(words):  # grow over consecutive words
                j += 1
                e = words[j][1]
            # Then cut back to EXACTLY the partner span's length. Growing alone overshoots:
            # the run stops at the first word boundary at or past `want`, which for V_att's
            # short single-word targets measured 1.68x the target coverage (n=12) -- the
            # control would have been deleting more text than the condition it exists to
            # control for. CoT targets are long quotes and barely moved (1.03x). Cutting
            # can end a control span mid-word, which is acceptable: V_cot's own quote spans
            # do too, and redaction rewrites the characters either way.
            e = min(e, s + want)
            if e > s and not _overlaps(s, e):
                placed.append([s, e])
                break

    # Top-up pass. When the target coverage is high the content-word pool saturates and the
    # loop above silently DROPS every span that can no longer find a free slot, so the
    # control ends up deleting less than the condition it controls for -- measured 0.492 vs
    # 0.616 on SynthPAI ner_cot_att, which flatters the cue condition on exactly the
    # comparison E2 exists to make. Total coverage is the axis that must match, so fill the
    # shortfall from whatever content words remain, accepting a coarser per-span length
    # match at saturation (there is no arrangement that preserves both).
    want_total = sum(e - s for s, e in merge_spans([list(t) for t in target_spans]))
    covered = sum(e - s for s, e in merge_spans(placed))
    if covered < want_total:
        free = [w for w in words if not _overlaps(w[0], w[1])]
        rng.shuffle(free)
        for ws, we in free:
            if covered >= want_total:
                break
            if _overlaps(ws, we):
                continue
            we = min(we, ws + (want_total - covered))
            if we > ws:
                placed.append([ws, we])
                covered = sum(e - s for s, e in merge_spans(placed))

    return merge_spans(placed)


def oracle_all_content(text: str) -> List[List[int]]:
    """E5's absolute removal-only ceiling: delete every content word, keeping only function
    words. Nothing a span-removal defense can do goes beyond this. If an attacker still
    beats random guessing here, the limit being measured belongs to the removal paradigm
    itself, not to the cue tagger -- which is the evidence RQ5 asks for. Costs no LLM call."""
    return merge_spans(content_word_spans(text))


__all__ = [
    "WORD_RE", "content_word_spans", "coverage", "is_functional_word", "merge_spans",
    "oracle_all_content", "random_spans_matched", "redact",
]
