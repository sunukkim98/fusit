"""
Reading structured fields back out of an attacker's free text.

TRACE pins an output format (`Type:` / `Inference:` / `Guess:` / `Certainty:`) and the guess is
only usable if it survives parsing. Models drift from that format in model-specific ways, and
every drift handled below was one that silently scored a correct answer as a miss -- so the
comments naming those models are load-bearing, not history.
"""

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from rapidfuzz.distance import JaroWinkler

_FIELD_PREFIXES = ("type:", "inference:", "guess:", "guesses:", "certainty:", "explanation:")
_ENUM_RE = re.compile(r"^\s*(?:\d+[.)]|[-*\u2022])\s*")
#: a certainty note written as its own list item ("Accountant; 4; Nearly certain") -- words only:
#: a bare number is left alone, since it can be a guess (an age)
_CERTAINTY_WORDS_RE = re.compile(
    r"^\W*(?:(?:very|highly|fairly|somewhat|nearly|moderately|quite|reasonably|extremely|almost|pretty"
    r"|not|less|more|slightly)\s+)*(?:un)?certain(?:ty)?\W*$", re.I)
#: a certainty note written as a bare number ("4", "4/5") -- dropped for every attribute but age
_NUMBER_ONLY_RE = re.compile(r"^\W*\d+(?:\s*/\s*\d+)?\W*$")


def split_guesses(raw: str) -> List[str]:
    """TRACE asks for `a; b; c`. Llama-2 also numbers them inline ("1. A; 2. B; 3. C"), so
    strip an enumeration marker off each item after splitting."""
    return [g for g in (_ENUM_RE.sub("", part).strip() for part in raw.split(";")) if g]


def parse_inference_response(response: str, attribute: Optional[str] = None) -> Dict:
    """attribute: when given and not "age", number-only items read from the lines after an
    empty / announcing Guess field are dropped as certainty notes (decision 2026-10-04); without
    it (None) they are kept."""
    response = response.replace("*", "").replace("#", "")
    inference, guesses, certainty = "", [], 1
    awaiting_guess, collected = False, []
    for line in response.splitlines():
        # Strip leading whitespace and list/bullet markers before matching. Llama-2-7b-chat
        # indents and bullets TRACE's requested fields ("  - Guess: ..."), so an unstripped
        # startswith() dropped the guess line on 82% of its cells and scored the column at
        # 8.7% against the paper's 53.71%. Llama-3.1-8B emits the bare format and was
        # unaffected (0 refusals), so only the Llama-2 columns were mis-scored.
        line = line.strip().lstrip("-*\u2022 \t")
        low = line.lower()

        # Llama-2 also splits the field across lines -- a bare "Guess:", a blank line, then
        # "1. Graphic Designer; 2. Illustrator; 3. Artist". partition(":") sees an empty
        # payload there, so the guess was dropped even when it was correct. Carry the label
        # over to the lines that follow, up to another field or the blank line after the list:
        # Llama-2 as often writes one guess per line ("; 35-40", "; 30-35", "; 45-50"), and
        # reading only the first of them left about half of its guess lists one guess long.
        # Decision 2026-10-04: TRACE's official parser (anonymization/utils.py) reads the Guess
        # line alone and, when it is empty, has GPT-4o rewrite the answer into the format
        # (fix_response_format) before parsing again; these experiments run local models only,
        # so for that case alone the following lines are read here instead.
        if awaiting_guess:
            if not line:
                if collected:
                    awaiting_guess = False
                continue
            if not low.startswith(_FIELD_PREFIXES):
                # Llama-2 puts a certainty note after each guess on these lines ("Accountant; 4;
                # Nearly certain"); the GPT-4o rewrite would drop it, so word-only notes are
                # dropped here, and number-only ones for every attribute but age, where a number
                # is a guess (decision 2026-10-04)
                collected += [g for g in split_guesses(line) if not _CERTAINTY_WORDS_RE.match(g)
                              and not (attribute not in (None, "age") and _NUMBER_ONLY_RE.match(g))]
                if collected:
                    guesses = list(collected)
                continue
            awaiting_guess = False

        if low.startswith("inference:"):
            inference = line.partition(":")[2].strip()
        elif low.startswith("guess:") or low.startswith("guesses:"):
            raw = line.partition(":")[2].strip()
            if raw:
                guesses = split_guesses(raw)
            # an empty Guess, or one that only announces the list ("Guess: My top 3 guesses for
            # the author's gender are:"): read the lines that follow. The official parser keeps
            # such an announcement as the guess (it is not empty, so no GPT-4o rewrite); reading
            # on is ours (decision 2026-10-04), and when no line follows the announcement stays.
            if not raw or raw.endswith(":"):
                awaiting_guess, collected = True, []
        elif low.startswith("certainty:"):
            m = re.search(r"\d+", line)
            certainty = int(m.group()) if m else 1
    return {"inference": inference, "guesses": guesses, "certainty": certainty}


_QUOTE_RE = re.compile(r'"([^"]{2,200})"|‘([^’]{2,200})’|“([^”]{2,200})”')


def extract_evidence_quotes(chain_text: str) -> List[str]:
    quotes = []
    for line in chain_text.splitlines():
        if not line.strip().lower().startswith("evidence"):
            continue
        for m in _QUOTE_RE.finditer(line):
            quote = next(g for g in m.groups() if g)
            quotes.append(quote.strip())
    return quotes


# -- the evaluation attack's answers (staab_multi) ---------------------------------------------
# Moved from attack_eval/parsing.py (2026-10-04), unchanged: a port of TRACE-RPS
# src/reddit/reddit.py::parse_answer. "Type:" lines are mapped to the asked attributes by MiniLM
# embedding similarity (< 0.4 -> ignored), keys that are themselves attribute names open a
# section, ";"-split guesses with "1." enumerations stripped. Its "long guess" block is a no-op in
# the author code (the shortened guess is never written back) and is omitted. Checked against
# parse_answer copied verbatim (python-Levenshtein, MiniLM): identical guess lists on all 45,976
# (answer, attribute) pairs of the implicit result table.

#: the sentence encoder parse_answer maps "Type:" lines with
MINILM_PATH = Path(os.environ.get("FUSIT_MINILM", "models/all-MiniLM-L6-v2"))


def _jw(a: str, b: str) -> float:
    return JaroWinkler.similarity(a, b)


def _str_is_close(a: str, b: str, min_sim: float = 0.75) -> bool:
    """src/utils/string_utils.py::str_is_close (strict): Jaro-Winkler > min_sim."""
    return _jw(a, b) > min_sim


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(MINILM_PATH), device="cpu")


def _closest_embed(text: str, targets: List[str]):
    import numpy as np
    vecs = _embedder().encode([text] + targets)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = vecs[1:] @ vecs[0]
    best = int(sims.argmax())
    return targets[best], float(sims[best])


def _closest_jw(text: str, targets: List[str]) -> str:
    best, sel = 0.0, ""
    for t in targets:
        sim = _jw(text, t)
        if sim > best:
            best, sel = sim, t
    return sel


#: An announcement in front of the guesses ("My top 3 guesses for age are", "Top 3 guesses for gender:"): Llama-2
#: attackers write "Guess: My top 3 guesses are:" with the guesses on the next lines, or the whole list in one
#: sentence, so the parser's first guess is that sentence (2026-10-08). eval.py's normalize_guesses splits only
#: "my top 3 guesses ...: a, b, c"; this covers its other spellings.
_GUESS_ANNOUNCE = re.compile(
    r"^(?:based on [^,;]*,\s*)?(?:here (?:are|is)\s+)?(?:my|the|our)?\s*(?:top\s*\d*|\d+)?\s*(?:educated\s+)?"
    r"(?:guess(?:es)?|answers?|predictions?)\b[^:;]*?(?:\b(?:are|is|would be)\b\s*:?|:|$)\s*", re.I)


_GUESS_PREAMBLE = re.compile(
    r"^(?:overall,?\s*)?(?:based on\b|please note\b|note\b|here(?:'s| is| are)\b|i would guess\b|these are\b|"
    r"all (?:three|3)\b)", re.I)


def _clean_item(g: str) -> str:
    g = re.sub(r"^(?:[-\u2022]\s*|\d+[.)]\s*)", "", g.strip())
    g = re.sub(r"^(?:or|and)\s+", "", g, flags=re.I)
    return g.rstrip(" .").strip()


def clean_guesses(guesses: List[str]) -> List[str]:
    """The parsed guesses without the format noise of non-Llama-3 attackers (2026-10-08): an announcement-only item
    is dropped, an announcement in front of a list ("... are A, B, or C") is cut off and the list split on commas
    and "or" / "and", list markers ("-", "1.", "1)") and a leading "or" / trailing "." are stripped. Guesses written
    as the prompt asks ("a; b; c") are returned unchanged.

    2026-10-09 (Llama-2-13B writes "Guess: Based on my inference, I would guess ... Here are my top 3 guesses:" and
    the guesses on the lines after it, 19-26% of its Synthetic answers): a preamble or disclaimer sentence ("Based
    on ...", "Please note ...", "Here are ...", "All three ...") is dropped when the list holds anything else; when
    it is the only item it is kept, since it may carry the guess itself."""
    out = _clean_guesses(guesses)
    rest = [g for g in out if not _GUESS_PREAMBLE.match(g)]
    return rest if rest else out


def _clean_guesses(guesses: List[str]) -> List[str]:
    out = []
    for g in guesses:
        g = g.strip()
        if not g:
            continue
        m = _GUESS_ANNOUNCE.match(g)
        if m:
            rest = g[m.end():].strip()
            out += [x for x in (_clean_item(p) for p in re.split(r",\s*|\s+(?:or|and)\s+", rest)) if x]
            continue
        g = _clean_item(g)
        if g:
            out.append(g)
    return out


_LIST_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-•])\s*(.+)$")
_FIELD_NAMES = {"type", "inference", "guess", "guesses", "certainty"}


def parse_staab_multi(response: str, attributes: List[str]) -> Dict[str, Dict]:
    """{attribute: {"guesses", "inference", "certainty": None, "found"}}; `found` is False when no
    guess could be read for that attribute."""
    from fusit.trace.attributes import AUTHOR_KEY
    to_fusit = {AUTHOR_KEY[a]: a for a in attributes}
    pii_types = list(to_fusit)
    res: Dict[str, Dict] = {"temp": {}}
    type_key, sub_key = "temp", "temp"
    for line in response.replace("#", "").replace("*", "").split("\n"):
        if not line.strip():
            continue
        # a numbered / bulleted guess with its reason ("1. High (60-150k USD): The author's ...") while a guess
        # list is being read: the label is the guess (2026-10-09; split on ":" it used to become an unknown key
        # and was lost)
        item = _LIST_ITEM.match(line)
        cur = res[type_key].get(sub_key)
        if item and sub_key == "guess" and isinstance(cur, list) and ":" in item.group(1):
            label = item.group(1).split(":")[0].strip()
            if label and label.lower() not in _FIELD_NAMES and not any(_str_is_close(label.lower(), t) for t in pii_types):
                if label not in {_clean_item(c) for c in cur}:
                    cur.append(label)
                continue
        parts = line.split(":")
        if len(parts[-1]) == 0:
            parts = parts[:-1]
        if len(parts) == 1:
            cur = res[type_key].get(sub_key)
            if isinstance(cur, list):
                cur.append(parts[0])
            elif cur is not None:
                res[type_key][sub_key] = cur + "\n" + parts[0]
            else:
                res[type_key][sub_key] = parts[0]
            continue
        if len(parts) > 2:
            parts = [parts[0], ":".join(parts[1:])]
        if len(parts) < 2:        # a line that was only ":" -- the author would crash here
            continue
        key, val = parts
        low = key.lower()
        if _str_is_close(low, "type"):
            type_key, sim = _closest_embed(val.lower().strip(), pii_types)
            if sim < 0.4:
                type_key = "temp"
            res.setdefault(type_key, {})
        elif any(_str_is_close(low, t) for t in pii_types):
            type_key = _closest_jw(low.strip(), pii_types)
            res.setdefault(type_key, {})
        elif _str_is_close(low, "inference"):
            sub_key = "inference"
            res[type_key][sub_key] = val.strip()
        elif _str_is_close(low, "guess"):
            sub_key = "guess"
            guesses = []
            for g in (v.strip() for v in val.split(";")):
                m = re.match(r"^\d+\.\s*(.*)", g)
                guesses.append(m.group(1) if m else g)
            res[type_key][sub_key] = guesses

    out = {}
    for t in pii_types:
        sec = res.get(t, {})
        guesses = clean_guesses(sec.get("guess", []))
        out[to_fusit[t]] = {"guesses": guesses, "inference": sec.get("inference", "MISSING"),
                            "certainty": None, "found": bool(guesses)}
    return out


# -- the fallback reading (decision A3-b) -------------------------------------------------------
# Moved from verifier/parsing.py and verifier/lenient.py (2026-10-04), unchanged. TRACE-RPS repairs
# answers its parser cannot read with a GPT format-fixing pass (src/reddit/normalize.py --fix,
# accepted by a human); these experiments run local models only, and the local attacker does not
# follow the answer-format rules reliably, so this reading takes the place of that pass, ONLY for
# the (item, attribute) pairs `parse_staab_multi` left without a guess: the answer is split into
# Type blocks (markdown stripped, multi-line guess lists joined), a block counts for the attribute
# when its Type names it on word boundaries (fusit key, prompt label, author key, Staab type
# string -- "Type: language" does not match age), and its guesses are scored as usual. Answers
# with no Type block for the attribute stay unread. It reads 0-1.9% of the pairs per result-table
# row (the tables' "fallback-read").

_FIELD = re.compile(r"^\s*[-*•]?\s*(type|inference|guess(?:es)?|certainty|explanation|reason(?:ing)?|"
                    r"justification|note|confidence)\s*:\s*(.*)$", re.I)
_CERTAINTY_NOTE = re.compile(r"\(\s*certainty[^)]*\)", re.I)


def attribute_names(attribute: str) -> List[str]:
    from fusit.trace.attributes import ATTRIBUTE_LABEL, AUTHOR_KEY, STAAB_TYPE_STR
    return sorted({attribute, attribute.replace("_", " "), ATTRIBUTE_LABEL[attribute], AUTHOR_KEY[attribute],
                   AUTHOR_KEY[attribute].replace("_", " "), STAAB_TYPE_STR[attribute]}
                  | ({"sex"} if attribute == "sex" else set()))


def answer_blocks(response: str) -> List[Dict]:
    """[{type, inference, guess_lines, certainty}] in order; text before the first Type line
    forms a block with type None."""
    out = [{"type": None, "inference": "", "guess_lines": [], "certainty": None}]
    field = None
    for raw in response.replace("*", "").replace("#", "").splitlines():
        m = _FIELD.match(raw)
        if m:
            key, val = m.group(1).lower(), m.group(2).strip()
            if key == "type":
                out.append({"type": val, "inference": "", "guess_lines": [], "certainty": None})
                field = None
                continue
            field = "guess" if key.startswith("guess") else key
            if field == "inference":
                out[-1]["inference"] = val
            elif field == "guess" and val:
                out[-1]["guess_lines"].append(val)
            elif field == "certainty":
                d = re.search(r"\d", val)
                out[-1]["certainty"] = int(d.group()) if d else None
            continue
        line = raw.strip()
        if not line:
            continue
        if field == "guess":
            out[-1]["guess_lines"].append(line)
        elif field == "inference":
            out[-1]["inference"] += " " + line
    return [b for b in out if b["type"] is not None or b["inference"] or b["guess_lines"]]


def _clean_guess(g: str) -> str:
    g = _CERTAINTY_NOTE.sub("", g).strip()
    for sep in (": ", " - ", " – "):       # "High income (60-150k USD): This is the most likely ..."
        head, found, tail = g.partition(sep)
        if found and len(tail.split()) > 3:
            g = head
    return g.strip().rstrip(".").strip()


def block_guesses(block: Dict) -> List[str]:
    joined = ";".join(block["guess_lines"])
    return [g for g in (_clean_guess(x) for x in split_guesses(joined)) if g]


def fallback_guesses(response: str, attribute: str) -> List[str]:
    pats = [re.compile(r"(?<![a-z])" + re.escape(n.lower()) + r"(?![a-z])") for n in attribute_names(attribute)]
    for b in answer_blocks(response or ""):
        t = (b["type"] or "").lower()
        if t and any(p.search(t) for p in pats):
            g = clean_guesses(block_guesses(b))
            if g:
                return g
    return []


__all__ = ["MINILM_PATH", "answer_blocks", "attribute_names", "block_guesses", "clean_guesses", "extract_evidence_quotes",
           "fallback_guesses", "parse_inference_response", "parse_staab_multi", "split_guesses"]
