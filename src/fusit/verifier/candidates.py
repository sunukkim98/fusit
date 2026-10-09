"""
Candidate sets for p(value | text).

  closed attributes   every option of the attack prompt, spelled as the prompt spells it
  open attributes     TRACE's own guesses, the first three distinct strings as the tagger read them
                      (decision 2026-10-04; until then trailing periods were stripped, case-
                      insensitive duplicates dropped and ages normalized the way the matcher reads
                      them -- clean=True)

Ground truth is never an input here.
"""

import re
from typing import List

from fusit.trace.attributes import ATTRIBUTE_OPTIONS

CLOSED = ("sex", "income_level", "education", "relationship_status")
OPEN = ("age", "occupation", "city_country", "birth_city_country")
N_OPEN_CANDIDATES = 3

_OPTIONS_PREFIX = "Choose from these options:"


def closed_options(attribute: str) -> List[str]:
    body = ATTRIBUTE_OPTIONS[attribute].strip()
    if not body.startswith(_OPTIONS_PREFIX):
        raise ValueError(f"{attribute}: no option list in {ATTRIBUTE_OPTIONS[attribute]!r}")
    return [o.strip() for o in body[len(_OPTIONS_PREFIX):].strip().rstrip(".").split(", ")]


def normalize_age(guess: str) -> str:
    ages = [a for a in (int(x) for x in re.findall(r"\d+", guess)) if a < 200]
    return "-".join(str(a) for a in ages[:2])


def open_candidates(attribute: str, guesses: List[str], n: int = N_OPEN_CANDIDATES, clean: bool = False) -> List[str]:
    if not clean:
        out = []
        for g in guesses:
            if g and g not in out:
                out.append(g)
            if len(out) == n:
                break
        return out
    out, seen = [], set()
    for g in guesses:
        g = normalize_age(g) if attribute == "age" else g.strip().rstrip(".").strip()
        if g and g.lower() not in seen:
            seen.add(g.lower())
            out.append(g)
        if len(out) == n:
            break
    return out


def candidate_set(attribute: str, cot_guesses: List[str], clean: bool = False) -> List[str]:
    if attribute in CLOSED:
        return closed_options(attribute)
    return open_candidates(attribute, cot_guesses, clean=clean)


__all__ = ["CLOSED", "N_OPEN_CANDIDATES", "OPEN", "candidate_set", "closed_options", "normalize_age",
           "open_candidates"]
