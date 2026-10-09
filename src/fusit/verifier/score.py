"""
Per-token candidate log-likelihoods for every V_cot word unit and for the null words of one item.
Nothing here decides; fusit.verifier.decide turns these into p, S and decisions.

Per (item, attribute) with >= 2 candidates:
    D, empty                  once
    D minus u, u alone        per unit
    D minus r, r alone        per null word r (units.null_words), all occurrences
Every condition stores the candidates' per-token log-probs ("tok"), so sum / mean / first-token
scoring are all derivable without rerunning the model.

V_att words (2026-10-06): score_item(att=True) makes them units too; add_att_units adds them to scores that
were made without them. The null words are the same either way, so the thresholds do not move.
"""

from typing import Dict, Optional

from fusit.trace.spans import redact
from fusit.verifier.candidates import candidate_set
from fusit.verifier.scoring import EMPTY_TEXT, token_logprobs
from fusit.verifier.units import att_units, null_words, word_positions, word_units


def _unit_row(key, u, tok, text, n_tokens):
    return {**key, **u, "unscored": False, "n_occ": len(u["spans"]), "len_tokens": n_tokens(u["text"]),
            "tok_Dminus": tok(redact(text, u["spans"])), "tok_alone": tok(u["text"])}


def score_item(model, tokenizer, dataset: str, item_id: str, text: str, cues: Dict[str, dict],
               labels: Optional[Dict[str, str]] = None, clean_candidates: bool = False,
               att: bool = False) -> Dict[str, list]:
    """{"pairs", "units", "null"} of one item. `cues` are the tag stage's records
    ({attribute: CueTagger.tag record}); `labels` (ground truth) ride along for analysis only.
    `att`: the V_att words are units too (units.att_units), scored like the V_cot ones."""
    labels = labels or {}
    n_tokens = lambda t: len(tokenizer(t, add_special_tokens=False)["input_ids"])  # noqa: E731
    out = {"pairs": [], "units": [], "null": []}
    for attr, a in cues.items():
        guesses = a["guess"]["guesses"]
        cands = candidate_set(attr, guesses, clean=clean_candidates)
        units = word_units(text, a["c"])
        if att:
            units += att_units(text, [w["word"] for w in a["att"]], units)
        key = {"dataset": dataset, "item_id": item_id, "attribute": attr}
        pair = {**key, "candidates": cands, "cot_guesses": guesses, "n_c": len(a["c"]),
                "n_units": len(units), "gt": labels.get(attr)}
        if len(cands) < 2:
            out["pairs"].append({**pair, "skipped": f"{len(cands)} candidate(s)"})
            out["units"] += [{**key, **u, "unscored": True} for u in units]
            continue
        tok = lambda t: token_logprobs(t, attr, cands, model, tokenizer)  # noqa: E731
        out["pairs"].append({**pair, "skipped": None, "tok_D": tok(text), "tok_empty": tok(EMPTY_TEXT)})
        for u in units:
            out["units"].append(_unit_row(key, u, tok, text, n_tokens))
        for k, w in enumerate(null_words(text, dataset, item_id, attr)):
            spans = word_positions(text, w)
            out["null"].append({**key, "k": k, "text": w, "spans": spans,
                                "tok_Dminus": tok(redact(text, spans)), "tok_alone": tok(w)})
    return out


def add_att_units(model, tokenizer, scores: Dict[str, list], text: str, cues: Dict[str, dict]) -> int:
    """Adds the V_att units to `scores` made by score_item(att=False), in place, scoring only them: each pair
    keeps its candidates, D / empty scores and null words, so thresholds and every V_cot decision stay as they
    were. Returns the number of units added."""
    n_tokens = lambda t: len(tokenizer(t, add_special_tokens=False)["input_ids"])  # noqa: E731
    added = 0
    for pair in scores["pairs"]:
        attr = pair["attribute"]
        key = {k: pair[k] for k in ("dataset", "item_id", "attribute")}
        cot = [u for u in scores["units"] if u["attribute"] == attr]
        new = att_units(text, [w["word"] for w in cues[attr]["att"]], cot)
        if pair["skipped"]:
            scores["units"] += [{**key, **u, "unscored": True} for u in new]
        else:
            tok = lambda t: token_logprobs(t, attr, pair["candidates"], model, tokenizer)  # noqa: E731
            scores["units"] += [_unit_row(key, u, tok, text, n_tokens) for u in new]
        added += len(new)
    return added


__all__ = ["add_att_units", "score_item"]
