"""
Stored per-token log-probs (fusit.verifier.score) -> distributions, S_cond / S_alone, null
thresholds and decisions. CPU only.

The scoring norm is per attribute: closed attributes use "sum" (as registered); open attributes
use "mean" (chosen at the gate).
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from fusit.verifier.candidates import CLOSED
from fusit.verifier.scoring import argmax, distribution_from, entropy, jsd

NORM_CLOSED = "sum"
NORM_OPEN = "mean"
PCTL_MAIN = 95
PCTL_SENS = 90
#: further sensitivity levels (2026-10-04): decision@85, decision@80
PCTL_EXTRA = (85, 80)


def norm_for(attribute: str, open_norm: Optional[str] = None) -> str:
    if attribute in CLOSED:
        return NORM_CLOSED
    return open_norm or NORM_OPEN


def pair_key(r):
    return r["dataset"], r["item_id"], r["attribute"]


def pair_dists(pair: dict, norm: str) -> dict:
    p_D = distribution_from(pair["candidates"], pair["tok_D"], norm)
    p_E = distribution_from(pair["candidates"], pair["tok_empty"], norm)
    return {"p_D": p_D, "p_empty": p_E, "a_star": argmax(p_D), "H_D": entropy(p_D),
            "H_empty": entropy(p_E), "jsd_D_empty": jsd(p_D, p_E)}


def unit_scores(u: dict, pair: dict, pd: dict, norm: str) -> dict:
    """S_cond, S_alone and the logged auxiliaries of one unit (or null word)."""
    p_Du = distribution_from(pair["candidates"], u["tok_Dminus"], norm)
    p_u = distribution_from(pair["candidates"], u["tok_alone"], norm)
    a = pd["a_star"]
    return {"p_Dminus": p_Du, "p_alone": p_u,
            "S_cond": jsd(pd["p_D"], p_Du), "S_alone": jsd(pd["p_empty"], p_u),
            "dp_cond": pd["p_D"][a] - p_Du[a], "dp_alone": p_u[a] - pd["p_empty"][a],
            "dH_cond": entropy(p_Du) - pd["H_D"], "dH_alone": entropy(p_u) - pd["H_empty"]}


def score_rows(rows: Dict[str, List[dict]], open_norm: Optional[str] = None) -> Dict[str, List[dict]]:
    """Adds distributions and S to every scored pair / unit / null row (in new dicts)."""
    pairs, pd_of = [], {}
    for p in rows["pairs"]:
        if p["skipped"]:
            pairs.append(dict(p))
            continue
        norm = norm_for(p["attribute"], open_norm)
        pd = pair_dists(p, norm)
        pd_of[pair_key(p)] = (p, pd, norm)
        pairs.append({**p, **pd, "norm": norm})
    out = {"pairs": pairs, "units": [], "null": []}
    for kind in ("units", "null"):
        for u in rows[kind]:
            if u.get("unscored"):
                out[kind].append(dict(u))
                continue
            p, pd, norm = pd_of[pair_key(u)]
            out[kind].append({**u, **unit_scores(u, p, pd, norm), "a_star": pd["a_star"]})
    return out


def thresholds(null_rows: List[dict], pctl: float) -> Dict[tuple, Dict[str, float]]:
    """{(dataset, attribute) and (dataset, "POOLED"): {"cond", "alone", "n"}}."""
    groups = defaultdict(list)
    for r in null_rows:
        if "S_cond" in r:
            groups[(r["dataset"], r["attribute"])].append(r)
            groups[(r["dataset"], "POOLED")].append(r)
    return {k: {"cond": float(np.percentile([r["S_cond"] for r in v], pctl)),
                "alone": float(np.percentile([r["S_alone"] for r in v], pctl)), "n": len(v)}
            for k, v in groups.items()}


def decide(u: dict, tau: Dict[tuple, Dict[str, float]], pooled: bool = False) -> bool:
    """OR rule. Units of skipped pairs (fewer than 2 candidates) are kept: no measurement, so protect."""
    if u.get("unscored"):
        return True
    t = tau[(u["dataset"], "POOLED" if pooled else u["attribute"])]
    return u["S_cond"] >= t["cond"] or u["S_alone"] >= t["alone"]


def decide_all(rows: Dict[str, List[dict]]):
    """(scored rows, thresholds {pctl: tau}): every unit gets decision@95 / @90 (per-attribute
    thresholds) and decision@95_pooled. `rows` pool every item of a dataset -- the thresholds are
    the null's percentiles over all of them."""
    scored = score_rows(rows)
    tau = {p: thresholds(scored["null"], p) for p in (PCTL_MAIN, PCTL_SENS) + PCTL_EXTRA}
    for u in scored["units"]:
        u["decision@95"] = decide(u, tau[PCTL_MAIN])
        u["decision@90"] = decide(u, tau[PCTL_SENS])
        u["decision@95_pooled"] = decide(u, tau[PCTL_MAIN], pooled=True)
        for p in PCTL_EXTRA:
            u[f"decision@{p}"] = decide(u, tau[p])
    return scored, tau


__all__ = ["NORM_CLOSED", "NORM_OPEN", "PCTL_EXTRA", "PCTL_MAIN", "PCTL_SENS", "decide", "decide_all", "norm_for",
           "pair_dists", "pair_key", "score_rows", "thresholds", "unit_scores"]
