"""
The explicit experiment's attack: DP-Fusion's token-recovery game (paper Section 3) with the LOSS
and Min-K% attacks (Section 4.3), by the official DP-Fusion-DPI protocol -- a replica of
dataset/dpfusion_official/Attack.py, which parses argv at import time and so cannot be imported.
Moved from scripts/table1/official.py and attack.py (2026-10-04), unchanged.

For each document and target group g in {PERSON, CODE, DATETIME} the attacker holds the released
text D', |C| = 5 candidate passages in which only g's mentions differ (the true one at index 0, the
rest drawn from candidate_set_100.json with Attack.py's per-type seeds) and the surrogate's weights.
Each candidate is scored by the surrogate's log-likelihood of D' given the official prompt holding
that passage; the best-scoring candidate is the guess. Validated on the paper's own paraphrases
against paper Table 1 (scripts/table1/official_check.py, results/official_check.md).

Decisions recorded against Attack.py (2026-10-04):
    C1-a  the surrogate is Llama-3.1-8B-Instruct (the attacker of the implicit tables); the official
          attacker is the defender's own Qwen2.5-7B.
    C1-c  only the logits that are read are computed (logits_to_keep); the numbers are the same.
    C1-d  Min-K% takes at most as many tokens as a paraphrase has (Attack.py would raise).
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from fusit.dataset import DATASET_DIR, TabDocument
from fusit.dp_fusion.prompting import official_prompt

OFFICIAL_DIR = DATASET_DIR / "dpfusion_official"
#: Attack.py's per-type candidate seeds
SEED = {"PERSON": 42, "CODE": 31, "LOC": 24, "ORG": 67, "DEM": 93, "DATETIME": 32,
        "QUANTITY": 36, "MISC": 91}
#: the groups the paper attacks
ATTACK_TYPES = ("PERSON", "CODE", "DATETIME")
N_CANDIDATES = 5
K_PERCENTS = (5, 10, 20, 30, 40)
#: the Min-K% variants the tables report
MINK = (5, 10, 20, 40)
MAX_PARA_TOKENS = 900


def load_candidate_pool(path: Path = OFFICIAL_DIR / "candidate_set_100.json") -> Dict[str, List[str]]:
    return json.loads(Path(path).read_text())


def candidates(doc: TabDocument, ent: str, pool: Dict[str, List[str]], n: int = N_CANDIDATES):
    """build_variants_for_entity_type_new: passages, index 0 the true one; rng re-seeded per
    (doc, type) with SEED[ent]."""
    mentions = sorted(((s.start, s.end) for s in doc.spans if s.entity_type == ent), key=lambda t: t[0])
    passage = doc.text
    surf = {passage[s:e] for s, e in mentions}
    rng = random.Random(SEED[ent])
    variants, mappings = [passage], [None]
    for _ in range(1, n):
        repls = rng.sample(pool[ent], k=len(surf))
        mapping = dict(zip(sorted(surf), repls))
        pieces, cursor = [], 0
        for s, e in mentions:
            pieces.append(passage[cursor:s])
            pieces.append(mapping[passage[s:e]])
            cursor = e
        pieces.append(passage[cursor:])
        variants.append("".join(pieces))
        mappings.append(mapping)
    return variants, mappings


@torch.inference_mode()
def logp_batch(prompts: List[str], paraphrase: str, model, tok, max_tokens: int = MAX_PARA_TOKENS):
    """batched_logp_causal_single_pass_batched: log P(paraphrase | prompt_i), one right-padded
    batch. Kept faithful on purpose: the paraphrase slice runs to min(start + max_tokens, L - 1) of
    the PADDED batch length L, so shorter candidates also score trailing pad tokens."""
    device = next(model.parameters()).device
    enc_c = tok([p + paraphrase for p in prompts], padding=True, truncation=True,
                add_special_tokens=False, return_tensors="pt")
    enc_p = tok(prompts, padding=True, truncation=True, add_special_tokens=False, return_tensors="pt")
    ids = enc_c.input_ids.to(device)
    mask = enc_c.attention_mask.to(device)
    L = ids.size(1)
    starts = [int(enc_p.attention_mask[i].sum()) for i in range(ids.size(0))]
    # only positions >= min(start) - 1 are ever read; lm_head is per-position, so computing just
    # those (logits_to_keep) gives the same numbers without the full [5, L, V] tensor (C1-c)
    keep = L - (min(starts) - 1)
    with torch.autocast(device.type, enabled=True, dtype=torch.float16):
        logits = model(input_ids=ids, attention_mask=mask, return_dict=True, logits_to_keep=keep).logits
    shift = L - keep
    logp_list, token_lls = [], []
    for i in range(ids.size(0)):
        start = starts[i]
        end = min(start + max_tokens, L - 1)
        tgt = ids[i, start:end + 1]
        lp = F.log_softmax(logits[i, start - 1 - shift:end - shift] / 1.0, dim=-1)
        tl = lp[torch.arange(tgt.size(0), device=device), tgt]
        logp_list.append(tl.sum().item())
        token_lls.append(tl.float().cpu())
    del logits
    torch.cuda.empty_cache()
    return logp_list, token_lls


def attack_scores(logp_list, token_lls, true_index: int = 0) -> Dict[str, Dict]:
    """attack_scores: "loss" = argmin of -sum log p / ln 10, Min-K% argmax; ties to the lowest index."""
    out = {}
    ppl = [-lp / math.log(10) for lp in logp_list]
    w = min(range(len(ppl)), key=ppl.__getitem__)
    out["loss"] = {"scores": ppl, "winner": w, "correct": int(w == true_index)}
    L = len(token_lls[0])
    for k in K_PERCENTS:
        kc = max(1, int(round(k / 100 * L)))
        sc = [torch.topk(t, min(kc, len(t)), largest=False).values.mean().item() for t in token_lls]
        w = max(range(len(sc)), key=lambda j: (sc[j], -j))
        out[f"min{k}"] = {"scores": sc, "winner": w, "correct": int(w == true_index)}
    return out


def attack_document(doc: TabDocument, released: str, model, tok, pool: Dict[str, List[str]],
                    types: Sequence[str] = ATTACK_TYPES) -> Dict[str, Dict]:
    """{group: result} for one released text: the official protocol for every attacked type the
    document has."""
    groups = {}
    for ent in types:
        if not doc.spans_of([ent]):
            continue
        variants, mappings = candidates(doc, ent, pool, N_CANDIDATES)
        logp, tl = logp_batch([official_prompt(tok, v, "_") for v in variants], released, model, tok)
        sc = attack_scores(logp, tl)
        groups[ent] = {"true_idx": 0, "mappings": mappings, "logp": logp, "n_tokens": [len(t) for t in tl],
                       "scores": sc, "pred": {m: sc[m]["winner"] for m in sc},
                       "correct": {m: sc[m]["correct"] for m in sc}}
    return groups


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval of k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


__all__ = ["ATTACK_TYPES", "K_PERCENTS", "MINK", "N_CANDIDATES", "attack_document", "attack_scores", "candidates",
           "load_candidate_pool", "logp_batch", "wilson"]
