"""
Perplexity of the original document under each Table 1 defence, DP-Fusion's utility metric.

    python -m fusit.perplexity --dataset synthetic --num-shards 20 --shard-index 0

DP-Fusion reports "perplexity, computed via teacher forcing on the ground truth document D":
the paraphrasing model is shown what a method lets it see, and D itself is forced as the
answer. It comes out at 1.03 when the model sees D, 1.46 when it sees D with named entities
redacted, and 1.43-1.46 for DP-Fusion -- so it measures how much of D the model can still
reproduce, not how fluent some output is. Two versions are computed:

    A  as the paper defines it, where that is defined. Redaction rows show the model the
       redacted document (the same placeholder-token contexts DP-Fusion builds). DP-Fusion
       rows force D through the mechanism itself: at every position the public and private
       contexts' next-token distributions are mixed with the lambda `find_lambda` would pick
       under that run's alpha*beta, and D's token is scored under the group average -- the
       distribution DP-Fusion samples from. TRACE and RPS have no such distribution, so they
       have no A value.
    B  on what each defence published, for every row alike: the published text goes in the
       document slot of the same prompt and D is forced. For DP-Fusion this scores the
       paraphrase it actually released, which A does not.

Both use Qwen2.5-7B-Instruct and DP-Fusion's paraphrasing prompt (`format_prompt_new_template`),
as the paper does; D is forced right after its pre-filled "Sure. Here is the paraphrased
document without underscores or placeholders:", preceded by a space. A document's perplexity
is exp of its mean token NLL; `--summarize` averages that over documents, as a per-document
utility.

Teacher forcing needs no decoding loop, so each context is one forward pass over prompt + D,
and lambda is found for all positions at once: `find_lambdas` is `find_lambda` vectorised,
with the same 20 bisection steps (its 1e-6 tolerance is first met on step 20).
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from fusit.dataset import Span
from fusit.dp_fusion import build_contexts
from fusit.dp_fusion.fusion import compute_renyi_divergence_clipped_symmetric
from fusit.dp_fusion.prompting import format_prompt_new_template
from fusit.main import (DTYPES, PIPELINE_DATASETS, load_model, load_records, log,
                        partition_by_source, write_records)
from fusit.utility import RUNS, load_run, outputs_for

#: The runs' DP-Fusion settings (fusit.main defaults, used by both runs).
ALPHA, TEMPERATURE = 2.0, 1.0


def find_lambdas(p_priv: torch.Tensor, p_pub: torch.Tensor, alpha: float, beta: float,
                 max_iter: int = 20) -> torch.Tensor:
    """`fusion.find_lambda` over a batch of positions: [N, V] x [N, V] -> [N]."""
    n = p_pub.shape[0]
    if beta <= 0:
        return torch.zeros(n, device=p_pub.device)
    at_one = compute_renyi_divergence_clipped_symmetric(p_priv, p_pub, alpha) <= beta
    left = torch.zeros(n, device=p_pub.device)
    right = torch.ones(n, device=p_pub.device)
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        mix = mid[:, None] * p_priv + (1 - mid)[:, None] * p_pub
        over = compute_renyi_divergence_clipped_symmetric(mix, p_pub, alpha) > beta
        right = torch.where(over, mid, right)
        left = torch.where(over, left, mid)
    return torch.where(at_one, torch.ones_like(left), left)


class Scorer:
    def __init__(self, model, tok, chunk: int):
        self.model, self.tok, self.chunk = model, tok, chunk
        self._hidden: Dict[tuple, torch.Tensor] = {}

    def target(self, document: str) -> List[int]:
        return self.tok(" " + document, add_special_tokens=False)["input_ids"]

    def prompt_ids(self, shown: str) -> List[int]:
        return self.tok(format_prompt_new_template(self.tok, shown, "_"), add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def hidden(self, context: Sequence[int], target: Sequence[int]) -> torch.Tensor:
        """Final hidden states at the positions that predict `target`, cached per context."""
        key = tuple(context)
        if key not in self._hidden:
            ids = torch.tensor([list(context) + list(target)], device=self.model.device)
            h = self.model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
            self._hidden[key] = h[len(context) - 1: len(context) + len(target) - 1]
        return self._hidden[key]

    def reset(self) -> None:
        self._hidden.clear()
        torch.cuda.empty_cache()

    def _probs(self, h: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.model.lm_head(h).float() / TEMPERATURE, dim=-1)

    @torch.no_grad()
    def plain(self, context: Sequence[int], target: Sequence[int]) -> Dict:
        h = self.hidden(context, target)
        t = torch.tensor(target, device=h.device)
        nll = 0.0
        for s in range(0, len(target), self.chunk):
            logp = F.log_softmax(self.model.lm_head(h[s:s + self.chunk]).float() / TEMPERATURE, dim=-1)
            nll -= logp.gather(1, t[s:s + self.chunk, None]).sum().item()
        return {"nll": nll, "tokens": len(target), "ppl": math.exp(nll / len(target))}

    @torch.no_grad()
    def fused(self, contexts: Dict[str, List[int]], target: Sequence[int], cap: float) -> Dict:
        """D forced through DP-Fusion: group-averaged lambda mixtures, as in the generator."""
        groups = [g for g in contexts if g != "PUBLIC"]
        hs = {g: self.hidden(contexts[g], target) for g in contexts}
        t = torch.tensor(target, device=hs["PUBLIC"].device)
        nll, lam_sum = 0.0, {g: 0.0 for g in groups}
        for s in range(0, len(target), self.chunk):
            p_pub = self._probs(hs["PUBLIC"][s:s + self.chunk])
            p_out = torch.zeros_like(p_pub)
            for g in groups:
                p_priv = self._probs(hs[g][s:s + self.chunk])
                lam = find_lambdas(p_priv, p_pub, ALPHA, cap)
                lam_sum[g] += lam.sum().item()
                p_out += lam[:, None] * p_priv + (1 - lam)[:, None] * p_pub
                del p_priv
            p_out /= len(groups)
            nll -= torch.log(p_out.gather(1, t[s:s + self.chunk, None]).clamp_min(1e-30)).sum().item()
        return {"nll": nll, "tokens": len(target), "ppl": math.exp(nll / len(target)),
                "lambda_mean": {g: v / len(target) for g, v in lam_sum.items()}}


def spans_of(text: str, spans: Sequence[Sequence[int]], label: str) -> List[Span]:
    return [Span(s, e, label, text[s:e]) for s, e in spans]


def dp_contexts(tok, text: str, partition: List[Span], types: Sequence[str]) -> Dict[str, List[int]]:
    ctx = build_contexts(tok, text, partition, entity_types=types)
    if len(ctx) == 1:  # nothing tagged: fusit.main paraphrases with one group equal to PUBLIC
        ctx["UNTAGGED"] = list(ctx["PUBLIC"])
    return ctx


def score_item(sc: Scorer, item: str, runs: Dict[str, Dict[str, dict]], done: Dict) -> Dict:
    t, d = runs["trace"][item], runs["dpfusion"][item]
    text = t["text"]
    target = sc.target(text)
    out = {"A": dict(done.get("A", {})), "B": dict(done.get("B", {}))}

    # A: the model's view under each defence
    trace_partition = partition_by_source(text, t["spans"], ["ner", "cot", "att"])
    dpf_partition = partition_by_source(text, d["spans"], ["ner"])
    trace_ctx = dp_contexts(sc.tok, text, trace_partition, ["ner", "cot", "att"])
    dpf_ctx = dp_contexts(sc.tok, text, dpf_partition, ["ner"])
    public_only = {
        "no_defense": sc.prompt_ids(text),
        "ner_redaction_spacy": build_contexts(sc.tok, text, spans_of(text, t["spans"].get("ner", []), "ner"))["PUBLIC"],
        "ner_redaction_presidio": dpf_ctx["PUBLIC"],
        "x_priv_redaction": trace_ctx["PUBLIC"],
    }
    for row, ctx in public_only.items():
        if row not in out["A"]:
            out["A"][row] = sc.plain(ctx, target)
    for cap in d["paraphrases"]:
        if f"dp_fusion_presidio@{cap}" not in out["A"]:
            out["A"][f"dp_fusion_presidio@{cap}"] = sc.fused(dpf_ctx, target, float(cap))
    for cap in t["paraphrases"]:
        if f"trace_dp_fusion@{cap}" not in out["A"]:
            out["A"][f"trace_dp_fusion@{cap}"] = sc.fused(trace_ctx, target, float(cap))
    sc.reset()

    # B: what each defence published
    for row, published in outputs_for(item, runs).items():
        if row not in out["B"]:
            out["B"][row] = sc.plain(sc.prompt_ids(published), target)
            sc.reset()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m fusit.perplexity", description=__doc__.split("\n\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", choices=PIPELINE_DATASETS, default="synthetic")
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--chunk", type=int, default=96, help="positions per vocabulary-sized softmax")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("perplexity_results.jsonl"))
    p.add_argument("--summarize", action="store_true",
                   help="print the table from every shard under --output's directory and exit")
    return p


ORDER = ["no_defense", "ner_redaction_spacy", "ner_redaction_presidio", "dp_fusion_presidio@0.01",
         "dp_fusion_presidio@0.1", "rps", "trace_r1", "trace_rps_r1", "x_priv_redaction",
         "trace_dp_fusion@0.01", "trace_dp_fusion@0.1"]


def summarize(out_dir: Path, dataset: str) -> str:
    vals: Dict[str, Dict[str, List[dict]]] = {"A": {}, "B": {}}
    n = 0
    for path in sorted(out_dir.glob("shard_*.jsonl")):
        for rec in load_records(path).values():
            n += 1
            for mode in ("A", "B"):
                for row, v in rec.get(mode, {}).items():
                    vals[mode].setdefault(row, []).append(v)
    lines = [f"== {dataset}: {n} item(s) ==",
             f"{'output':<26} {'A mean':>8} {'A median':>9} {'B mean':>8} {'B median':>9}"]
    rows = sorted(set(vals["A"]) | set(vals["B"]), key=lambda r: ORDER.index(r) if r in ORDER else len(ORDER))
    for row in rows:
        cells = []
        for mode in ("A", "B"):
            v = sorted(x["ppl"] for x in vals[mode].get(row, []))
            cells += ([f"{sum(v) / len(v):8.3f}", f"{v[len(v) // 2]:9.3f}"] if v else [f"{'-':>8}", f"{'-':>9}"])
        lines.append(f"{row:<26} " + " ".join(cells))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.summarize:
        print(summarize(args.output.parent, args.dataset))
        return 0

    runs = {name: load_run(args.results, name, args.dataset) for name in RUNS}
    items = sorted(set.intersection(*(set(r) for r in runs.values())))[args.shard_index::args.num_shards]
    records = load_records(args.output)
    log(f"{len(items)} item(s) from {args.dataset} (shard {args.shard_index}/{args.num_shards}); "
        f"{sum(1 for i in items if i in records)} already in {args.output}")

    model, tok = load_model(args.model, DTYPES[args.dtype], args.device)
    sc = Scorer(model, tok, args.chunk)
    expected = len(ORDER)
    for n, item in enumerate(items, 1):
        rec = records.get(item, {"item": item, "dataset": args.dataset,
                                 "config": {"model": args.model, "dtype": args.dtype}})
        if len(rec.get("B", {})) >= expected and len(rec.get("A", {})) >= 8:
            continue
        t0 = time.perf_counter()
        rec.update(score_item(sc, item, runs, rec))
        records[item] = rec
        write_records(args.output, records)
        log(f"perplexity {n}/{len(items)} {item}: {rec['A'].get('trace_dp_fusion@0.01', {}).get('ppl', float('nan')):.3f} "
            f"({rec['B']['no_defense']['tokens']} tokens, {time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
