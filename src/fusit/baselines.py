"""
TRACE and TRACE-RPS as baselines for the Table 1 run, under the same models and attacker as
`fusit.main`.

    python -m fusit.baselines --dataset synthetic --n-items 20 --k-att 10

    1. trace   TRACE's iterative rewrite (`fusit.trace.anonymize`): the tagger model infers,
               reads attention and writes the leakage chain; the paraphrase model rewrites.
               Up to 5 rounds.
    2. rps     RPS (`fusit.trace.rps`) optimizes a refusal suffix on the attacker model, on
               the original text (RPS) and on TRACE's outputs (TRACE-RPS), then the attacker
               attacks every output.

TRACE is scored twice: after its first round (`trace_r1`, `trace_rps_r1`) and after the full
loop (`trace`, `trace_rps`). The first round analyses the original text
with the same model and prompts as the cue tagger and rewrites once, so it is the comparison
matched to TRACE x DP-Fusion, which also tags once and generates once; the full loop is TRACE
as published, which keeps querying the attacker's inference between rewrites. `--max-rounds 1`
stops after the first; rerunning the same output with the default picks the loop up at round
2 and adds the full-loop conditions.

Stage 1 alternates two 7B models, which do not fit on a 24 GB card together, so it goes round
by round: every unfinished item gets its analysis under the tagger model, then the rewriting
model is loaded and every item that needs it is rewritten. The attacker's no-defense score is
not repeated here -- `fusit.main` already records it with the same model.

The output JSONL is a checkpoint in the same way as `fusit.main`'s: records are rewritten
after every item and each phase skips work already stored.

LEGACY (decision RP4, 2026-10-04): this module produced the earlier results/table1 runs only -- RPS
at the official 2 x 10,000 iterations over `optimization_prompt`, attacked with the
pre-2026-10-04 `fusit.main` attack (the TRACE guess prompt, 400 tokens). Kept to preserve that
code; the main table runs TRACE@1 and RPS through `fusit.main --trace / --rps` (decisions T*, R1-R5,
RP1-RP3) and evaluates with the result tables' protocol. Whether to delete it is open.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

from fusit.main import (DTYPES, PIPELINE_DATASETS, free_model, load_model, load_records, log,
                        select_items, write_records)
from fusit.trace import guess_attribute, score_hits
from fusit.trace import anonymize, rps

#: Conditions scored after TRACE's first round, and the ones only a full loop adds.
ROUND1_CONDITIONS = ("trace_r1", "rps", "trace_rps_r1")
FULL_CONDITIONS = ("trace", "trace_rps")


def conditions_for(args) -> tuple:
    return ROUND1_CONDITIONS + (FULL_CONDITIONS if args.max_rounds == anonymize.MAX_ROUNDS else ())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m fusit.baselines", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", choices=PIPELINE_DATASETS, default="synthetic")
    p.add_argument("--n-items", type=int, default=1, help="0 means the whole corpus")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--item-ids", nargs="+", default=None, metavar="ID")
    p.add_argument("--tagger-model", default="NousResearch/Llama-2-7b-chat-hf",
                   help="TRACE's inference, attention and leakage chain")
    p.add_argument("--paraphrase-model", default="Qwen/Qwen2.5-7B-Instruct", help="TRACE's rewrite")
    p.add_argument("--attacker-model", default="NousResearch/Meta-Llama-3.1-8B-Instruct",
                   help="RPS target and the attacker")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--k-att", type=int, default=10, help="TRACE top-K attention words")
    p.add_argument("--max-rounds", type=int, choices=range(1, anonymize.MAX_ROUNDS + 1),
                   default=anonymize.MAX_ROUNDS,
                   help="stop TRACE after this many rounds; below the full 5 only the round-1 "
                        "conditions are scored, and a later full run resumes the loop")
    p.add_argument("--rewrite-ratio", type=float, default=1.3,
                   help="rewrite budget = ratio x document tokens + 512 (explanation, then the text)")
    p.add_argument("--rewrite-cap", type=int, default=6144)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("baseline_results.jsonl"))
    return p


CONFIG_KEYS = ("tagger_model", "paraphrase_model", "attacker_model", "dtype", "k_att",
               "rewrite_ratio", "rewrite_cap")


def config_of(args) -> dict:
    return {k: getattr(args, k) for k in CONFIG_KEYS}


# ---------------------------------------------------------------------------
# stage 1: TRACE
# ---------------------------------------------------------------------------

def trace_state(rec: dict) -> dict:
    return rec.setdefault("trace", {"text": rec["text"], "rounds": [], "done": False, "stop": None})


def stage_trace(args, recs: Sequence[dict], records: Dict[str, dict]) -> None:
    for r in range(1, args.max_rounds + 1):
        # A: attacker side, on items whose previous round is complete
        todo = [rec for rec in recs if not trace_state(rec)["done"] and len(rec["trace"]["rounds"]) == r - 1]
        if todo:
            model, tok = load_model(args.tagger_model, DTYPES[args.dtype], args.device)
            for n, rec in enumerate(todo, 1):
                st, t0 = rec["trace"], time.perf_counter()
                analysis = anonymize.analyze(st["text"], list(rec["truth"]), model, tok, k=args.k_att)
                st["rounds"].append({"analysis": analysis})
                if not anonymize.needs_rewrite(analysis):
                    st["done"], st["stop"] = True, "certainty"
                write_records(args.output, records)
                log(f"trace round {r} analyze {n}/{len(todo)} {rec['item']}: "
                    f"{sum('chain' in a for a in analysis.values())}/{len(analysis)} confident "
                    f"({time.perf_counter() - t0:.1f}s)")
            del model, tok
            free_model()

        # B: rewrite, on items whose round r has an analysis but no rewrite yet
        todo = [rec for rec in recs if not rec["trace"]["done"] and len(rec["trace"]["rounds"]) == r
                and "rewrite" not in rec["trace"]["rounds"][-1]]
        if todo:
            model, tok = load_model(args.paraphrase_model, DTYPES[args.dtype], args.device)
            for n, rec in enumerate(todo, 1):
                st, t0 = rec["trace"], time.perf_counter()
                n_tok = len(tok(st["text"], add_special_tokens=False)["input_ids"])
                budget = min(args.rewrite_cap, round(args.rewrite_ratio * n_tok) + 512)
                out = anonymize.rewrite(st["text"], st["rounds"][-1]["analysis"], model, tok, budget)
                st["rounds"][-1]["rewrite"] = {"text": out["text"], "raw": out["raw"], "parsed": out["parsed"],
                                               "echo": out.get("echo", False), "budget": budget}
                if out["text"] == st["text"]:
                    st["done"], st["stop"] = True, "unchanged"
                else:
                    st["text"] = out["text"]
                    if r == anonymize.MAX_ROUNDS:
                        st["done"], st["stop"] = True, "max_rounds"
                write_records(args.output, records)
                log(f"trace round {r} rewrite {n}/{len(todo)} {rec['item']}: parsed={out['parsed']} "
                    f"{len(st['text'])}/{len(rec['text'])} chars ({time.perf_counter() - t0:.1f}s)")
            del model, tok
            free_model()


# ---------------------------------------------------------------------------
# stage 2: RPS and the attack
# ---------------------------------------------------------------------------

def round1_text(rec: dict) -> str:
    """TRACE's text after one round: the first rewrite, or the original if the loop stopped
    before rewriting."""
    first = rec["trace"]["rounds"][0]
    return first["rewrite"]["text"] if "rewrite" in first else rec["text"]


def outputs_of(rec: dict) -> Dict[str, Optional[str]]:
    suffixed = rec.get("rps", {})
    return {
        "trace_r1": round1_text(rec),
        "trace": rec["trace"]["text"],
        "rps": suffixed.get("rps", {}).get("text"),
        "trace_rps_r1": suffixed.get("trace_rps_r1", {}).get("text"),
        "trace_rps": suffixed.get("trace_rps", {}).get("text"),
    }


def stage_rps_attack(args, recs: Sequence[dict], records: Dict[str, dict]) -> None:
    conditions = conditions_for(args)
    todo = [rec for rec in recs
            if any(c not in rec.get("attack", {}).get(a, {}) for a in rec["truth"] for c in conditions)]
    if not todo:
        log("rps/attack: every item already done")
        return
    model, tok = load_model(args.attacker_model, DTYPES[args.dtype], args.device)
    for n, rec in enumerate(todo, 1):
        attrs = list(rec["truth"])
        for key, source in (("rps", rec["text"]), ("trace_rps_r1", round1_text(rec)),
                            ("trace_rps", rec["trace"]["text"])):
            if key not in conditions or key in rec.setdefault("rps", {}):
                continue
            t0 = time.perf_counter()
            res = rps.optimize(source, attrs, model, tok, log=log)
            res["seconds"] = round(time.perf_counter() - t0, 1)
            rec["rps"][key] = res
            write_records(args.output, records)
            log(f"{key} {n}/{len(todo)} {rec['item']}: success={res['rule_success']} ({res['seconds']}s)")

        t0 = time.perf_counter()
        attacks = rec.setdefault("attack", {})
        for attribute, truth in rec["truth"].items():
            cells = attacks.setdefault(attribute, {})
            for name, txt in outputs_of(rec).items():
                if name not in conditions or name in cells:
                    continue
                guesses = (guess_attribute(txt, attribute, model, tok)["guesses"][:3]
                           if txt and txt.strip() else [])
                cells[name] = {"guesses": guesses, **score_hits(attribute, guesses, truth)}
        write_records(args.output, records)
        log(f"attack {n}/{len(todo)} {rec['item']}: {len(attrs)} attribute(s) ({time.perf_counter() - t0:.1f}s)")
    del model, tok
    free_model()


def main(argv: Optional[Sequence[str]] = None) -> int:
    import faulthandler, signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)

    args = build_parser().parse_args(argv)
    items = select_items(args)
    records = load_records(args.output)
    for item in items:
        rec = records.setdefault(item.username, {
            "item": item.username, "dataset": args.dataset, "text": item.text,
            "truth": dict(item.relevant_pii), "config": config_of(args),
        })
        if rec["config"] != config_of(args):
            raise SystemExit(f"{item.username}: output was produced with {rec['config']}, "
                             f"this run asks for {config_of(args)}. Use a different --output.")
    recs = [records[i.username] for i in items]
    log(f"{len(items)} item(s) from {args.dataset} (shard {args.shard_index}/{args.num_shards})")

    stage_trace(args, recs, records)
    stage_rps_attack(args, recs, records)

    n = sum(len(r["attack"]) for r in recs)
    for c in conditions_for(args):
        hits = sum(r["attack"][a][c]["top1"] for r in recs for a in r["attack"])
        print(f"{c:<10} {100 * hits / n:6.2f}% top-1 over {n} attribute(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
