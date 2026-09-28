"""
Grouping diagnostic: DP-Fusion with all of X_priv in one privacy group, against one group per
cue source.

    python -m fusit.diag_grouping config    --root results/diag_grouping
    python -m fusit.diag_grouping gen       --task single --dataset synthetic --num-shards 20 --shard-index 0
    python -m fusit.diag_grouping eval      --dataset synthetic --num-shards 20 --shard-index 0
    python -m fusit.diag_grouping summarize --root results/diag_grouping

With one group per source DP-Fusion samples from mean_i [lambda_i p_i + (1 - lambda_i) p_pub]:
no distribution in that mixture has seen X_priv whole, so even at lambda_i = 1 the output is
an average of three partial views, and raising alpha*beta cannot bring it to a paraphrase of
the original. `fusit.main --grouping single` puts all of X_priv in one group (DP-Fusion
Appendix A.19), whose private context is the unredacted document. This runs both over one
alpha*beta grid and tabulates them:

    config     writes ROOT/config.json: the grid, seeds, every fusit.main argument, the reused
               runs, the git state
    gen        one fusit.main shard -- tag (reused from results/table1), paraphrase, attack
    eval       utility of every paraphrase -- the Llama-3.1 judge, BLEU, ROUGE, SBERT, as in
               fusit.utility -- and its perplexity, as in fusit.perplexity: A ("mechanism")
               forces D through the grouping's own mixture at that alpha*beta, B ("release")
               scores the published paraphrase
    summarize  alpha*beta x grouping x dataset table, per-document CSV, the seed variance
               table and the Acc/Comb curves

Source grouping at 0.01 and 0.1 is results/table1 itself, so those rows are read from
table1, table1_utility and table1_perplexity rather than generated again. Every new
paraphrase is seeded per (seed, item, alpha*beta) -- seed 0 for the grid, 1-3 for the
variance runs -- and stores its per-step lambda and divergence.

epsilon is DP-Fusion Theorem 4 with m = the number of groups (3 for source, 1 for single) and
beta = divergence / alpha, the convention of Definition 4 and of the table1 runs; per
document the largest over groups. Appendix A.19 writes the single-group constraint as
D_alpha(P_out || P_pub) <= beta_i instead, which read literally makes beta the divergence
itself; `eps_a19` is that reading, computed from the stored divergences, for single rows.
"""

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from fusit.main import DTYPES, dp_fusion_contexts, load_records, log, write_records
from fusit.trace.scoring import score_hits

#: alpha*beta values, spelled as fusit.main keys its paraphrases (str of the float)
GRID = ("0.001", "0.01", "0.1", "1.0", "10.0")
#: no cap: lambda = 1 at every step. For single that is a plain paraphrase of the original
#: (A.19's "No DPI - Original Document"), for source the mean of the three partial views --
#: the limits each grouping's curve can approach. Not a DP run, so it gets no epsilon.
ANCHOR = "inf"
CAPS = GRID + (ANCHOR,)
GROUPINGS = ("source", "single")
#: source at these caps is results/table1, generated with the same models and settings
REUSED = {"source": ("0.01", "0.1")}
MAIN_SEED = 0
#: sampling-variance runs: synthetic only, a fixed 100-item subset, both groupings
VARIANCE = {"dataset": "synthetic", "n_items": 100, "sample_seed": 0, "caps": ("0.01", "1.0"),
            "seeds": (1, 2, 3), "num_shards": 5}
PRECEDENCE = ["ner", "cot", "att"]
#: shard counts of the table1 run, kept so a shard here covers the same items
SHARDS = {"synthetic": 20, "synthpai": 30}
K_ATT = {"synthetic": 10, "synthpai": 30}  # experiment plan 2.1, as scripts/table1

MODELS = {
    "tagger": "NousResearch/Llama-2-7b-chat-hf",
    "paraphrase": "Qwen/Qwen2.5-7B-Instruct",
    "attacker": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "judge": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "perplexity": "Qwen/Qwen2.5-7B-Instruct",
}
ALPHA, DELTA = 2.0, 1e-3  # fusit.main defaults, which table1 ran with


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def gen_tasks() -> Dict[str, dict]:
    """{task: what it generates}. A task is one fusit.main output directory under gen/ or seeds/."""
    tasks = {
        "source": {"dir": "gen/source", "grouping": "source",
                   "caps": [c for c in CAPS if c not in REUSED["source"]], "gen_seed": MAIN_SEED},
        "single": {"dir": "gen/single", "grouping": "single", "caps": list(CAPS), "gen_seed": MAIN_SEED},
    }
    for g in GROUPINGS:
        for s in VARIANCE["seeds"]:
            tasks[f"{g}_s{s}"] = {"dir": f"seeds/{g}_s{s}", "grouping": g, "caps": list(VARIANCE["caps"]),
                                  "gen_seed": s, "variance": True}
    return tasks


def gen_argv(root: Path, results: Path, task: str, dataset: str, num_shards: int, shard_index: int,
             n_items: Optional[int] = None) -> List[str]:
    t = gen_tasks()[task]
    if t.get("variance"):
        if dataset != VARIANCE["dataset"]:
            raise SystemExit(f"variance task {task} runs on {VARIANCE['dataset']} only")
        n, sample_seed = n_items or VARIANCE["n_items"], VARIANCE["sample_seed"]
    else:
        n, sample_seed = n_items or 0, 0
    tags = sorted((results / "table1" / dataset).glob("shard_*.jsonl"))
    return [
        "--dataset", dataset, "--n-items", str(n), "--seed", str(sample_seed),
        "--num-shards", str(num_shards), "--shard-index", str(shard_index),
        "--tagger-model", MODELS["tagger"], "--sources", *PRECEDENCE, "--k-att", str(K_ATT[dataset]),
        "--precedence", *PRECEDENCE, "--tags-from", *map(str, tags),
        "--paraphrase-model", MODELS["paraphrase"], "--attacker-model", MODELS["attacker"],
        "--alpha", str(ALPHA), "--delta", str(DELTA),
        "--grouping", t["grouping"], "--alpha-beta", *t["caps"], "--gen-seed", str(t["gen_seed"]),
        "--save-token-trace", "--attack-conditions", "dp_fusion",
        "--output", str(root / t["dir"] / dataset / f"shard_{shard_index}.jsonl"),
    ]


def write_config(root: Path, results: Path, n_items: Optional[int]) -> Path:
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()

    config = {
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": {"head": git("rev-parse", "HEAD"), "dirty": git("status", "--porcelain").splitlines()},
        "grid": list(GRID), "anchor": ANCHOR, "groupings": list(GROUPINGS), "main_gen_seed": MAIN_SEED,
        "n_items": n_items or "all",
        "reused": {
            "tags": str(results / "table1" / "{dataset}" / "shard_*.jsonl"),
            "source_caps": list(REUSED["source"]),
            "source_rows_from": [str(results / r) for r in ("table1", "table1_utility", "table1_perplexity")],
        },
        "variance": {**VARIANCE, "caps": list(VARIANCE["caps"]), "seeds": list(VARIANCE["seeds"])},
        "models": MODELS, "alpha": ALPHA, "delta": DELTA, "shards": SHARDS, "k_att": K_ATT,
        "epsilon": "Theorem 4, m = groups, beta = divergence / alpha, max over groups per document; "
                   "eps_a19 (single only): beta = divergence (Appendix A.19 read literally)",
        # the exact fusit.main call, shown for synthetic shard 0 of 1; the --tags-from list is
        # every results/table1 shard of the dataset
        "tasks": {name: {**t, "argv": ["python", "-m", "fusit.main",
                                       *gen_argv(root, results, name, "synthetic", 1, 0, n_items)]}
                  for name, t in gen_tasks().items()},
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------
# collecting outputs
# ---------------------------------------------------------------------------

def load_dir(path: Path) -> Dict[str, dict]:
    records = {}
    for p in sorted(path.glob("shard_*.jsonl")):
        records.update(load_records(p))
    return records


def row_name(prefix: str, cap: str) -> str:
    return f"{prefix}@{cap}"


def parse_row(row: str):
    """'single_s2@1.0' -> ('single', 2, '1.0'); main grid rows have seed MAIN_SEED."""
    prefix, cap = row.split("@")
    grouping, _, seed = prefix.partition("_s")
    return grouping, int(seed) if seed else MAIN_SEED, cap


def collect(root: Path, results: Path, dataset: str, reused: bool) -> Dict[str, Dict[str, dict]]:
    """{row: {item: record}} for every paraphrase, row = '<grouping>[_s<seed>]@<cap>'.

    The record is fusit.main's, shared between the rows of one run; a row reads its own cap
    out of it. With `reused`, source rows at REUSED caps come from results/table1."""
    rows: Dict[str, Dict[str, dict]] = {}

    def add(prefix, records, caps=None):
        for item, rec in records.items():
            for cap in rec.get("paraphrases", {}):
                if caps is None or cap in caps:
                    rows.setdefault(row_name(prefix, cap), {})[item] = rec

    if reused:
        add("source", load_dir(results / "table1" / dataset), REUSED["source"])
    for name, t in gen_tasks().items():
        add(name if t.get("variance") else t["grouping"], load_dir(root / t["dir"] / dataset))
    return rows


def published(rec: dict, cap: str) -> str:
    para = rec["paraphrases"][cap]
    return rec["text"] if para.get("text") is None else para["text"]


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def run_eval(args) -> None:
    rows = collect(args.root, args.results, args.dataset, reused=False)
    items = sorted({i for r in rows.values() for i in r})[args.shard_index::args.num_shards]
    output = args.root / "eval" / args.dataset / f"shard_{args.shard_index}.jsonl"
    records = load_records(output)
    log(f"eval: {len(items)} item(s) from {args.dataset} over {len(rows)} row(s); "
        f"{sum(1 for i in items if i in records)} already in {output}")

    def rec_for(item):
        return records.setdefault(item, {"item": item, "dataset": args.dataset,
                                         "config": {"judge_model": MODELS["judge"],
                                                    "perplexity_model": MODELS["perplexity"], "dtype": args.dtype},
                                         "utility": {}, "A": {}, "B": {}})

    def mine(item):
        return {row: recs[item] for row, recs in rows.items() if item in recs}

    # stage 1: the judge and the lexical scores
    todo = [i for i in items if any(r not in records.get(i, {}).get("utility", {}) for r in mine(i))]
    if todo:
        from fusit.main import free_model, load_model
        from fusit.utility import Lexical, judge_batch, judge_prompt, parse_judge

        model, tok = load_model(MODELS["judge"], DTYPES[args.dtype], args.device)
        tok.padding_side = "left"
        lexical = Lexical(args.device)
        for n, item in enumerate(todo, 1):
            t0, rec, outputs = time.perf_counter(), rec_for(item), mine(item)
            original = next(iter(outputs.values()))["text"]
            texts = {row: published(r, parse_row(row)[2]) for row, r in outputs.items() if row not in rec["utility"]}
            prompts = {row: judge_prompt(original, t) for row, t in texts.items()}
            lengths = {row: len(tok(p, add_special_tokens=False)["input_ids"]) for row, p in prompts.items()}
            batch: List[str] = []
            # fusit.utility's batching: shortest first, padded length x batch size under a budget
            for row in sorted(prompts, key=lengths.get) + [None]:
                if row is not None and (not batch or (len(batch) + 1) * max(lengths[b] for b in batch + [row]) <= args.batch_tokens):
                    batch.append(row)
                    continue
                answers = judge_batch([prompts[b] for b in batch], model, tok, args.judge_max_new_tokens)
                for b, answer in zip(batch, answers):
                    rec["utility"][b] = {**parse_judge(answer), **lexical(original, texts[b]),
                                         "judge_raw": answer, "chars": len(texts[b])}
                write_records(output, records)
                batch = [row] if row is not None else []
            log(f"judge {n}/{len(todo)} {item}: {len(texts)} output(s) ({time.perf_counter() - t0:.1f}s)")
        del model, tok, lexical
        free_model()

    # stage 2: perplexity, A through the mechanism (grid rows only: it does not depend on the
    # sampling seed) and B on the published text
    def ppl_todo(item):
        outs = mine(item)
        done = records.get(item, {})
        return ([r for r in outs if parse_row(r)[1] == MAIN_SEED and r not in done.get("A", {})],
                [r for r in outs if r not in done.get("B", {})])

    todo = [i for i in items if any(ppl_todo(i))]
    if todo:
        from fusit.main import free_model, load_model
        from fusit.perplexity import Scorer

        model, tok = load_model(MODELS["perplexity"], DTYPES[args.dtype], args.device)
        sc = Scorer(model, tok, args.chunk)
        for n, item in enumerate(todo, 1):
            t0, rec, outputs = time.perf_counter(), rec_for(item), mine(item)
            want_a, want_b = ppl_todo(item)
            some = next(iter(outputs.values()))
            target = sc.target(some["text"])
            for row in want_a:
                grouping, _, cap = parse_row(row)
                ctx = dp_fusion_contexts(tok, outputs[row], grouping, PRECEDENCE)
                if len(ctx) == 1:  # nothing tagged: fusit.main paraphrases with one group equal to PUBLIC
                    ctx["UNTAGGED"] = list(ctx["PUBLIC"])
                rec["A"][row] = sc.fused(ctx, target, float(cap))
            sc.reset()
            for row in want_b:
                rec["B"][row] = sc.plain(sc.prompt_ids(published(outputs[row], parse_row(row)[2])), target)
                sc.reset()
            write_records(output, records)
            log(f"perplexity {n}/{len(todo)} {item}: A {len(want_a)}, B {len(want_b)} "
                f"({len(target)} tokens, {time.perf_counter() - t0:.1f}s)")
        del model, tok, sc
        free_model()


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

#: reference rows: (label, attack run, attack condition, utility/perplexity key)
REFERENCE = [
    ("No defense", "table1", "no_defense", "no_defense"),
    ("Redaction: X_priv", "table1", "x_priv_redaction", "x_priv_redaction"),
    ("DP-Fusion (Presidio) @0.01", "table1_dpfusion", "dp_fusion@0.01", "dp_fusion_presidio@0.01"),
    ("DP-Fusion (Presidio) @0.1", "table1_dpfusion", "dp_fusion@0.1", "dp_fusion_presidio@0.1"),
]


def utility_metrics(s: Optional[dict]) -> dict:
    """The table's utility columns from one stored score; judge scores are re-parsed from the
    raw answer and left out when it fell back, as fusit.utility.summarize does."""
    if s is None:
        return {}
    from fusit.utility import parse_judge

    out = {"sbert": s["sbert"], "bleu": s["bleu"], "rouge1": s["rouge1"]}
    j = {**s, **parse_judge(s["judge_raw"])}
    if j["parse"] != "fallback":
        read, mean = min(max(j["readability"], 0), 10), min(max(j["meaning"], 0), 10)
        out.update(readability=read, meaning=mean, hallucinations=min(max(j["hallucinations"], 0), 1),
                   comb=(read / 10 + mean / 10 + min(max(s["rouge1"], 0), 1)) / 3)
    return out


def epsilon_a19(divergences: Sequence[float]) -> float:
    """Theorem 4 at m = 1 with beta = the divergence itself (Appendix A.19 read literally)."""
    return sum(4 * d for d in divergences) + math.log(1 / DELTA) / (ALPHA - 1)


def per_document(root: Path, results: Path, dataset: str) -> List[dict]:
    """One row per (row, item): attack hits, utility, perplexity, lambda, epsilon."""
    rows = collect(root, results, dataset, reused=True)
    evals = load_dir(root / "eval" / dataset)
    t1_util = load_dir(results / "table1_utility" / dataset)
    t1_ppl = load_dir(results / "table1_perplexity" / dataset)

    out = []
    for row, recs in rows.items():
        grouping, seed, cap = parse_row(row)
        reused = grouping == "source" and seed == MAIN_SEED and cap in REUSED["source"]
        for item, rec in recs.items():
            para = rec["paraphrases"][cap]
            if reused:
                key = f"trace_dp_fusion@{cap}"
                util = t1_util.get(item, {}).get("scores", {}).get(key)
                ppl_a = t1_ppl.get(item, {}).get("A", {}).get(key)
                ppl_b = t1_ppl.get(item, {}).get("B", {}).get(key)
            else:
                ev = evals.get(item, {})
                util, ppl_a, ppl_b = ev.get("utility", {}).get(row), ev.get("A", {}).get(row), ev.get("B", {}).get(row)
            lams = [v for v in para["lambda_mean"].values() if v is not None]
            trace = para.get("trace")
            steps = [x for v in trace["lambda"].values() for x in v] if trace else []
            if cap == ANCHOR:  # nothing bounded the divergence, so there is no guarantee to state
                para, trace = {**para, "epsilon": None}, None
            d = {"dataset": dataset, "row": row, "grouping": grouping, "seed": seed, "alpha_beta": cap,
                 "item": item, "reused_from_table1": int(reused), "tokens": para["tokens"],
                 "lambda": sum(lams) / len(lams) if lams else None,
                 "lam_one": sum(x >= 0.999 for x in steps) / len(steps) if steps else None,
                 "lam_low": sum(x < 0.1 for x in steps) / len(steps) if steps else None,
                 "epsilon": max(para["epsilon"].values()) if para.get("epsilon") else None,
                 "eps_a19": (epsilon_a19(trace["divergence"]["x_priv"])
                             if grouping == "single" and trace and "x_priv" in trace["divergence"] else None),
                 "ppl_mech": ppl_a["ppl"] if ppl_a else None, "ppl_release": ppl_b["ppl"] if ppl_b else None,
                 **utility_metrics(util)}
            for attribute, cells in rec.get("attack", {}).items():
                cell = cells.get(f"dp_fusion@{cap}")
                if cell is not None:
                    d[f"top1:{attribute}"] = score_hits(attribute, cell["guesses"], rec["truth"][attribute])["top1"]
            out.append(d)

    # reference rows, from the table1 runs
    runs = {r: load_dir(results / r / dataset) for r in ("table1", "table1_dpfusion")}
    for label, run, cond, key in REFERENCE:
        for item, rec in runs[run].items():
            d = {"dataset": dataset, "row": label, "grouping": "reference", "item": item,
                 "ppl_mech": t1_ppl.get(item, {}).get("A", {}).get(key, {}).get("ppl"),
                 "ppl_release": t1_ppl.get(item, {}).get("B", {}).get(key, {}).get("ppl"),
                 **utility_metrics(t1_util.get(item, {}).get("scores", {}).get(key))}
            for attribute, cells in rec.get("attack", {}).items():
                if cond in cells:
                    d[f"top1:{attribute}"] = score_hits(attribute, cells[cond]["guesses"], rec["truth"][attribute])["top1"]
            out.append(d)
    return out


COLUMNS = [  # (key, header, format)
    ("acc", "Acc", "{:.2f}"), ("comb", "Comb", "{:.2f}"), ("readability", "Read", "{:.2f}"),
    ("meaning", "Mean", "{:.2f}"), ("sbert", "SBERT", "{:.3f}"), ("bleu", "BLEU", "{:.3f}"),
    ("rouge1", "R-1", "{:.3f}"), ("lambda", "λ", "{:.3f}"), ("lam_one", "λ=1 %", "{:.1f}"),
    ("lam_low", "λ<0.1 %", "{:.1f}"), ("epsilon", "ε", "{:.1f}"),
    ("eps_a19", "ε (A.19)", "{:.1f}"), ("ppl_mech", "Mech PPL", "{:.3f}"), ("ppl_release", "Release PPL", "{:.2f}"),
]


#: columns stored as fractions and shown as %
PERCENT = {"comb", "lam_one", "lam_low"}


def aggregate(docs: List[dict], rows: Sequence[str], items: Optional[set] = None) -> Dict[str, dict]:
    """{row: {column: mean}} over the items and attribute attacks every one of `rows` has, per
    column, so a row missing a metric does not shrink the others' sample. Acc and Comb are %."""
    by = {r: {d["item"]: d for d in docs if d["row"] == r and (items is None or d["item"] in items)} for r in rows}
    out = {r: {} for r in rows}
    attack_keys = [set((i, k) for i, d in by[r].items() for k in d if k.startswith("top1:")) for r in rows]
    shared = set.intersection(*attack_keys) if attack_keys and all(attack_keys) else set()
    for r in rows:
        if shared:
            out[r]["acc"] = 100 * sum(by[r][i][k] for i, k in shared) / len(shared)
            out[r]["n_attacks"] = len(shared)
    for key, _, _ in COLUMNS[1:]:
        have = [r for r in rows if any(d.get(key) is not None for d in by[r].values())]
        if not have:
            continue
        common = set.intersection(*(set(i for i, d in by[r].items() if d.get(key) is not None) for r in have))
        for r in have:
            v = [by[r][i][key] for i in common]
            if v:
                out[r][key] = (100 if key in PERCENT else 1) * sum(v) / len(v)
                out[r][f"n_{key}"] = len(v)
                if key in ("ppl_mech", "ppl_release"):
                    out[r][f"{key}_median"] = statistics.median(v)
    return out


def fmt(v, f) -> str:
    return "–" if v is None else f.format(v)


def main_table(docs: List[dict], dataset: str):
    grid_rows = [row_name(g, c) for c in CAPS for g in GROUPINGS]
    grid_rows = [r for r in grid_rows if any(d["row"] == r for d in docs)]
    refs = [label for label, *_ in REFERENCE if any(d["row"] == label for d in docs)]
    items = set.intersection(*(set(d["item"] for d in docs if d["row"] == r) for r in grid_rows)) if grid_rows else set()
    agg = aggregate(docs, grid_rows + refs, items)
    n = {k: max((agg[r].get(f"n_{k}", 0) for r in grid_rows), default=0) for k in ("attacks", "comb", "ppl_mech")}
    lines = [f"## {dataset}", "",
             f"{len(items)} documents every grid row has; Acc over {n['attacks']} attribute attacks, "
             f"judge columns over {n['comb']} documents whose judgements parsed for every row, "
             f"Mech PPL over {n['ppl_mech']}. Acc and Comb are %. Reference rows are over the same documents.", "",
             "| αβ | grouping | " + " | ".join(h for _, h, _ in COLUMNS) + " |",
             "|---|---|" + "---:|" * len(COLUMNS)]
    table = []
    for r in grid_rows + refs:
        if r in refs:
            cap, grouping = "", r
        else:
            grouping, _, cap = parse_row(r)
            grouping += " †" if grouping == "source" and cap in REUSED["source"] else ""
        lines.append(f"| {cap} | {grouping} | " + " | ".join(fmt(agg[r].get(k), f) for k, _, f in COLUMNS) + " |")
        table.append({"dataset": dataset, "row": r, "alpha_beta": cap, "grouping": grouping.rstrip(" †"),
                      **{k: agg[r].get(k) for k, _, _ in COLUMNS},
                      **{k: v for k, v in agg[r].items() if k.startswith("n_") or k.endswith("_median")}})
    lines += ["", "† generated by the results/table1 run and reused (same models, settings and tags; unseeded)."]
    return table, "\n".join(lines), agg


def variance_table(docs: List[dict]):
    """Seed 1-3 runs on the synthetic subset: mean and sd over seeds, per (grouping, cap)."""
    ds = VARIANCE["dataset"]
    docs = [d for d in docs if d["dataset"] == ds]
    seed_rows = sorted({d["row"] for d in docs if d.get("seed") not in (None, MAIN_SEED)})
    if not seed_rows:
        return [], ""
    items = set.intersection(*(set(d["item"] for d in docs if d["row"] == r) for r in seed_rows))
    main_rows = [row_name(g, c) for g in GROUPINGS for c in VARIANCE["caps"]]
    main_rows = [r for r in main_rows if any(d["row"] == r for d in docs)]
    agg = aggregate(docs, seed_rows + main_rows, items)
    keys = ["acc", "comb", "readability", "meaning", "sbert", "lambda", "epsilon", "ppl_release"]
    heads = {k: h for k, h, _ in COLUMNS}
    lines = [f"## Sampling variance ({ds}, {len(items)} documents, seeds {', '.join(map(str, VARIANCE['seeds']))})", "",
             "mean ± sd over seeds; `grid run` is the table's seed-0 (single) or table1 (source) run on the same documents.", "",
             "| αβ | grouping | " + " | ".join(heads[k] for k in keys) + " |", "|---|---|" + "---:|" * len(keys)]
    table = []
    for g in GROUPINGS:
        for c in VARIANCE["caps"]:
            per_seed = [agg[r] for r in seed_rows if parse_row(r)[0] == g and parse_row(r)[2] == c]
            if not per_seed:
                continue
            cells, rec = [], {"grouping": g, "alpha_beta": c, "seeds": len(per_seed), "n_items": len(items)}
            for k in keys:
                v = [a[k] for a in per_seed if k in a]
                if not v:
                    cells.append("–")
                    continue
                m, s = statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)
                rec[k], rec[f"{k}_sd"] = m, s
                f = dict((k2, f2) for k2, _, f2 in COLUMNS)[k]
                cells.append(f"{f.format(m)} ± {f.format(s)}")
            lines.append(f"| {c} | {g} | " + " | ".join(cells) + " |")
            table.append(rec)
            main = agg.get(row_name(g, c))
            if main:
                lines.append(f"| {c} | {g} (grid run) | " + " | ".join(fmt(main.get(k), dict((k2, f2) for k2, _, f2 in COLUMNS)[k]) for k in keys) + " |")
    return table, "\n".join(lines)


def plot_curves(aggs: Dict[str, dict], variance: List[dict], path: Path) -> None:
    """Acc and Comb against alpha*beta, one line per grouping, one column per dataset. The
    lambda = 1 anchor sits past the axis break as a hollow marker: the limit of each curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"source": "#2a78d6", "single": "#eb6834"}  # reference palette slots 1-2
    labels = {"source": "source (3 groups)", "single": "single (1 group)"}
    sd = {(v["grouping"], v["alpha_beta"], k): v.get(f"{k}_sd") for v in variance for k in ("acc", "comb")}
    x_anchor = 200.0

    plt.rcParams.update({"font.size": 9})
    datasets = list(aggs)
    fig, axes = plt.subplots(2, len(datasets), figsize=(4.8 * len(datasets), 6.6), squeeze=False)
    for j, ds in enumerate(datasets):
        agg = aggs[ds]
        for i, (key, title) in enumerate((("acc", "Attack top-1 accuracy (%)"), ("comb", "Comb (%)"))):
            ax = axes[i][j]
            for g in GROUPINGS:
                pts = [(float(c), agg[row_name(g, c)][key]) for c in GRID
                       if row_name(g, c) in agg and key in agg[row_name(g, c)]]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=colors[g], lw=2, solid_capstyle="round", solid_joinstyle="round",
                        marker="o", ms=6, mec="white", mew=1.5, label=labels[g], zorder=3)
                for c in VARIANCE["caps"]:
                    e = sd.get((g, c, key)) if ds == VARIANCE["dataset"] else None
                    if e is not None and row_name(g, c) in agg:
                        ax.errorbar([float(c)], [agg[row_name(g, c)][key]], yerr=[e], color=colors[g],
                                    lw=1, capsize=3, zorder=2)
                anchor = agg.get(row_name(g, ANCHOR), {}).get(key)
                if anchor is not None:
                    ax.plot([xs[-1], x_anchor], [ys[-1], anchor], color=colors[g], lw=1, ls=(0, (1, 2)), zorder=2)
                    ax.plot([x_anchor], [anchor], marker="o", ms=7, mfc="white", mec=colors[g], mew=2, zorder=3)
                    ax.annotate(f"{anchor:.1f}", (x_anchor, anchor), xytext=(7, 0), textcoords="offset points",
                                va="center", fontsize=8, color=ink)
            # No defense labelled above its line, Presidio below, so equal values do not collide
            for label, style, dy, va in (("No defense", (0, (1, 2)), 3, "bottom"),
                                         ("DP-Fusion (Presidio) @0.01", (0, (4, 2)), -3, "top")):
                if label in agg and key in agg[label]:
                    y = agg[label][key]
                    ax.axhline(y, color=muted, lw=1, ls=style, zorder=1)
                    ax.annotate(label.replace(" @0.01", " 0.01"), (0.0009, y), xytext=(0, dy), textcoords="offset points",
                                fontsize=7.5, color=muted, va=va)
            ax.set_xscale("log")
            ax.set_xticks([float(c) for c in GRID] + [x_anchor])
            ax.set_xticklabels(list(GRID) + ["∞\n(λ=1)"])
            ax.minorticks_off()
            ax.set_xlim(0.0007, 600)
            ax.axvspan(25, 80, color="white", zorder=4)  # axis break between the grid and the anchor
            ax.grid(True, color=grid, lw=0.8)
            ax.set_axisbelow(True)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            for s_ in ("left", "bottom"):
                ax.spines[s_].set_color(grid)
            ax.tick_params(colors=muted, labelsize=8)
            ax.set_title(f"{ds} — {title}", fontsize=9.5, color=ink, loc="left")
            if i == 1:
                ax.set_xlabel("αβ (divergence cap)", color=muted)
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=2, frameon=False, fontsize=8.5)
    n_var = max((v.get("n_items", 0) for v in variance), default=0)
    fig.text(0.01, 0.005, "Hollow markers: no cap (λ = 1 every step). "
             + (f"Error bars: ± sd over seeds {', '.join(map(str, VARIANCE['seeds']))} on {n_var} "
                f"{VARIANCE['dataset']} documents. " if n_var else "")
             + "\nDotted line: no defense; dashed: DP-Fusion with Presidio NER at αβ = 0.01.",
             fontsize=7.5, color=muted)
    fig.tight_layout(rect=(0, 0.04, 1, 0.965))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def write_csv(path: Path, rows: List[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(args) -> None:
    out = args.root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    all_docs, tables, mds, aggs = [], [], [], {}
    for ds in args.datasets:
        docs = per_document(args.root, args.results, ds)
        if not any(d["grouping"] in GROUPINGS for d in docs):
            continue
        table, md, agg = main_table(docs, ds)
        all_docs += docs
        tables += table
        mds.append(md)
        aggs[ds] = agg
    if not aggs:
        raise SystemExit(f"no generation under {args.root}")
    variance, variance_md = variance_table(all_docs)

    write_csv(out / "per_document.csv", all_docs)
    write_csv(out / "table.csv", tables)
    if variance:
        write_csv(out / "variance.csv", variance)
    md = "# DP-Fusion grouping diagnostic: source (3 groups) vs single (1 group)\n\n" + "\n\n".join(mds)
    if variance_md:
        md += "\n\n" + variance_md
    (out / "table.md").write_text(md + "\n")
    plot_curves(aggs, variance, out / "curves.png")
    print(md)
    print(f"\nwrote {', '.join(str(out / f) for f in ('table.md', 'table.csv', 'per_document.csv', 'variance.csv', 'curves.png'))}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m fusit.diag_grouping", description=__doc__.split("\n\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--root", type=Path, default=Path("results/diag_grouping"))
        sp.add_argument("--results", type=Path, default=Path("results"),
                        help="where the table1 runs being reused live")

    sp = sub.add_parser("config", help="write ROOT/config.json")
    common(sp)
    sp.add_argument("--n-items", type=int, default=None, help="smoke runs: items per dataset")

    sp = sub.add_parser("gen", help="run one fusit.main shard of a task")
    common(sp)
    sp.add_argument("--task", choices=sorted(gen_tasks()), required=True)
    sp.add_argument("--dataset", choices=sorted(SHARDS), required=True)
    sp.add_argument("--num-shards", type=int, required=True)
    sp.add_argument("--shard-index", type=int, required=True)
    sp.add_argument("--n-items", type=int, default=None, help="smoke runs: items per dataset")

    sp = sub.add_parser("eval", help="utility and perplexity of one shard of every paraphrase")
    common(sp)
    sp.add_argument("--dataset", choices=sorted(SHARDS), required=True)
    sp.add_argument("--num-shards", type=int, default=1)
    sp.add_argument("--shard-index", type=int, default=0)
    sp.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    sp.add_argument("--device", default="cuda:0")
    sp.add_argument("--batch-tokens", type=int, default=24000, help="as fusit.utility")
    sp.add_argument("--judge-max-new-tokens", type=int, default=512, help="as fusit.utility")
    sp.add_argument("--chunk", type=int, default=96, help="as fusit.perplexity")

    sp = sub.add_parser("summarize", help="tables, per-document CSV and curves into ROOT/summary")
    common(sp)
    sp.add_argument("--datasets", nargs="+", default=["synthetic", "synthpai"])
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "config":
        print(write_config(args.root, args.results, args.n_items))
    elif args.command == "gen":
        from fusit.main import main as fusit_main
        argv = gen_argv(args.root, args.results, args.task, args.dataset, args.num_shards,
                        args.shard_index, args.n_items)
        tags = [a for a in argv if a.startswith(str(args.results / "table1"))]
        log("fusit.main " + " ".join(a for a in argv if a not in tags).replace(
            "--tags-from", f"--tags-from <{len(tags)} table1 shards>"))
        return fusit_main(argv)
    elif args.command == "eval":
        run_eval(args)
    elif args.command == "summarize":
        summarize(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
