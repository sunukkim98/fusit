"""
Collect the Table 1 runs into one table per dataset, next to the paper's numbers.

    python scripts/table1/summarize.py [results]

Two runs feed it, each under its own directory:

    results/table1            TRACE cues (NER u V_cot u V_att, spaCy NER) -> DP-Fusion,
                              grouped by cue source; also scores the undefended text
    results/table1_dpfusion   DP-Fusion as the paper runs it with a real tagger: Presidio +
                              BERT-NER at threshold 0.5, single group (Appendices A.17-A.19)
    results/table1_tracerps   TRACE and TRACE-RPS as baselines (fusit.baselines): TRACE's
                              rewrite with the same tagger and rewriting models, RPS suffixes
                              optimized on the attacker

Values are top-1 attribute inference accuracy (%) of the attacker, Llama-3.1-8B-Instruct,
recomputed from the stored guesses with fusit.trace.scoring (TRACE-RPS's model-free rules),
so a scorer change applies without rerunning any model. Rows are only compared over the
attribute attacks every row has, so a partial run cannot tilt one row against another.

Each row carries its cost -- text touched, or for DP-Fusion the epsilon and mean lambda --
because an accuracy drop means little without it: deleting half the document lowers
accuracy whether or not the right words were chosen.

Paper numbers are TRACE-RPS Table 1, Llama3.1-8B-Instruct column. Its anonymisation rows use
GPT-3.5/GPT-4o as the rewriting model, so they are context, not a like-for-like baseline.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from fusit.trace.scoring import score_hits

PAPER = {  # TRACE-RPS Table 1, Llama3.1-8B-Instruct column
    "synthetic": {"No Defense": "57.14", "D-Defense": "49.71", "RPS": "0 (13.48)",
                  "FgAA (GPT-3.5)": "29.52", "TRACE (GPT-3.5)": "22.86",
                  "TRACE-RPS (GPT-3.5)": "0 (13.48)", "TRACE (GPT-4o)": "18.48"},
    "synthpai": {"No Defense": "54.52", "D-Defense": "47.85", "RPS": "0 (14.10)",
                 "FgAA (GPT-3.5)": "40.29", "TRACE (GPT-3.5)": "36.17",
                 "TRACE-RPS (GPT-3.5)": "0 (14.10)", "TRACE (GPT-4o)": "30.71"},
}
EXPECTED = {"synthetic": 525, "synthpai": 298}

# (run directory, stored condition) -> row label. dp_fusion@<cap> is expanded per cap.
RUNS = {
    "table1": {
        "no_defense": "No Defense",
        "ner_redaction": "Redaction: spaCy NER",
        "x_priv_redaction": "Redaction: TRACE X_priv",
        "dp_fusion": "TRACE x DP-Fusion",
    },
    "table1_dpfusion": {
        "ner_redaction": "Redaction: Presidio BERT-NER",
        "dp_fusion": "DP-Fusion (Presidio BERT-NER)",
    },
    "table1_tracerps": {
        "trace_r1": "TRACE@1 round",
        "trace": "TRACE@5 rounds",
        "rps": "RPS",
        "trace_rps_r1": "TRACE-RPS@1 round",
        "trace_rps": "TRACE-RPS@5 rounds",
    },
}


def load(root: Path, run: str, dataset: str) -> dict:
    records = {}
    for path in sorted((root / run / dataset).glob("shard_*.jsonl")):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    records[r["item"]] = r
    return records


def ci95(k: int, n: int) -> float:
    return float("nan") if n == 0 else 196 * math.sqrt((k / n) * (1 - k / n) / n)


def rows_for(root: Path, dataset: str):
    """-> {label: {(item, attribute): top1}}, {label: cost string}, per-run item counts."""
    cells, costs, counts = defaultdict(dict), {}, {}
    for run, labels in RUNS.items():
        records = load(root, run, dataset)
        attacked = [r for r in records.values() if "attack" in r]
        counts[run] = len(attacked)
        if not attacked:
            continue
        for r in attacked:
            for attribute, conds in r["attack"].items():
                for cond, cell in conds.items():
                    kind, _, cap = cond.partition("@")
                    if kind not in labels:
                        continue
                    label = f"{labels[kind]} ab={cap}" if cap else labels[kind]
                    top1 = score_hits(attribute, cell["guesses"], r["truth"][attribute])["top1"]
                    cells[label][(r["item"], attribute)] = top1

        def mean_cov(key):
            return 100 * sum(r["coverage"].get(key, 0) for r in attacked) / len(attacked)

        for kind, label in labels.items():
            if kind == "ner_redaction":
                costs[label] = f"{mean_cov('ner'):5.1f}% text"
            elif kind == "x_priv_redaction":
                costs[label] = f"{mean_cov('x_priv'):5.1f}% text"
        if run == "table1_tracerps":
            rounds = [len(r["trace"]["rounds"]) for r in attacked]
            costs[labels["trace"]] = f"{sum(rounds) / len(rounds):.2f} rounds on average"
            firsts = [r["trace"]["rounds"][0].get("rewrite") for r in attacked]
            ratios = sorted(len(f["text"]) / len(r["text"]) for f, r in zip(firsts, attacked) if f)
            costs[labels["trace_r1"]] = (f"rewritten {100 * len(ratios) / len(attacked):5.1f}%, "
                                         f"median length x{ratios[len(ratios) // 2]:.2f}")
            for kind in ("rps", "trace_rps_r1", "trace_rps"):
                ok = [r["rps"][kind]["rule_success"] for r in attacked if kind in r.get("rps", {})]
                cells_ = [c[kind] for r in attacked for c in r["attack"].values() if kind in c]
                refused = sum(not c["guesses"] for c in cells_)
                costs[labels[kind]] = (f"suffix success {100 * sum(ok) / max(1, len(ok)):5.1f}%, "
                                       f"no guess {100 * refused / max(1, len(cells_)):5.1f}%")
        caps = {cap for r in attacked for cap in r.get("paraphrases", {})}
        for cap in caps:
            paras = [r["paraphrases"][cap] for r in attacked if cap in r.get("paraphrases", {})]
            paras = [p for p in paras if p.get("epsilon")]
            if not paras:
                continue
            eps = [max(p["epsilon"].values()) for p in paras]
            lam = [v for p in paras for v in p["lambda_mean"].values() if v is not None]
            costs[f"{labels['dp_fusion']} ab={cap}"] = (
                f"eps {sum(eps) / len(eps):6.2f}, lambda {sum(lam) / len(lam):.3f}"
                + (f", tagged text {mean_cov('x_priv'):4.1f}%" if run == "table1_dpfusion" else ""))
    return cells, costs, counts


def summarize(root: Path, dataset: str) -> str:
    cells, costs, counts = rows_for(root, dataset)
    head = f"== {dataset}: " + ", ".join(
        f"{run} {n}/{EXPECTED[dataset]} items" for run, n in counts.items()) + " =="
    if not cells:
        return head + "\n(no attacks yet)"

    shared = set.intersection(*(set(v) for v in cells.values()))
    n = len(shared)
    lines = [head, f"compared over the {n} attribute attacks every row has",
             f"{'defence':<40} {'top-1 acc':>9} {'95% CI':>7}   cost", "-" * 92]

    def order(label):
        rank = ["No Defense", "Redaction: spaCy", "Redaction: Presidio", "DP-Fusion (Presidio",
                "RPS", "TRACE@", "TRACE-RPS@", "Redaction: TRACE", "TRACE x DP-Fusion"]
        return (next(i for i, p in enumerate(rank) if label.startswith(p)), label)

    for label in sorted(cells, key=order):
        k = sum(cells[label][key] for key in shared)
        lines.append(f"{label:<40} {100 * k / n:8.2f}% {ci95(k, n):7.2f}   {costs.get(label, '')}")

    lines += ["", "paper, Llama3.1-8B-Instruct column (context, not like-for-like):"]
    lines += [f"  {name:<28} {v}" for name, v in PAPER[dataset].items()]
    return "\n".join(lines)


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    print(summarize(root, "synthetic"))
    print()
    print(summarize(root, "synthpai"))
