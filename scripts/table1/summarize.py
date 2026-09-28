"""
Collect the Table 1 runs into one table per dataset, next to the paper's numbers.

    python scripts/table1/summarize.py [results] [--floor results/diag_floor]

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

--floor DIR reads a fusit.floor run and adds what the attacker scores with the text's
information gone, writing four files into DIR:

    per_attribute.{csv,md}   accuracy per attribute of (a) the empty comments slot, (b) every
                             word masked, their max (the floor), (c) two label-prior answers
                             that need no model, and every Table 1 row; a row below its
                             attribute's floor is marked
    table1_norm.{csv,md}     Table 1 with norm_reduction = (Acc_ND - Acc) / (Acc_ND - Acc_floor),
                             Acc_floor = max(Acc_empty, Acc_masked) over the same attacks

(c) answers one constant per attribute, fitted on the whole dataset's labels (in-sample):
`majority_label` is the most frequent ground-truth string; `majority_scored` is the label
value the scorer credits most often, which differs because the scorer folds labels together
(single/widowed, every bachelor's degree, ages within 5). Ties go to the first in sorted order.
"No guess" is an attack whose guess list parsed empty, as in the RPS rows' cost; a top-1 that
names no value ("Unknown") is counted apart as non-committal (fusit.floor.NONCOMMITTAL_RE).
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
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


def order(label):
    rank = ["No Defense", "Redaction: spaCy", "Redaction: Presidio", "DP-Fusion (Presidio",
            "RPS", "TRACE@", "TRACE-RPS@", "Redaction: TRACE", "TRACE x DP-Fusion"]
    return (next(i for i, p in enumerate(rank) if label.startswith(p)), label)


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

    for label in sorted(cells, key=order):
        k = sum(cells[label][key] for key in shared)
        lines.append(f"{label:<40} {100 * k / n:8.2f}% {ci95(k, n):7.2f}   {costs.get(label, '')}")

    lines += ["", "paper, Llama3.1-8B-Instruct column (context, not like-for-like):"]
    lines += [f"  {name:<28} {v}" for name, v in PAPER[dataset].items()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# --floor: the attacker without the text
# ---------------------------------------------------------------------------

FLOOR_CONDITIONS = {"empty": "(a) empty text", "masked": "(b) all words masked"}
FLOOR_ROW = "floor = max(a, b)"
PRIORS = {"majority_label": "(c) majority label", "majority_scored": "(c) majority, scorer's classes"}


def majority_answers(dataset: str) -> dict:
    """{attribute: {"majority_label": value, "majority_scored": value}}, fitted on every label
    of the dataset. max() keeps the first of equals, so ties go to the first in sorted order."""
    from fusit.dataset import get_dataset
    labels = defaultdict(list)
    for item in get_dataset(dataset).items():
        for attribute, value in item.relevant_pii.items():
            labels[attribute].append(str(value))
    out = {}
    for attribute, values in labels.items():
        counts = Counter(values)
        candidates = sorted(counts)
        credited = {g: sum(score_hits(attribute, [g], t)["top1"] for t in values) for g in candidates}
        out[attribute] = {"majority_label": max(candidates, key=counts.get),
                          "majority_scored": max(candidates, key=credited.get)}
    return out


def floor_report(root: Path, floor_dir: Path, dataset: str):
    """-> (per-attribute rows, Table 1 rows with norm_reduction, markdown), or None when either
    side has no attacks yet. Every number is over the attacks every Table 1 row and both floor
    conditions have."""
    from fusit.floor import is_noncommittal

    cells, costs, _ = rows_for(root, dataset)
    floor_records = load(floor_dir.parent, floor_dir.name, dataset)
    fcells, refused = defaultdict(dict), defaultdict(dict)
    for r in floor_records.values():
        for attribute, conds in r["attack"].items():
            for cond, label in FLOOR_CONDITIONS.items():
                if cond in conds:
                    guesses, key = conds[cond]["guesses"], (r["item"], attribute)
                    fcells[label][key] = score_hits(attribute, guesses, r["truth"][attribute])["top1"]
                    refused[label][key] = (not guesses, is_noncommittal(guesses))
    if not cells or len(fcells) < len(FLOOR_CONDITIONS):
        return None

    shared = set.intersection(*(set(v) for v in [*cells.values(), *fcells.values()]))
    truth = {(r["item"], a): v for r in floor_records.values() for a, v in r["truth"].items()}
    answers = majority_answers(dataset)
    pcells = {PRIORS[name]: {k: score_hits(k[1], [answers[k[1]][name]], truth[k])["top1"] for k in shared}
              for name in PRIORS}
    groups = {a: sorted(k for k in shared if k[1] == a) for a in sorted({a for _, a in shared})}
    groups["overall"] = sorted(shared)

    def acc(c, keys):
        return 100 * sum(c[k] for k in keys) / len(keys)

    empty, masked = (fcells[label] for label in FLOOR_CONDITIONS.values())
    floor = {g: max(acc(empty, keys), acc(masked, keys)) for g, keys in groups.items()}
    methods = sorted(cells, key=order)

    # per attribute, long form
    per_attr = []
    for g, keys in groups.items():
        base = {"dataset": dataset, "attribute": g, "n": len(keys), "floor": round(floor[g], 2)}
        for label, c in fcells.items():
            per_attr.append({**base, "row": label, "kind": "floor", "acc": round(acc(c, keys), 2),
                             "no_guess_pct": round(100 * sum(refused[label][k][0] for k in keys) / len(keys), 2),
                             "noncommittal_pct": round(100 * sum(refused[label][k][1] for k in keys) / len(keys), 2)})
        per_attr.append({**base, "row": FLOOR_ROW, "kind": "floor", "acc": round(floor[g], 2)})
        for name, label in PRIORS.items():
            per_attr.append({**base, "row": label, "kind": "prior", "acc": round(acc(pcells[label], keys), 2),
                             "answer": answers[g][name] if g != "overall" else ""})
        for label in methods:
            a = acc(cells[label], keys)
            per_attr.append({**base, "row": label, "kind": "method", "acc": round(a, 2),
                             "below_floor": int(a < floor[g])})

    # Table 1 with norm_reduction
    n = len(shared)
    nd = acc(cells["No Defense"], shared) if "No Defense" in cells else float("nan")
    a_empty, a_masked = acc(empty, shared), acc(masked, shared)
    table = []
    for label in methods:
        k = sum(cells[label][key] for key in shared)
        a = 100 * k / n
        denom = nd - floor["overall"]
        table.append({"dataset": dataset, "method": label, "n": n, "acc": round(a, 2),
                      "ci95": round(ci95(k, n), 2), "acc_no_defense": round(nd, 2),
                      "acc_floor_empty": round(a_empty, 2), "acc_floor_masked": round(a_masked, 2),
                      "acc_floor": round(floor["overall"], 2),
                      "norm_reduction": round((nd - a) / denom, 4) if denom > 0 else float("nan"),
                      "cost": costs.get(label, "").strip()})

    # markdown
    cols = list(groups)
    head = "| row | " + " | ".join(f"{g} (n={len(groups[g])})" for g in cols) + " |"
    rule = "|---|" + "---:|" * len(cols)
    by = {(r["row"], r["attribute"]): r for r in per_attr}
    lines = [f"## {dataset}", "",
             f"Top-1 accuracy (%) over the {n} attribute attacks every row has. "
             "`▼` marks a Table 1 row below its attribute's floor.", "", head, rule]
    for row in [*FLOOR_CONDITIONS.values(), FLOOR_ROW, *PRIORS.values(), *methods]:
        vals = []
        for g in cols:
            r = by[(row, g)]
            vals.append(f"{r['acc']:.2f}" + (" ▼" if r.get("below_floor") else ""))
        name = f"**{row}**" if row == FLOOR_ROW else row
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    for label in FLOOR_CONDITIONS.values():
        for field, what in (("no_guess_pct", "no guess %"), ("noncommittal_pct", "non-committal top-1 %")):
            lines.append(f"| {label}: {what} | " + " | ".join(f"{by[(label, g)][field]:.1f}" for g in cols) + " |")
    lines += ["", "(c) answers, fitted on every label of the dataset:", "",
              "| attribute | majority label | majority, scorer's classes |", "|---|---|---|"]
    lines += [f"| {a} | {answers[a]['majority_label']} | {answers[a]['majority_scored']} |"
              for a in cols if a != "overall"]
    flagged = [r for r in per_attr if r.get("below_floor")]
    lines += ["", f"Table 1 rows below their attribute's floor: {len(flagged)}", ""]
    lines += [f"- {r['attribute']}: {r['row']} {r['acc']:.2f}% < floor {r['floor']:.2f}%" for r in flagged]
    per_attr_md = "\n".join(lines)

    lines = [f"## {dataset}", "",
             f"Over {n} attribute attacks. Acc_ND {nd:.2f}%, Acc_empty {a_empty:.2f}%, "
             f"Acc_masked {a_masked:.2f}%, Acc_floor {floor['overall']:.2f}%. "
             "norm_reduction = (Acc_ND - Acc) / (Acc_ND - Acc_floor).", "",
             "| defence | top-1 acc | 95% CI | norm_reduction | cost |", "|---|---:|---:|---:|---|"]
    lines += [f"| {r['method']} | {r['acc']:.2f} | {r['ci95']:.2f} | {r['norm_reduction']:.3f} | {r['cost']} |"
              for r in table]
    return per_attr, table, per_attr_md, "\n".join(lines)


def write_csv(path: Path, rows) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("root", nargs="?", type=Path, default=Path("results"))
    parser.add_argument("--floor", type=Path, default=None, metavar="DIR",
                        help="a fusit.floor run; adds the floor tables and writes them into DIR")
    args = parser.parse_args()
    print(summarize(args.root, "synthetic"))
    print()
    print(summarize(args.root, "synthpai"))

    if args.floor is not None:
        reports = {ds: floor_report(args.root, args.floor, ds) for ds in ("synthetic", "synthpai")}
        reports = {ds: r for ds, r in reports.items() if r is not None}
        if not reports:
            raise SystemExit(f"no floor attacks under {args.floor}")
        per_attr_md = "# Accuracy per attribute, floor conditions and Table 1 rows\n\n" + \
            "\n\n".join(r[2] for r in reports.values()) + "\n"
        norm_md = "# Table 1 with norm_reduction\n\n" + "\n\n".join(r[3] for r in reports.values()) + "\n"
        write_csv(args.floor / "per_attribute.csv", [row for r in reports.values() for row in r[0]])
        write_csv(args.floor / "table1_norm.csv", [row for r in reports.values() for row in r[1]])
        (args.floor / "per_attribute.md").write_text(per_attr_md)
        (args.floor / "table1_norm.md").write_text(norm_md)
        print()
        print(norm_md)
        print(per_attr_md)
        print(f"wrote {', '.join(str(args.floor / f) for f in ('per_attribute.csv', 'per_attribute.md', 'table1_norm.csv', 'table1_norm.md'))}")
