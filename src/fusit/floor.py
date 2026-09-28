"""
The attacker's floor: what it scores from the prompt alone, with the text's information gone.

    python -m fusit.floor --dataset synthetic --n-items 20 --output results/diag_floor_smoke/synthetic/shard_0.jsonl

A defence's accuracy only means something against the accuracy an attacker reaches without
reading anything -- a prior over ages, a lean towards "male" or "middle income". Two LLM
conditions measure that floor with the Table 1 attacker, prompt and scorer unchanged:

    empty    the comments slot left empty; the rest of TRACE-RPS's prompt is kept. The prompt
             then depends on the attribute alone, so under greedy decoding every item gets
             the same answer: it is generated once per attribute and scored against each
             item. `--empty-cache` shares that answer between workers, so shards on
             different GPUs cannot disagree about what a constant condition answered.
    masked   every whitespace-delimited run replaced by "_" with `fusit.trace.redact`, the
             redaction the Table 1 rows use: character length and line structure survive,
             words do not. Under the Llama-3.1 tokenizer this keeps the token count within
             0.8-1.2x of the original (median 0.94 Synthetic, 1.04 SynthPAI); each record
             carries both counts.
    no_defense  the original text, as `fusit.main` attacks it. Not part of the floor: run it
             on a few items to check this module reproduces the stored Table 1 guesses.

The label-prior baseline that needs no model (answer the most frequent value every time) is
computed from the dataset in scripts/table1/summarize.py, next to these.

Output is a JSONL checkpoint like `fusit.main`'s: records are rewritten after every item and
a restart skips the cells already stored. The unparsed answer is kept in every cell so a
refusal can be told from a wrong guess after the fact (`NONCOMMITTAL_RE`). The run's settings
go to config.json beside the output, and a later run into the same directory under different
settings is refused.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

from fusit.main import (DTYPES, PIPELINE_DATASETS, free_model, load_model, load_records, log,
                        select_items, write_records)
from fusit.trace import guess_attribute, redact, score_hits
from fusit.trace.spans import WORD_RE

CONDITIONS = ("empty", "masked", "no_defense")

#: Settings every cell's value depends on; a record made under others is refused.
CELL_CONFIG = ("attacker_model", "dtype")

#: Settings shared by every shard of one run, written to config.json.
RUN_CONFIG = CELL_CONFIG + ("dataset", "n_items", "seed", "item_ids", "num_shards", "conditions",
                            "empty_cache")

#: A top-1 guess that names no value. Parsed guesses like these score 0 like any wrong answer;
#: summarize.py reports them beside the empty-guess rate. On the 20-item smoke run none
#: occurred: Llama-3.1-8B's refusals ("I cannot provide information that could be used to
#: identify an individual", or asking for the comments on the empty prompt) carry no Guess
#: line at all and parse to an empty list.
NONCOMMITTAL_RE = re.compile(
    r"\b(unknown|unclear|undetermined|indeterminate|not (?:specified|mentioned|determinable|"
    r"available|applicable|enough|possible|provided)|cannot|can't|unable|no information|"
    r"insufficient|n/?a|none)\b", re.IGNORECASE)


def is_noncommittal(guesses) -> bool:
    return bool(guesses) and bool(NONCOMMITTAL_RE.search(guesses[0]))


def mask_all(text: str) -> str:
    return redact(text, [[m.start(), m.end()] for m in WORD_RE.finditer(text)])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m fusit.floor", description=__doc__.split("\n\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", choices=PIPELINE_DATASETS, default="synthetic")
    p.add_argument("--n-items", type=int, default=1, help="0 means the whole corpus")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--item-ids", nargs="+", default=None, metavar="ID")
    p.add_argument("--attacker-model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["empty", "masked"])
    p.add_argument("--empty-cache", type=Path, default=None,
                   help="JSON file holding the empty-prompt answer per attribute, shared by every "
                        "worker of a run; without it each process generates its own")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("floor_results.jsonl"))
    return p


def git_commit() -> str:
    """HEAD, suffixed "+dirty" when the source tree has uncommitted changes."""
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    try:
        return git("rev-parse", "HEAD") + ("+dirty" if git("status", "--porcelain", "--", "src", "scripts") else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_run_config(args) -> None:
    """config.json beside the output: written by the first shard, checked by every later one."""
    path = args.output.parent / "config.json"
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k in RUN_CONFIG}
    if path.exists():
        stored = json.loads(path.read_text())
        if {k: stored.get(k) for k in RUN_CONFIG} != config:
            raise SystemExit(f"{path} was written for {stored}, this run asks for {config}. "
                             "Use a different output directory.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"module": "fusit.floor", **config, "git_commit": git_commit(),
                               "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2) + "\n")
    try:
        os.link(tmp, path)  # atomic: of several shards starting together, one writes
    except FileExistsError:
        pass
    finally:
        tmp.unlink()
    write_run_config(args)  # a shard that lost the race still checks the winner's settings


def empty_answer(attribute: str, args, model, tok, cache: Dict[str, dict]) -> dict:
    """The empty-prompt answer for `attribute`. With --empty-cache, each attribute's answer is
    published once as `<cache>.<attribute>.lock` (created atomically, so of two workers
    generating it together only the first is kept) and every worker uses that one; the cache
    file itself is a merged view for reading."""
    if attribute in cache:
        return cache[attribute]
    path = args.empty_cache
    lock = None if path is None else path.with_name(f"{path.name}.{attribute}.lock")
    if lock is None or not lock.exists():
        out = guess_attribute("", attribute, model, tok, return_response=True)
        entry = {"guesses": out["guesses"][:3], "certainty": out["certainty"],
                 "response": out["response"], **{k: getattr(args, k) for k in CELL_CONFIG}}
        if lock is None:
            cache[attribute] = entry
            return entry
        lock.parent.mkdir(parents=True, exist_ok=True)
        tmp = lock.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False))
        try:
            os.link(tmp, lock)
        except FileExistsError:
            pass
        finally:
            tmp.unlink()
        merged = {p.name[len(path.name) + 1:-len(".lock")]: json.loads(p.read_text())
                  for p in sorted(path.parent.glob(f"{path.name}.*.lock"))}
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=1) + "\n")
        tmp.replace(path)
    entry = json.loads(lock.read_text())
    if any(entry[k] != getattr(args, k) for k in CELL_CONFIG):
        raise SystemExit(f"{lock} was generated with {({k: entry[k] for k in CELL_CONFIG})}; "
                         "use a different --empty-cache.")
    cache[attribute] = entry
    return entry


def main(argv: Optional[Sequence[str]] = None) -> int:
    import faulthandler, signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)

    args = build_parser().parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    write_run_config(args)

    items = select_items(args)
    records = load_records(args.output)
    cell_config = {k: getattr(args, k) for k in CELL_CONFIG}
    todo = []
    for item in items:
        rec = records.setdefault(item.username, {
            "item": item.username, "dataset": args.dataset, "truth": dict(item.relevant_pii),
            "config": cell_config, "attack": {},
        })
        if rec["config"] != cell_config:
            raise SystemExit(f"{item.username}: output was produced with {rec['config']}, "
                             f"this run asks for {cell_config}. Use a different --output.")
        if any(c not in rec["attack"].get(a, {}) for a in rec["truth"] for c in args.conditions):
            todo.append((item, rec))
    log(f"{len(items)} item(s) from {args.dataset} (shard {args.shard_index}/{args.num_shards}); "
        f"{len(items) - len(todo)} already done in {args.output}")

    if todo:
        model, tok = load_model(args.attacker_model, DTYPES[args.dtype], args.device)
        cache: Dict[str, dict] = {}
        for n, (item, rec) in enumerate(todo, 1):
            t0 = time.perf_counter()
            masked = mask_all(item.text)
            if "masked" in args.conditions and "tokens" not in rec:
                rec["tokens"] = {"original": len(tok(item.text, add_special_tokens=False)["input_ids"]),
                                 "masked": len(tok(masked, add_special_tokens=False)["input_ids"])}
            for attribute, truth in rec["truth"].items():
                cells = rec["attack"].setdefault(attribute, {})
                for cond in args.conditions:
                    if cond in cells:
                        continue
                    if cond == "empty":
                        out = empty_answer(attribute, args, model, tok, cache)
                    else:
                        out = guess_attribute(masked if cond == "masked" else item.text, attribute,
                                              model, tok, return_response=True)
                    guesses = out["guesses"][:3]
                    cells[cond] = {"guesses": guesses, "certainty": out["certainty"],
                                   "response": out["response"], **score_hits(attribute, guesses, truth)}
            write_records(args.output, records)
            log(f"floor {n}/{len(todo)} {item.username}: {len(rec['truth'])} attribute(s) "
                f"({time.perf_counter() - t0:.1f}s)")
        del model, tok
        free_model()

    recs = [records[i.username] for i in items]
    n = sum(len(r["truth"]) for r in recs)
    for c in args.conditions:
        cells = [r["attack"][a][c] for r in recs for a in r["truth"]]
        print(f"{c:<11} top-1 {100 * sum(x['top1'] for x in cells) / n:6.2f}%  "
              f"no guess {100 * sum(not x['guesses'] for x in cells) / n:5.1f}%  "
              f"non-committal {100 * sum(is_noncommittal(x['guesses']) for x in cells) / n:5.1f}%  "
              f"over {n} attribute(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
