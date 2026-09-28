"""
TRACE x DP-Fusion, end to end, from the command line.

    python -m fusit.main --dataset synthetic --n-items 20 --alpha-beta 0.01 0.1 1.0

Three stages, each with its own model, run over every selected item before the next stage
starts:

    1. tag         the cue tagger marks what leaks each attribute: NER(D) u V_cot u V_att
    2. paraphrase  DP-Fusion rewrites the document once per alpha*beta, one privacy group
                   per cue source
    3. attack      an attacker tries to recover each attribute from the original, from
                   redacted copies, and from every paraphrase

Staging by model rather than by item is what makes this fit on one card: three 7-8B models
do not fit on 24 GB together, so each stage loads its model, runs all items, and frees it.
It also means a crash in stage 3 does not cost the generation already done, since every
record is written once its stage completes.

Results go to a JSONL file, one line per item, carrying the raw spans, paraphrases and
attacker guesses as well as the scores, so a different scorer can be applied later without
rerunning any model. Scoring is TRACE-RPS's (`fusit.trace.scoring`): top-1/2/3 with age
ranges, category synonyms and fuzzy matching on free text, which is what makes accuracies
comparable to its Table 1.

For cluster runs, `--num-shards/--shard-index` split the selection round-robin, and the
output file doubles as a checkpoint: records are rewritten after every item, and on restart
each stage skips the items it already finished.

This implements the pipeline in scripts/pipeline_prototype.ipynb; that notebook explains
the design choices step by step.
"""

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from fusit.dataset import Span, get_dataset
from fusit.dp_fusion import build_contexts, dp_fusion_groups_incremental
from fusit.trace import SOURCES, CueTagger, guess_attribute, merge_spans, redact, score_hits
from fusit.trace.ner import BACKENDS

#: Corpora whose items carry attribute ground truth. TAB-ECHR marks private *spans* instead,
#: so there is no attribute for the attacker to recover.
PIPELINE_DATASETS = ("synthetic", "synthpai")

#: Attack conditions, by kind. `dp_fusion` expands to one condition per --alpha-beta value.
CONDITION_KINDS = ("no_defense", "ner_redaction", "x_priv_redaction", "dp_fusion")

#: Sources that need the tagger LLM; `ner` alone runs without loading it.
LLM_SOURCES = {"cot", "att"}

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


# ---------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m fusit.main",
        description="Tag implicit-PII cues, paraphrase with DP-Fusion, then attack the result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = p.add_argument_group("data")
    data.add_argument("--dataset", choices=PIPELINE_DATASETS, default="synthetic")
    data.add_argument("--n-items", type=int, default=1,
                      help="how many items to sample; 0 means the whole corpus")
    data.add_argument("--seed", type=int, default=0, help="sampling seed")
    data.add_argument("--item-ids", nargs="+", default=None, metavar="ID",
                      help="run exactly these usernames instead of sampling")

    models = p.add_argument_group("models")
    models.add_argument("--tagger-model", default="NousResearch/Llama-2-7b-chat-hf",
                        help="attention (V_att) and inference chain (V_cot)")
    models.add_argument("--paraphrase-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="DP-Fusion generation")
    models.add_argument("--attacker-model", default="NousResearch/Meta-Llama-3.1-8B-Instruct",
                        help="attribute inference against every output")
    models.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    models.add_argument("--device", default="cuda:0")

    tag = p.add_argument_group("cue tagger")
    tag.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES),
                     help="signal sources forming X_priv")
    tag.add_argument("--k-att", type=int, default=10, help="top-K attention words per attribute")
    tag.add_argument("--ner-backend", choices=BACKENDS, default="spacy")
    tag.add_argument("--ner-score-threshold", type=float, default=0.0,
                     help="presidio backend only: drop detections scoring below this. DP-Fusion "
                          "Appendix A.17 runs its BERT-NER tagger at 0.5")
    tag.add_argument("--ner-with-dates", action="store_true",
                     help="presidio backend only: union spaCy dates back in")
    tag.add_argument("--precedence", nargs="+", choices=SOURCES, default=["ner", "cot", "att"],
                     help="which source keeps a span two sources both marked")

    dpf = p.add_argument_group("DP-Fusion")
    dpf.add_argument("--alpha-beta", type=float, nargs="+", default=[0.1], metavar="CAP",
                     help="divergence cap(s), i.e. the paper's alpha*beta; one paraphrase each")
    dpf.add_argument("--alpha", type=float, default=2.0, help="Renyi order")
    dpf.add_argument("--delta", type=float, default=1e-3)
    dpf.add_argument("--temperature", type=float, default=1.0)
    dpf.add_argument("--max-new-tokens", type=int, default=0,
                     help="generation budget per paraphrase; 0 sizes it to the document "
                          "(see --length-ratio), since a fixed budget truncates long "
                          "SynthPAI profiles into summaries and confounds the attack")
    dpf.add_argument("--length-ratio", type=float, default=1.0,
                     help="with --max-new-tokens 0: budget = ratio x document tokens")
    dpf.add_argument("--max-new-tokens-cap", type=int, default=2048,
                     help="with --max-new-tokens 0: upper bound on the sized budget")

    run = p.add_argument_group("run")
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0,
                     help="this worker runs every num-shards-th item starting here")
    run.add_argument("--output", type=Path, default=Path("pipeline_results.jsonl"))
    run.add_argument("--attack-conditions", nargs="+", choices=CONDITION_KINDS,
                     default=list(CONDITION_KINDS),
                     help="which outputs to attack; e.g. skip no_defense when another run with "
                          "the same attacker already scored the original text")
    run.add_argument("--skip-attack", action="store_true",
                     help="stop after stage 2; records are written without attack results")

    return p


def validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.alpha <= 1:
        parser.error("--alpha must be > 1")
    if not 0 < args.delta < 1:
        parser.error("--delta must be in (0, 1)")
    if any(c <= 0 for c in args.alpha_beta):
        parser.error("--alpha-beta values must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.max_new_tokens < 0 or args.length_ratio <= 0 or args.max_new_tokens_cap < 1:
        parser.error("generation budget arguments must be positive")
    if args.n_items < 0:
        parser.error("--n-items must be >= 0")
    if sorted(args.precedence) != sorted(set(args.precedence)) or not set(args.sources) <= set(args.precedence):
        parser.error("--precedence must list every source in --sources, each once")
    if args.ner_score_threshold and args.ner_backend != "presidio":
        parser.error("--ner-score-threshold only applies to --ner-backend presidio")
    if args.ner_with_dates and args.ner_backend != "presidio":
        parser.error("--ner-with-dates only applies to --ner-backend presidio")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"--device {args.device} requested but CUDA is not available")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def load_model(model_id: str, dtype: torch.dtype, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device)
    model.eval()
    log(f"loaded {model_id} in {time.perf_counter() - t0:.0f}s")
    return model, tok


def free_model() -> None:
    """Return a stage's GPU memory. Call it after deleting the stage's own references: the
    memory only comes back once nothing holds the weights."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log(f"freed; {torch.cuda.memory_allocated() / 1e9:.2f} GB still allocated")


def partition_by_source(text: str, by_source: Dict[str, List[List[int]]],
                        precedence: Sequence[str]) -> List[Span]:
    """Turn per-source spans into a partition: every private character in exactly one group.

    DP-Fusion's analysis assumes "the tagger assigns every sensitive token to exactly one
    privacy group", and sources overlap, so the earlier source in `precedence` keeps a shared
    character. The group name is the source, which is what lets NER entities, chain evidence
    and attention words carry separate epsilons.
    """
    claimed, out = set(), []
    for src in precedence:
        for s, e in merge_spans(by_source.get(src, [])):
            free = sorted(set(range(s, e)) - claimed)
            if not free:
                continue
            claimed.update(free)
            start = prev = free[0]
            for i in free[1:] + [None]:
                if i is None or i != prev + 1:
                    out.append(Span(start, prev + 1, src, text[start:prev + 1]))
                    start = i
                if i is not None:
                    prev = i
    return sorted(out, key=lambda s: s.start)


def theorem4_epsilon(divergences: Sequence[float], alpha: float, delta: float, m: int) -> float:
    """DP-Fusion Theorem 4 for one group over a T-token transcript.

    `divergences` are what the generation loop logged, capped at alpha*beta; the formula
    takes the paper's beta, hence the division. Computed here directly because the package's
    two accountants disagree about which of those they expect.
    """
    per_step = sum(
        (1 / (alpha - 1)) * math.log((m - 1) / m + (1 / m) * math.exp((alpha - 1) * 4 * (d / alpha)))
        for d in divergences
    )
    return per_step + math.log(1 / delta) / (alpha - 1)


def coverage(text: str, spans: List[List[int]]) -> float:
    return sum(e - s for s, e in merge_spans(spans)) / len(text) if text else 0.0


def select_items(args) -> list:
    ds = get_dataset(args.dataset)
    if args.item_ids:
        by_id = {i.username: i for i in ds.items()}
        missing = [u for u in args.item_ids if u not in by_id]
        if missing:
            raise SystemExit(f"unknown item ids in {args.dataset}: {missing}")
        items = [by_id[u] for u in args.item_ids]
    else:
        items = ds.select(n=args.n_items or None, seed=args.seed)
    # round-robin rather than contiguous blocks: long and short items are then spread
    # across workers instead of one shard inheriting every long profile
    return items[args.shard_index::args.num_shards]


#: Arguments a stage's stored output depends on. Resuming under different values would
#: silently mix two experiments in one file, so a mismatch is refused.
STAGE_CONFIG = {
    "tag": ("tagger_model", "sources", "k_att", "ner_backend", "ner_score_threshold",
            "ner_with_dates"),
    "paraphrase": ("paraphrase_model", "precedence", "alpha", "delta", "temperature",
                   "max_new_tokens", "length_ratio", "max_new_tokens_cap"),
    "attack": ("attacker_model",),
}


def stage_config(args, stage: str) -> dict:
    return {k: getattr(args, k) for k in STAGE_CONFIG[stage]}


def load_records(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    records = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["item"]] = r
    return records


def write_records(path: Path, records: Dict[str, dict]) -> None:
    """Atomic: a worker killed mid-write leaves the previous checkpoint intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in records.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def check_config(rec: dict, args, stage: str) -> None:
    stored = rec.get("config", {}).get(stage)
    if stored is not None and stored != stage_config(args, stage):
        raise SystemExit(
            f"{rec['item']}: stage '{stage}' in the output was produced with {stored}, "
            f"but this run asks for {stage_config(args, stage)}. Use a different --output."
        )


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_tag(args, items, records: Dict[str, dict]) -> None:
    todo = []
    for item in items:
        rec = records.get(item.username)
        if rec is not None:
            check_config(rec, args, "tag")
            continue
        todo.append(item)
    if not todo:
        log("tag: every item already done")
        return

    # ner alone never touches the LLM, so do not spend a GPU load on it
    needs_llm = bool(set(args.sources) & LLM_SOURCES)
    model, tok = load_model(args.tagger_model, DTYPES[args.dtype], args.device) if needs_llm else (None, None)
    ner_options = {}
    if args.ner_backend == "presidio":
        ner_options = {"score_threshold": args.ner_score_threshold, "with_dates": args.ner_with_dates}

    tagger = None
    for n, item in enumerate(todo, 1):
        tagger = CueTagger(model, tok, attributes=list(item.relevant_pii), sources=args.sources,
                           k=args.k_att, ner_backend=args.ner_backend, ner_options=ner_options)
        t0 = time.perf_counter()
        by_source = tagger.explain(item.text)
        x_priv = merge_spans([s for spans in by_source.values() for s in spans])
        records[item.username] = {
            "item": item.username,
            "dataset": args.dataset,
            "text": item.text,
            "truth": dict(item.relevant_pii),
            "config": {"tag": stage_config(args, "tag")},
            "spans": by_source,
            "x_priv": x_priv,
            "coverage": {**{s: coverage(item.text, sp) for s, sp in by_source.items()},
                         "x_priv": coverage(item.text, x_priv)},
        }
        write_records(args.output, records)
        log(f"tag {n}/{len(todo)} {item.username}: X_priv "
            f"{100 * records[item.username]['coverage']['x_priv']:.1f}% ({time.perf_counter() - t0:.1f}s)")

    del tagger, model, tok
    if needs_llm:
        free_model()


def generation_budget(args, tok, text: str) -> int:
    if args.max_new_tokens:
        return args.max_new_tokens
    doc_tokens = len(tok(text, add_special_tokens=False)["input_ids"])
    return max(64, min(args.max_new_tokens_cap, round(args.length_ratio * doc_tokens)))


def stage_paraphrase(args, items, records: Dict[str, dict]) -> None:
    caps = [str(c) for c in args.alpha_beta]
    todo = []
    for item in items:
        rec = records[item.username]
        check_config(rec, args, "paraphrase")
        if not all(c in rec.get("paraphrases", {}) for c in caps):
            todo.append(rec)
    if not todo:
        log("paraphrase: every item already done")
        return

    model, tok = load_model(args.paraphrase_model, DTYPES[args.dtype], args.device)

    for n, rec in enumerate(todo, 1):
        text = rec["text"]
        rec.setdefault("config", {})["paraphrase"] = stage_config(args, "paraphrase")
        rec.setdefault("paraphrases", {})
        partition = partition_by_source(text, rec["spans"], args.precedence)

        # contexts use THIS model's tokenizer -- length alignment holds for one tokenization
        ctx = build_contexts(tok, text, partition, entity_types=args.precedence)
        groups = [k for k in ctx if k != "PUBLIC"]
        rec["groups"] = groups
        if not groups:
            # Nothing tagged. DP-Fusion does not release the document unchanged here -- it
            # still paraphrases, with nothing private to mix in, so every private distribution
            # equals the public one, lambda is 1 and the divergence 0. Reproduce exactly that
            # with a single group identical to PUBLIC; the loop needs at least one private group.
            # With NER alone this is common (spaCy tagged nothing on 66 of 525 synthetic items),
            # so treating it as "attack the original" would overstate the attacker.
            ctx["UNTAGGED"] = list(ctx["PUBLIC"])
            groups = ["UNTAGGED"]
        prompt_len = len(ctx["PUBLIC"])
        budget = generation_budget(args, tok, text)

        for cap in caps:
            if cap in rec["paraphrases"]:
                continue
            io = {k: torch.tensor(v) for k, v in ctx.items()}
            t0 = time.perf_counter()
            _, lambdas, divergences = dp_fusion_groups_incremental(
                token_ids_groups=io, beta_dict={g: float(cap) for g in groups}, alpha=args.alpha,
                model=model, tokenizer=tok, temperature=args.temperature, max_new_tokens=budget,
            )
            # the returned text is the whole PUBLIC sequence, and io is mutated in place
            out = tok.decode(io["PUBLIC"][prompt_len:].tolist(), skip_special_tokens=True)
            rec["paraphrases"][cap] = {
                "text": out,
                "budget": budget,
                # counted off the sequence: the loop logs no divergence for the first token
                "tokens": len(io["PUBLIC"]) - prompt_len,
                "lambda_mean": {g: sum(v) / len(v) if v else None for g, v in lambdas.items()},
                "divergence_max": {g: max(v) if v else None for g, v in divergences.items()},
                "epsilon": {g: theorem4_epsilon(v, args.alpha, args.delta, len(groups))
                            for g, v in divergences.items()},
            }
            write_records(args.output, records)
            log(f"paraphrase {n}/{len(todo)} {rec['item']} cap={cap}: "
                f"{rec['paraphrases'][cap]['tokens']}/{budget} tokens ({time.perf_counter() - t0:.1f}s)")
            del io
            torch.cuda.empty_cache()

    del model, tok
    free_model()


def conditions_for(rec: dict, kinds: Sequence[str] = CONDITION_KINDS) -> Dict[str, Optional[str]]:
    text = rec["text"]
    conds = {}
    if "no_defense" in kinds:
        conds["no_defense"] = text
    if "ner_redaction" in kinds:
        conds["ner_redaction"] = redact(text, rec["spans"].get("ner", []))
    if "x_priv_redaction" in kinds:
        conds["x_priv_redaction"] = redact(text, rec["x_priv"])
    if "dp_fusion" in kinds:
        for cap, para in rec.get("paraphrases", {}).items():
            # records from before untagged items were paraphrased carry text=None
            conds[f"dp_fusion@{cap}"] = text if para.get("text") is None else para["text"]
    return conds


def stage_attack(args, items, records: Dict[str, dict]) -> None:
    todo = []
    for item in items:
        rec = records[item.username]
        check_config(rec, args, "attack")
        wanted = conditions_for(rec, args.attack_conditions)
        done = rec.get("attack", {})
        if any(c not in done.get(a, {}) for a in rec["truth"] for c in wanted):
            todo.append(rec)
    if not todo:
        log("attack: every item already done")
        return

    model, tok = load_model(args.attacker_model, DTYPES[args.dtype], args.device)

    for n, rec in enumerate(todo, 1):
        rec.setdefault("config", {})["attack"] = stage_config(args, "attack")
        attacks = rec.setdefault("attack", {})
        t0 = time.perf_counter()
        for attribute, truth in rec["truth"].items():
            cells = attacks.setdefault(attribute, {})
            for name, txt in conditions_for(rec, args.attack_conditions).items():
                if name in cells:
                    continue
                guesses = (guess_attribute(txt, attribute, model, tok)["guesses"][:3]
                           if txt and txt.strip() else [])
                cells[name] = {"guesses": guesses, **score_hits(attribute, guesses, truth)}
        write_records(args.output, records)
        log(f"attack {n}/{len(todo)} {rec['item']}: {len(rec['truth'])} attribute(s) "
            f"({time.perf_counter() - t0:.1f}s)")

    del model, tok
    free_model()


def summarize(records: Sequence[dict]) -> str:
    """Top-1 accuracy per condition over every attribute attacked -- Table 1's metric."""
    totals, n = {}, 0
    for rec in records:
        for cells in rec.get("attack", {}).values():
            n += 1
            for name, cell in cells.items():
                totals[name] = totals.get(name, 0) + cell["top1"]
    if not n:
        return "no attacks recorded"
    lines = [f"{'condition':<24} top-1 accuracy over {n} attribute(s)"]
    lines += [f"{name:<24} {100 * h / n:6.2f}%" for name, h in totals.items()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    # `kill -USR1 <pid>` dumps every thread's Python stack to stderr without stopping the run
    # -- the only way to see where a cluster worker spends its time when ptrace is disallowed.
    import faulthandler, signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)

    parser = build_parser()
    args = parser.parse_args(argv)
    validate(args, parser)

    items = select_items(args)
    records = load_records(args.output)
    log(f"{len(items)} item(s) from {args.dataset} (shard {args.shard_index}/{args.num_shards}); "
        f"{sum(1 for i in items if i.username in records)} already in {args.output}")

    stage_tag(args, items, records)
    stage_paraphrase(args, items, records)
    if args.skip_attack:
        log("--skip-attack: stopping after stage 2")
        return 0
    stage_attack(args, items, records)

    print(summarize([records[i.username] for i in items]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
