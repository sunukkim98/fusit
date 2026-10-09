"""
TRACE x DP-Fusion, end to end, from the command line.

    python -m fusit.main --dataset synthetic --n-items 20 --alpha-beta 0.01 0.1 1.0

Three stages, each with its own model, run over every selected item before the next stage
starts:

    1. tag         the cue tagger marks what leaks each attribute: NER(D) u V_cot u V_att
       verify      (--verify) the verifier scores every V_cot word; --decide then sets the
                   thresholds over all shards and writes the X_priv conditions
       trace       (--trace) TRACE@1: one TRACE rewrite from the tag stage's cues
    2. paraphrase  DP-Fusion rewrites the document once per alpha*beta, one privacy group
                   per cue source (--grouping single: one group holding all of X_priv)
    3. attack      the result tables' attack (TRACE-RPS Table 1's protocol, fusit.trace) on every
                   released text: the original, "_"-masked X_priv conditions, the paraphrases,
                   TRACE@1; answers are stored raw and scored by --summarize
       judge       (--judge) Staab et al.'s LLM-as-judge utility and text similarity (fusit.utility)
       ppl         (--ppl) perplexity of D, mechanism and release (fusit.perplexity)
    --summarize    the result table over every shard

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
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from fusit.dataset import Span, get_dataset
from fusit.dp_fusion import (aligned_token_ids, build_contexts, dp_fusion_groups_incremental, epsilon_single_group,
                             generate_single_group, official_prompt)
from fusit.trace import SOURCES, CueTagger, merge_spans
from fusit.trace.ner import BACKENDS
from fusit.utils import seed_for

#: Corpora whose items carry attribute ground truth. TAB-ECHR marks private *spans* instead,
#: so there is no attribute for the attacker to recover.
PIPELINE_DATASETS = ("synthetic", "synthpai", "tab")

#: TAB (the explicit experiment) has no attribute labels: the cue tagger runs every attribute
#: (decision X5, 2026-10-04), in the evaluation attack's order.
TAB_ATTRIBUTES = ("income_level", "age", "sex", "education", "relationship_status",
                  "occupation", "city_country", "birth_city_country")

#: Released texts the evaluation scores, by kind (`released_texts`): the original, the "_"-masked
#: copies of the --mask-conditions X_priv conditions, every DP-Fusion paraphrase, TRACE@1.
CONDITION_KINDS = ("no_defense", "mask", "dp_fusion", "trace", "aa", "rps")

#: The texts RPS defends (--rps-bases): the original (the RPS row), TRACE@1 (TRACE-RPS@1) and every
#: DP-Fusion paraphrase (+ RPS).
RPS_BASES = ("no_defense", "trace", "dp_fusion")

#: The X_priv conditions masked by default: NER (Presidio), X_priv, X_priv verified @95.
MASK_CONDITIONS = ("ner_only", "ner_both", "ner_att_vcot95")

#: The evaluation attack's generation budget (TRACE-RPS's llama3_8b configs).
ATTACK_MAX_NEW = {"synthpai": 1000, "synthetic": 500}
ATTACK_TEMPLATES = ("author_llama3", "chat", "chat_nosystem", "deepseek_r1")
#: a chat-template attack whose prompt leaves fewer new tokens than this in the attacker's context
#: is not run (status context_overflow), as before
ATTACK_MIN_NEW = 200

#: Sources that need the tagger LLM; `ner` alone runs without loading it.
LLM_SOURCES = {"cot", "att"}

#: How X_priv becomes DP-Fusion privacy groups (--grouping).
GROUPINGS = ("source", "single")

#: The one group --grouping single builds.
SINGLE_GROUP = "x_priv"

#: Ground-truth labels: fusit.dataset's own, or TRACE-RPS's attack/eval labels (decision A6).
LABELS = ("fusit", "tracerps")

#: Stages, in order; --stop-after ends the run after one of them.
STAGES = ("tag", "verify", "trace", "aa", "paraphrase", "rps", "attack", "judge", "ppl")

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

    p.add_argument("--config", type=Path, default=None, metavar="YAML",
                   help="experiment config (configs/*.yaml, fusit.config): its `run` section sets these options' "
                        "defaults -- the command line still wins -- and its `xpriv` section what --decide builds")

    data = p.add_argument_group("data")
    data.add_argument("--dataset", choices=PIPELINE_DATASETS, default="synthetic")
    data.add_argument("--n-items", type=int, default=1,
                      help="how many items to sample; 0 means the whole corpus")
    data.add_argument("--seed", type=int, default=0, help="sampling seed")
    data.add_argument("--item-ids", nargs="+", default=None, metavar="ID",
                      help="run exactly these usernames instead of sampling")
    data.add_argument("--labels", choices=LABELS, default="fusit",
                      help="ground truth: fusit.dataset's, or TRACE-RPS's attack/eval labels (the "
                           "result tables', decision A6)")

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
    tag.add_argument("--ner-backend", choices=tuple(BACKENDS) + ("gold",), default="spacy",
                     help="gold: the dataset's own annotated spans (TAB only; decision X4)")
    tag.add_argument("--ner-score-threshold", type=float, default=0.0,
                     help="presidio backend only: drop detections scoring below this. DP-Fusion "
                          "Appendix A.17 runs its BERT-NER tagger at 0.5")
    tag.add_argument("--ner-with-dates", action="store_true",
                     help="presidio backend only: union spaCy dates back in")
    tag.add_argument("--precedence", nargs="+", choices=SOURCES, default=["ner", "cot", "att"],
                     help="which source keeps a span two sources both marked")
    tag.add_argument("--cue-strip-system", action="store_true",
                     help="trim the system prompt of the V_cot guess / chain calls, as TRACE-RPS's Llama-2 "
                          "template does (decision (b))")
    tag.add_argument("--quote-match", choices=("exact", "fuzzy"), default="exact",
                     help="locating V_cot quotes in D: exact = case-sensitive verbatim; fuzzy = "
                          "case-insensitive, else rapidfuzz partial_ratio >= --fuzzy-threshold (decision (a))")
    tag.add_argument("--fuzzy-threshold", type=float, default=90)
    tag.add_argument("--chain-max-new", type=int, default=500, help="V_cot chain generation budget (TRACE: 500)")
    tag.add_argument("--fit-context", action="store_true",
                     help="never cut a document in the V_cot calls: a shorter guess budget, and the chain run "
                          "per line-bounded piece when it would not fit (decision T8)")
    tag.add_argument("--tags-from", type=Path, nargs="+", default=None, metavar="JSONL",
                     help="take stage 1 from these earlier outputs instead of rerunning the "
                          "tagger; their tag config must match this run's")

    ver = p.add_argument_group("verifier")
    ver.add_argument("--verify", action="store_true",
                     help="stage 2: score every V_cot word unit and the null words (fusit.verifier.score)")
    ver.add_argument("--verifier-model", default="models/Llama-2-7b-chat-hf",
                     help="the verifier's attacker (p(value | text) by candidate log-likelihood)")
    ver.add_argument("--verify-att", action="store_true",
                     help="also score the V_att words as units (2026-10-06); on an output verified without them "
                          "only the V_att units are scored and added, so thresholds and V_cot decisions stay")
    ver.add_argument("--decide", type=Path, nargs="+", default=None, metavar="JSONL",
                     help="no stage runs: pool the verify scores of these outputs (every shard of a "
                          "dataset), set the null thresholds and decisions, and write each record's "
                          "X_priv conditions (fusit.verifier.decide / xpriv) back into its file")

    tr = p.add_argument_group("TRACE@1")
    tr.add_argument("--trace", action="store_true",
                    help="TRACE as a defense, one round: the anonymizer rewrites D from the tag stage's "
                         "guesses, chains and V_att words (fusit.trace.anonymize, decisions V-j, TJ-1..4)")
    tr.add_argument("--anonymizer-model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="the TRACE@1 rewriter (greedy); also AA@1's")
    tr.add_argument("--aa", action="store_true",
                    help="AA@1 as a defense (2026-10-06): one round of Staab et al.'s adversarial anonymization "
                         "from the tag stage's inferences, AA's own prompt (fusit.trace.anonymize.aa_rewrite)")

    rp = p.add_argument_group("RPS")
    rp.add_argument("--rps", action="store_true",
                    help="RPS (fusit.trace.rps): a refusal suffix optimized on the evaluation prompt for each "
                         "--rps-bases text; the suffixed texts join the evaluation as rps@<base>")
    rp.add_argument("--rps-model", default="NousResearch/Meta-Llama-3.1-8B-Instruct", help="the RPS target")
    rp.add_argument("--rps-bases", nargs="+", choices=RPS_BASES, default=list(RPS_BASES))
    rp.add_argument("--rps-restarts", type=int, default=1, help="decision R1 (official 2)")
    rp.add_argument("--rps-iterations", type=int, default=1000, help="decision R1 (official 10000)")
    rp.add_argument("--rps-skip-rule", choices=("official", "legacy"), default="official", help="decision R3")
    rp.add_argument("--rps-kv-cache", action="store_true", help="decision R4: reuse the prompt prefix (not exact)")
    rp.add_argument("--rps-xpriv-conditions", nargs="+", default=None, metavar="CONDITION",
                    help="give a suffix only to these X_priv conditions' paraphrases (default: every paraphrase)")
    rp.add_argument("--rps-no-think-prefill", dest="rps_think_prefill", action="store_false",
                    help="DeepSeek-R1 targets: drop the \"<think>\\n\" the current chat template opens the answer with "
                         "(templates from before the model card's recommendation did not), so the suffix may make "
                         "the model skip thinking (2026-10-07)")

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
    dpf.add_argument("--grouping", choices=GROUPINGS, default="source",
                     help="source: one private group per cue source, output = mean over groups "
                          "of lambda_i p_i + (1 - lambda_i) p_pub. single: all of X_priv in one "
                          "group, so the private context is the whole document (DP-Fusion "
                          "Appendix A.19)")
    dpf.add_argument("--gen-seed", type=int, default=None,
                     help="reseed sampling before every paraphrase from (seed, item, cap), so a "
                          "paraphrase does not depend on which shard or order produced it; "
                          "unset, the generator is never reseeded")
    dpf.add_argument("--prompt", choices=("fusit", "official"), default="fusit",
                     help="fusit's chat-template prompt, or the official DP-Fusion-DPI template "
                          "(decision D9; --grouping single only)")
    dpf.add_argument("--xpriv-conditions", nargs="+", default=None, metavar="CONDITION",
                     help="paraphrase once per verifier X_priv condition (record['x_priv_conditions'], "
                          "written by --decide) instead of the tag stage's X_priv; keys condition@cap")
    dpf.add_argument("--budget-rule", choices=("fusit", "ceil"), default="fusit",
                     help="fusit: max(64, min(cap, round(ratio x doc tokens))); ceil: "
                          "ceil(ratio x doc tokens), no floor or cap (the result tables, 2026-10-03)")
    dpf.add_argument("--seed-rule", choices=("fusit", "item"), default="fusit",
                     help="fusit: --gen-seed as below; item: seed_for(dataset, item, condition, cap)")
    dpf.add_argument("--save-token-trace", action="store_true",
                     help="store every step's lambda and divergence per group, not only their "
                          "mean, max and the epsilon they give")

    run = p.add_argument_group("run")
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0,
                     help="this worker runs every num-shards-th item starting here")
    run.add_argument("--output", type=Path, default=Path("pipeline_results.jsonl"))
    run.add_argument("--attack-conditions", nargs="+", choices=CONDITION_KINDS,
                     default=list(CONDITION_KINDS),
                     help="which released texts the evaluation (attack, judge, PPL) scores")
    run.add_argument("--mask-conditions", nargs="+", default=list(MASK_CONDITIONS), metavar="CONDITION",
                     help="X_priv conditions (written by --decide) evaluated as '_'-masked texts")
    run.add_argument("--skip-attack", action="store_true",
                     help="stop before the attack; records are written without attack results")
    ev = p.add_argument_group("evaluation (the result tables' protocol)")
    ev.add_argument("--attack-max-batch-tokens", type=int, default=20000,
                    help="padded tokens per attack batch (prompt + new tokens); fixes the batch layout")
    ev.add_argument("--attack-template", choices=ATTACK_TEMPLATES, default="author_llama3",
                    help="how the attack prompt is rendered: author_llama3 = the result tables' Llama-3 template "
                         "(the attacker must be Llama-3 Instruct); chat = the attacker's own chat template; "
                         "chat_nosystem = the same with the system prompt folded into the user turn, as "
                         "DeepSeek-R1 asks; deepseek_r1 = TRACE-RPS's DeepSeek-R1-Distill template (user turn only, "
                         "answer opened with <think>). Except author_llama3, the new-token budget is cut to the attacker's context "
                         "(Llama-2: 4096) and a reasoning model's <think> block is dropped at scoring (2026-10-07)")
    ev.add_argument("--attack-max-new", type=int, default=0,
                    help="new-token budget of every attack answer; 0 = the result tables' "
                         "(Synthetic 500, SynthPAI 1000). A reasoning attacker needs room for its <think> block")
    ev.add_argument("--judge", action="store_true",
                    help="LLM-as-judge utility (Staab et al.) and text similarity of every released text")
    ev.add_argument("--judge-model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ev.add_argument("--judge-max-batch-tokens", type=int, default=16000, help="decision E2")
    ev.add_argument("--sbert-model", default="sentence-transformers/paraphrase-MiniLM-L6-v2",
                    help="TRACE-RPS's SBERT similarity (decision E3); 'none' skips it")
    ev.add_argument("--ppl", action="store_true",
                    help="perplexity of D: mechanism (what each method shows the model) and release")
    ev.add_argument("--ppl-model", default="Qwen/Qwen2.5-7B-Instruct")
    ev.add_argument("--summarize", type=Path, nargs="+", default=None, metavar="JSONL",
                    help="no stage runs: the result table over these outputs (every shard of a dataset)")
    run.add_argument("--stop-after", choices=STAGES, default=None,
                     help="end the run after this stage (e.g. tag + verify, then --decide)")

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
    if (args.ner_backend == "gold") != (args.dataset == "tab") and not (args.decide or args.summarize):
        parser.error("--ner-backend gold is TAB's (its annotated spans), and TAB runs with it")
    if args.ner_with_dates and args.ner_backend != "presidio":
        parser.error("--ner-with-dates only applies to --ner-backend presidio")
    if args.tags_from and not all(p.exists() for p in args.tags_from):
        parser.error(f"--tags-from: missing {[str(p) for p in args.tags_from if not p.exists()]}")
    if args.prompt == "official" and args.grouping != "single":
        parser.error("--prompt official needs --grouping single")
    # --decide / --summarize run no model
    if args.device.startswith("cuda") and not torch.cuda.is_available() and not (args.decide or args.summarize):
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


def dp_fusion_contexts(tok, rec: dict, grouping: str, precedence: Sequence[str]) -> Dict[str, List[int]]:
    """The length-matched contexts one item is paraphrased from, PUBLIC first.

    Both groupings hide the same characters -- all of X_priv -- from PUBLIC; they differ in
    what a private context reveals. `source` gives each cue source its own context revealing
    only that source's spans, so no single distribution sees X_priv whole. `single` has one
    context revealing all of it, which is the unredacted prompt.
    """
    text = rec["text"]
    if grouping == "single":
        spans = [Span(s, e, SINGLE_GROUP, text[s:e]) for s, e in merge_spans(rec["x_priv"])]
        return build_contexts(tok, text, spans, entity_types=[SINGLE_GROUP])
    partition = partition_by_source(text, rec["spans"], precedence)
    return build_contexts(tok, text, partition, entity_types=precedence)


def generation_seed(seed: int, item: str, cap: str) -> int:
    """A fixed seed per paraphrase, stable across processes (unlike `hash`)."""
    return zlib.crc32(f"{seed}:{item}:{cap}".encode())


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
    ds = get_dataset(args.dataset if args.labels == "fusit" or args.dataset == "tab" else f"{args.dataset}_tracerps")
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
    "tag": ("tagger_model", "sources", "k_att", "ner_backend", "ner_with_dates"),
    "verify": ("verifier_model", "verify_att"),
    "trace": ("anonymizer_model",),
    "aa": ("anonymizer_model",),
    "rps": ("rps_model", "rps_restarts", "rps_iterations", "rps_skip_rule", "rps_kv_cache"),
    "paraphrase": ("paraphrase_model", "precedence", "alpha", "delta", "temperature",
                   "max_new_tokens", "length_ratio", "max_new_tokens_cap"),
    "attack": ("attacker_model",),
    "judge": ("judge_model",),
    "ppl": ("ppl_model",),
}


#: Paraphrase arguments recorded only when set away from their default, so records written
#: before they existed still match a default run.
#: results/table1 predates --ner-score-threshold, so its tag config lacks that key.
OPTIONAL_CONFIG = {
    "tag": {"ner_score_threshold": 0.0, "labels": "fusit", "cue_strip_system": False, "quote_match": "exact",
            "fuzzy_threshold": 90, "chain_max_new": 500, "fit_context": False},
    "verify": {"verify_att": False},
    "rps": {"rps_think_prefill": True},
    "attack": {"attack_template": "author_llama3", "attack_max_new": 0},
    "paraphrase": {"grouping": "source", "gen_seed": None, "prompt": "fusit", "budget_rule": "fusit",
                   "seed_rule": "fusit", "xpriv_conditions": None},
}


def without_defaults(config: dict, stage: str) -> dict:
    defaults = OPTIONAL_CONFIG.get(stage, {})
    return {k: v for k, v in config.items() if not (k in defaults and v == defaults[k])}


def stage_config(args, stage: str) -> dict:
    keys = [*STAGE_CONFIG[stage], *OPTIONAL_CONFIG.get(stage, {})]
    return without_defaults({k: getattr(args, k) for k in keys}, stage)


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
    # which X_priv conditions to paraphrase does not change how any one is made, so conditions may be
    # added to an output later (2026-10-04)
    drop = lambda c: {k: v for k, v in c.items() if k != "xpriv_conditions"}  # noqa: E731
    if stored is not None and drop(without_defaults(stored, stage)) != drop(stage_config(args, stage)):
        raise SystemExit(
            f"{rec['item']}: stage '{stage}' in the output was produced with {stored}, "
            f"but this run asks for {stage_config(args, stage)}. Use a different --output."
        )


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

#: What stage 1 writes, and all --tags-from copies.
TAG_FIELDS = ("item", "dataset", "text", "truth", "spans", "x_priv", "coverage", "cues")


def reuse_tags(args, todo: list, records: Dict[str, dict]) -> list:
    """Fill `records` from --tags-from for the items it has; returns the ones it lacks."""
    pool = {}
    for path in args.tags_from:
        pool.update(load_records(path))
    missing = []
    for item in todo:
        old = pool.get(item.username)
        if old is None:
            missing.append(item)
            continue
        check_config(old, args, "tag")
        if old["text"] != item.text:
            raise SystemExit(f"{item.username}: --tags-from text differs from the dataset's")
        records[item.username] = {**{k: old[k] for k in TAG_FIELDS if k in old},
                                  "config": {"tag": old["config"]["tag"]}}
    if len(missing) < len(todo):
        write_records(args.output, records)
        log(f"tag: reused {len(todo) - len(missing)} item(s) from --tags-from")
    return missing


def stage_tag(args, items, records: Dict[str, dict]) -> None:
    todo = []
    for item in items:
        rec = records.get(item.username)
        if rec is not None:
            check_config(rec, args, "tag")
            continue
        todo.append(item)
    if todo and args.tags_from:
        todo = reuse_tags(args, todo, records)
    if not todo:
        log("tag: every item already done")
        return

    # ner alone never touches the LLM, so do not spend a GPU load on it
    needs_llm = bool(set(args.sources) & LLM_SOURCES)
    model, tok = load_model(args.tagger_model, DTYPES[args.dtype], args.device) if needs_llm else (None, None)
    ner_options = {}
    if args.ner_backend == "presidio":
        ner_options = {"score_threshold": args.ner_score_threshold, "with_dates": args.ner_with_dates}

    gold = args.ner_backend == "gold"
    tagger = None
    for n, item in enumerate(todo, 1):
        attributes = list(TAB_ATTRIBUTES) if args.dataset == "tab" else list(item.relevant_pii)
        # gold NER: the tagger runs the other sources; the dataset's spans are added below
        tagger = CueTagger(model, tok, attributes=attributes, sources=[s for s in args.sources if not (gold and s == "ner")],
                           k=args.k_att, ner_backend="spacy" if gold else args.ner_backend, ner_options=ner_options,
                           strip_system=args.cue_strip_system, quote_match=args.quote_match,
                           fuzzy_threshold=args.fuzzy_threshold, chain_max_new=args.chain_max_new,
                           fit_context=args.fit_context)
        t0 = time.perf_counter()
        tagged = tagger.tag(item.text)
        by_source = tagger.spans_by_source(tagged)        # = tagger.explain(item.text)
        if gold and "ner" in args.sources:
            tagged = {"ner": {"backend": "gold", "spans": merge_spans(item.offsets())}, **tagged}
            by_source = {"ner": tagged["ner"]["spans"], **by_source}
        x_priv = merge_spans([s for spans in by_source.values() for s in spans])
        records[item.username] = {
            "item": item.username,
            "dataset": args.dataset,
            "text": item.text,
            "truth": dict(getattr(item, "relevant_pii", {})),
            "config": {"tag": stage_config(args, "tag")},
            "spans": by_source,
            "x_priv": x_priv,
            "coverage": {**{s: coverage(item.text, sp) for s, sp in by_source.items()},
                         "x_priv": coverage(item.text, x_priv)},
            # what each source produced, kept apart: NER spans, and per attribute the V_att words
            # and the V_cot guess / chain / quotes (CueTagger.tag) -- what the verifier reads
            "cues": tagged,
        }
        if args.dataset == "tab":     # typed gold spans: the explicit attack builds its candidates from them
            records[item.username]["entities"] = [[s.start, s.end, s.entity_type, s.text] for s in item.spans]
        write_records(args.output, records)
        log(f"tag {n}/{len(todo)} {item.username}: X_priv "
            f"{100 * records[item.username]['coverage']['x_priv']:.1f}% ({time.perf_counter() - t0:.1f}s)")

    del tagger, model, tok
    if needs_llm:
        free_model()


def stage_verify(args, items, records: Dict[str, dict]) -> None:
    """Stage 2: the verifier's measurements for every V_cot word unit and the null words of each
    item (fusit.verifier.score), from the tag stage's records. Deciding needs the null of every
    item of the dataset, so it is a separate pass over all shards: --decide."""
    from fusit.verifier.inputs import verifier_cues
    from fusit.verifier.score import add_att_units, score_item

    todo, upgrade = [], []
    want = stage_config(args, "verify")
    for item in items:
        rec = records[item.username]
        if "cues" not in rec:
            raise SystemExit(f"{rec['item']}: the tag record carries no cue records (guesses, chains, quotes); "
                             "it was tagged before 2026-10-04 -- tag it again")
        stored = rec.get("config", {}).get("verify")
        if (args.verify_att and stored is not None and "scores" in rec.get("verify", {})
                and {**without_defaults(stored, "verify"), "verify_att": True} == want
                and not stored.get("verify_att")):
            upgrade.append(rec)          # verified without V_att: score only the V_att units (--verify-att)
            continue
        check_config(rec, args, "verify")
        if "scores" not in rec.get("verify", {}):
            todo.append(rec)
    if not todo and not upgrade:
        log("verify: every item already done")
        return
    model, tok = load_model(args.verifier_model, DTYPES[args.dtype], args.device)
    for n, rec in enumerate(todo, 1):
        t0 = time.perf_counter()
        rec.setdefault("config", {})["verify"] = want
        # null words are seeded by the dataset name, not the label set (verifier.score)
        scores = score_item(model, tok, rec["dataset"], rec["item"], rec["text"], verifier_cues(rec["cues"]),
                            rec["truth"], att=args.verify_att)
        rec["verify"] = {"scores": scores}
        write_records(args.output, records)
        log(f"verify {n}/{len(todo)} {rec['item']}: {len(scores['units'])} unit(s), {len(scores['null'])} null "
            f"({time.perf_counter() - t0:.1f}s)")
    for n, rec in enumerate(upgrade, 1):
        t0 = time.perf_counter()
        k = add_att_units(model, tok, rec["verify"]["scores"], rec["text"], verifier_cues(rec["cues"]))
        rec["config"]["verify"] = want
        write_records(args.output, records)
        log(f"verify (V_att) {n}/{len(upgrade)} {rec['item']}: {k} V_att unit(s) added ({time.perf_counter() - t0:.1f}s)")
    del model, tok
    free_model()


def stage_trace(args, items, records: Dict[str, dict]) -> None:
    """TRACE@1: one round of TRACE's rewrite from the tag stage's records (the guesses, chains
    and V_att words the TRACE x DP-Fusion rows use), so the two defenses share their cues."""
    from fusit.trace.anonymize import analysis_from_cues, needs_rewrite, rewrite, rewrite_budget

    todo = []
    for item in items:
        rec = records[item.username]
        if "cues" not in rec:
            raise SystemExit(f"{rec['item']}: the tag record carries no cue records; tag it again")
        check_config(rec, args, "trace")
        if "trace" not in rec:
            todo.append(rec)
    if not todo:
        log("trace: every item already done")
        return
    model, tok = load_model(args.anonymizer_model, DTYPES[args.dtype], args.device)
    for n, rec in enumerate(todo, 1):
        t0 = time.perf_counter()
        rec.setdefault("config", {})["trace"] = stage_config(args, "trace")
        analysis = analysis_from_cues(rec["cues"])
        out = {"round": 1, "attributes": sorted(a for a, v in analysis.items() if "chain" in v),
               "certainty": {a: v["certainty"] for a, v in analysis.items()}}
        if not any(v["guesses"] for v in analysis.values()):
            out.update(stop="no attribute guessed", text=rec["text"], raw=None, parsed=False)
        elif not needs_rewrite(analysis):
            out.update(stop="every certainty <= 2", text=rec["text"], raw=None, parsed=False)
        else:
            budget = rewrite_budget(rec["text"], tok)
            r = rewrite(rec["text"], analysis, model, tok, budget)
            out.update(stop=None, text=r["text"], raw=r["raw"], parsed=r["parsed"], max_new_tokens=budget)
        out["seconds"] = round(time.perf_counter() - t0, 1)
        rec["trace"] = out
        write_records(args.output, records)
        log(f"trace {n}/{len(todo)} {rec['item']}: {len(out['attributes'])} attribute(s), stop={out['stop']}, "
            f"parsed={out['parsed']} ({out['seconds']}s)")
    del model, tok
    free_model()


def stage_aa(args, items, records: Dict[str, dict]) -> None:
    """AA@1 (2026-10-06): one round of Staab et al.'s adversarial anonymization from the tag stage's
    inferences (the same attacker side as TRACE@1 and the TRACE x DP-Fusion rows), with AA's own prompt;
    the rewriter and its generation are TRACE@1's (fusit.trace.anonymize.aa_rewrite, decision AA-1)."""
    from fusit.trace.anonymize import aa_rewrite, analysis_from_cues, rewrite_budget

    todo = []
    for item in items:
        rec = records[item.username]
        if "cues" not in rec:
            raise SystemExit(f"{rec['item']}: the tag record carries no cue records; tag it again")
        check_config(rec, args, "aa")
        if "aa" not in rec:
            todo.append(rec)
    if not todo:
        log("aa: every item already done")
        return
    model, tok = load_model(args.anonymizer_model, DTYPES[args.dtype], args.device)
    for n, rec in enumerate(todo, 1):
        t0 = time.perf_counter()
        rec.setdefault("config", {})["aa"] = stage_config(args, "aa")
        analysis = analysis_from_cues(rec["cues"])
        out = {"round": 1, "attributes": sorted(a for a, v in analysis.items() if v["guesses"])}
        if not out["attributes"]:
            out.update(stop="no attribute guessed", text=rec["text"], raw=None, parsed=False)
        else:
            budget = rewrite_budget(rec["text"], tok)
            r = aa_rewrite(rec["text"], analysis, model, tok, budget)
            out.update(stop=None, text=r["text"], raw=r["raw"], parsed=r["parsed"], max_new_tokens=budget)
        out["seconds"] = round(time.perf_counter() - t0, 1)
        rec["aa"] = out
        write_records(args.output, records)
        log(f"aa {n}/{len(todo)} {rec['item']}: {len(out['attributes'])} attribute(s), stop={out['stop']}, "
            f"parsed={out['parsed']} ({out['seconds']}s)")
    del model, tok
    free_model()


def run_decide(paths: Sequence[Path], spec: Optional[dict] = None) -> None:
    """--decide: decisions over the pooled verify scores of `paths`, and every record's X_priv
    conditions (fusit.verifier.xpriv.build_item), written back into its file. Thresholds are
    per (dataset, attribute), so all shards of a dataset must be given together. `spec` is the
    config's `xpriv` section (which experiment-only conditions to build); None keeps the legacy set."""
    from fusit.verifier.decide import decide_all
    from fusit.verifier.inputs import verifier_cues
    from fusit.verifier.xpriv import build_item

    files = {path: load_records(path) for path in paths}
    rows = {"pairs": [], "units": [], "null": []}
    for recs in files.values():
        for rec in recs.values():
            if "scores" not in rec.get("verify", {}):
                raise SystemExit(f"{rec['item']}: no verify scores -- run stage 2 (--verify) first")
            if (spec or {}).get("verified_att") and not rec.get("config", {}).get("verify", {}).get("verify_att"):
                raise SystemExit(f"{rec['item']}: xpriv.verified_att needs V_att units -- verify with --verify-att first")
            for k in rows:
                rows[k] += rec["verify"]["scores"][k]
    scored, tau = decide_all(rows)
    units = {}
    for u in scored["units"]:
        units.setdefault((u["dataset"], u["item_id"]), []).append(
            {k: u[k] for k in u if k in ("attribute", "u_id", "text", "spans", "source") or k.startswith("decision@")})
    for path, recs in files.items():
        for rec in recs.values():
            us = units.get((rec["dataset"], rec["item"]), [])
            rec["verify"]["decisions"] = us
            built = build_item(rec["dataset"], rec["item"], rec["text"], verifier_cues(rec["cues"]),
                               rec["spans"].get("ner", []), us, spec)
            # conditions an earlier decide built and this spec does not are kept, not dropped: paraphrases
            # and attacks may already have been run on them
            rec["x_priv_conditions"] = {**rec.get("x_priv_conditions", {}), **built}
        write_records(path, recs)
    log(f"decide: {sum(len(r) for r in files.values())} record(s), {len(scored['units'])} unit(s); thresholds "
        + ", ".join(f"{k[1]} {v['cond']:.4f}/{v['alone']:.4f}" for k, v in sorted(tau[95].items())))


def generation_budget(args, tok, text: str) -> int:
    if args.max_new_tokens:
        return args.max_new_tokens
    if args.budget_rule == "ceil":
        # the result tables' rule (verifier/dpfusion.py): the tokenizer's default call
        return math.ceil(args.length_ratio * len(tok(text)["input_ids"]))
    doc_tokens = len(tok(text, add_special_tokens=False)["input_ids"])
    return max(64, min(args.max_new_tokens_cap, round(args.length_ratio * doc_tokens)))


def paraphrase_keys(args) -> List[str]:
    caps = [str(c) for c in args.alpha_beta]
    if args.xpriv_conditions:
        return [f"{cond}@{cap}" for cond in args.xpriv_conditions for cap in caps]
    return caps


def paraphrase_condition(args, model, tok, rec: dict, cond: str, cap: str) -> dict:
    """One DP-Fusion run on a verifier X_priv condition: single group, the official prompt
    (decision D9), the generator of fusit.dp_fusion.run (D1-D4), seeded per (dataset, item,
    condition, cap) -- the result tables' runs (verifier/dpfusion.py), reproduced."""
    text = rec["text"]
    xp = rec["x_priv_conditions"][cond]
    ab = float(cap)
    if args.prompt == "official":
        prompt = official_prompt(tok, text, "_")
    else:
        from fusit.dp_fusion.prompting import format_prompt_new_template
        prompt = format_prompt_new_template(tok, text, "_")
    private, public = aligned_token_ids(tok, prompt, text, xp["x_priv"], "_")
    budget = generation_budget(args, tok, text)
    seed = (seed_for(rec["dataset"], rec["item"], cond, ab) if args.seed_rule == "item"
            else generation_seed(args.gen_seed or 0, rec["item"], cap))
    t0 = time.perf_counter()
    gen, lam, div = generate_single_group(model, tok, private, public, ab, seed, budget,
                                          alpha=args.alpha, temperature=args.temperature)
    eps = epsilon_single_group(div, ab, alpha=args.alpha, delta=args.delta) if div else {}
    return {"condition": cond, "ab": ab, "seed": seed, "prompt_tokens": len(private), "max_new_tokens": budget,
            "n_public_tokens_replaced": sum(a != b for a, b in zip(private, public)), "coverage": xp["coverage"],
            "lambdas": lam, "divergences": div, "eps_data": eps.get("empirical"), "eps_theo": eps.get("theoretical"),
            "eps_T": eps.get("T"), "generated_ids": gen, "text": tok.decode(gen, skip_special_tokens=True),
            "n_generated": len(gen), "hit_eos": bool(gen) and gen[-1] == tok.eos_token_id,
            "seconds": round(time.perf_counter() - t0, 1)}


def stage_paraphrase(args, items, records: Dict[str, dict]) -> None:
    caps = [str(c) for c in args.alpha_beta]
    keys = paraphrase_keys(args)
    todo = []
    for item in items:
        rec = records[item.username]
        check_config(rec, args, "paraphrase")
        if args.xpriv_conditions and "x_priv_conditions" not in rec:
            raise SystemExit(f"{rec['item']}: no X_priv conditions -- run --decide over every shard first")
        if not all(k in rec.get("paraphrases", {}) for k in keys):
            todo.append(rec)
    if not todo:
        log("paraphrase: every item already done")
        return

    model, tok = load_model(args.paraphrase_model, DTYPES[args.dtype], args.device)

    for n, rec in enumerate(todo, 1):
        text = rec["text"]
        rec.setdefault("config", {})["paraphrase"] = stage_config(args, "paraphrase")
        rec.setdefault("paraphrases", {})
        if args.xpriv_conditions:
            for key in keys:
                if key in rec["paraphrases"]:
                    continue
                cond, cap = key.rsplit("@", 1)
                rec["paraphrases"][key] = paraphrase_condition(args, model, tok, rec, cond, cap)
                write_records(args.output, records)
                p = rec["paraphrases"][key]
                log(f"paraphrase {n}/{len(todo)} {rec['item']} {key}: {p['n_generated']}/{p['max_new_tokens']} "
                    f"tokens ({p['seconds']}s)")
                torch.cuda.empty_cache()
            continue

        # contexts use THIS model's tokenizer -- length alignment holds for one tokenization
        ctx = dp_fusion_contexts(tok, rec, args.grouping, args.precedence)
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
            if args.gen_seed is not None:
                torch.manual_seed(generation_seed(args.gen_seed, rec["item"], cap))
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
            if args.save_token_trace:
                # step t here is generated token t+1: the loop logs nothing for the first
                rec["paraphrases"][cap]["trace"] = {"lambda": lambdas, "divergence": divergences}
            write_records(args.output, records)
            log(f"paraphrase {n}/{len(todo)} {rec['item']} cap={cap}: "
                f"{rec['paraphrases'][cap]['tokens']}/{budget} tokens ({time.perf_counter() - t0:.1f}s)")
            del io
            torch.cuda.empty_cache()

    del model, tok
    free_model()


def released_texts(rec: dict, args) -> Dict[str, str]:
    """{condition: released text, plain} -- what the evaluation scores for one record. Names:
    no_defense, mask_<x_priv condition> ("_" for every masked char, line breaks kept),
    dp_fusion@<paraphrase key> (condition@cap), trace (TRACE@1), aa (AA@1)."""
    from fusit.trace.spans import mask_lines
    kinds, text, out = args.attack_conditions, rec["text"], {}
    if "no_defense" in kinds:
        out["no_defense"] = text
    if "mask" in kinds:
        for c in args.mask_conditions:
            if c in rec.get("x_priv_conditions", {}):
                out[f"mask_{c}"] = mask_lines(text, rec["x_priv_conditions"][c]["x_priv"])
    if "dp_fusion" in kinds:
        for key, para in rec.get("paraphrases", {}).items():
            # records from before untagged items were paraphrased carry text=None
            out[f"dp_fusion@{key}"] = text if para.get("text") is None else para["text"]
    if "trace" in kinds and "trace" in rec:
        out["trace"] = rec["trace"]["text"]
    if "aa" in kinds and "aa" in rec:
        out["aa"] = rec["aa"]["text"]
    if "rps" in kinds:
        for name, r in rec.get("rps", {}).items():
            out[name] = r["text"]
    return out


def attack_text(rec: dict, condition: str, text: str) -> str:
    """create_prompts' formatting: Synthetic's date prefix on the original, its masks and TRACE's
    rewrite (line by line); DP-Fusion paraphrases are free text and stay as they are; an RPS text
    is its base so formatted plus the suffix (`rps_attack_text`)."""
    from fusit.trace.prompts import attack_comments
    if condition.startswith("rps@"):
        return rec["rps"][condition]["attack_text"]
    if condition.startswith("dp_fusion@"):
        return text
    return attack_comments(rec["dataset"], text)


def rps_attack_text(rec: dict, base: str, base_attack: str, suffix: str) -> str:
    """How the evaluation shows a base text with its RPS suffix (decision RP2). rps.py publishes
    response + " \\n\\n" + suffix, and TRACE-RPS's Synthetic loader makes every non-empty line of
    that a dated comment -- the suffix becomes "2014-05-19: <suffix>", the very line it was
    optimized as. So for Synthetic bases shown as dated comments (the original, TRACE@1) the suffix
    is added as one more dated line; DP-Fusion paraphrases (no dates) and SynthPAI get the official
    " \\n\\n" + suffix."""
    from fusit.trace.prompts import SYNTHETIC_DATE
    if rec["dataset"] == "synthetic" and not base.startswith("dp_fusion@"):
        return base_attack + "\n" + f"{SYNTHETIC_DATE}: {suffix.strip()}"
    return base_attack + " \n\n" + suffix


def stage_rps(args, items, records: Dict[str, dict]) -> None:
    """RPS (TRACE-RPS rps/rps.py, fusit.trace.rps.search) for every --rps-bases text: the suffix is
    optimized in the evaluation prompt (staab_multi over the item's labelled attributes, as the
    attack sees the base text) with "2014-05-19: {defense}" inserted before "\\nFirst reason
    step-by-step", as rps.py does (decision R5 / RP1). R1 one restart of 1,000 iterations, R3 the
    official skipped-step rule, R4 full-forward probes, by default."""
    from fusit.trace.prompts import staab_multi_messages
    from fusit.trace.rps import SENTINEL, Target, defense_init_for, profile_of, published_text, search

    todo = []
    for item in items:
        rec = records[item.username]
        check_config(rec, args, "rps")
        texts = released_texts(rec, argparse.Namespace(**{**vars(args), "attack_conditions": list(args.rps_bases)}))
        if args.rps_xpriv_conditions:
            # only these X_priv conditions' paraphrases get a suffix (2026-10-07; default: every paraphrase)
            texts = {c: t for c, t in texts.items()
                     if not c.startswith("dp_fusion@") or c.split("@")[1] in args.rps_xpriv_conditions}
        todo += [(rec, c, t) for c, t in texts.items() if f"rps@{c}" not in rec.get("rps", {})]
    if not todo:
        log("rps: every base already defended")
        return
    model, tok = load_model(args.rps_model, DTYPES[args.dtype], args.device)
    target = Target(model, tok, kv_cache=args.rps_kv_cache, profile=profile_of(args.rps_model),
                    think_prefill=args.rps_think_prefill)
    for n, (rec, base, text) in enumerate(todo, 1):
        t0 = time.perf_counter()
        base_attack = attack_text(rec, base, text)
        _, user = staab_multi_messages(base_attack, list(rec["truth"]))
        pos = user.find("\nFirst reason step-by-step")
        if pos < 0:
            raise ValueError("footer anchor not found in the evaluation prompt")
        tmpl = user[:pos] + "2014-05-19: " + SENTINEL + "\n" + user[pos:]
        context = getattr(model.config, "max_position_embeddings", None)
        n_prompt = len(target.ids(tmpl.replace(SENTINEL, defense_init_for(args.rps_model))))
        if context and n_prompt > context - ATTACK_MIN_NEW:
            # Llama-2's 4096 tokens (2026-10-07): the attack cannot answer this prompt either (context_overflow), so
            # no suffix is searched; the base text is published as it is and the record says why
            rec.setdefault("config", {})["rps"] = stage_config(args, "rps")
            rec.setdefault("rps", {})[f"rps@{base}"] = {
                "base": base, "suffix": "", "text": text, "attack_text": base_attack, "success": False,
                "skipped": "context", "n_prompt_tokens": n_prompt, "seconds": 0.0}
            write_records(args.output, records)
            log(f"rps {n}/{len(todo)} {rec['item']} {base}: skipped, prompt {n_prompt} tokens > context {context}")
            continue
        r = search(target, tmpl, n_restarts=args.rps_restarts,
                   n_iterations=args.rps_iterations, skip_rule=args.rps_skip_rule,
                   defense_init=defense_init_for(args.rps_model))
        rec.setdefault("config", {})["rps"] = stage_config(args, "rps")
        rec.setdefault("rps", {})[f"rps@{base}"] = {
            "base": base, "suffix": r["best_defense"], "text": published_text(text, r["best_defense"]),
            "attack_text": rps_attack_text(rec, base, base_attack, r["best_defense"]),
            **{k: r[k] for k in ("success", "iterations", "stage", "best_p_first", "best_p_second", "judge_calls",
                                 "final_response_text", "restarts")},
            "seconds": round(time.perf_counter() - t0, 1)}
        write_records(args.output, records)
        log(f"rps {n}/{len(todo)} {rec['item']} {base}: success={r['success']} it={r['iterations']} "
            f"({time.perf_counter() - t0:.1f}s)")
    del target, model, tok
    free_model()


def stage_attack(args, items, records: Dict[str, dict]) -> None:
    """The result tables' attack (TRACE-RPS Table 1's: staab_multi, author template, greedy,
    1000 / 500 new tokens) on every released text; answers are stored raw and scored at summary
    time (`summarize`), so a scoring change never needs the GPU."""
    from fusit.trace.chat import HFGenerator, generate_all
    from fusit.trace.prompts import render_author, render_chat, render_deepseek_r1, staab_multi_messages

    if args.dataset == "tab":
        return stage_attack_tab(args, items, records)
    todo = []
    for item in items:
        rec = records[item.username]
        check_config(rec, args, "attack")
        if "attack" in rec and "attack_answers" not in rec:
            raise SystemExit(f"{rec['item']}: attacked with the pre-2026-10-04 attack; use a different --output")
        done = rec.get("attack_answers", {})
        todo += [(rec, c, t) for c, t in released_texts(rec, args).items() if c not in done]
    if not todo:
        log("attack: every text already attacked")
        return
    gen = HFGenerator(args.attacker_model, dtype=args.dtype, device=args.device)
    prompts, prefill = [], []
    for rec, cond, text in todo:
        system, user = staab_multi_messages(attack_text(rec, cond, text), list(rec["truth"]))
        budget = args.attack_max_new or ATTACK_MAX_NEW[rec["dataset"]]
        if args.attack_template == "author_llama3":
            prompts.append((gen.encode(render_author(system, user), add_special_tokens=True), budget))
            prefill.append("")
            continue
        if args.attack_template == "deepseek_r1":
            rendered = render_deepseek_r1(user)
            ids = gen.encode(rendered, add_special_tokens=True)
        else:
            rendered = render_chat(gen.tokenizer, system, user, args.attack_template == "chat_nosystem")
            ids = gen.encode(rendered, add_special_tokens=False)
        # DeepSeek-R1's template opens the answer with "<think>\n"; it is put back in front of the stored
        # answer so that scoring (drop_think) sees the whole block, closed or not
        prefill.append("<think>\n" if rendered.rstrip().endswith("<think>") else "")
        # Llama-2's 4096-token context: the answer gets what the prompt leaves, never less than ATTACK_MIN_NEW
        room = gen.context_len - len(ids) if gen.context_len else budget
        prompts.append((ids, min(budget, room) if room >= ATTACK_MIN_NEW else budget))
    n_done = [0]

    def save(idx, outs):
        for i, g in zip(idx, outs):
            rec, cond, _ = todo[i]
            rec.setdefault("config", {})["attack"] = stage_config(args, "attack")
            rec.setdefault("attack_answers", {})[cond] = {
                "response": g.text if g.text is None else prefill[i] + g.text, "status": g.status,
                "n_prompt_tokens": g.n_prompt_tokens,
                "n_new_tokens": g.n_new_tokens, "hit_max_new_tokens": g.hit_max_new_tokens,
                "max_new_tokens": prompts[i][1]}
        write_records(args.output, records)
        n_done[0] += len(idx)
        log(f"attack {n_done[0]}/{len(todo)} answers")

    generate_all(gen, prompts, 8, args.attack_max_batch_tokens, save)
    del gen
    free_model()


def stage_attack_tab(args, items, records: Dict[str, dict]) -> None:
    """The explicit experiment's attack (fusit.dp_fusion.attack, the official DP-Fusion-DPI
    token-recovery game, LOSS and Min-K%) on every released text, surrogate --attacker-model (C1-a)."""
    from fusit.dataset import Span, TabDocument
    from fusit.dp_fusion.attack import attack_document, load_candidate_pool

    todo = []
    for item in items:
        rec = records[item.username]
        check_config(rec, args, "attack")
        done = rec.get("attack_tab", {})
        todo += [(rec, c, t) for c, t in released_texts(rec, args).items() if c not in done]
    if not todo:
        log("attack: every text already attacked")
        return
    model, tok = load_model(args.attacker_model, DTYPES[args.dtype], args.device)
    pool = load_candidate_pool()
    for n, (rec, cond, text) in enumerate(todo, 1):
        t0 = time.perf_counter()
        doc = TabDocument(username=rec["item"], text=rec["text"], spans=[Span(*e) for e in rec["entities"]])
        groups = attack_document(doc, text, model, tok, pool)
        rec.setdefault("config", {})["attack"] = stage_config(args, "attack")
        rec.setdefault("attack_tab", {})[cond] = {
            g: {k: v[k] for k in ("logp", "n_tokens", "pred", "correct")} for g, v in groups.items()}
        write_records(args.output, records)
        log(f"attack {n}/{len(todo)} {rec['item']} {cond}: "
            + " ".join(f"{g}:{v['correct']['loss']}" for g, v in groups.items())
            + f" ({time.perf_counter() - t0:.1f}s)")
    del model, tok
    free_model()


def summarize_tab(recs: Dict[str, dict]) -> List[str]:
    """The explicit table over TAB records: ASR of LOSS and Min-K% over (document, group) trials
    with 95% Wilson intervals (scripts/table1/score.py), coverage, cosine, judge utility, PPL."""
    import statistics as st
    from fusit.dp_fusion.attack import MINK, wilson
    from fusit.perplexity import ppl_of
    from fusit.utility import utility_cell

    metrics = ["loss"] + [f"min{k}" for k in MINK]
    conds = []
    for rec in recs.values():
        conds += [c for c in rec.get("attack_tab", {}) if c not in conds]
    lines = ["| condition | n | " + " | ".join(f"{m} ASR [95% CI]" for m in metrics)
             + " | coverage | cosine | utility | mech PPL | release PPL |", "|---" * (len(metrics) + 7) + "|"]
    for cond in conds:
        trials = [g for r in recs.values() for g in r.get("attack_tab", {}).get(cond, {}).values()]
        cells = []
        for m in metrics:
            k = sum(g["correct"][m] for g in trials)
            lo, hi = wilson(k, len(trials))
            cells.append(f"{100 * k / len(trials):.1f} [{100 * lo:.1f}, {100 * hi:.1f}]" if trials else "—")
        base = cond[len("dp_fusion@"):].rsplit("@", 1)[0] if cond.startswith("dp_fusion@") else (
            cond[len("mask_"):] if cond.startswith("mask_") else None)
        cov = [r["x_priv_conditions"][base]["coverage"]["char"] for r in recs.values()
               if base and base in r.get("x_priv_conditions", {})]
        judged = [r["judge"][cond] for r in recs.values() if cond in r.get("judge", {})]
        u = utility_cell([j["response"] for j in judged], judged) if judged else {}
        mkey = "no_defense" if cond == "no_defense" else (cond if cond.startswith("mask_") else (
            "mask_" + cond[len("dp_fusion@"):] if cond.startswith("dp_fusion@") and cond.count("@") == 2 else None))
        mech = [ppl_of(r["ppl"]["mechanism"][mkey]) for r in recs.values() if mkey and mkey in r.get("ppl", {}).get("mechanism", {})]
        rel = [ppl_of(r["ppl"]["release"][cond]) for r in recs.values() if cond in r.get("ppl", {}).get("release", {})]
        f = lambda v, fmt: format(v, fmt) if v is not None else "—"  # noqa: E731
        lines.append(f"| {cond} | {len(trials)} | " + " | ".join(cells)
                     + f" | {f(100 * st.mean(cov) if cov else None, '.1f')} | {f(u.get('cosine'), '.3f')} "
                     f"| {f(u.get('util') if judged and any(j.get('response') for j in judged) else None, '.3f')} "
                     f"| {f(st.mean(mech) if mech else None, '.3f')} | {f(st.mean(rel) if rel else None, '.3f')} |")
    return lines


def stage_judge(args, items, records: Dict[str, dict]) -> None:
    """Staab et al.'s LLM-as-judge utility and the text similarities of every released text
    against D (fusit.utility, decisions B1, E1-E4)."""
    from fusit.trace.chat import HFGenerator
    from fusit.utility import TextSimilarity, judge_texts

    todo = []
    for item in items:
        rec = records[item.username]
        check_config(rec, args, "judge")
        done = rec.get("judge", {})
        todo += [(rec, c, t) for c, t in released_texts(rec, args).items() if c not in done]
    if not todo:
        log("judge: every text already judged")
        return
    gen = HFGenerator(args.judge_model, dtype=args.dtype, device=args.device)
    sim = TextSimilarity(device="cpu", sbert=None if args.sbert_model == "none" else args.sbert_model)
    n_done = [0]

    def save(idx, outs):
        for i, g in zip(idx, outs):
            rec, cond, text = todo[i]
            rec.setdefault("config", {})["judge"] = stage_config(args, "judge")
            rec.setdefault("judge", {})[cond] = {"response": g.text, "status": g.status,
                                                 "hit_max_new_tokens": g.hit_max_new_tokens,
                                                 **sim(rec["text"], text)}
        write_records(args.output, records)
        n_done[0] += len(idx)
        log(f"judge {n_done[0]}/{len(todo)} answers")

    judge_texts(gen, [(rec["text"], t) for rec, _, t in todo], 8, args.judge_max_batch_tokens, save)
    del gen, sim
    free_model()


def stage_ppl(args, items, records: Dict[str, dict]) -> None:
    """D teacher-forced (fusit.perplexity, decisions B3, D9, E5): mechanism -- the original, every
    mask, and the single-group DP-Fusion mixture at each paraphrased cap -- and release, D given
    each released text."""
    from fusit.perplexity import mechanism_ppl, release_ppl

    todo = []
    for item in items:
        rec = records[item.username]
        check_config(rec, args, "ppl")
        want = released_texts(rec, args)
        if any(c not in rec.get("ppl", {}).get("release", {}) for c in want):
            todo.append(rec)
    if not todo:
        log("ppl: every item already done")
        return
    model, tok = load_model(args.ppl_model, DTYPES[args.dtype], args.device)
    for n, rec in enumerate(todo, 1):
        t0 = time.perf_counter()
        rec.setdefault("config", {})["ppl"] = stage_config(args, "ppl")
        out = rec.setdefault("ppl", {"mechanism": {}, "release": {}})
        masks = {f"mask_{c}": rec["x_priv_conditions"][c]["x_priv"] for c in args.mask_conditions
                 if c in rec.get("x_priv_conditions", {})}
        caps: Dict[str, List[float]] = {}
        for key in rec.get("paraphrases", {}):
            if "@" in key:
                cond, cap = key.rsplit("@", 1)
                caps.setdefault(f"mask_{cond}", []).append(float(cap))
        need = {m: s for m, s in masks.items() if m not in out["mechanism"]
                or any(f"{m}@{c}" not in out["mechanism"] for c in caps.get(m, []))}
        if "no_defense" not in out["mechanism"] or need:
            out["mechanism"].update(mechanism_ppl(model, tok, rec["text"], need, caps))
        for cond, text in released_texts(rec, args).items():
            if cond not in out["release"]:
                out["release"][cond] = (out["mechanism"]["no_defense"] if cond == "no_defense"
                                        else release_ppl(model, tok, rec["text"], text))
        write_records(args.output, records)
        log(f"ppl {n}/{len(todo)} {rec['item']} ({time.perf_counter() - t0:.1f}s)")
    del model, tok
    free_model()


def summarize(paths: Sequence[Path]) -> str:
    """The result table over every record of `paths` (all shards of a dataset), per dataset:
    attack top-1 with its bootstrap CI, not-answered and fallback-read shares (fusit.trace.scoring
    `score_answer` / `table_cell`); judge utility (fusit.utility `utility_cell`); mechanism and
    release PPL (mean over documents of exp(mean NLL))."""
    import statistics as st
    from fusit.perplexity import ppl_of
    from fusit.trace.scoring import score_answer, table_cell
    from fusit.utility import utility_cell

    by_ds: Dict[str, Dict[str, dict]] = {}
    for path in paths:
        for rec in load_records(path).values():
            by_ds.setdefault(rec["dataset"], {})[rec["item"]] = rec
    lines = []
    for ds, recs in by_ds.items():
        if ds == "tab":
            lines += [f"## tab ({len(recs)} documents)", ""] + summarize_tab(recs) + [""]
            continue
        order = [i.username for i in get_dataset(f"{ds}_tracerps").items() if i.username in recs]
        conds = []
        for rec in recs.values():
            conds += [c for c in rec.get("attack_answers", {}) if c not in conds]
            conds += [c for c in rec.get("judge", {}) if c not in conds]
            conds += [c for c in rec.get("ppl", {}).get("release", {}) if c not in conds]
        # table order: original, masks, TRACE@1, DP-Fusion by condition then cap
        pos = lambda x: MASK_CONDITIONS.index(x) if x in MASK_CONDITIONS else len(MASK_CONDITIONS)  # noqa: E731
        def rank(c):
            if c.startswith("rps@"):            # each RPS row after its base's block
                return (5,) + rank(c[len("rps@"):])
            return ((0,) if c == "no_defense" else (1, pos(c[5:]), c) if c.startswith("mask_")
                    else (2, c) if c in ("trace", "aa")
                    else (3, pos(c.split("@")[1]), c.split("@")[1], float(c.rsplit("@", 1)[1])) if c.count("@") == 2
                    else (4, c))
        conds.sort(key=rank)
        lines += [f"## {ds} ({len(recs)} items)", "",
                  "| condition | top-1 | 95% CI | not answered | fallback-read | utility_comb | utility | ROUGE-1 "
                  "| cosine | no-halluc. | mech PPL | release PPL |", "|---" * 12 + "|"]
        for cond in conds:
            rows = {}
            for i in order:
                a = recs[i].get("attack_answers", {}).get(cond)
                if a is not None:
                    rows[i] = list(score_answer(a["response"], a["status"], list(recs[i]["truth"]),
                                                recs[i]["truth"]).values())
            cell = table_cell(rows, [i for i in order if i in rows]) if rows else None
            judged = [recs[i]["judge"][cond] for i in order if cond in recs[i].get("judge", {})]
            u = utility_cell([j["response"] for j in judged], judged) if judged else None
            base = cond[len("rps@"):] if cond.startswith("rps@") else cond     # RPS only appends a suffix
            mkey = None if cond.startswith("rps@") and not base.startswith("dp_fusion@") else (
                "no_defense" if cond == "no_defense" else (
                cond if cond.startswith("mask_") else
                ("mask_" + base[len("dp_fusion@"):] if base.startswith("dp_fusion@") and base.count("@") == 2 else None)))
            mech = [ppl_of(recs[i]["ppl"]["mechanism"][mkey]) for i in order
                    if mkey and mkey in recs[i].get("ppl", {}).get("mechanism", {})]
            rel = [ppl_of(recs[i]["ppl"]["release"][cond]) for i in order if cond in recs[i].get("ppl", {}).get("release", {})]
            f = lambda v, fmt: format(v, fmt) if v is not None else "—"  # noqa: E731
            lines.append(
                f"| {cond} | {f(cell and 100 * cell['top1'], '.2f')} | "
                + (f"[{100 * cell['ci'][0]:.1f}, {100 * cell['ci'][1]:.1f}]" if cell else "—")
                + f" | {f(cell and 100 * cell['not_answered'], '.1f')} | {f(cell and 100 * cell['lenient'], '.1f')} "
                f"| {f(u and u['comb'], '.3f')} | {f(u and u['util'], '.3f')} | {f(u and u.get('rouge1'), '.3f')} "
                f"| {f(u and u.get('cosine'), '.3f')} | {f(u and 100 * u['hall'], '.1f')} "
                f"| {f(st.mean(mech) if mech else None, '.3f')} | {f(st.mean(rel) if rel else None, '.3f')} |")
        lines.append("")
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
    cfg = {}
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        from fusit.config import load_config, per_dataset, run_defaults
        cfg = load_config(pre.config)
        defaults = run_defaults(cfg, parser)
        parser.set_defaults(**defaults)
        parser.set_defaults(**per_dataset(defaults, parser.parse_known_args(argv)[0].dataset))
    args = parser.parse_args(argv)
    validate(args, parser)
    if args.decide:
        run_decide(args.decide, cfg.get("xpriv") or None)
        return 0
    if args.summarize:
        print(summarize(args.summarize))
        return 0

    items = select_items(args)
    records = load_records(args.output)
    log(f"{len(items)} item(s) from {args.dataset} (shard {args.shard_index}/{args.num_shards}); "
        f"{sum(1 for i in items if i.username in records)} already in {args.output}")

    stage_tag(args, items, records)
    if args.stop_after == "tag":
        return 0
    if args.verify:
        stage_verify(args, items, records)
    if args.stop_after == "verify":
        return 0
    if args.trace:
        stage_trace(args, items, records)
    if args.stop_after == "trace":
        return 0
    if args.aa:
        stage_aa(args, items, records)
    if args.stop_after == "aa":
        return 0
    stage_paraphrase(args, items, records)
    if args.stop_after == "paraphrase":
        log("stopping before the attack")
        return 0
    if args.rps:
        stage_rps(args, items, records)
    if args.skip_attack or args.stop_after == "rps":
        log("stopping before the attack")
        return 0
    stage_attack(args, items, records)
    if args.stop_after == "attack":
        return 0
    if args.judge:
        stage_judge(args, items, records)
    if args.stop_after == "judge":
        return 0
    if args.ppl:
        stage_ppl(args, items, records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
