"""
Utility of every Table 1 output, measured the way the two papers being compared measure it.

    python -m fusit.utility --dataset synthetic --num-shards 20 --shard-index 0

For each item it gathers the text every defence published -- redactions, DP-Fusion
paraphrases, TRACE's rewrite, RPS suffixes -- and scores each against the original:

    judge   Staab et al.'s LLM utility judge (llm-anonymization, MIT), which TRACE-RPS uses for
            TRACE: readability 1-10, meaning 1-10, hallucinations 0/1, as JSON. The prompt is
            copied byte for byte, including its missing closing brace.
    bleu    NLTK sentence BLEU on whitespace tokens, smoothing method4 (its compute_bleu)
    rouge   ROUGE-1 / ROUGE-L F1 with stemming (its compute_rouge)
    sbert   cosine similarity under paraphrase-MiniLM-L6-v2, TRACE-RPS's metric for RPS. The
            model truncates at 128 word pieces, so on long profiles it compares openings.

Staab et al.'s combined score is (readability/10 + meaning/10 + ROUGE-1) / 3, computed at
summary time from the stored parts.

The judge is Llama-3.1-8B-Instruct run locally and greedily, where the papers used GPT-4.
It wrote none of the outputs being judged: Qwen2.5-7B wrote the paraphrases and rewrites, so
using it would let it grade its own text.

Not reproduced from DP-Fusion: its pairwise GPT-4o-mini win rate (a tournament over this many
methods is a different experiment) and its perplexity, which teacher-forces the original
document through the private generator and so only exists for DP-Fusion outputs.
"""

import argparse
import json
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from fusit.main import DTYPES, PIPELINE_DATASETS, load_model, load_records, log, write_records
from fusit.trace import redact
from fusit.trace.chat import format_chat

# -- Staab et al., src/anonymized/anonymized.py `score_anonymization_utility_prompt` --
UTILITY_SYSTEM_PROMPT = "You are an expert text similarity scorer that carefully compares two texts and gives a score based on how similar they are. You follow the instructions and format precisely and you try to give a score as close to the ground truth as possible."
UTILITY_HEADER = textwrap.dedent(
    """
    You are a judge that must decide on a scale from 1-10 whether two sets of comments, the original and the adapted, have the same meaning and content. \
    10 means they have excatly the same content and message while 1 means they are nothing alike. Before giving your final \
    score give a brief step-by-step reasoning of your decision. Give your answer in the JSON format below:\n\
    {
    "readability": {
        "explanation": <Is the adapted text as readable and understandable as the original text? Could a human read it without issues? Focus only on the adapted text without your knowledge of the original one.>,
        "score": <Number between 1 (unreadable) to 10 (equally readable as the original text)>
    },
    "meaning": {
        "explanation": <Does the adapted text have the same meaning as the original text? Does it convey the same message?>,
        "score": <Number between 1 (different meaning) to 10 (same meaning)>
    },
    "hallucinations": {
        "explanation": <Does the adapted text contain any information that was not present in the original text and that does not reflect the original text or did it correctly abstract and generalize the original text?>,
        "score": Either 0 (contains new information) or 1 (contains no new information)>
    }
    """
)
UTILITY_FOOTER = "Only answer in the given format and do not add any additional information."


#: Where each Table 1 run lives, relative to --results.
RUNS = {"trace": "table1", "dpfusion": "table1_dpfusion", "tracerps": "table1_tracerps"}


def load_run(root: Path, run: str, dataset: str) -> Dict[str, dict]:
    records = {}
    for path in sorted((root / RUNS[run] / dataset).glob("shard_*.jsonl")):
        records.update(load_records(path))
    return records


def outputs_for(item: str, runs: Dict[str, Dict[str, dict]]) -> Dict[str, str]:
    """{condition: published text} for one item, across the three runs. Condition names say
    which tagger or run they come from, since two runs both have an ner_redaction."""
    t, d, r = runs["trace"][item], runs["dpfusion"][item], runs["tracerps"][item]
    text = t["text"]
    out = {
        "no_defense": text,
        "ner_redaction_spacy": redact(text, t["spans"].get("ner", [])),
        "ner_redaction_presidio": redact(text, d["spans"].get("ner", [])),
        "x_priv_redaction": redact(text, t["x_priv"]),
    }
    for cap, para in d["paraphrases"].items():
        out[f"dp_fusion_presidio@{cap}"] = text if para.get("text") is None else para["text"]
    for cap, para in t["paraphrases"].items():
        out[f"trace_dp_fusion@{cap}"] = text if para.get("text") is None else para["text"]
    first = r["trace"]["rounds"][0]
    out["trace_r1"] = first["rewrite"]["text"] if "rewrite" in first else r["text"]
    out["rps"] = r["rps"]["rps"]["text"]
    out["trace_rps_r1"] = r["rps"]["trace_rps_r1"]["text"]
    return out


def judge_prompt(original: str, adapted: str) -> str:
    # llm-anonymization's Prompt.template "{header}\n{shots}\n{intermediate}\n\n{footer}\n\n{answer}"
    intermediate = f"Original text:\n\n{original}\nAdapted text:\n\n{adapted}"
    return f"{UTILITY_HEADER}\n\n{intermediate}\n\n{UTILITY_FOOTER}\n\n"


def parse_judge(answer: str) -> Dict:
    """Staab et al.'s `parse_utility_answer`: the outermost {...} as JSON, and on a decoding
    error every score falls back to 1. Two recoveries are tried first -- appending the closing
    brace the prompt's own template omits, then reading the scores out of JSON-like text --
    and `parse` records which path produced the scores."""
    start = answer.find("{")
    end = answer.rfind("}") + 1
    for candidate, how in ((answer[start:end], "ok"), (answer[start:] + "}", "repaired")):
        try:
            obj = json.loads(candidate)
            scores = {k: obj[k]["score"] for k in ("readability", "meaning", "hallucinations")}
            scores = {k: float(v) for k, v in scores.items()}
            return {**scores, "parse": how}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    # The template writes each explanation as an unquoted <...>, and Llama-3.1 copies that:
    # the scores are all there but the answer is not JSON. Read each block's score directly
    # rather than let the fallback grade a well-formed judgement 1/10.
    # Each block runs from its key to the next key; its score is the last "score" in it, quoted
    # or not ("Score: 4" also appears inside explanations). A block the answer never reached --
    # the judge ran out of tokens -- leaves that score unknown, and the answer is then marked
    # `fallback` with Staab et al.'s 1s, which `summarize` leaves out of the means.
    keys = ("readability", "meaning", "hallucinations")
    starts = {k: m.start() for k in keys if (m := re.search(rf'"?{k}"?\s*:', answer, re.IGNORECASE))}
    scores = {}
    for k, start in starts.items():
        end = min([v for v in starts.values() if v > start] + [len(answer)])
        found = re.findall(r'score"?\s*[:=]\s*"?(\d+(?:\.\d+)?)', answer[start:end], re.IGNORECASE)
        if found:
            scores[k] = float(found[-1])
    if len(scores) == 3:
        return {**scores, "parse": "regex"}
    return {"readability": 1.0, "meaning": 1.0, "hallucinations": 1.0, "parse": "fallback"}


class Lexical:
    def __init__(self, device: str):
        from nltk.translate import bleu
        from nltk.translate.bleu_score import SmoothingFunction
        from rouge_score import rouge_scorer
        from sentence_transformers import SentenceTransformer

        self._bleu, self._smooth = bleu, SmoothingFunction().method4
        self._rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL", "rougeLsum"], use_stemmer=True)
        self._sbert = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2", device=device)

    def __call__(self, original: str, adapted: str) -> Dict[str, float]:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # NLTK warns on hypotheses with no n-gram overlap
            bleu = self._bleu([original.split(" ")], adapted.split(" "), smoothing_function=self._smooth)
        rouge = self._rouge.score(original, adapted)
        emb = self._sbert.encode([original, adapted], convert_to_tensor=True, normalize_embeddings=True)
        return {"bleu": float(bleu), "rouge1": rouge["rouge1"].fmeasure, "rougeL": rouge["rougeL"].fmeasure,
                "sbert": float(emb[0] @ emb[1])}


@torch.no_grad()
def judge_batch(prompts: List[str], model, tok, max_new_tokens: int) -> List[str]:
    texts = [format_chat(tok, model, UTILITY_SYSTEM_PROMPT, p) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                         top_p=None, pad_token_id=tok.pad_token_id)
    return tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m fusit.utility", description=__doc__.split("\n\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", choices=PIPELINE_DATASETS, default="synthetic")
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--judge-model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--batch-tokens", type=int, default=24000,
                   help="prompt tokens per judge batch (padded length x batch size)")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("utility_results.jsonl"))
    p.add_argument("--summarize", action="store_true",
                   help="print the table from every shard under --output's directory and exit")
    return p


def run(args) -> None:
    runs = {name: load_run(args.results, name, args.dataset) for name in RUNS}
    items = sorted(set.intersection(*(set(r) for r in runs.values())))
    items = items[args.shard_index::args.num_shards]
    records = load_records(args.output)
    log(f"{len(items)} item(s) from {args.dataset} (shard {args.shard_index}/{args.num_shards}); "
        f"{sum(1 for i in items if i in records)} already in {args.output}")

    model, tok = load_model(args.judge_model, DTYPES[args.dtype], args.device)
    tok.padding_side = "left"
    lexical = Lexical(args.device)

    for n, item in enumerate(items, 1):
        rec = records.setdefault(item, {"item": item, "dataset": args.dataset,
                                        "config": {"judge_model": args.judge_model, "dtype": args.dtype},
                                        "scores": {}})
        outputs = outputs_for(item, runs)
        original = outputs["no_defense"]
        todo = [c for c in outputs if c not in rec["scores"]]
        if not todo:
            continue
        t0 = time.perf_counter()
        prompts = {c: judge_prompt(original, outputs[c]) for c in todo}
        lengths = {c: len(tok(prompts[c], add_special_tokens=False)["input_ids"]) for c in todo}
        order = sorted(todo, key=lengths.get)
        batch: List[str] = []
        for c in order + [None]:
            if c is not None and (not batch or (len(batch) + 1) * max(lengths[b] for b in batch + [c]) <= args.batch_tokens):
                batch.append(c)
                continue
            answers = judge_batch([prompts[b] for b in batch], model, tok, args.max_new_tokens)
            for b, answer in zip(batch, answers):
                rec["scores"][b] = {**parse_judge(answer), **lexical(original, outputs[b]), "judge_raw": answer,
                                    "chars": len(outputs[b])}
            write_records(args.output, records)
            batch = [c] if c is not None else []
        log(f"utility {n}/{len(items)} {item}: {len(todo)} output(s) ({time.perf_counter() - t0:.1f}s)")


ORDER = ["no_defense", "ner_redaction_spacy", "ner_redaction_presidio", "dp_fusion_presidio@0.01",
         "dp_fusion_presidio@0.1", "rps", "trace_r1", "trace_rps_r1", "x_priv_redaction",
         "trace_dp_fusion@0.01", "trace_dp_fusion@0.1"]


def summarize(out_dir: Path, dataset: str) -> str:
    rows: Dict[str, List[dict]] = {}
    for path in sorted(out_dir.glob("shard_*.jsonl")):
        for rec in load_records(path).values():
            for cond, s in rec["scores"].items():
                # re-parse from the raw answer, so a parser fix applies without re-judging
                s = {**s, **parse_judge(s["judge_raw"])}
                rows.setdefault(cond, []).append(s)
    if not rows:
        return f"== {dataset}: no utility scores yet =="
    n_items = max(len(v) for v in rows.values())
    lines = [f"== {dataset}: {n_items} item(s) ==",
             f"{'output':<26} {'read':>5} {'mean':>5} {'hall':>5} {'comb':>6} {'BLEU':>6} {'R-1':>6} {'R-L':>6} {'SBERT':>6} {'unread':>6}"]
    for cond in sorted(rows, key=lambda c: ORDER.index(c) if c in ORDER else len(ORDER)):
        bad = sum(x["parse"] == "fallback" for x in rows[cond])
        v = [x for x in rows[cond] if x["parse"] != "fallback"]
        mean = lambda k: sum(x[k] for x in v) / len(v)
        # hallucinations is 0/1, but the judge sometimes answers it on the 1-10 scale of the
        # other two; anything above 1 is read as "no new information"
        hall = sum(min(max(x["hallucinations"], 0), 1) for x in v) / len(v)
        comb = sum((min(max(x["readability"], 0), 10) / 10 + min(max(x["meaning"], 0), 10) / 10
                    + min(max(x["rouge1"], 0), 1)) / 3 for x in v) / len(v)
        lines.append(f"{cond:<26} {mean('readability'):5.2f} {mean('meaning'):5.2f} {hall:5.2f} "
                     f"{100 * comb:5.1f}% {mean('bleu'):6.3f} {mean('rouge1'):6.3f} {mean('rougeL'):6.3f} "
                     f"{mean('sbert'):6.3f} {bad:6d}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.summarize:
        print(summarize(args.output.parent, args.dataset))
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
