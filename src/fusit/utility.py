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

Decisions recorded against Staab et al.'s code (2026-10-04); the implicit result table
(verifier/judge_utility.py) scores through the prompt and parser here:
    B1-a  judge Llama-3.1-8B-Instruct, greedy (official: gpt-4-1106-preview, temperature 0.1).
    B1-b  original and adapted texts are the comments joined by newlines WITHOUT dates; the
          reference joins str(Comment) = "YYYY-MM-DD: text" per comment, for the judge and for
          ROUGE. The outputs judged here (one paraphrase or rewrite per document) are not
          per-comment, so there is nothing to date.
    B1-c  an answer the original parser cannot decode (the template omits the outer closing
          brace and the local judge copies that; 45% of the result table's answers) is read key
          by key instead of being scored 1 -- the local judge does not follow the format reliably.
    B1-d  an answer from which a score still cannot be read keeps the original 1 (1.7%).
    B1-e  one prompt and parser, here; the header is the official string (see UTILITY_HEADER).
    B1-f  600 new tokens; no result-table answer reached it.

    E1  (2026-10-04) 600 new tokens (B1-f), the result tables' value; the legacy runner below used 512.
    E2  the judge is batched over every query at once, longest first (`fusit.trace.chat.make_batches`),
        capped at 16000 prompt + new tokens: "_"-masked texts tokenize long and the result tables'
        40000 ran out of memory on a 24 GB card.
    E3  both similarity families are computed (`TextSimilarity`): ROUGE-1 and all-MiniLM-L6-v2
        cosine (DP-Fusion A.18), which the result tables report, and BLEU, ROUGE-L and
        paraphrase-MiniLM-L6-v2 SBERT (TRACE-RPS's RPS metric), kept alongside.
    E4  the hallucination score is clipped to [0, 1] before averaging (the judge now and then
        answers it on the 1-10 scale; 2 of the result tables' 37,858 answers).

Not reproduced from DP-Fusion: its pairwise GPT-4o-mini win rate (a tournament over this many
methods is a different experiment) and its perplexity, which teacher-forces the original
document through the private generator and so only exists for DP-Fusion outputs.

Two runners use the prompt and parser above:
    `judge_texts` + `TextSimilarity` + `utility_cell`   the result tables' (fusit.main --judge, 2026-10-04)
    `run` / `summarize` (python -m fusit.utility)       LEGACY: the earlier results/table1 runs only
        (outputs of the pre-2026-10-04 fusit.main: spaCy NER, per-source groups, 512 tokens). Kept
        to preserve that code; the main table uses the runner above. Whether to delete it is open.
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
# the value of the original textwrap.dedent(...) expression (its 8-space indentation survives the
# backslash line continuations as runs of spaces), computed from the source and pasted as a literal.
# Until 2026-10-04 this module re-ran textwrap.dedent on a 4-space-indented copy, which joined the
# continued lines with 4 spaces instead of 9 (decision B1-e).
UTILITY_HEADER = '\nYou are a judge that must decide on a scale from 1-10 whether two sets of comments, the original and the adapted, have the same meaning and content.         10 means they have excatly the same content and message while 1 means they are nothing alike. Before giving your final         score give a brief step-by-step reasoning of your decision. Give your answer in the JSON format below:\n{\n"readability": {\n    "explanation": <Is the adapted text as readable and understandable as the original text? Could a human read it without issues? Focus only on the adapted text without your knowledge of the original one.>,\n    "score": <Number between 1 (unreadable) to 10 (equally readable as the original text)>\n},\n"meaning": {\n    "explanation": <Does the adapted text have the same meaning as the original text? Does it convey the same message?>,\n    "score": <Number between 1 (different meaning) to 10 (same meaning)>\n},\n"hallucinations": {\n    "explanation": <Does the adapted text contain any information that was not present in the original text and that does not reflect the original text or did it correctly abstract and generalize the original text?>,\n    "score": Either 0 (contains new information) or 1 (contains no new information)>\n}\n'
UTILITY_FOOTER = "Only answer in the given format and do not add any additional information."


#: LEGACY (the earlier results/table1 runs; see the module docstring). Where each run lives, relative to --results.
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


JUDGE_KEYS = ("readability", "meaning", "hallucinations")


def parse_utility_answer(answer_str: str) -> dict:
    """anonymized.py::parse_utility_answer, plus numeric scores (a non-numeric score counts as a
    decoding error for that key, i.e. 1, the original fallback)."""
    try:
        start, end = answer_str.find("{"), answer_str.rfind("}") + 1
        answer = json.loads(answer_str[start:end])
        ok = True
    except json.decoder.JSONDecodeError:
        answer = {k: {"explanation": "decoding_error", "score": 1} for k in JUDGE_KEYS}
        ok = False
    scores = {}
    for k in JUDGE_KEYS:
        try:
            scores[k] = float(answer[k]["score"])
        except (KeyError, TypeError, ValueError):
            scores[k], ok = 1.0, False
    return {"scores": scores, "parsed": ok}


_KEY_SCORE = {k: re.compile(r'"' + k + r'"\s*:\s*\{.*?"score"\s*:\s*"?(-?\d+(?:\.\d+)?)', re.S) for k in JUDGE_KEYS}


def parse_fallback(answer_str: str) -> dict:
    """For answers the original parser cannot decode: the prompt's own JSON template never closes
    its outer brace, and the judge sometimes copies that. Each key's "score" is read directly;
    a key with no readable score keeps the original fallback of 1 (decisions B1-c, B1-d)."""
    strict = parse_utility_answer(answer_str)
    if strict["parsed"]:
        return {**strict, "fallback": False}
    scores, found = {}, 0
    for k in JUDGE_KEYS:
        m = _KEY_SCORE[k].search(answer_str or "")
        scores[k] = float(m.group(1)) if m else 1.0
        found += bool(m)
    return {"scores": scores, "parsed": found == len(JUDGE_KEYS), "fallback": True, "strict_scores": strict["scores"]}


def parse_judge(answer: str) -> Dict:
    """`parse_fallback` in this module's record format: the three scores plus `parse` -- "ok"
    (the original parser read it), "regex" (read key by key), "fallback" (a score could not be
    read and keeps the original 1). Until 2026-10-04 this module had its own recovery (append the
    missing brace, then a block-wise regex) and left "fallback" answers out of the means."""
    p = parse_fallback(answer or "")
    how = "ok" if not p["fallback"] else ("regex" if p["parsed"] else "fallback")
    return {**p["scores"], "parse": how}


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


# -- the result tables' runner (fusit.main --judge) ----------------------------------------------

#: E1 / B1-f
JUDGE_MAX_NEW = 600
#: E2
JUDGE_MAX_BATCH_TOKENS = 16000


def judge_texts(gen, pairs: Sequence[tuple], batch_size: int = 8, max_batch_tokens: int = JUDGE_MAX_BATCH_TOKENS,
                on_batch=None):
    """[(original, adapted)] -> [Generation] through the judge's own chat template (system + user,
    tokenized without extra special tokens), as verifier/judge_utility.py ran it."""
    from fusit.trace.chat import generate_all
    enc = []
    for original, adapted in pairs:
        s = gen.tokenizer.apply_chat_template([{"role": "system", "content": UTILITY_SYSTEM_PROMPT},
                                               {"role": "user", "content": judge_prompt(original, adapted)}],
                                              tokenize=False, add_generation_prompt=True)
        enc.append((gen.encode(s, add_special_tokens=False), JUDGE_MAX_NEW))
    return generate_all(gen, enc, batch_size, max_batch_tokens, on_batch)


#: DP-Fusion A.18's sentence encoder (the result tables' cosine) and TRACE-RPS's (E3)
COSINE_MODEL = Path("models/all-MiniLM-L6-v2")
SBERT_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"


class TextSimilarity:
    """Released text vs the original: ROUGE-1 / ROUGE-L F (rouge_score, stemming -- Staab et al.'s
    compute_rouge, scorer.score(original, adapted)), BLEU (compute_bleu), all-MiniLM-L6-v2 cosine
    and paraphrase-MiniLM-L6-v2 SBERT (E3). `sbert=None` skips the latter (no local copy)."""

    def __init__(self, device: str = "cpu", cosine_model=COSINE_MODEL, sbert: Optional[str] = SBERT_MODEL):
        from nltk.translate import bleu
        from nltk.translate.bleu_score import SmoothingFunction
        from rouge_score import rouge_scorer
        from sentence_transformers import SentenceTransformer

        self._bleu, self._smooth = bleu, SmoothingFunction().method4
        self._rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL", "rougeLsum"], use_stemmer=True)
        self._cos = SentenceTransformer(str(cosine_model), device=device)
        self._sbert = SentenceTransformer(sbert, device=device) if sbert else None

    def __call__(self, original: str, adapted: str) -> Dict[str, float]:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # NLTK warns on hypotheses with no n-gram overlap
            bleu = self._bleu([original.split(" ")], adapted.split(" "), smoothing_function=self._smooth)
        rouge = self._rouge.score(original, adapted)
        e = self._cos.encode([original, adapted], normalize_embeddings=True)
        out = {"rouge1": rouge["rouge1"].fmeasure, "rougeL": rouge["rougeL"].fmeasure, "bleu": float(bleu),
               "cosine": float(e[0] @ e[1])}
        if self._sbert is not None:
            e = self._sbert.encode([original, adapted], normalize_embeddings=True)
            out["sbert"] = float(e[0] @ e[1])
        return out


def utility_cell(answers: Sequence[str], similarity: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """One result-table utility cell (verifier/judge_utility.py `score`): judge answers read by
    `parse_fallback`, readability / meaning clipped to [0, 10], hallucinations to [0, 1] (E4);
    utility = (readability/10 + meaning/10) / 2, utility_comb = (readability/10 + meaning/10 +
    ROUGE-1) / 3 per item (plot_anonymized.py), means over items."""
    import numpy as np
    ps = [parse_fallback(a or "") for a in answers]
    sc = {k: np.array([p["scores"][k] for p in ps]) for k in JUDGE_KEYS}
    strict = [parse_utility_answer(a or "") for a in answers]
    ss = {k: np.array([p["scores"][k] for p in strict]) for k in JUDGE_KEYS}
    read, mean = sc["readability"].clip(0, 10), sc["meaning"].clip(0, 10)
    rg = np.array([s["rouge1"] for s in similarity]).clip(0, 1)
    out = {"n": len(answers), "read": float(read.mean()), "mean": float(mean.mean()),
           "hall": float(sc["hallucinations"].clip(0, 1).mean()),
           "util": float(((read / 10 + mean / 10) / 2).mean()),
           "util_strict": float(((ss["readability"].clip(0, 10) / 10 + ss["meaning"].clip(0, 10) / 10) / 2).mean()),
           "comb": float(((read / 10 + mean / 10 + rg) / 3).mean()),
           "fail": float(1 - np.mean([p["parsed"] for p in ps])),
           "fallback": float(np.mean([p["fallback"] and p["parsed"] for p in ps]))}
    for k in ("rouge1", "rougeL", "bleu", "cosine", "sbert"):
        v = [s[k] for s in similarity if k in s]
        if v:
            out[k] = float(np.mean(v))
    return out


# -- LEGACY runner: the earlier results/table1 runs only (see the module docstring) --------------

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
        v = rows[cond]          # unreadable answers keep the original 1s (decision B1-d)
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
