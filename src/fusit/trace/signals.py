"""
The three signal sources. Each answers "which characters of this text leak the attribute?"
and returns char-offset spans, so they can be unioned.

    NER(D)   `ner_spans`         entities. Deliberately NOT LLM-prompted: a prompt-based NER
                                 would blur into another V_cot and make the NER-vs-cues
                                 ablation circular, since both signals would then come from
                                 the same model as the attacker. Implemented in
                                 `fusit.trace.ner`, which offers spaCy and Presidio+BERT-NER
                                 backends; re-exported here so the three sources read
                                 together.
    V_cot    `infer_and_chain`   the attacker profiles the author, then justifies its own
                                 guess step by step and quotes the text at each step; the
                                 quotes are the cue. Two LLM calls per attribute.
    V_att    `attention_spans`   the attribute's question is appended to the text and the
                                 attention from the final token back over the context is read
                                 off the last layer. One forward pass, no generation.

`oracle_gt_spans` is V_cot's achievable ceiling: the same chain, but asked to justify the
ground-truth value instead of the attacker's guess. Not deployable -- it reads the label --
which is exactly what makes it a ceiling.
"""

import re
from typing import Dict, List, Tuple

import torch

from fusit.utils import find_phrase_offsets
from fusit.trace.attributes import ATTRIBUTE_OPTIONS, label_of, question_of
from fusit.trace.chat import chat, format_chat
from fusit.trace.parsing import extract_evidence_quotes, parse_inference_response
from fusit.trace.prompts import (
    ADVERSARIAL_INFERENCE_QUERY_PROMPT_TEMPLATE,
    ADVERSARIAL_INFERENCE_SYSTEM_PROMPT,
    PRIVACY_LEAKAGE_CHAIN_PROMPT_TEMPLATE,
    PRIVACY_LEAKAGE_CHAIN_SYSTEM_PROMPT,
)
from fusit.trace.ner import NER_LABELS, ner_spans, spacy_ner_spans
from fusit.trace.spans import WORD_RE, is_functional_word, match_quote, merge_spans, occurrences

# ---------------------------------------------------------------------------
# V_cot
# ---------------------------------------------------------------------------

def _n_tokens(tokenizer, s: str) -> int:
    return len(tokenizer(s, add_special_tokens=False)["input_ids"])


def _context(model) -> int:
    return getattr(model.config, "max_position_embeddings", 8192)


def chain_pieces(text: str, overhead: int, budget_tokens: int, tokenizer) -> List[Tuple[int, int]]:
    """[(start, end)] of `text`: all of it when it fits `budget_tokens`, else line-bounded pieces
    that each fit (a line alone too long is split on words) -- the cue tagger's way of never
    cutting a document (decision T8)."""
    n_tokens = lambda t: _n_tokens(tokenizer, t)  # noqa: E731
    budget = budget_tokens - overhead
    if budget < 64:
        raise ValueError(f"prompt overhead {overhead} leaves no room for the document")
    if n_tokens(text) <= budget:
        return [(0, len(text))]
    units, pos = [], 0
    for line in text.split("\n"):
        s, e = pos, pos + len(line)
        pos = e + 1
        if n_tokens(line) <= budget:
            units.append((s, e))
            continue
        cur = None
        for m in WORD_RE.finditer(line):
            ws, we = s + m.start(), s + m.end()
            if cur and n_tokens(text[cur[0]:we]) > budget:
                units.append(cur)
                cur = None
            cur = (ws, we) if cur is None else (cur[0], we)
        if cur:
            units.append(cur)
    pieces, cur = [], None
    for s, e in units:
        if cur and n_tokens(text[cur[0]:e]) > budget:
            pieces.append(cur)
            cur = None
        cur = (s, e) if cur is None else (cur[0], e)
    pieces.append(cur)
    return pieces


def guess_attribute(text: str, attribute: str, model, tokenizer, return_response: bool = False,
                    strip_system: bool = False, fit_context: bool = False) -> Dict:
    """The attacker's guess step alone (no privacy-leakage chain) -- reused both as the
    first half of infer_and_chain (V_cot) and standalone for Q2's implicit-ASR attack
    against the paraphrased output in repro/synthpai_eval.py.

    return_response: also return the unparsed answer under "response", for callers that
    need to tell a refusal from a guess after the fact (fusit.floor).
    fit_context: never cut the document -- when prompt + 400 new tokens would not fit the context,
    generate fewer tokens instead of letting `chat` drop the prompt's head (decision T8 for the
    cue tagger). Off by default.
    strip_system: trim the system prompt, as TRACE-RPS's Llama-2 template does ("<<SYS>>\\nYou
    are ..."); the cue tagger's setting (decision (b)), off for every other caller."""
    label = label_of(attribute)
    options = ATTRIBUTE_OPTIONS.get(attribute, "")
    user_prompt = ADVERSARIAL_INFERENCE_QUERY_PROMPT_TEMPLATE.format(
        target_attribute=label, target_attribute_options=options, comments=text
    )
    system = ADVERSARIAL_INFERENCE_SYSTEM_PROMPT.strip() if strip_system else ADVERSARIAL_INFERENCE_SYSTEM_PROMPT
    max_new = 400
    if fit_context:
        n_prompt = _n_tokens(tokenizer, format_chat(tokenizer, model, system, user_prompt))
        # chat() keeps the prompt whole while prompt <= context - max_new - 16
        max_new = min(max_new, _context(model) - n_prompt - 16)
        if max_new < 1:
            raise ValueError(f"the attack prompt alone ({n_prompt} tokens) fills the context")
    response = chat(model, tokenizer, system, user_prompt, max_new_tokens=max_new)
    parsed = parse_inference_response(response, attribute)
    return {**parsed, "response": response} if return_response else parsed


def infer_and_chain(text: str, attribute: str, model, tokenizer, strip_system: bool = False,
                    quote_match: str = "exact", fuzzy_threshold: float = 90, chain_max_new: int = 500,
                    fit_context: bool = False) -> Dict:
    """Returns {inference, guesses, certainty, guess_response, chain, quotes, evidence_spans}
    (guess_response: the guess step's unparsed answer). evidence_spans is
    V_cot for this attribute: char offsets of the chain's quoted evidence, located back in `text`
    via fusit.utils.find_phrase_offsets (skips quotes the model didn't reproduce
    verbatim -- silent skip, not a crash, since exact-quote fidelity from a 7B model is not
    guaranteed). `chain` is the privacy-leakage chain as generated and `quotes` its Evidence
    quotes, each with the string it was located as ("located", None when not found) and the
    offsets it counts at -- what the verifier reads; evidence_spans is their union.

    quote_match="exact": find_phrase_offsets, every case-sensitive verbatim occurrence (as before).
    quote_match="fuzzy": decision (a) -- the quote is located case-insensitively, else by its best
    rapidfuzz partial_ratio alignment >= fuzzy_threshold (match_quote); the located string, edge
    whitespace trimmed, then counts at every case-insensitive word-bounded occurrence.
    strip_system: decision (b), as in guess_attribute.
    chain_max_new: the chain's generation budget (TRACE's 500; `chat` keeps the prompt's tail when
    prompt + budget would exceed the context, so a larger budget cuts more of a long document).
    fit_context: never cut the document (decision T8 for the cue tagger): the guess as in
    guess_attribute, and a chain prompt that would not fit is run per line-bounded piece of the
    document (chain_pieces), every piece with the same guesses; "chain" is then the pieces'
    chains joined by blank lines and "chain_pieces" lists them. Quotes are located in the whole
    document either way."""
    label = label_of(attribute)
    parsed = guess_attribute(text, attribute, model, tokenizer, return_response=True, strip_system=strip_system,
                             fit_context=fit_context)
    # kept only for the record (decision T4: a certainty the answer never states is parsed as 1,
    # so the raw answer is what tells the two apart); nothing below reads it
    parsed["guess_response"] = parsed.pop("response")

    if not parsed["guesses"]:
        return {**parsed, "chain": None, "quotes": [], "evidence_spans": []}

    system = PRIVACY_LEAKAGE_CHAIN_SYSTEM_PROMPT.strip() if strip_system else PRIVACY_LEAKAGE_CHAIN_SYSTEM_PROMPT

    def chain_prompt(comments: str) -> str:
        return PRIVACY_LEAKAGE_CHAIN_PROMPT_TEMPLATE.format(
            comments=comments,
            target_attribute=label,
            inference=parsed["inference"],
            guess="; ".join(parsed["guesses"]),
        )

    pieces = [(0, len(text))]
    if fit_context:
        overhead = _n_tokens(tokenizer, format_chat(tokenizer, model, system, chain_prompt("")))
        # a piece's prompt stays below what chat() keeps whole (context - max_new - 16)
        pieces = chain_pieces(text, overhead, _context(model) - chain_max_new - 32, tokenizer)
    chains = [{"piece": [s, e], "text": chat(model, tokenizer, system, chain_prompt(text[s:e]), max_new_tokens=chain_max_new)}
              for s, e in pieces]
    chain_response = "\n\n".join(c["text"] for c in chains)
    quotes = [q for c in chains for q in extract_evidence_quotes(c["text"])]
    if quote_match == "exact":
        evidence_spans = find_phrase_offsets(text, quotes) if quotes else []
        located = []
        for q in quotes:
            spans = find_phrase_offsets(text, [q])
            located.append({"quote": q, "match": "exact" if spans else "none", "located": q if spans else None,
                            "spans": spans})
    elif quote_match == "fuzzy":
        located = []
        for q in quotes:
            kind, score, span = match_quote(q, text, fuzzy_threshold)
            spans, found = [], None
            if span:
                s, e = span
                while s < e and text[s].isspace():
                    s += 1
                while e > s and text[e - 1].isspace():
                    e -= 1
                if e > s:
                    found = text[s:e]
                    spans = merge_spans([[s, e]] + occurrences(text, found))
            located.append({"quote": q, "match": kind, "score": score, "located": found, "spans": spans})
        evidence_spans = merge_spans([sp for x in located for sp in x["spans"]])
    else:
        raise ValueError(f"quote_match {quote_match!r}, expected 'exact' or 'fuzzy'")

    out = {**parsed, "chain": chain_response, "quotes": located, "evidence_spans": evidence_spans}
    if fit_context:
        out["chain_pieces"] = chains
    return out


# ---------------------------------------------------------------------------
# V_att
# ---------------------------------------------------------------------------

@torch.no_grad()
def attention_spans(
    text: str, question: str, model, tokenizer, k: int = 10, return_weights: bool = False
):
    """Getting attn_weights out of `model(...)` needs attn_implementation="eager" -- the
    default "sdpa"/"flash_attention_2" backends silently return None for out.attentions even
    when output_attentions=True is passed. But loading the WHOLE model under "eager" broke
    dp_fusion_groups_incremental itself: on Qwen2.5-7B-Instruct's batched incremental-cache
    decoding, eager attention produced NaN fused probabilities (crashed
    torch.multinomial on the very first SLURM pilot, job 119949) -- independent of dtype
    (reproduced in both fp16 and bf16). Root cause not fully chased down; the fix is to keep
    the model on its normal backend for everything else and flip to "eager" only for this
    one plain forward pass, via HF's set_attn_implementation (transformers>=4.48), which
    updates the model's internal mask-construction path consistently (unlike poking
    model.config._attn_implementation directly).

    return_weights: also return the per-word attention mass as
    [(start, end, word, weight), ...] over EVERY word (functional words included), ordered
    as they appear in the text. Only for inspection -- scripts/trace_prototype.ipynb reads
    it to show what the ranking is actually built from. Callers in the sweeps ignore it."""
    prompt = text + " " + question
    enc = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = enc.pop("offset_mapping")[0].tolist()
    enc = enc.to(model.device)

    original_impl = model.config._attn_implementation
    if original_impl != "eager":
        model.set_attn_implementation("eager")

    # Capture ONLY the final layer's attention, via a forward hook on its self_attn module.
    #
    # The obvious `model(**enc, output_attentions=True)` makes the model retain a
    # [1, heads, seq, seq] tensor for EVERY layer: on Qwen2.5-7B (28 layers x 28 heads) that
    # is 6.3 GB at seq=2000 and 14.1 GB at seq=3000, on top of a ~15 GB model -- which is
    # what OOM-killed 352 SynthPAI generations on 24 GB cards (SynthPAI profiles run to
    # 13.5k chars; Synthetic comments, median 611 chars, never tripped it). Only the last
    # layer is ever read, so the hook keeps 1/28th of that: 0.22 GB at seq=2000. Under eager
    # attention the module returns real weights whether or not output_attentions is set.
    captured = {}

    def _grab(module, inputs, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            captured["attn"] = output[1].detach()

    handle = model.model.layers[-1].self_attn.register_forward_hook(_grab)
    try:
        # Only the hooked attention weights are read, so skip what a normal forward also keeps:
        # the KV cache (1.4 GB for Llama-2-7B at ~3.5k tokens) and logits for every position.
        # On the longest SynthPAI profile those two were the difference between fitting a
        # 24 GB card and OOM.
        out = model(**enc, use_cache=False, logits_to_keep=1)
        if "attn" not in captured:  # backend returned no weights; fall back
            out = model(**enc, output_attentions=True, use_cache=False, logits_to_keep=1)
            captured["attn"] = out.attentions[-1]
    finally:
        handle.remove()
        if original_impl != "eager":
            model.set_attn_implementation(original_impl)

    last_layer = captured["attn"]  # [1, heads, seq, seq]
    pooled = last_layer.mean(dim=1)  # [1, seq, seq]
    last_token_att = pooled[0, -1, :].float().cpu().numpy()  # attention FROM last token TO context

    # Only attend to tokens that fall inside `text` (exclude the appended question).
    text_len = len(text)

    word_weights: List[Tuple[int, int, str, float]] = []
    for m in WORD_RE.finditer(text):
        ws, we, word = m.start(), m.end(), m.group()
        weight = 0.0
        for i, (ts, te) in enumerate(offset_mapping):
            if te == 0 and ts == 0:
                continue  # special/pad token
            if te > text_len:
                continue
            if ts < we and te > ws:  # token overlaps this word
                weight += float(last_token_att[i])
        word_weights.append((ws, we, word, weight))

    candidates = [(ws, we, w) for ws, we, w, wt in word_weights if not is_functional_word(w)]
    weights_only = {(ws, we): wt for ws, we, w, wt in word_weights}
    candidates.sort(key=lambda x: weights_only[(x[0], x[1])], reverse=True)

    spans = [[ws, we] for ws, we, w in candidates[:k]]
    return (spans, word_weights) if return_weights else spans


# ---------------------------------------------------------------------------
# Ceiling
# ---------------------------------------------------------------------------

def oracle_gt_spans(text: str, attribute: str, ground_truth: str, model, tokenizer) -> List[List[int]]:
    """E5's *achievable* ceiling for CoT-derived cues: the same privacy-leakage chain as
    V_cot, except the chain is asked to justify the GROUND-TRUTH value instead of the
    attacker's own guess. This upper-bounds V_cot's cue quality by removing its dependence
    on the attacker guessing right in the first place (in the pilot the attacker's top-1 was
    correct only ~39-46% of the time, so V_cot spends most of its budget explaining wrong
    answers). Not a deployable defense -- it reads the label -- which is exactly why it is
    the ceiling."""
    label = label_of(attribute)
    chain_prompt = PRIVACY_LEAKAGE_CHAIN_PROMPT_TEMPLATE.format(
        comments=text,
        target_attribute=label,
        inference=f"The author's {label} is {ground_truth}.",
        guess=ground_truth,
    )
    chain_response = chat(model, tokenizer, PRIVACY_LEAKAGE_CHAIN_SYSTEM_PROMPT, chain_prompt, max_new_tokens=500)
    quotes = extract_evidence_quotes(chain_response)
    return find_phrase_offsets(text, quotes) if quotes else []


__all__ = [
    "NER_LABELS", "attention_spans", "guess_attribute", "infer_and_chain", "ner_spans",
    "oracle_gt_spans",
]
