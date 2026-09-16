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
from fusit.trace.chat import chat
from fusit.trace.parsing import extract_evidence_quotes, parse_inference_response
from fusit.trace.prompts import (
    ADVERSARIAL_INFERENCE_QUERY_PROMPT_TEMPLATE,
    ADVERSARIAL_INFERENCE_SYSTEM_PROMPT,
    PRIVACY_LEAKAGE_CHAIN_PROMPT_TEMPLATE,
    PRIVACY_LEAKAGE_CHAIN_SYSTEM_PROMPT,
)
from fusit.trace.ner import NER_LABELS, ner_spans, spacy_ner_spans
from fusit.trace.spans import WORD_RE, is_functional_word

# ---------------------------------------------------------------------------
# V_cot
# ---------------------------------------------------------------------------

def guess_attribute(text: str, attribute: str, model, tokenizer) -> Dict:
    """The attacker's guess step alone (no privacy-leakage chain) -- reused both as the
    first half of infer_and_chain (V_cot) and standalone for Q2's implicit-ASR attack
    against the paraphrased output in repro/synthpai_eval.py."""
    label = label_of(attribute)
    options = ATTRIBUTE_OPTIONS.get(attribute, "")
    user_prompt = ADVERSARIAL_INFERENCE_QUERY_PROMPT_TEMPLATE.format(
        target_attribute=label, target_attribute_options=options, comments=text
    )
    response = chat(model, tokenizer, ADVERSARIAL_INFERENCE_SYSTEM_PROMPT, user_prompt, max_new_tokens=400)
    return parse_inference_response(response)


def infer_and_chain(text: str, attribute: str, model, tokenizer) -> Dict:
    """Returns {inference, guesses, certainty, evidence_spans}. evidence_spans is V_cot
    for this attribute: char offsets of the chain's quoted evidence, located back in `text`
    via fusit.utils.find_phrase_offsets (skips quotes the model didn't reproduce
    verbatim -- silent skip, not a crash, since exact-quote fidelity from a 7B model is not
    guaranteed)."""
    label = label_of(attribute)
    parsed = guess_attribute(text, attribute, model, tokenizer)

    if not parsed["guesses"]:
        return {**parsed, "evidence_spans": []}

    chain_prompt = PRIVACY_LEAKAGE_CHAIN_PROMPT_TEMPLATE.format(
        comments=text,
        target_attribute=label,
        inference=parsed["inference"],
        guess="; ".join(parsed["guesses"]),
    )
    chain_response = chat(model, tokenizer, PRIVACY_LEAKAGE_CHAIN_SYSTEM_PROMPT, chain_prompt, max_new_tokens=500)
    quotes = extract_evidence_quotes(chain_response)
    evidence_spans = find_phrase_offsets(text, quotes) if quotes else []

    return {**parsed, "evidence_spans": evidence_spans}


# ---------------------------------------------------------------------------
# V_att
# ---------------------------------------------------------------------------

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
        out = model(**enc) if True else None
        if "attn" not in captured:  # backend returned no weights; fall back
            out = model(**enc, output_attentions=True)
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
