"""
TRACE as a defense on its own: the iterative rewrite of anonymization/trace.py, with local
models in place of the OpenAI calls.

One round, per TRACE-RPS `adversarial_anonymization`:

    analyze   the attacker guesses each attribute from the current text; for every attribute
              it still guesses with certainty above 2, read the top-K attention words and ask
              for the privacy-leakage chain behind the guess
    rewrite   the anonymizer rewrites the text given the inferences, the words and the chains

and the loop stops after 5 rounds, when no attribute is guessed confidently any more, or when
the rewrite returns the text unchanged.

The two halves are separate functions because they run on different models (the tagger model
and the rewriting model), which do not fit on one 24 GB card together; a caller runs `analyze`
for a batch of documents, swaps models, and runs `rewrite` for the same batch.

Where this departs from the reference code, and why:

- Several attributes per document. trace.py hands the loop one attribute (Synthetic has one
  per profile) and, when given more, only checks the certainty of the last. Here every
  attribute of the item is analysed each round and one rewrite addresses them all -- the same
  attributes the cue tagger targets, so TRACE and TRACE x DP-Fusion see the same task.
- No format-fixing call. trace.py sends an unparseable inference to GPT to reformat;
  `fusit.trace.parsing` already recovers the formats Llama-2 drifts into. (The reference's
  fix call cannot run as published either: utils.py imports
  `fix_response_user_prompt_with_certainty`, which prompts.py does not define.)

Decisions recorded against the official code (2026-10-04), shared with the experiment scripts
(verifier/trace_rows.py, scripts/table1/trace_defense.py):
    T1  local models: the experiments infer and chain with Llama-2-7B-Chat (implicit, the cue
        cache) or Qwen2.5-7B (explicit), and rewrite with Qwen2.5-7B, all greedy
        (official: GPT-4o, temperature 0).
    T2  the result tables report one round (TRACE@1); the explicit experiment also ran 5.
    T3  every labelled attribute is analysed and ONE rewrite addresses them all (official: the
        profile's first attribute only, and only the last attribute's certainty is checked).
    T4  stop when no attribute is guessed or every certainty <= 2; the implicit table treats a
        certainty Llama-2 did not write as unknown rather than as 1.
    T5  the important words of the implicit table are V_att (verifier/cues.py); the explicit
        experiment uses `attention_spans` (fusit's questions and no BOS, kept as is).
    T6  attribute names in prompts are `label_of` ("relationship status", "place of birth"),
        not the reference's keys ("relationship_status", "birth_city_country").
    T8  generation lengths are the callers' (see their docstrings), not GPT's 2000/500.
    T9  inference answers are read by fusit's lenient parser, no fix call (above).
    T11 the rewrite is parsed by the official rule plus two exceptions (`parse_anonymized`).
    V-j `clean_chain` strips the answer before the official rule: Llama-2's chains start with a
        space (" Inference Chain:\n..."), so the official startswith never removed the header
        (0 of the implicit table's 1684 chains; all 1684 once stripped).
    TJ-1 the rewrite addresses the attributes still guessed with certainty above 2 -- the official
        stop rule applied per attribute; a certainty the answer does not state counts as above 2
        (T4, `stated_certainty`).
    TJ-2 several attributes in one prompt are told apart: each inference opens with
        "Type: <label>" and each chain with "[<label>]" (verifier/trace_rows.py's layout).
    TJ-3 every guess goes to the prompt, as the official code passes the guess string whole.
    TJ-4 the rewrite may generate max(2000, tokens(D) + 512) tokens (`rewrite_budget`), so a long
        SynthPAI profile is not cut (GPT's 2000 in the official code).
"""

import re
from typing import Dict, List, Optional, Sequence

from fusit.trace.attributes import label_of, question_of
from fusit.trace.chat import chat
from fusit.trace.prompts import (
    AA_SYSTEM_PROMPT,
    ADVERSARIAL_ANONYMIZATION_SYSTEM_PROMPT,
    ADVERSARIAL_REASONING_CHAIN_ANONYMIZATION_QUERY_PROMPT_TEMPLATE,
    PRIVACY_LEAKAGE_CHAIN_PROMPT_TEMPLATE,
    PRIVACY_LEAKAGE_CHAIN_SYSTEM_PROMPT,
    aa_user_prompt,
)
from fusit.trace.signals import attention_spans, guess_attribute

#: trace.py's `adversarial_anonymization(max_iterations=5)`.
MAX_ROUNDS = 5

#: trace.py stops once `inference['certainty'] <= 2`.
CERTAINTY_STOP = 2

#: trace.py's GPT generation length; the floor of `rewrite_budget` (TJ-4).
REWRITE_MIN_NEW = 2000

def clean_chain(response: str) -> str:
    """trace.py `privacy_leakage_chain`'s post-processing of the chain answer, with the answer
    stripped first (decision V-j)."""
    response = response.replace("*", "").replace("#", "").strip()
    if response.startswith("Inference Chain:\n"):
        response = response[len("Inference Chain:\n"):].strip()
    return response


def is_word_list(s: str) -> bool:
    """One line of at least three comma-separated items of at most three words each -- the
    important-words list (T11-b), not a rewrite of comments."""
    s = s.strip()
    if not s or "\n" in s:
        return False
    items = [x.strip() for x in s.split(",") if x.strip()]
    return len(items) >= 3 and all(len(x.split()) <= 3 for x in items)


def parse_anonymized(response: str, text: str):
    """trace.py `anonymization_model`'s parsing: the text after the first '#', with a leading
    "Comments:" and anything from "Inference for comments:" on removed; the input `text`
    unchanged when there is no '#'. Returns (released text, parsed).

    Decision T11 (2026-10-04): the official rule plus ONE exception. On the implicit table's
    822 Qwen2.5 rewrites (verifier/out/trace):
      - APPLIED: nothing after the '#'. Qwen sometimes writes the anonymized comments first and
        ends with a lone '#' (the order the prompt asks for reversed; 99 rewrites, the '#' the
        last character in every one). The official rule, written for GPT, would release an
        empty string; the text before the '#' -- the rewrite itself -- is released instead.
      - NOT applied: the prompt restated after the '#' ("Below I give you a set of comments
        from a user", then "Comments:" and the text, then another '#'; 30 rewrites). The
        official rule releases it, instructions included (fusit's `rewrite` cut it back to
        the first comments block until 2026-10-04).
    The explicit experiment's 501 rewrites (scripts/table1/trace_defense.py) hit neither.

    Decision T11-b (2026-10-04): a second exception. Qwen also writes the rewrite first and puts
    the prompt's important-words list after the '#' (24 of the implicit table's 823 rewrites;
    33 of 140 in the fusit.main sample of 2026-10-04); the official rule releases the word list.
    When what follows the '#' is such a list (`is_word_list`) AND the text before it is at least
    half as long as the input -- a rewrite, not a one-line explanation of the changes -- the text
    before the '#' is released instead."""
    lines = response.splitlines()
    if lines and "explanation" in lines[0].lower():
        response = "\n".join(lines[1:]).strip()
    if "#" not in response:
        return text, False
    before, after = response.split("#", 1)
    anonymized = after.strip()
    if not anonymized and before.strip():          # the T11 exception
        anonymized = before.strip()
    elif is_word_list(anonymized) and len(before.strip()) >= len(text) / 2:     # T11-b
        anonymized = before.strip()
    if anonymized.startswith("Comments:\n"):
        anonymized = anonymized[len("Comments:\n"):]
    if "Inference for comments:" in anonymized:
        anonymized = anonymized.split("Inference for comments:", 1)[0]
    return anonymized, True


def leakage_chain(text: str, attribute: str, inference: str, guesses: Sequence[str],
                  model, tokenizer) -> str:
    """trace.py's `privacy_leakage_chain`, returning the chain text itself (the cue tagger
    only keeps the quotes)."""
    prompt = PRIVACY_LEAKAGE_CHAIN_PROMPT_TEMPLATE.format(
        comments=text, inference=inference, guess="; ".join(guesses),
        target_attribute=label_of(attribute),
    )
    return clean_chain(chat(model, tokenizer, PRIVACY_LEAKAGE_CHAIN_SYSTEM_PROMPT, prompt, max_new_tokens=500))


def analyze(text: str, attributes: Sequence[str], model, tokenizer, k: int) -> Dict[str, Dict]:
    """One round's attacker side: {attribute: {inference, guesses, certainty[, words, chain]}}.

    `words` and `chain` are only computed for attributes still guessed with certainty above
    CERTAINTY_STOP -- the ones the rewrite has to address.
    """
    out = {}
    for attr in attributes:
        parsed = guess_attribute(text, attr, model, tokenizer)
        entry = {k_: parsed[k_] for k_ in ("inference", "guesses", "certainty")}
        if parsed["guesses"] and parsed["certainty"] > CERTAINTY_STOP:
            spans = attention_spans(text, question_of(attr), model, tokenizer, k=k)
            entry["words"] = [text[s:e] for s, e in spans]
            entry["chain"] = leakage_chain(text, attr, parsed["inference"], parsed["guesses"],
                                           model, tokenizer)
        out[attr] = entry
    return out


def stated_certainty(response: Optional[str]) -> Optional[int]:
    """The certainty the guess answer states (its last "Certainty:" line with a digit, as
    fusit.trace.parsing reads it), or None when it states none -- where the parser falls back
    to 1 (decision T4)."""
    if response is None:
        raise ValueError("no guess answer stored; tag again (records from before 2026-10-04 lack it)")
    found = None
    for line in response.replace("*", "").replace("#", "").splitlines():
        line = line.strip().lstrip("-*\u2022 \t")
        if line.lower().startswith("certainty:"):
            m = re.search(r"\d+", line)
            found = int(m.group()) if m else found
    return found


def analysis_from_cues(cues: Dict) -> Dict[str, Dict]:
    """One round's attacker side read from a CueTagger.tag record instead of recomputed, so
    TRACE@1 sees the very guesses, chains and V_att words of the TRACE x DP-Fusion rows. Same
    shape as `analyze`; `chain` (cleaned per piece, V-j) and `words` only for the attributes
    the rewrite addresses (TJ-1)."""
    out = {}
    for attr, rec in cues["attributes"].items():
        cot = rec["cot"]
        cert = stated_certainty(cot.get("guess_response"))
        entry = {"inference": cot["inference"], "guesses": cot["guesses"], "certainty": cert}
        if cot["guesses"] and cot["chain"] is not None and (cert is None or cert > CERTAINTY_STOP):
            pieces = [p["text"] for p in cot["chain_pieces"]] if cot.get("chain_pieces") else [cot["chain"]]
            entry["words"] = rec["att"]["words"]
            entry["chain"] = "\n\n".join(clean_chain(p) for p in pieces)
        out[attr] = entry
    return out


def rewrite_budget(text: str, tokenizer) -> int:
    """TJ-4: max(2000, tokens(D) + 512)."""
    return max(REWRITE_MIN_NEW, len(tokenizer(text, add_special_tokens=False)["input_ids"]) + 512)


def needs_rewrite(analysis: Dict[str, Dict]) -> bool:
    return any("chain" in a for a in analysis.values())


def rewrite(text: str, analysis: Dict[str, Dict], model, tokenizer, max_new_tokens: int) -> Dict:
    """One round's anonymizer side. Returns {text, raw, parsed}; `parsed` is False when the
    response had no `#` separator, in which case `text` is the input unchanged -- trace.py's
    behaviour, which also ends the loop. Parsed by `parse_anonymized` (official rule, T11)."""
    active = {attr: a for attr, a in analysis.items() if "chain" in a}
    # TJ-2: label each attribute's inference and chain; TJ-3: every guess
    inference = "\n\n".join(f"Type: {label_of(attr)}\n{a['inference']}\nGuess: {'; '.join(a['guesses'])}"
                            for attr, a in active.items())
    words: List[str] = []
    for a in active.values():
        words += [w for w in a["words"] if w not in words]
    chain = "\n\n".join(f"[{label_of(attr)}]\n{a['chain']}" for attr, a in active.items())

    prompt = ADVERSARIAL_REASONING_CHAIN_ANONYMIZATION_QUERY_PROMPT_TEMPLATE.format(
        comments=text, inference=inference, important_words=", ".join(words), reasoning_chain=chain,
    )
    response = chat(model, tokenizer, ADVERSARIAL_ANONYMIZATION_SYSTEM_PROMPT, prompt,
                    max_new_tokens=max_new_tokens)

    anonymized, parsed = parse_anonymized(response, text)
    return {"text": anonymized, "raw": response, "parsed": parsed}


def aa_inference_string(analysis: Dict[str, Dict]) -> str:
    """AA's inference block (LLMFullAnonymizer._create_anon_prompt): every attribute with a guess, as
    "Type: <key>\\nInference: <inference>\\nGuess: <guess>\\n" -- the guess list printed as AA prints it (a
    Python list). No certainty gate: AA anonymizes against every inference it has."""
    return "".join(f"Type: {attr}\nInference: {a['inference']}\nGuess: {a['guesses']}\n"
                   for attr, a in analysis.items() if a["guesses"])


def aa_rewrite(text: str, analysis: Dict[str, Dict], model, tokenizer, max_new_tokens: int) -> Dict:
    """AA@1 (2026-10-06): one round of Staab et al.'s adversarial anonymization -- the anonymizer sees the
    comments and the inferences only (no V_att words, no chain), with AA's own prompt (prompt_level 3).
    The inferences are the tag stage's (`analysis_from_cues`), as TRACE@1's; generation (greedy) and parsing
    (`parse_anonymized`) are TRACE@1's, so the two rewriting defenses differ only in what they are told
    (decision AA-1). Returns {text, raw, parsed}."""
    response = chat(model, tokenizer, AA_SYSTEM_PROMPT, aa_user_prompt(text, aa_inference_string(analysis)),
                    max_new_tokens=max_new_tokens)
    anonymized, parsed = parse_anonymized(response, text)
    return {"text": anonymized, "raw": response, "parsed": parsed}


__all__ = ["CERTAINTY_STOP", "MAX_ROUNDS", "REWRITE_MIN_NEW", "aa_inference_string", "aa_rewrite",
           "analysis_from_cues", "analyze", "clean_chain",
           "is_word_list",
           "leakage_chain", "needs_rewrite", "rewrite_budget", "stated_certainty",
           "parse_anonymized", "rewrite"]
