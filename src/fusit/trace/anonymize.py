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
  `fusit.trace.parsing` already recovers the formats Llama-2 drifts into.
- A rewrite that restates the prompt is cut back to its comments (see `rewrite`).
"""

from typing import Dict, List, Sequence

from fusit.trace.attributes import label_of, question_of
from fusit.trace.chat import chat
from fusit.trace.prompts import (
    ADVERSARIAL_ANONYMIZATION_SYSTEM_PROMPT,
    ADVERSARIAL_REASONING_CHAIN_ANONYMIZATION_QUERY_PROMPT_TEMPLATE,
    PRIVACY_LEAKAGE_CHAIN_PROMPT_TEMPLATE,
    PRIVACY_LEAKAGE_CHAIN_SYSTEM_PROMPT,
)
from fusit.trace.signals import attention_spans, guess_attribute

#: trace.py's `adversarial_anonymization(max_iterations=5)`.
MAX_ROUNDS = 5

#: trace.py stops once `inference['certainty'] <= 2`.
CERTAINTY_STOP = 2

#: How the rewriting prompt opens; see `rewrite` for why a response may start with it.
PROMPT_ECHO = "Below I give you a set of comments from a user"


def leakage_chain(text: str, attribute: str, inference: str, guesses: Sequence[str],
                  model, tokenizer) -> str:
    """trace.py's `privacy_leakage_chain`, returning the chain text itself (the cue tagger
    only keeps the quotes)."""
    prompt = PRIVACY_LEAKAGE_CHAIN_PROMPT_TEMPLATE.format(
        comments=text, inference=inference, guess="; ".join(guesses),
        target_attribute=label_of(attribute),
    )
    response = chat(model, tokenizer, PRIVACY_LEAKAGE_CHAIN_SYSTEM_PROMPT, prompt, max_new_tokens=500)
    response = response.replace("*", "").replace("#", "")
    if response.startswith("Inference Chain:\n"):
        response = response[len("Inference Chain:\n"):].strip()
    return response


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


def needs_rewrite(analysis: Dict[str, Dict]) -> bool:
    return any("chain" in a for a in analysis.values())


def rewrite(text: str, analysis: Dict[str, Dict], model, tokenizer, max_new_tokens: int) -> Dict:
    """One round's anonymizer side. Returns {text, raw, parsed}; `parsed` is False when the
    response had no `#` separator, in which case `text` is the input unchanged -- trace.py's
    behaviour, which also ends the loop."""
    active = [a for a in analysis.values() if "chain" in a]
    inference = "\n\n".join(f"{a['inference']}\nGuess: {'; '.join(a['guesses'])}" for a in active)
    words: List[str] = []
    for a in active:
        words += [w for w in a["words"] if w not in words]
    chain = "\n\n".join(a["chain"] for a in active)

    prompt = ADVERSARIAL_REASONING_CHAIN_ANONYMIZATION_QUERY_PROMPT_TEMPLATE.format(
        comments=text, inference=inference, important_words=", ".join(words), reasoning_chain=chain,
    )
    response = chat(model, tokenizer, ADVERSARIAL_ANONYMIZATION_SYSTEM_PROMPT, prompt,
                    max_new_tokens=max_new_tokens)

    parsed_text = response
    lines = parsed_text.splitlines()
    if lines and "explanation" in lines[0].lower():
        parsed_text = "\n".join(lines[1:]).strip()
    if "#" not in parsed_text:
        return {"text": text, "raw": response, "parsed": False}
    anonymized = parsed_text.split("#", 1)[1].strip()
    # Qwen2.5-7B sometimes restates the task after the `#` -- the instruction paragraph, then
    # "Comments:" and the text, then another `#` and the whole thing again -- which the
    # reference parser (written against GPT) would publish, instructions included, at twice
    # the length. Keep only the first comments block; `echo` records that it happened.
    echo = anonymized.startswith(PROMPT_ECHO) and "Comments:\n" in anonymized
    if echo:
        anonymized = anonymized.split("Comments:\n", 1)[1].split("\n#\n", 1)[0].strip()
    if anonymized.startswith("Comments:\n"):
        anonymized = anonymized[len("Comments:\n"):]
    if "Inference for comments:" in anonymized:
        anonymized = anonymized.split("Inference for comments:", 1)[0]
    return {"text": anonymized, "raw": response, "parsed": True, "echo": echo}


__all__ = ["CERTAINTY_STOP", "MAX_ROUNDS", "analyze", "leakage_chain", "needs_rewrite", "rewrite"]
