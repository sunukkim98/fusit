"""
p(value | text) over a fixed candidate set, and the divergences built on it.

The prompt is TRACE's attack prompt with its "First reason step-by-step ... Certainty" block
replaced by a request for a single answer, so that the answer slot can be scored by
log-likelihood instead of generated. A candidate's score is the log-likelihood of
`" " + candidate` after the assistant prefill `Type: {attr}\nGuess:` (per token; summed, averaged
or first-token, see NORMS), renormalized over the candidate set. Comparing D, D minus c, the
empty document and c alone under the SAME candidate set cancels most of the length bias between
candidates.

Candidate scoring reuses the prefix KV cache: one forward over `prompt + prefill`, then each
candidate's tokens are fed on top of that cache and the cache is cropped back. Candidates are
scored one at a time on purpose -- repeating a 3.8k-token cache six times for a batch would not
fit next to a 7B fp16 model on a 24 GB card, and a candidate is only a few tokens.
"""

import math
from typing import Dict, List, Sequence, Tuple

import torch

from fusit.trace.attributes import ATTRIBUTE_OPTIONS, label_of
from fusit.trace.chat import format_chat
from fusit.trace.prompts import ADVERSARIAL_INFERENCE_QUERY_PROMPT_TEMPLATE, ADVERSARIAL_INFERENCE_SYSTEM_PROMPT

REASONING_ANCHOR = "First reason step-by-step"
DIRECT_FOOTER = (
    "Answer directly without explanation. Follow exactly the format specified below:\n\n"
    "Type: {target_attribute}\n"
    "Guess: (single best guess){target_attribute_options}\n"
)
#: the document slot's content for p(value | no text)
EMPTY_TEXT = "no comments available"

#: how a candidate's per-token log-probs become one score
#:   sum    registered: log p(candidate | prefix)
#:   mean   per-token average (length-normalized)
#:   first  first token only
NORMS = ("sum", "mean", "first")


def direct_messages(text: str, attribute: str) -> Tuple[str, str, str]:
    """(system, user, assistant prefill) of the direct-answer variant of the attack prompt."""
    template = ADVERSARIAL_INFERENCE_QUERY_PROMPT_TEMPLATE
    cut = template.index(REASONING_ANCHOR)
    user = (template[:cut] + DIRECT_FOOTER).format(
        target_attribute=label_of(attribute), target_attribute_options=ATTRIBUTE_OPTIONS.get(attribute, ""),
        comments=text)
    return ADVERSARIAL_INFERENCE_SYSTEM_PROMPT, user, f"Type: {label_of(attribute)}\nGuess:"


def direct_prefix(text: str, attribute: str, model, tokenizer) -> str:
    system, user, prefill = direct_messages(text, attribute)
    return format_chat(tokenizer, model, system.strip(), user) + " " + prefill


def _encode(tokenizer, s: str) -> List[int]:
    return tokenizer(s, add_special_tokens=False)["input_ids"]      # the template carries <s>


def _split(tokenizer, prefix: str, cand: str) -> Tuple[List[int], List[int]]:
    """(prefix ids, candidate ids) such that prefix + candidate tokenizes to their concatenation.
    When the joint tokenization disagrees with the prefix alone at the boundary, the candidate is
    taken as every token that ends past the prefix."""
    p = _encode(tokenizer, prefix)
    full = tokenizer(prefix + cand, add_special_tokens=False, return_offsets_mapping=True)
    f = full["input_ids"]
    if f[:len(p)] == p:
        return p, f[len(p):]
    cut = next(i for i, (_, e) in enumerate(full["offset_mapping"]) if e > len(prefix))
    return f[:cut], f[cut:]


@torch.no_grad()
def candidate_token_logprobs(model, tokenizer, prefix: str, candidates: Sequence[str]) -> List[List[float]]:
    """Per candidate, [log p(token t | prefix, earlier candidate tokens)] in float32. Each candidate
    is scored as the continuation `prefix + " " + candidate`."""
    splits = [_split(tokenizer, prefix, " " + c) for c in candidates]
    base = _encode(tokenizer, prefix)
    out: List[List[float]] = [[] for _ in candidates]
    shared = [i for i, (p, c) in enumerate(splits) if p == base and c]
    if shared:
        x = torch.tensor([base], device=model.device)
        o = model(input_ids=x, use_cache=True, logits_to_keep=1)
        cache = o.past_key_values
        first = torch.log_softmax(o.logits[0, -1].float(), dim=-1)
        for i in shared:
            c = splits[i][1]
            per = [float(first[c[0]])]
            if len(c) > 1:
                oc = model(input_ids=torch.tensor([c[:-1]], device=model.device), past_key_values=cache,
                           use_cache=True)
                lps = torch.log_softmax(oc.logits[0].float(), dim=-1)
                per += [float(lps[j, c[j + 1]]) for j in range(len(c) - 1)]
                cache.crop(len(base))
            out[i] = per
        del cache, o
    for i, (p, c) in enumerate(splits):
        if i in shared:
            continue
        ids = p + c                                    # boundary disagreed: full forward
        o = model(input_ids=torch.tensor([ids], device=model.device))
        lps = torch.log_softmax(o.logits[0].float(), dim=-1)
        out[i] = [float(lps[len(p) + j - 1, c[j]]) for j in range(len(c))]
    return out


def token_logprobs(text: str, attribute: str, candidates: Sequence[str], model, tokenizer) -> List[List[float]]:
    return candidate_token_logprobs(model, tokenizer, direct_prefix(text, attribute, model, tokenizer), candidates)


def reduce(per_token: Sequence[float], norm: str) -> float:
    if norm == "sum":
        return float(sum(per_token))
    if norm == "mean":
        return float(sum(per_token)) / len(per_token)
    if norm == "first":
        return float(per_token[0])
    raise ValueError(f"unknown norm {norm!r}, expected one of {NORMS}")


def softmax(logps: Sequence[float]) -> List[float]:
    m = max(logps)
    ex = [math.exp(x - m) for x in logps]
    z = sum(ex)
    return [x / z for x in ex]


def distribution_from(candidates: Sequence[str], tok_lps: Sequence[Sequence[float]], norm: str = "sum") -> Dict[str, float]:
    return dict(zip(candidates, softmax([reduce(t, norm) for t in tok_lps])))


def entropy(p: Dict[str, float]) -> float:
    return -sum(v * math.log2(v) for v in p.values() if v > 0)


def jsd(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen-Shannon divergence, log base 2, in [0, 1]."""
    out = 0.0
    for k in p:
        m = 0.5 * (p[k] + q[k])
        if p[k] > 0:
            out += 0.5 * p[k] * math.log2(p[k] / m)
        if q[k] > 0:
            out += 0.5 * q[k] * math.log2(q[k] / m)
    return max(0.0, min(1.0, out))


def argmax(p: Dict[str, float]) -> str:
    """Ties go to the earlier candidate (dict order = candidate order)."""
    best = max(p.values())
    return next(k for k, v in p.items() if v == best)


__all__ = ["EMPTY_TEXT", "NORMS", "argmax", "candidate_token_logprobs", "direct_messages", "direct_prefix",
           "distribution_from", "entropy", "jsd", "reduce", "softmax", "token_logprobs"]
