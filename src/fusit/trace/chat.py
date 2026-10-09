"""
One chat call, portable across the attacker models this package is run with.

TRACE-RPS calls OpenAI; everything here runs a local HF chat model instead, which means two
things the OpenAI path never had to handle: tokenizers that ship no chat template, and context
windows short enough for a long profile to overflow. Both are handled here so that every
caller in the package gets the same treatment and a model swap stays a one-line change.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

LLAMA2_CHAT_TEMPLATE = "<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"


def format_chat(tokenizer, model, system_prompt: str, user_prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    if getattr(model.config, "model_type", "") == "llama":
        return LLAMA2_CHAT_TEMPLATE.format(system=system_prompt.strip(), user=user_prompt)
    raise ValueError(
        f"{model.config.model_type!r} tokenizer has no chat_template and there is no fallback "
        "for it; add one here the way LLAMA2_CHAT_TEMPLATE does."
    )


@torch.no_grad()
def chat(model, tokenizer, system_prompt: str, user_prompt: str, max_new_tokens: int = 500) -> str:
    prompt = format_chat(tokenizer, model, system_prompt, user_prompt)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    # Llama-2's context is 4096 total and a long SynthPAI profile plus TRACE's prompt can
    # exceed it; generate() would then index past the position embeddings and crash. Keep the
    # tail, where the format instructions sit -- same trade repro/attackers.py makes.
    budget = getattr(model.config, "max_position_embeddings", 8192) - max_new_tokens - 16
    if inputs["input_ids"].shape[1] > budget:
        for key in ("input_ids", "attention_mask"):
            if key in inputs:
                inputs[key] = inputs[key][:, -budget:]

    inputs = inputs.to(model.device)
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = gen[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# -- batched greedy generation (the evaluation attack and the utility judge) --------------------
# Moved from attack_eval/backend.py (2026-10-04), unchanged. HF transformers, left-padded batches,
# greedy, the model's own generation_config EOS set. Greedy decoding under left padding is
# deterministic for a fixed batch composition, but fp16 kernels can pick different reduction
# orders for different batch shapes, so a query's output can differ in rare tokens between batch
# layouts (on the result tables, 14-25% of attacker answers differ in some token between two
# layouts of the same texts, flipping about one top-1 hit per row). `make_batches` therefore fixes
# the layout: longest first, fixed limits.

@dataclass
class Generation:
    text: Optional[str]
    n_prompt_tokens: int
    n_new_tokens: int
    hit_max_new_tokens: bool
    status: str                     # "ok" | "context_overflow"


class HFGenerator:
    def __init__(self, model_path: str, dtype: str = "float16", device: str = "cuda:0", seed: int = 10):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.model_path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype), device_map=device)
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        self.context_len = getattr(self.model.config, "max_position_embeddings", None)

    def encode(self, prompt: str, add_special_tokens: bool) -> List[int]:
        return self.tokenizer(prompt, add_special_tokens=add_special_tokens)["input_ids"]

    @torch.no_grad()
    def generate(self, batch: Sequence[Tuple[List[int], int]]) -> List[Generation]:
        """batch: [(prompt ids, max_new_tokens)]. One generate() call; max_new_tokens is the
        batch maximum, and each row is cut back to its own budget afterwards."""
        out: List[Optional[Generation]] = [None] * len(batch)
        live = []
        for i, (ids, max_new) in enumerate(batch):
            if self.context_len and len(ids) + max_new > self.context_len:
                out[i] = Generation(None, len(ids), 0, False, "context_overflow")
            else:
                live.append(i)
        if live:
            width = max(len(batch[i][0]) for i in live)
            pad = self.tokenizer.pad_token_id
            input_ids = torch.tensor([[pad] * (width - len(batch[i][0])) + batch[i][0] for i in live])
            mask = torch.tensor([[0] * (width - len(batch[i][0])) + [1] * len(batch[i][0]) for i in live])
            max_new = max(batch[i][1] for i in live)
            gen = self.model.generate(
                input_ids=input_ids.to(self.model.device), attention_mask=mask.to(self.model.device),
                max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=pad,
            )
            for row, i in enumerate(live):
                new = gen[row, width:].tolist()[: batch[i][1]]
                stop = next((j for j, t in enumerate(new) if t in self.eos_ids), None)
                kept = new if stop is None else new[:stop]
                out[i] = Generation(self.tokenizer.decode(kept, skip_special_tokens=True),
                                    len(batch[i][0]), len(kept), stop is None, "ok")
        return out


def make_batches(lengths: Sequence[int], max_new: Sequence[int], batch_size: int,
                 max_batch_tokens: int) -> List[List[int]]:
    """Indices grouped longest-first so an OOM shows up in the first batch, capped by row count
    and by padded tokens (rows x (longest prompt + max_new_tokens))."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches, cur = [], []
    for i in order:
        cand = cur + [i]
        width = max(lengths[j] for j in cand) + max(max_new[j] for j in cand)
        if cur and (len(cand) > batch_size or len(cand) * width > max_batch_tokens):
            batches.append(cur)
            cand = [i]
        cur = cand
    if cur:
        batches.append(cur)
    return batches


def generate_all(gen: "HFGenerator", prompts: Sequence[Tuple[List[int], int]], batch_size: int,
                 max_batch_tokens: int, on_batch=None) -> List[Generation]:
    """Every (ids, max_new) through `make_batches`, in input order; `on_batch(indices, outputs)`
    after each batch (for checkpointing)."""
    out: List[Optional[Generation]] = [None] * len(prompts)
    for idx in make_batches([len(p[0]) for p in prompts], [p[1] for p in prompts], batch_size, max_batch_tokens):
        res = gen.generate([prompts[i] for i in idx])
        for i, g in zip(idx, res):
            out[i] = g
        if on_batch is not None:
            on_batch(idx, res)
    return out


__all__ = ["Generation", "HFGenerator", "LLAMA2_CHAT_TEMPLATE", "chat", "format_chat", "generate_all", "make_batches"]
