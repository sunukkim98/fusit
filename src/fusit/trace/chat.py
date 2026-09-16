"""
One chat call, portable across the attacker models this package is run with.

TRACE-RPS calls OpenAI; everything here runs a local HF chat model instead, which means two
things the OpenAI path never had to handle: tokenizers that ship no chat template, and context
windows short enough for a long profile to overflow. Both are handled here so that every
caller in the package gets the same treatment and a model swap stays a one-line change.
"""

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


__all__ = ["LLAMA2_CHAT_TEMPLATE", "chat", "format_chat"]
