"""
Running DP-Fusion on one document, as every experiment calls it.

    generate_single_group   m = 1: "PRIVATE" reveals X_priv, "PUBLIC" hides it (equal-length ids
                            from contexts.aligned_token_ids); one cap alpha*beta
    generate_multi_group    m groups, every group the same cap
    epsilon_single_group    (eps, delta) of a run, Theorem 4 with m = 1

Differences from the official DP-Fusion-DPI code (DP-FUSION_Defense.py):
    - distributions are softmaxed in float32 (fusion.py); the official code softmaxes fp16 logits
    - no fp16 cast of the KV cache: the model already runs in fp16, so it would be a no-op
    - the first forward keeps only the last position's logits (less memory, same tokens)
    - epsilon follows the paper: Algorithm 1 bounds D_alpha by alpha*beta and Theorem 4 is
      written in beta, so an observed divergence d enters as beta = d / alpha. The official
      code passes d itself as beta, which reports epsilon alpha times larger.
As in the official code, the first generated token's lambda / divergence is not recorded, so
epsilon covers tokens 2..T.
"""

import torch

from fusit.dp_fusion.epsilon import compute_epsilon_single_group
from fusit.dp_fusion.fusion import dp_fusion_groups_incremental

__all__ = ["epsilon_single_group", "generate_multi_group", "generate_single_group"]


def generate_single_group(model, tokenizer, private, public, ab, seed, max_new_tokens,
                          alpha=2.0, temperature=1.0):
    """`(generated_ids, lambdas, divergences)` of one DP-Fusion run with one private group."""
    io = {"PUBLIC": torch.tensor(public), "PRIVATE": torch.tensor(private)}
    torch.manual_seed(seed)
    _, lam, div = dp_fusion_groups_incremental(
        token_ids_groups=io, beta_dict={"PRIVATE": ab}, alpha=alpha, model=model,
        tokenizer=tokenizer, temperature=temperature, max_new_tokens=max_new_tokens,
    )
    return io["PUBLIC"][len(public):].tolist(), lam.get("PRIVATE", []), div.get("PRIVATE", [])


def generate_multi_group(model, tokenizer, groups_ids, ab, seed, max_new_tokens,
                         alpha=2.0, temperature=1.0):
    """`(generated_ids, lambdas_by_group, divergences_by_group)`; every group gets the cap `ab`."""
    io = {g: torch.tensor(v) for g, v in groups_ids.items()}
    n_prompt = len(groups_ids["PUBLIC"])
    torch.manual_seed(seed)
    _, lam, div = dp_fusion_groups_incremental(
        token_ids_groups=io, beta_dict={g: ab for g in groups_ids if g != "PUBLIC"}, alpha=alpha,
        model=model, tokenizer=tokenizer, temperature=temperature, max_new_tokens=max_new_tokens,
    )
    return io["PUBLIC"][n_prompt:].tolist(), lam, div


def epsilon_single_group(divergences, ab, alpha=2.0, delta=1e-3):
    """compute_epsilon_single_group with beta = ab / alpha (see the module docstring)."""
    return compute_epsilon_single_group(divergences, alpha=alpha, delta=delta, beta=ab / alpha)
