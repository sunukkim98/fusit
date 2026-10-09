"""
DP-Fusion: token-level differentially private inference.

Generation mixes the next-token distribution from a private context with the one from a
redacted public context, capping the per-step Renyi drift so the whole sequence carries a
formal (epsilon, delta) guarantee.

    Thareja et al. "DP-Fusion: Token-Level Differentially Private
    Inference for Large Language Models" (arXiv:2507.04531)

Modules:
    core       DPFusion, the user-facing wrapper
    contexts   length-matched public/per-group token sequences
    fusion     the mechanism: divergence, lambda search, the decoding loop
    epsilon    turns per-step divergences into an (epsilon, delta) guarantee
    prompting  the prompt templates (the official DP-Fusion-DPI one and fusit's chat-template one)
    run        one DP-Fusion run per document, as the experiments call it (+ decisions D1-D4)
    tagger     Document Privacy API client for automatic phrase extraction
"""

from fusit.dp_fusion.contexts import aligned_token_ids, build_aligned_tokens, build_contexts, locate_document
from fusit.dp_fusion.core import DPFusion, generate_dp_text
from fusit.dp_fusion.epsilon import compute_dp_epsilon, compute_epsilon_single_group
from fusit.dp_fusion.fusion import (
    DEFAULT_BETA_DICT,
    compute_renyi_divergence_clipped_symmetric,
    dp_fusion_groups_incremental,
    find_lambda,
)
from fusit.dp_fusion.prompting import format_prompt_new_template, official_nodpi_prompt, official_prompt
from fusit.dp_fusion.run import epsilon_single_group, generate_multi_group, generate_single_group
from fusit.dp_fusion.tagger import Tagger

__all__ = [
    "DEFAULT_BETA_DICT",
    "DPFusion",
    "Tagger",
    "aligned_token_ids",
    "build_aligned_tokens",
    "epsilon_single_group",
    "generate_multi_group",
    "generate_single_group",
    "official_nodpi_prompt",
    "official_prompt",
    "build_contexts",
    "compute_dp_epsilon",
    "compute_epsilon_single_group",
    "compute_renyi_divergence_clipped_symmetric",
    "dp_fusion_groups_incremental",
    "find_lambda",
    "format_prompt_new_template",
    "generate_dp_text",
    "locate_document",
]
