"""
DP-Fusion: token-level differentially private inference.

Generation mixes the next-token distribution from a private context with the one from a
redacted public context, capping the per-step Renyi drift so the whole sequence carries a
formal (epsilon, delta) guarantee.

    Thareja et al. "DP-Fusion: Token-Level Differentially Private
    Inference for Large Language Models" (arXiv:2507.04531)

Modules:
    core       DPFusion, the user-facing wrapper
    fusion     the mechanism: divergence, lambda search, the decoding loop
    epsilon    turns per-step divergences into an (epsilon, delta) guarantee
    prompting  the shared private/public prompt template
    tagger     Document Privacy API client for automatic phrase extraction
"""

from fusit.dp_fusion.core import DPFusion, generate_dp_text
from fusit.dp_fusion.epsilon import compute_dp_epsilon, compute_epsilon_single_group
from fusit.dp_fusion.fusion import (
    DEFAULT_BETA_DICT,
    compute_renyi_divergence_clipped_symmetric,
    dp_fusion_groups_incremental,
    find_lambda,
)
from fusit.dp_fusion.prompting import format_prompt_new_template
from fusit.dp_fusion.tagger import Tagger

__all__ = [
    "DEFAULT_BETA_DICT",
    "DPFusion",
    "Tagger",
    "compute_dp_epsilon",
    "compute_epsilon_single_group",
    "compute_renyi_divergence_clipped_symmetric",
    "dp_fusion_groups_incremental",
    "find_lambda",
    "format_prompt_new_template",
    "generate_dp_text",
]
