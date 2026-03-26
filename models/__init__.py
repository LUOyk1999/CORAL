"""
CORAL Models Package

SmolVLM-VLA model for LIBERO robot manipulation with CORAL LoRA experts.
"""

from .configuration_smolvlm_vla import SmolVLMVLAConfig
from .modeling_smolvlm_vla import SmolVLMVLA
from .processing_smolvlm_vla import SmolVLMVLAProcessor

from .action_hub import (
    BaseActionSpace,
    build_action_space,
    register_action,
    LiberoJointActionSpace,
    ACTION_REGISTRY,
)

from .transformer_smolvlm import (
    SmolVLMActionTransformer,
    TransformerBlock,
    DiTBlock,
    FinalLayer,
    Attention,
    Mlp,
    timestep_embedding,
)

__all__ = [
    "SmolVLMVLAConfig",
    "SmolVLMVLA",
    "SmolVLMVLAProcessor",
    "BaseActionSpace",
    "build_action_space",
    "register_action",
    "LiberoJointActionSpace",
    "ACTION_REGISTRY",
    "SmolVLMActionTransformer",
    "TransformerBlock",
    "DiTBlock",
    "FinalLayer",
    "Attention",
    "Mlp",
    "timestep_embedding",
]
