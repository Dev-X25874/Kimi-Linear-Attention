"""Kimi Linear: An Expressive, Efficient Attention Architecture."""

from .config import KimiLinearConfig, KimiLinearTrainingConfig
from .kda import KimiDeltaAttention
from .mla import MultiHeadLatentAttention
from .model import KimiLinearModel

__version__ = "0.1.0"

__all__ = [
    "KimiLinearConfig",
    "KimiLinearTrainingConfig",
    "KimiDeltaAttention",
    "MultiHeadLatentAttention",
    "KimiLinearModel",
]
