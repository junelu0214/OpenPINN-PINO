"""Adaptive-sampling variants."""

from .residual_adaptive import VARIANT
from .gradient_movement import VARIANT as GRADIENT_MOVEMENT
from .importance_sampling import VARIANT as IMPORTANCE_SAMPLING
from .residual_movement import VARIANT as RESIDUAL_MOVEMENT

VARIANTS = (
    VARIANT,
    RESIDUAL_MOVEMENT,
    GRADIENT_MOVEMENT,
    IMPORTANCE_SAMPLING,
)

__all__ = ["VARIANT", "VARIANTS"]
