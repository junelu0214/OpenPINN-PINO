"""Adaptive-loss-weight variants."""

from .self_adaptive import VARIANT
from .relobralo import VARIANT as RELOBRALO
from .magnitude_power import VARIANT as MAGNITUDE_POWER
from .pointwise_self_adaptive import VARIANT as POINTWISE_SELF_ADAPTIVE
from .uncertainty_weighting import VARIANT as UNCERTAINTY_WEIGHTING

VARIANTS = (
    VARIANT,
    UNCERTAINTY_WEIGHTING,
    RELOBRALO,
    MAGNITUDE_POWER,
    POINTWISE_SELF_ADAPTIVE,
)

__all__ = ["VARIANT", "VARIANTS"]
