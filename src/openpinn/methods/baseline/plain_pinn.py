"""Default plain PINN used as the comparison control."""

from .._types import MethodVariantSpec

VARIANT = MethodVariantSpec(
    name="plain_pinn",
    title="普通 PINN",
    component_file="models.py",
    variant_file="methods/baseline/plain_pinn.py",
    description="普通 MLP、均匀采样、固定损失权重和 Adam。",
    aliases=("plain", "none"),
)
