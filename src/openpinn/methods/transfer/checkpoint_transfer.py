"""Full or partial transfer from a compatible OpenPINN checkpoint."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .._types import MethodVariantSpec


def _options(values: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = dict(values)
    checkpoint = selected.pop(
        "checkpoint", selected.pop("initial_checkpoint", None)
    )
    last_layers = selected.pop("trainable_last_linear_layers", None)
    if selected:
        raise ValueError(
            "transfer options support checkpoint and trainable_last_linear_layers; got: "
            + ", ".join(sorted(selected))
        )
    if checkpoint in {None, ""}:
        raise ValueError(
            "transfer requires method_options={'checkpoint': 'path/to/checkpoint.pt'}."
        )
    patch: dict[str, Any] = {
        "training": {"initial_checkpoint": str(Path(checkpoint))}
    }
    if last_layers is not None:
        patch["model"] = {"trainable_last_linear_layers": int(last_layers)}
    return patch


VARIANT = MethodVariantSpec(
    name="checkpoint_transfer",
    title="兼容检查点迁移",
    component_file="training.py, models.py",
    variant_file="methods/transfer/checkpoint_transfer.py",
    description="加载兼容检查点，微调整网或最后若干线性层。",
    aliases=("transfer_learning", "warm_start"),
    requires_checkpoint=True,
    option_patch=_options,
)
