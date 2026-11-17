"""Shared metadata contract for one concrete method variant."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

OptionPatch = Callable[[Mapping[str, Any]], Mapping[str, Any]]
ConfigFinalizer = Callable[[dict[str, Any], int], dict[str, Any]]


@dataclass(frozen=True)
class MethodVariantSpec:
    """One implementation inside a broader OpenPINN method family."""

    name: str
    title: str
    component_file: str
    variant_file: str
    description: str
    aliases: tuple[str, ...] = ()
    compatible_with: tuple[str, ...] = ("static", "dynamic", "custom")
    requires_checkpoint: bool = False
    default_patch: Mapping[str, Any] = field(default_factory=dict)
    option_patch: OptionPatch | None = None
    finalize_config: ConfigFinalizer | None = None
