"""Typed component factories for configuration-driven construction.

The registry is intentionally small: it maps a component *category* and a
normalized name to one factory, then validates the factory result against the
category contract.  It is not a global Trick manager and contains no training
or experiment orchestration logic.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from torch import nn

from .formulations import FormulationProtocol
from .optimizers import OptimizerAdapter
from .problems import PDEProblem
from .sampling import SamplerProtocol


ComponentKind: TypeAlias = Literal[
    "problem",
    "model",
    "feature",
    "constraint",
    "formulation",
    "sampler",
    "optimizer",
    "loss",
    "derivative",
]
ComponentFactory: TypeAlias = Callable[..., Any]
ComponentValidator: TypeAlias = Callable[[Any], bool]

_KINDS: tuple[ComponentKind, ...] = (
    "problem",
    "model",
    "feature",
    "constraint",
    "formulation",
    "sampler",
    "optimizer",
    "loss",
    "derivative",
)

# Public second-development kinds that MethodSpec/YAML names resolve through.
PUBLIC_EXTENSION_KINDS: tuple[ComponentKind, ...] = (
    "model",
    "sampler",
    "loss",
    "optimizer",
    "derivative",
)


def component_kinds() -> tuple[ComponentKind, ...]:
    """Return every first-class component category, including built-in kinds."""

    return _KINDS


def _is_loss_composer(value: Any) -> bool:
    if callable(getattr(value, "compose", None)) or callable(
        getattr(value, "compose_pointwise", None)
    ):
        return True
    return callable(value)


def _is_derivative(value: Any) -> bool:
    has_autodiff = callable(getattr(value, "grad", None)) and callable(
        getattr(value, "hessian", None)
    )
    has_grid = callable(getattr(value, "partial", None))
    return has_autodiff or has_grid


def normalize_component_name(value: Any) -> str:
    """Normalize a YAML/public name without introducing fuzzy matching."""

    normalized = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    if not normalized:
        raise ValueError("Component names must contain at least one letter or digit.")
    return normalized


@dataclass(frozen=True)
class _ComponentSpec:
    canonical_name: str
    factory: ComponentFactory
    aliases: tuple[str, ...]


class ComponentRegistry:
    """Category-separated registry with duplicate and result-type checks.

    Registries are explicit objects rather than mutable process globals.  Use
    :func:`openpinn.default_component_registry` to obtain an isolated registry
    containing the built-in components, then register project-specific
    factories on that object and pass it to :class:`openpinn.Experiment`.
    """

    def __init__(self) -> None:
        self._specs: dict[ComponentKind, dict[str, _ComponentSpec]] = {
            kind: {} for kind in _KINDS
        }
        self._aliases: dict[ComponentKind, dict[str, str]] = {
            kind: {} for kind in _KINDS
        }
        self._validators: dict[ComponentKind, tuple[ComponentValidator, str]] = {
            "problem": (lambda value: isinstance(value, PDEProblem), "PDEProblem"),
            "model": (lambda value: isinstance(value, nn.Module), "torch.nn.Module"),
            "feature": (lambda value: isinstance(value, nn.Module), "torch.nn.Module"),
            "constraint": (
                lambda value: isinstance(value, nn.Module),
                "torch.nn.Module",
            ),
            "formulation": (
                lambda value: isinstance(value, FormulationProtocol),
                "FormulationProtocol",
            ),
            "sampler": (
                lambda value: isinstance(value, SamplerProtocol),
                "SamplerProtocol",
            ),
            "optimizer": (
                lambda value: isinstance(value, OptimizerAdapter),
                "OptimizerAdapter",
            ),
            "loss": (
                _is_loss_composer,
                "loss composer with compose(), compose_pointwise(), or a terms callable",
            ),
            "derivative": (
                _is_derivative,
                "derivative backend with grad()/hessian() or partial()",
            ),
        }

    @staticmethod
    def _kind(value: str) -> ComponentKind:
        if value not in _KINDS:
            supported = ", ".join(_KINDS)
            raise ValueError(
                f"Unknown component category {value!r}; choose one of: {supported}."
            )
        return value  # type: ignore[return-value]

    def register(
        self,
        kind: ComponentKind,
        name: str,
        factory: ComponentFactory,
        *,
        aliases: Iterable[str] = (),
        replace: bool = False,
    ) -> None:
        """Register one factory and optional aliases inside one category.

        Duplicate names and aliases are rejected unless ``replace=True``.
        Replacing a canonical entry removes all aliases of the previous entry,
        so stale aliases cannot continue constructing the old factory.
        """

        selected_kind = self._kind(kind)
        if not callable(factory):
            raise TypeError("Component factory must be callable.")
        canonical = normalize_component_name(name)
        alias_names: set[str] = set()
        for alias in aliases:
            normalized_alias = normalize_component_name(alias)
            if normalized_alias != canonical:
                alias_names.add(normalized_alias)
        normalized_aliases = tuple(sorted(alias_names))
        specs = self._specs[selected_kind]
        alias_map = self._aliases[selected_kind]

        existing = specs.get(canonical)
        if existing is not None and not replace:
            raise ValueError(
                f"Component {kind}:{canonical!r} is already registered."
            )
        proposed = (canonical, *normalized_aliases)
        conflicts = sorted(
            key
            for key in proposed
            if key in alias_map and alias_map[key] != canonical
        )
        if conflicts:
            raise ValueError(
                f"Component alias conflict in {kind}: {', '.join(conflicts)}."
            )
        if existing is not None:
            for key, target in tuple(alias_map.items()):
                if target == canonical:
                    del alias_map[key]

        spec = _ComponentSpec(canonical, factory, normalized_aliases)
        specs[canonical] = spec
        for key in proposed:
            alias_map[key] = canonical

    def contains(self, kind: ComponentKind, name: str) -> bool:
        """Return whether a normalized name or alias is registered."""

        selected_kind = self._kind(kind)
        try:
            normalized = normalize_component_name(name)
        except ValueError:
            return False
        return normalized in self._aliases[selected_kind]

    def canonical_name(self, kind: ComponentKind, name: str) -> str:
        """Resolve a name or alias to its canonical registered name."""

        selected_kind = self._kind(kind)
        normalized = normalize_component_name(name)
        try:
            return self._aliases[selected_kind][normalized]
        except KeyError as exc:
            choices = ", ".join(self.available(selected_kind)) or "<none>"
            raise ValueError(
                f"Component {kind}:{name!r} is not registered; available: {choices}."
            ) from exc

    def available(self, kind: ComponentKind) -> tuple[str, ...]:
        """Return sorted canonical component names for a category."""

        selected_kind = self._kind(kind)
        return tuple(sorted(self._specs[selected_kind]))

    def factory(self, kind: ComponentKind, name: str) -> ComponentFactory:
        """Return a registered factory without constructing its component."""

        selected_kind = self._kind(kind)
        canonical = self.canonical_name(selected_kind, name)
        return self._specs[selected_kind][canonical].factory

    def create(self, kind: ComponentKind, name: str, /, **kwargs: Any) -> Any:
        """Construct and category-check one component."""

        selected_kind = self._kind(kind)
        canonical = self.canonical_name(selected_kind, name)
        spec = self._specs[selected_kind][canonical]
        component = spec.factory(**kwargs)
        validator, expected = self._validators[selected_kind]
        if not validator(component):
            raise TypeError(
                f"Factory for {kind}:{canonical!r} returned "
                f"{type(component).__name__}, expected {expected}."
            )
        return component

    def copy(self) -> "ComponentRegistry":
        """Return an independent registry containing the same factories."""

        duplicate = ComponentRegistry()
        duplicate._specs = {
            kind: dict(specs) for kind, specs in self._specs.items()
        }
        duplicate._aliases = {
            kind: dict(aliases) for kind, aliases in self._aliases.items()
        }
        return duplicate
