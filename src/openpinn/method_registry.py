"""High-level method names mapped to the components they configure.

The numerical implementations remain in their responsibility modules
(``models.py``, ``features.py``, ``sampling.py`` and so on).  This module is
only the small switchboard that lets users select one method by name without
copying a training program.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .geometry import SpaceTimeDomain
from .methods._types import ConfigFinalizer, MethodVariantSpec, OptionPatch
from .methods.adam_lbfgs import VARIANT as ADAM_LBFGS
from .methods.adaptive_activation import VARIANT as ADAPTIVE_ACTIVATION
from .methods.adaptive_sampling import VARIANTS as ADAPTIVE_SAMPLING_VARIANTS
from .methods.adaptive_weights import VARIANTS as ADAPTIVE_WEIGHTS_VARIANTS
from .methods.baseline import VARIANT as BASELINE
from .methods.domain_decomposition import VARIANT as DOMAIN_DECOMPOSITION
from .methods.fourier import VARIANT as FOURIER
from .methods.hard_boundary import VARIANT as HARD_BOUNDARY
from .methods.modified_mlp import VARIANT as MODIFIED_MLP
from .methods.time_window import VARIANT as TIME_WINDOW
from .methods.transfer import VARIANT as TRANSFER
from .problems import PDEProblem


def normalize_method_name(value: Any) -> str:
    """Normalize a public method name while keeping matching explicit."""

    normalized = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    if not normalized:
        raise ValueError("Method names must contain at least one letter or digit.")
    return normalized


def deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""

    result = copy.deepcopy(dict(base))
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


@dataclass(frozen=True)
class MethodSpec:
    """One method family resolved to one concrete default/selected variant."""

    name: str
    title: str
    component_file: str
    description: str
    variant_file: str = ""
    aliases: tuple[str, ...] = ()
    compatible_with: tuple[str, ...] = ("static", "dynamic", "custom")
    requires_checkpoint: bool = False
    default_patch: Mapping[str, Any] = field(default_factory=dict)
    option_patch: OptionPatch | None = None
    finalize_config: ConfigFinalizer | None = None
    default_variant: str = "default"
    variant: str = "default"
    variant_title: str = ""
    variants: tuple[MethodVariantSpec, ...] = ()


CompatibilityStatus = Literal["applicable", "conditional", "incompatible"]


@dataclass(frozen=True)
class MethodCompatibility:
    """Compatibility result for one concrete method variant."""

    method: str
    variant: str
    title: str
    status: CompatibilityStatus
    reason: str


@dataclass(frozen=True)
class ProblemCompatibilityReport:
    """Human- and program-readable method report for one problem."""

    problem: str
    family: str
    methods: tuple[MethodCompatibility, ...]

    @property
    def applicable(self) -> tuple[MethodCompatibility, ...]:
        return tuple(item for item in self.methods if item.status == "applicable")

    @property
    def conditional(self) -> tuple[MethodCompatibility, ...]:
        return tuple(item for item in self.methods if item.status == "conditional")

    @property
    def incompatible(self) -> tuple[MethodCompatibility, ...]:
        return tuple(item for item in self.methods if item.status == "incompatible")

    def to_text(self) -> str:
        """Return a concise Chinese report suitable for terminal output."""

        headings = (
            ("applicable", "结构上可直接尝试"),
            ("conditional", "满足附加条件后可尝试"),
            ("incompatible", "当前不适用"),
        )
        lines = [
            f"问题：{self.problem}",
            f"结构类型：{self.family}",
            "说明：这是训练前结构检查，不是精度保证；仍需 1–5 步短跑。",
        ]
        for status, heading in headings:
            lines.append(f"\n{heading}：")
            selected = [item for item in self.methods if item.status == status]
            if not selected:
                lines.append("  （无）")
                continue
            for item in selected:
                lines.append(
                    f"  - {item.method}/{item.variant}（{item.title}）：{item.reason}"
                )
        return "\n".join(lines)


class MethodRegistry:
    """Registry for method families and their selectable implementations."""

    def __init__(self) -> None:
        self._specs: dict[str, MethodSpec] = {}
        self._aliases: dict[str, str] = {}
        self._variant_aliases: dict[str, dict[str, str]] = {}

    def register(self, spec: MethodSpec) -> None:
        canonical = normalize_method_name(spec.name)
        if canonical in self._specs or canonical in self._aliases:
            raise ValueError(f"Method {spec.name!r} is already registered.")
        aliases = {canonical, *(normalize_method_name(item) for item in spec.aliases)}
        conflicts = sorted(alias for alias in aliases if alias in self._aliases)
        if conflicts:
            raise ValueError(f"Method alias conflict: {', '.join(conflicts)}.")
        variants = spec.variants
        if not variants:
            variants = (
                MethodVariantSpec(
                    name=spec.default_variant,
                    title=spec.variant_title or spec.title,
                    component_file=spec.component_file,
                    variant_file=spec.variant_file,
                    description=spec.description,
                    compatible_with=spec.compatible_with,
                    requires_checkpoint=spec.requires_checkpoint,
                    default_patch=spec.default_patch,
                    option_patch=spec.option_patch,
                    finalize_config=spec.finalize_config,
                ),
            )
        variant_map: dict[str, str] = {}
        variant_names = {normalize_method_name(item.name) for item in variants}
        default_variant = normalize_method_name(spec.default_variant)
        if default_variant not in variant_names:
            raise ValueError(
                f"Default variant {spec.default_variant!r} is not registered for {spec.name!r}."
            )
        for item in variants:
            item_name = normalize_method_name(item.name)
            for alias in {item_name, *(normalize_method_name(value) for value in item.aliases)}:
                if alias in variant_map:
                    raise ValueError(
                        f"Variant alias {alias!r} is duplicated in method family {spec.name!r}."
                    )
                variant_map[alias] = item_name
        self._specs[canonical] = replace(spec, variants=tuple(variants))
        self._variant_aliases[canonical] = variant_map
        for alias in aliases:
            self._aliases[alias] = canonical

    def resolve(self, name: str, variant: str | None = None) -> MethodSpec:
        normalized = normalize_method_name(name)
        try:
            family_name = self._aliases[normalized]
            family = self._specs[family_name]
        except KeyError as exc:
            choices = ", ".join(self.available())
            raise ValueError(
                f"Unknown OpenPINN method {name!r}; available methods: {choices}."
            ) from exc
        requested = family.default_variant if variant is None else variant
        requested_normalized = normalize_method_name(requested)
        try:
            selected_name = self._variant_aliases[family_name][requested_normalized]
        except KeyError as exc:
            choices = ", ".join(self.available_variants(family.name))
            raise ValueError(
                f"Unknown variant {requested!r} for method family {family.name!r}; "
                f"available variants: {choices}."
            ) from exc
        selected = next(
            item
            for item in family.variants
            if normalize_method_name(item.name) == selected_name
        )
        return replace(
            family,
            component_file=selected.component_file,
            variant_file=selected.variant_file,
            compatible_with=selected.compatible_with,
            requires_checkpoint=selected.requires_checkpoint,
            default_patch=selected.default_patch,
            option_patch=selected.option_patch,
            finalize_config=selected.finalize_config,
            variant=selected.name,
            variant_title=selected.title,
        )

    def available(self) -> tuple[str, ...]:
        return tuple(spec.name for _, spec in sorted(self._specs.items()))

    def specs(self) -> tuple[MethodSpec, ...]:
        return tuple(self.resolve(self._specs[key].name) for key in sorted(self._specs))

    def available_variants(self, name: str) -> tuple[str, ...]:
        normalized = normalize_method_name(name)
        try:
            family_name = self._aliases[normalized]
        except KeyError as exc:
            choices = ", ".join(self.available())
            raise ValueError(
                f"Unknown OpenPINN method {name!r}; available methods: {choices}."
            ) from exc
        return tuple(item.name for item in self._specs[family_name].variants)

    def variant_specs(self, name: str) -> tuple[MethodVariantSpec, ...]:
        normalized = normalize_method_name(name)
        try:
            family_name = self._aliases[normalized]
        except KeyError as exc:
            choices = ", ".join(self.available())
            raise ValueError(
                f"Unknown OpenPINN method {name!r}; available methods: {choices}."
            ) from exc
        return self._specs[family_name].variants


def _family(
    name: str,
    title: str,
    description: str,
    variant: MethodVariantSpec,
    *,
    aliases: tuple[str, ...] = (),
    variants: tuple[MethodVariantSpec, ...] | None = None,
) -> MethodSpec:
    """Build one family around a default plus optional sibling variants."""

    registered_variants = tuple(variants or (variant,))

    return MethodSpec(
        name=name,
        title=title,
        component_file=variant.component_file,
        variant_file=variant.variant_file,
        description=description,
        aliases=aliases,
        compatible_with=variant.compatible_with,
        requires_checkpoint=variant.requires_checkpoint,
        default_patch=variant.default_patch,
        option_patch=variant.option_patch,
        finalize_config=variant.finalize_config,
        default_variant=variant.name,
        variant=variant.name,
        variant_title=variant.title,
        variants=registered_variants,
    )


def default_method_registry() -> MethodRegistry:
    """Return ten method families plus the plain-PINN comparison family.

    Operator-learning names (coordinate models and grid operators) are not
    registered here. DeepONet and FNO models plus their training mode are selected with
    ``build_operator_config`` and ``train_operator``. See
    :mod:`openpinn.operator_protocol`.
    """

    registry = MethodRegistry()
    families = (
        _family(
            "baseline",
            "普通 PINN",
            "普通 PINN 对照家族。",
            BASELINE,
            aliases=("plain", "plain_pinn", "none"),
        ),
        _family(
            "modified_mlp",
            "改进 MLP",
            "网络结构改进家族。",
            MODIFIED_MLP,
            aliases=("gated_mlp",),
        ),
        _family(
            "adaptive_activation",
            "自适应激活",
            "可训练激活函数家族。",
            ADAPTIVE_ACTIVATION,
            aliases=("adaptive_tanh",),
        ),
        _family(
            "fourier",
            "Fourier 特征",
            "坐标特征映射家族。",
            FOURIER,
            aliases=("fourier_features", "rff"),
        ),
        _family(
            "hard_boundary",
            "强制边界",
            "把条件直接写进输出的约束家族。",
            HARD_BOUNDARY,
            aliases=("hard_constraint", "exact_bc"),
        ),
        _family(
            "adaptive_weights",
            "自适应权重",
            "训练中调整损失项权重的家族。",
            ADAPTIVE_WEIGHTS_VARIANTS[0],
            aliases=("self_adaptive_weights", "self_adaptive"),
            variants=ADAPTIVE_WEIGHTS_VARIANTS,
        ),
        _family(
            "adaptive_sampling",
            "自适应采样",
            "依据训练状态更新训练点的采样家族。",
            ADAPTIVE_SAMPLING_VARIANTS[0],
            aliases=("residual_adaptive", "rar"),
            variants=ADAPTIVE_SAMPLING_VARIANTS,
        ),
        _family(
            "adam_lbfgs",
            "组合优化器",
            "按阶段组合优化器的家族。",
            ADAM_LBFGS,
            aliases=("adam_then_lbfgs", "staged_optimizer"),
        ),
        _family(
            "time_window",
            "时间窗",
            "按时间顺序组织动态问题训练的家族。",
            TIME_WINDOW,
            aliases=("causal", "causal_training"),
        ),
        _family(
            "transfer",
            "迁移学习",
            "从兼容检查点初始化目标问题的家族。",
            TRANSFER,
            aliases=("transfer_learning", "warm_start"),
        ),
        _family(
            "domain_decomposition",
            "区域分解",
            "用多个网络处理不同子区域的家族。",
            DOMAIN_DECOMPOSITION,
            aliases=("xpinn", "multi_network"),
        ),
    )
    for family in families:
        registry.register(family)
    return registry


def problem_family(problem: str | PDEProblem) -> str:
    """Return the compatibility family used by the method selector."""

    if isinstance(problem, PDEProblem):
        if isinstance(problem.domain, SpaceTimeDomain):
            return "dynamic"
        kind = normalize_method_name(problem.metadata.get("kind", "custom"))
        if kind == "heterogeneousstatic":
            return "heterogeneous_poisson"
        if kind in {"static", "customstatic", "elliptic", "steadystate"}:
            return "static"
        return "custom"
    normalized = normalize_method_name(problem)
    if normalized in {"poisson", "poisson1d", "static"}:
        return "poisson"
    if normalized in {"burgers", "burgers1d", "burger", "dynamic"}:
        return "burgers"
    if normalized in {
        "tunnelmodel1",
        "model1",
        "tunnelconsolidationmodel1",
        "twintunnelconsolidation",
    }:
        return "dynamic"
    if normalized in {
        "heterogeneouspoisson",
        "heterogeneouspoisson1d",
        "piecewisepoisson",
        "hetero",
    }:
        return "heterogeneous_poisson"
    if normalized in {"coupledmanufactured", "coupledellipticmanufactured"}:
        return "custom"
    return "custom"


def _problem_family_aliases(problem: str | PDEProblem) -> tuple[str, set[str]]:
    family = problem_family(problem)
    aliases = {family}
    if family == "burgers":
        aliases.add("dynamic")
    if family in {"poisson", "heterogeneous_poisson"}:
        aliases.add("static")
    return family, aliases


def check_method_compatibility(
    problem: str | PDEProblem,
    *,
    has_checkpoint: bool = False,
    registry: MethodRegistry | None = None,
) -> ProblemCompatibilityReport:
    """List structurally applicable method variants before training.

    This deliberately does not claim numerical accuracy.  A returned
    ``applicable`` method has passed the registry's problem-family contract;
    every user-defined equation still needs a short smoke run.
    """

    selected_registry = registry or default_method_registry()
    family, aliases = _problem_family_aliases(problem)
    problem_name = problem.name if isinstance(problem, PDEProblem) else str(problem)
    results: list[MethodCompatibility] = []

    for method_spec in selected_registry.specs():
        for variant in selected_registry.variant_specs(method_spec.name):
            matched = bool(aliases.intersection(variant.compatible_with))
            if not matched:
                allowed = "、".join(variant.compatible_with)
                status: CompatibilityStatus = "incompatible"
                reason = f"实现仅声明支持：{allowed}"
            elif variant.requires_checkpoint:
                status = "conditional"
                reason = (
                    "已声明有 checkpoint，仍须核对网络输入、输出及结构"
                    if has_checkpoint
                    else "需要与当前网络输入、输出及结构匹配的 checkpoint.pt"
                )
            elif method_spec.name == "domain_decomposition":
                status = "conditional"
                reason = "当前只实现一维两材料界面，还需提供左右材料参数和界面位置"
            else:
                status = "applicable"
                reason = "问题结构与该实现的注册条件相符"
            results.append(
                MethodCompatibility(
                    method=method_spec.name,
                    variant=variant.name,
                    title=variant.title,
                    status=status,
                    reason=reason,
                )
            )

    return ProblemCompatibilityReport(
        problem=problem_name,
        family=family,
        methods=tuple(results),
    )


def apply_method(
    config: Mapping[str, Any],
    method: str,
    *,
    variant: str | None = None,
    problem: str | PDEProblem,
    steps: int,
    options: Mapping[str, Any] | None = None,
    registry: MethodRegistry | None = None,
) -> tuple[dict[str, Any], MethodSpec]:
    """Resolve one public method name and apply its minimal component patch."""

    selected_registry = registry or default_method_registry()
    spec = selected_registry.resolve(method, variant=variant)
    family, family_aliases = _problem_family_aliases(problem)
    compatible = bool(family_aliases.intersection(spec.compatible_with))
    if not compatible:
        allowed = ", ".join(spec.compatible_with)
        raise ValueError(
            f"Method {spec.name!r}/{spec.variant!r} is not compatible with "
            f"problem family {family!r}; "
            f"supported families: {allowed}."
        )

    resolved = deep_merge(config, spec.default_patch)
    from .method_options import validate_method_options

    selected_options = validate_method_options(spec.name, spec.variant, options)
    if spec.option_patch is not None:
        resolved = deep_merge(resolved, spec.option_patch(selected_options))
    elif selected_options:
        raise ValueError(
            f"Method {spec.name!r}/{spec.variant!r} does not define configurable options."
        )
    if spec.finalize_config is not None:
        resolved = spec.finalize_config(resolved, steps)
    resolved["method"] = {
        "name": spec.name,
        "variant": spec.variant,
        "variant_title": spec.variant_title,
        "title": spec.title,
        "component_file": spec.component_file,
        "variant_file": spec.variant_file,
        "options": selected_options,
        "resolved": True,
    }
    return resolved, spec


def list_methods() -> tuple[MethodSpec, ...]:
    """List method metadata for help pages and user interfaces."""

    return default_method_registry().specs()


def list_method_variants(method: str) -> tuple[MethodVariantSpec, ...]:
    """List concrete implementations currently registered under one family."""

    return default_method_registry().variant_specs(method)
