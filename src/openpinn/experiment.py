"""Configuration-driven benchmark experiment orchestration."""

from __future__ import annotations

import copy
import hashlib
import math
import platform
import time
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .artifacts import ArtifactWriter
from .benchmarks import (
    make_burgers_problem,
    make_coupled_manufactured_problem,
    make_piecewise_poisson_problem,
    make_poisson_problem,
    make_tunnel_consolidation_model1_problem,
)
from .config import load_config
from .conditions import DirichletBC
from .constraints import HardConstraint, IdentityOutput
from .features import FourierFeatures, IdentityFeature
from .formulations import (
    FormulationProtocol,
    InterfaceStrongFormulation,
    StrongFormulation,
)
from .geometry import (
    ClosedSpline,
    Disk,
    Ellipse,
    Interval,
    Polygon,
    RectangleWithCircularHoles,
    RoundedRectangle,
    SpaceTimeDomain,
)
from .losses import (
    AdaptiveLossComposer,
    LossComposer,
    MagnitudePowerLossComposer,
    PointwiseAdaptiveLossComposer,
    ReLoBRaLoLossComposer,
    UncertaintyLossComposer,
)
from .metrics import (
    max_absolute_error,
    mean_absolute_error,
    r2_score,
    relative_l2_error,
    rmse,
)
from .models import (
    DomainDecompositionPINN,
    MLP,
    ModifiedMLP,
    PINNModel,
    freeze_all_but_last_linear_layers,
)
from .optimizers import (
    AdamAdapter,
    AdamWAdapter,
    LBFGSAdapter,
    OptimizerAdapter,
    StagedOptimizerAdapter,
)
from .problems import PDEProblem
from .registry import ComponentKind, ComponentRegistry, normalize_component_name
from .sampling import (
    AdaptiveMovementSampler,
    ImportanceSamplingSampler,
    ResidualAdaptiveSampler,
    SamplerProtocol,
    TimeWindowSampler,
    UniformSampler,
)
from .training import Trainer, TrainingHistory


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} configuration must be a mapping.")
    return dict(value)


def _normalized_name(value: Any) -> str:
    return normalize_component_name(value)


def _require_component(
    registry: ComponentRegistry,
    kind: ComponentKind,
    name: str,
    label: str,
) -> str:
    """Resolve a configured component or fail without baseline fallback."""

    if not registry.contains(kind, name):
        choices = ", ".join(registry.available(kind)) or "<none>"
        raise ValueError(
            f"{label} {name!r} is not implemented; registered {kind}s: {choices}."
        )
    return registry.canonical_name(kind, name)


def _problem_component_name(problem_config: Mapping[str, Any]) -> str:
    return _normalized_name(
        problem_config.get("name", problem_config.get("kind", "poisson"))
    )


def _named_component_config(
    value: Any, config_name: str, default: str
) -> tuple[str, dict[str, Any]]:
    if isinstance(value, str) or value is None:
        values: dict[str, Any] = {"name": value or default}
    else:
        values = _mapping(value, config_name)
    name = _normalized_name(values.get("name", values.get("kind", default)))
    return name, values


def _reject_unknown(values: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(f"Unsupported {name} field(s): {', '.join(unknown)}")


def _assert_finite(value: Any, name: str) -> None:
    if isinstance(value, Tensor):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} contains NaN or infinite tensor values.")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise FloatingPointError(f"{name} contains NaN or infinite array values.")
        return
    if isinstance(value, Number) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"{name} contains a non-finite number.")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{name}[{index}]")


def _validate_config(
    config: Mapping[str, Any], registry: ComponentRegistry
) -> None:
    """Reject unsupported starter options instead of silently running a baseline."""

    _reject_unknown(
        config,
        {
            "schema_version",
            "experiment",
            "seed",
            "device",
            "dtype",
            "problem",
            "method",
            "methods",
            "model",
            "sampling",
            "sampler",
            "loss",
            "optimizer",
            "differentiator",
            "training",
            "evaluation",
            "output",
            "output_dir",
        },
        "top-level configuration",
    )
    if int(config.get("schema_version", 1)) != 1:
        raise ValueError("The starter accepts only schema_version: 1.")
    if "sampling" in config and "sampler" in config:
        raise ValueError("Pass sampling or sampler, not both.")

    method = _mapping(config.get("method"), "method")
    _reject_unknown(
        method,
        {
            "name",
            "variant",
            "variant_title",
            "title",
            "component_file",
            "variant_file",
            "options",
            "resolved",
        },
        "method",
    )
    if method and method.get("resolved") is not True:
        raise ValueError(
            "A method selector must be resolved through openpinn.solve() or "
            "openpinn.solve_config() before constructing Experiment directly."
        )
    methods = config.get("methods", [])
    if not isinstance(methods, (list, tuple)):
        raise TypeError("methods must be a list of resolved method records.")
    for index, method_item in enumerate(methods):
        selected_method = _mapping(method_item, f"methods[{index}]")
        _reject_unknown(
            selected_method,
            {
                "name",
                "variant",
                "variant_title",
                "title",
                "component_file",
                "variant_file",
                "options",
                "resolved",
            },
            f"methods[{index}]",
        )
        if selected_method.get("resolved") is not True:
            raise ValueError(f"methods[{index}] was not resolved by openpinn.solve().")

    experiment = _mapping(config.get("experiment"), "experiment")
    _reject_unknown(
        experiment, {"name", "description", "seed", "device", "dtype"}, "experiment"
    )
    problem = _mapping(config.get("problem"), "problem")
    _reject_unknown(
        problem,
        {
            "name",
            "kind",
            "domain",
            "parameters",
            "equation",
            "exact_solution",
            "boundary_conditions",
            "initial_condition",
        },
        "problem",
    )
    parameters = _mapping(problem.get("parameters"), "problem.parameters")
    problem_name = _problem_component_name(problem)
    problem_component = _require_component(
        registry, "problem", problem_name, "Problem"
    )
    if problem_component == "heterogeneouspoisson":
        allowed_parameters = {
            "interface",
            "left_diffusivity",
            "right_diffusivity",
            "a_left",
            "a_right",
        }
    elif problem_component == "poisson":
        allowed_parameters = {"wave_number", "k", "amplitude"}
    elif problem_component == "burgers":
        allowed_parameters = {"viscosity", "nu", "viscosity_coefficient"}
    elif problem_component == "coupledellipticmanufactured":
        allowed_parameters = {"coupling_strength", "kappa"}
    else:
        allowed_parameters = None
    if allowed_parameters is not None:
        _reject_unknown(parameters, allowed_parameters, "problem.parameters")

    model = _mapping(config.get("model"), "model")
    _reject_unknown(
        model,
        {
            "name",
            "kind",
            "input_dim",
            "output_dim",
            "hidden_layers",
            "layers",
            "depth",
            "activation",
            "activation_options",
            "initialization",
            "bias",
            "feature",
            "feature_transform",
            "output_transform",
            "output",
            "constraint",
            "trainable_last_linear_layers",
            "options",
        },
        "model",
    )
    model_name = _normalized_name(model.get("name", model.get("kind", "mlp")))
    model_component = _require_component(registry, "model", model_name, "Model")
    feature_aliases = [key for key in ("feature", "feature_transform") if key in model]
    if len(feature_aliases) > 1:
        raise ValueError("Pass model.feature or model.feature_transform, not both.")
    output_aliases = [
        key for key in ("output_transform", "output", "constraint") if key in model
    ]
    if len(output_aliases) > 1:
        raise ValueError(
            "Pass only one of model.output_transform, model.output, or model.constraint."
        )
    initialization = _normalized_name(model.get("initialization", "xavier_uniform"))
    if model_component == "mlp" and initialization not in {
        "xavieruniform",
        "glorotuniform",
    }:
        raise ValueError("The starter MLP currently supports only xavier_uniform initialization.")

    if model_component == "mlp":
        feature_value = model.get("feature", model.get("feature_transform", "identity"))
        feature_name, _ = _named_component_config(
            feature_value, "model.feature", "identity"
        )
        _require_component(
            registry, "feature", feature_name, "Feature transform"
        )
        output_key = output_aliases[0] if output_aliases else None
        output_value = model.get(output_key, "identity") if output_key else "identity"
        constraint_name, _ = _named_component_config(
            output_value, "model.output_transform", "identity"
        )
        _require_component(
            registry, "constraint", constraint_name, "Output constraint"
        )

    sampling = _mapping(config.get("sampling", config.get("sampler")), "sampling")
    _reject_unknown(
        sampling,
        {
            "name",
            "kind",
            "interior",
            "n_interior",
            "boundary",
            "n_boundary",
            "initial",
            "n_initial",
            "seed",
            "options",
        },
        "sampling",
    )
    sampling_name = _normalized_name(sampling.get("name", sampling.get("kind", "uniform")))
    _require_component(registry, "sampler", sampling_name, "Sampling method")

    loss = _mapping(config.get("loss"), "loss")
    _reject_unknown(
        loss, {"formulation", "weights", "weighting", "reduction", "options"}, "loss"
    )
    formulation = _normalized_name(loss.get("formulation", "strong"))
    _require_component(registry, "formulation", formulation, "Formulation")
    weighting_name, _ = _named_component_config(
        loss.get("weighting", "fixed"), "loss.weighting", "fixed"
    )
    _require_component(registry, "loss", weighting_name, "Loss")

    optimizer = _mapping(config.get("optimizer"), "optimizer")
    _reject_unknown(
        optimizer,
        {
            "name",
            "kind",
            "lr",
            "learning_rate",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "capturable",
            "differentiable",
            "fused",
            "max_iter",
            "max_eval",
            "tolerance_grad",
            "tolerance_change",
            "history_size",
            "line_search_fn",
            "stages",
            "options",
        },
        "optimizer",
    )
    optimizer_name = _normalized_name(
        optimizer.get("name", optimizer.get("kind", "adam"))
    )
    _require_component(registry, "optimizer", optimizer_name, "Optimizer")
    differentiator = _mapping(config.get("differentiator"), "differentiator")
    _reject_unknown(
        differentiator, {"mode", "name", "options"}, "differentiator"
    )
    differentiator_name = _normalized_name(
        differentiator.get("mode", differentiator.get("name", "autodiff"))
    )
    _require_component(
        registry, "derivative", differentiator_name, "Differentiator"
    )
    if optimizer_name == "adamlbfgs":
        stages = optimizer.get("stages")
        if not isinstance(stages, (list, tuple)) or len(stages) != 2:
            raise ValueError("adam_lbfgs requires exactly two optimizer stages.")
        normalized_stages: list[str] = []
        stage_steps = 0
        allowed_stage_fields = {
            "name",
            "kind",
            "steps",
            "lr",
            "learning_rate",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "capturable",
            "differentiable",
            "fused",
            "max_iter",
            "max_eval",
            "tolerance_grad",
            "tolerance_change",
            "history_size",
            "line_search_fn",
        }
        for index, stage in enumerate(stages):
            if not isinstance(stage, Mapping):
                raise TypeError(f"optimizer.stages[{index}] must be a mapping.")
            stage_values = dict(stage)
            _reject_unknown(
                stage_values, allowed_stage_fields, f"optimizer.stages[{index}]"
            )
            normalized_stages.append(
                _normalized_name(stage_values.get("name", stage_values.get("kind", "")))
            )
            steps_value = int(stage_values.get("steps", 0))
            if steps_value <= 0:
                raise ValueError("Every optimizer stage requires positive steps.")
            stage_steps += steps_value
        if normalized_stages != ["adam", "lbfgs"]:
            raise ValueError("adam_lbfgs stages must be ordered Adam then L-BFGS.")
    training = _mapping(config.get("training"), "training")
    _reject_unknown(
        training,
        {"steps", "epochs", "log_every", "resample_every", "initial_checkpoint"},
        "training",
    )
    if optimizer_name == "adamlbfgs":
        configured_steps = int(training.get("steps", training.get("epochs", stage_steps)))
        if configured_steps != stage_steps:
            raise ValueError(
                "training.steps must equal the sum of optimizer stage steps."
            )
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    _reject_unknown(
        evaluation,
        {
            "points",
            "time_slices",
            "x_points",
            "time_points",
            "boundary_points",
            "initial_points",
            "metrics",
            "note",
        },
        "evaluation",
    )
    output = _mapping(config.get("output"), "output")
    _reject_unknown(
        output,
        {
            "root",
            "directory",
            "dir",
            "run_name",
            "save_config",
            "save_status",
            "save_metrics",
            "save_history",
            "save_predictions",
            "save_compute_ledger",
            "save_checkpoint",
        },
        "output",
    )
    required_flags = (
        "save_config",
        "save_status",
        "save_metrics",
        "save_history",
        "save_predictions",
        "save_compute_ledger",
    )
    disabled = [flag for flag in required_flags if output.get(flag, True) is False]
    if disabled:
        raise ValueError(
            "The starter output contract requires these artifacts: " + ", ".join(disabled)
        )
    _assert_finite(config, "configuration")


def _dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    name = str(value or "float32").lower().replace("torch.", "")
    choices = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "fp64": torch.float64,
    }
    if name not in choices:
        raise ValueError("dtype must be float32 or float64 in the starter.")
    return choices[name]


def _device(value: Any) -> torch.device:
    name = str(value or "cpu").lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    selected = torch.device(name)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return selected


def _interval(value: Any, default: tuple[float, float]) -> Interval:
    if value is None:
        return Interval(*default)
    if isinstance(value, Interval):
        return value
    if isinstance(value, Mapping):
        values = dict(value)
        if "bounds" in values:
            value = values["bounds"]
        elif "lower" in values and "upper" in values:
            return Interval(float(values["lower"]), float(values["upper"]))
        elif "min" in values and "max" in values:
            return Interval(float(values["min"]), float(values["max"]))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return Interval(float(value[0]), float(value[1]))
    raise ValueError("Interval domain must be [lower, upper] or a matching mapping.")


def _spatial_interval(problem_config: Mapping[str, Any]) -> Interval:
    domain_config = problem_config.get("domain")
    if isinstance(domain_config, Mapping):
        domain_values = dict(domain_config)
        interval_config = domain_values.get(
            "x", domain_values.get("space", domain_config)
        )
    else:
        interval_config = domain_config
    return _interval(interval_config, (-1.0, 1.0))


def _poisson_problem_factory(*, config: Mapping[str, Any]) -> PDEProblem:
    parameters = _mapping(config.get("parameters"), "problem.parameters")
    return make_poisson_problem(
        _spatial_interval(config),
        wave_number=float(parameters.get("wave_number", parameters.get("k", 1.0))),
        amplitude=float(parameters.get("amplitude", 1.0)),
    )


def _heterogeneous_poisson_problem_factory(
    *, config: Mapping[str, Any]
) -> PDEProblem:
    parameters = _mapping(config.get("parameters"), "problem.parameters")
    return make_piecewise_poisson_problem(
        _spatial_interval(config),
        interface=float(parameters.get("interface", 0.0)),
        left_diffusivity=float(
            parameters.get("left_diffusivity", parameters.get("a_left", 1.0))
        ),
        right_diffusivity=float(
            parameters.get("right_diffusivity", parameters.get("a_right", 4.0))
        ),
    )


def _burgers_problem_factory(*, config: Mapping[str, Any]) -> PDEProblem:
    parameters = _mapping(config.get("parameters"), "problem.parameters")
    domain_config = config.get("domain")
    if isinstance(domain_config, SpaceTimeDomain):
        domain = domain_config
    else:
        values = (
            _mapping(domain_config, "problem.domain")
            if domain_config is not None
            else {}
        )
        domain = SpaceTimeDomain(
            _interval(values.get("space", values.get("x")), (-1.0, 1.0)),
            _interval(values.get("time", values.get("t")), (0.0, 1.0)),
        )
    viscosity = parameters.get(
        "viscosity",
        parameters.get(
            "nu", parameters.get("viscosity_coefficient", 0.01 / math.pi)
        ),
    )
    return make_burgers_problem(domain, viscosity=float(viscosity))


def _coupled_manufactured_problem_factory(
    *, config: Mapping[str, Any]
) -> PDEProblem:
    parameters = _mapping(config.get("parameters"), "problem.parameters")
    coupling_strength = parameters.get(
        "coupling_strength", parameters.get("kappa", 1.0)
    )
    return make_coupled_manufactured_problem(
        _spatial_interval(config), coupling_strength=float(coupling_strength)
    )


def _tunnel_model1_problem_factory(*, config: Mapping[str, Any]) -> PDEProblem:
    parameters = _mapping(config.get("parameters"), "problem.parameters")
    centers_value = parameters.get("tunnel_centers_m", ((10.0, 20.0), (30.0, 20.0)))
    try:
        centers = tuple((float(center[0]), float(center[1])) for center in centers_value)
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(
            "problem.parameters.tunnel_centers_m must contain two [x, y] pairs."
        ) from exc
    return make_tunnel_consolidation_model1_problem(
        width_m=float(parameters.get("width_m", 40.0)),
        height_m=float(parameters.get("height_m", 40.0)),
        duration_years=float(parameters.get("duration_years", 40.0)),
        tunnel_radius_m=float(parameters.get("tunnel_radius_m", 4.0)),
        tunnel_centers_m=centers,
        initial_pressure_kpa=float(parameters.get("initial_pressure_kpa", 20.0)),
        consolidation_x_m2_per_year=float(
            parameters.get("consolidation_x_m2_per_year", 0.30)
        ),
        consolidation_y_m2_per_year=float(
            parameters.get("consolidation_y_m2_per_year", 0.60)
        ),
    )


def _problem(
    config: Mapping[str, Any], registry: ComponentRegistry
) -> PDEProblem:
    problem_config = _mapping(config.get("problem"), "problem")
    name = _problem_component_name(problem_config)
    return registry.create("problem", name, config=problem_config)


def _component_options(config: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(config)
    values.pop("name", None)
    values.pop("kind", None)
    nested = values.pop("options", None)
    if nested is not None:
        nested_values = _mapping(nested, "component.options")
        overlap = sorted(set(values).intersection(nested_values))
        if overlap:
            raise ValueError(
                "Component options duplicate top-level field(s): "
                + ", ".join(overlap)
            )
        values.update(nested_values)
    return values


def _identity_feature_factory(
    *, config: Mapping[str, Any], input_dim: int
) -> nn.Module:
    values = _component_options(config)
    if values:
        raise ValueError(
            "Identity feature received unsupported option(s): "
            + ", ".join(sorted(values))
        )
    return IdentityFeature(input_dim)


def _fourier_feature_factory(
    *, config: Mapping[str, Any], input_dim: int
) -> nn.Module:
    values = _component_options(config)
    allowed = {
        "num_frequencies",
        "mapping_size",
        "scale",
        "seed",
        "include_input",
        "trainable",
    }
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(
            "Fourier feature received unsupported option(s): " + ", ".join(unknown)
        )
    return FourierFeatures(input_dim, **values)


def _feature(
    model_config: Mapping[str, Any],
    input_dim: int,
    registry: ComponentRegistry,
) -> nn.Module:
    feature_config = model_config.get(
        "feature", model_config.get("feature_transform", "identity")
    )
    name, values = _named_component_config(
        feature_config, "model.feature", "identity"
    )
    return registry.create("feature", name, config=values, input_dim=input_dim)


def _hard_transform(problem: PDEProblem) -> HardConstraint:
    domain = problem.domain
    if isinstance(domain, Interval):
        lower, upper = domain.bounds

        boundary_condition = next(
            (
                condition
                for condition in problem.conditions
                if isinstance(condition, DirichletBC)
            ),
            None,
        )
        if boundary_condition is None:
            raise ValueError("An interval hard constraint requires Dirichlet boundary data.")

        def extension(coordinates: Tensor) -> Tensor:
            endpoints = coordinates.new_tensor([[lower], [upper]])
            endpoint_values = boundary_condition.value(endpoints) if callable(
                boundary_condition.value
            ) else boundary_condition.value
            if not isinstance(endpoint_values, Tensor):
                endpoint_values = coordinates.new_full((2, 1), float(endpoint_values))
            else:
                endpoint_values = endpoint_values.to(
                    device=coordinates.device, dtype=coordinates.dtype
                )
                if endpoint_values.ndim == 0:
                    endpoint_values = endpoint_values.expand(2, 1)
                elif endpoint_values.ndim == 1:
                    endpoint_values = endpoint_values.reshape(-1, 1)
                if endpoint_values.shape[0] == 1:
                    endpoint_values = endpoint_values.expand(2, -1)
                if endpoint_values.shape[0] != 2:
                    raise ValueError(
                        "Dirichlet boundary data must provide values at both interval endpoints."
                    )
            alpha = (coordinates[:, :1] - lower) / (upper - lower)
            return (1.0 - alpha) * endpoint_values[0:1] + alpha * endpoint_values[1:2]

        def distance(coordinates: Tensor) -> Tensor:
            x = coordinates[:, :1]
            return (x - lower) * (upper - x)

        return HardConstraint(distance, extension)

    lower, upper = domain.space.bounds
    time_lower, time_upper = domain.time.bounds

    def initial(coordinates: Tensor) -> Tensor:
        return -torch.sin(math.pi * coordinates[:, :1])

    def extension(coordinates: Tensor) -> Tensor:
        tau = (coordinates[:, 1:2] - time_lower) / (time_upper - time_lower)
        return (1.0 - tau) * initial(coordinates)

    def distance(coordinates: Tensor) -> Tensor:
        x = coordinates[:, :1]
        tau = (coordinates[:, 1:2] - time_lower) / (time_upper - time_lower)
        return tau * (x - lower) * (upper - x)

    return HardConstraint(distance, extension)


def _identity_constraint_factory(
    *, config: Mapping[str, Any], problem: PDEProblem
) -> nn.Module:
    del problem
    values = _component_options(config)
    if values:
        raise ValueError(
            "Identity output constraint received unsupported option(s): "
            + ", ".join(sorted(values))
        )
    return IdentityOutput()


def _hard_constraint_factory(
    *, config: Mapping[str, Any], problem: PDEProblem
) -> nn.Module:
    values = _component_options(config)
    if values:
        raise ValueError(
            "Hard output constraint received unsupported option(s): "
            + ", ".join(sorted(values))
        )
    return _hard_transform(problem)


def _constraint(
    model_config: Mapping[str, Any],
    problem: PDEProblem,
    registry: ComponentRegistry,
) -> nn.Module:
    output_aliases = [
        key
        for key in ("output_transform", "output", "constraint")
        if key in model_config
    ]
    if len(output_aliases) > 1:
        raise ValueError(
            "Pass only one of model.output_transform, model.output, or model.constraint."
        )
    output_config = (
        model_config.get(output_aliases[0], "identity")
        if output_aliases
        else "identity"
    )
    name, values = _named_component_config(
        output_config, "model.output_transform", "identity"
    )
    return registry.create("constraint", name, config=values, problem=problem)


def _feature_output_dimension(feature: nn.Module, input_dim: int) -> int:
    method = getattr(feature, "output_dimension", None)
    if callable(method):
        output_dim = int(method(input_dim))
    else:
        declared = getattr(feature, "out_features", None)
        if declared is None:
            raise TypeError(
                "Feature components must define output_dimension(input_dim) "
                "or an integer out_features attribute."
            )
        output_dim = int(declared)
    if output_dim <= 0:
        raise ValueError("Feature output dimension must be positive.")
    return output_dim


def _mlp_model_factory(
    *,
    config: Mapping[str, Any],
    problem: PDEProblem,
    registry: ComponentRegistry,
) -> nn.Module:
    model_config = _component_options(config)
    allowed = {
        "input_dim",
        "output_dim",
        "hidden_layers",
        "layers",
        "depth",
        "activation",
        "activation_options",
        "initialization",
        "bias",
        "feature",
        "feature_transform",
        "output_transform",
        "output",
        "constraint",
        "trainable_last_linear_layers",
    }
    _reject_unknown(model_config, allowed, "MLP model")
    configured_input = model_config.get("input_dim")
    if configured_input is not None and int(configured_input) != problem.domain.dim:
        raise ValueError(
            f"model.input_dim={configured_input} does not match problem dimension "
            f"{problem.domain.dim}."
        )
    initialization = _normalized_name(
        model_config.get("initialization", "xavier_uniform")
    )
    if initialization not in {"xavieruniform", "glorotuniform"}:
        raise ValueError(
            "The starter MLP currently supports only xavier_uniform initialization."
        )
    feature_aliases = [
        key for key in ("feature", "feature_transform") if key in model_config
    ]
    if len(feature_aliases) > 1:
        raise ValueError("Pass model.feature or model.feature_transform, not both.")
    feature = _feature(model_config, problem.domain.dim, registry)
    input_width = _feature_output_dimension(feature, problem.domain.dim)
    hidden = model_config.get("hidden_layers", model_config.get("layers", (64, 64, 64)))
    if isinstance(hidden, int):
        depth = int(model_config.get("depth", 3))
        hidden = [hidden] * depth
    output_dim = int(model_config.get("output_dim", problem.output_dim))
    network = MLP(
        input_width,
        output_dim,
        hidden_layers=hidden,
        activation=model_config.get("activation", "tanh"),
        bias=bool(model_config.get("bias", True)),
        activation_options=model_config.get("activation_options"),
    )
    transform = _constraint(model_config, problem, registry)
    model = PINNModel(network, feature_transform=feature, output_transform=transform)
    trainable_last = model_config.get("trainable_last_linear_layers")
    if trainable_last is not None:
        freeze_all_but_last_linear_layers(model, int(trainable_last))
    return model


def _modified_mlp_model_factory(
    *,
    config: Mapping[str, Any],
    problem: PDEProblem,
    registry: ComponentRegistry,
) -> nn.Module:
    """Build the gated Modified MLP behind the same model configuration shape."""

    model_config = _component_options(config)
    allowed = {
        "input_dim",
        "output_dim",
        "hidden_layers",
        "layers",
        "depth",
        "activation",
        "activation_options",
        "initialization",
        "bias",
        "feature",
        "feature_transform",
        "output_transform",
        "output",
        "constraint",
    }
    _reject_unknown(model_config, allowed, "Modified MLP model")
    configured_input = model_config.get("input_dim")
    if configured_input is not None and int(configured_input) != problem.domain.dim:
        raise ValueError(
            f"model.input_dim={configured_input} does not match problem dimension "
            f"{problem.domain.dim}."
        )
    initialization = _normalized_name(
        model_config.get("initialization", "xavier_uniform")
    )
    if initialization not in {"xavieruniform", "glorotuniform"}:
        raise ValueError(
            "ModifiedMLP currently supports only xavier_uniform initialization."
        )
    feature_aliases = [
        key for key in ("feature", "feature_transform") if key in model_config
    ]
    if len(feature_aliases) > 1:
        raise ValueError("Pass model.feature or model.feature_transform, not both.")
    feature = _feature(model_config, problem.domain.dim, registry)
    input_width = _feature_output_dimension(feature, problem.domain.dim)
    hidden = model_config.get("hidden_layers", model_config.get("layers", (64, 64, 64)))
    if isinstance(hidden, int):
        hidden = [hidden] * int(model_config.get("depth", 3))
    output_dim = int(model_config.get("output_dim", problem.output_dim))
    network = ModifiedMLP(
        input_width,
        output_dim,
        hidden_layers=hidden,
        activation=model_config.get("activation", "tanh"),
        bias=bool(model_config.get("bias", True)),
        activation_options=model_config.get("activation_options"),
    )
    transform = _constraint(model_config, problem, registry)
    return PINNModel(network, feature_transform=feature, output_transform=transform)


def _domain_decomposition_model_factory(
    *,
    config: Mapping[str, Any],
    problem: PDEProblem,
    registry: ComponentRegistry,
) -> nn.Module:
    """Build two 1D subnetworks behind the common model registry."""

    del registry
    model_config = _component_options(config)
    allowed = {
        "input_dim",
        "output_dim",
        "hidden_layers",
        "layers",
        "depth",
        "activation",
        "activation_options",
        "initialization",
        "bias",
        "interface",
    }
    _reject_unknown(model_config, allowed, "domain-decomposition model")
    if problem.metadata.get("kind") != "heterogeneous_static":
        raise ValueError(
            "The starter domain-decomposition model requires heterogeneous Poisson."
        )
    configured_input = int(model_config.get("input_dim", problem.domain.dim))
    if configured_input != problem.domain.dim:
        raise ValueError("model.input_dim does not match the problem dimension.")
    output_dim = int(model_config.get("output_dim", problem.output_dim))
    hidden = model_config.get(
        "hidden_layers", model_config.get("layers", (32, 32, 32))
    )
    if isinstance(hidden, int):
        hidden = [hidden] * int(model_config.get("depth", 3))
    initialization = _normalized_name(
        model_config.get("initialization", "xavier_uniform")
    )
    if initialization not in {"xavieruniform", "glorotuniform"}:
        raise ValueError(
            "DomainDecompositionPINN supports only xavier_uniform initialization."
        )
    interface = float(
        model_config.get("interface", problem.metadata["interface"])
    )
    return DomainDecompositionPINN(
        configured_input,
        output_dim,
        hidden_layers=hidden,
        activation=model_config.get("activation", "tanh"),
        interface=interface,
        bias=bool(model_config.get("bias", True)),
        activation_options=model_config.get("activation_options"),
    )


def _model_output_dimension(model: nn.Module) -> int:
    declared = getattr(model, "output_dim", None)
    if declared is None:
        network = getattr(model, "network", None)
        declared = getattr(network, "output_dim", None)
    if declared is None:
        raise TypeError(
            "Model components must expose an integer output_dim attribute."
        )
    output_dim = int(declared)
    if output_dim <= 0:
        raise ValueError("Model output_dim must be positive.")
    return output_dim


def _model(
    config: Mapping[str, Any],
    problem: PDEProblem,
    registry: ComponentRegistry | None = None,
) -> nn.Module:
    registry = default_component_registry() if registry is None else registry
    model_config = _mapping(config.get("model"), "model")
    configured_output = int(model_config.get("output_dim", problem.output_dim))
    if configured_output != problem.output_dim:
        raise ValueError(
            f"model.output_dim={configured_output} does not match problem.output_dim="
            f"{problem.output_dim}."
        )
    name = _normalized_name(model_config.get("name", model_config.get("kind", "mlp")))
    model = registry.create(
        "model",
        name,
        config=model_config,
        problem=problem,
        registry=registry,
    )
    actual_output = _model_output_dimension(model)
    if actual_output != problem.output_dim:
        raise ValueError(
            f"Constructed model.output_dim={actual_output} does not match "
            f"problem.output_dim={problem.output_dim}."
        )
    return model


def _strong_formulation_factory(
    *,
    config: Mapping[str, Any],
    problem: PDEProblem,
    derivative_backend: Any = None,
) -> StrongFormulation:
    values = dict(config)
    values.pop("formulation", None)
    values.pop("weights", None)
    values.pop("weighting", None)
    values.pop("reduction", None)
    options = values.pop("options", None)
    if options is not None:
        values.update(_mapping(options, "loss.options"))
    if values:
        raise ValueError(
            "Strong formulation received unsupported option(s): "
            + ", ".join(sorted(values))
        )
    return StrongFormulation(problem, derivative_backend)


def _interface_strong_formulation_factory(
    *,
    config: Mapping[str, Any],
    problem: PDEProblem,
    derivative_backend: Any = None,
) -> InterfaceStrongFormulation:
    values = dict(config)
    values.pop("formulation", None)
    values.pop("weights", None)
    values.pop("weighting", None)
    values.pop("reduction", None)
    options = values.pop("options", None)
    if options is not None:
        values.update(_mapping(options, "loss.options"))
    if values:
        raise ValueError(
            "Interface strong formulation received unsupported option(s): "
            + ", ".join(sorted(values))
        )
    return InterfaceStrongFormulation(problem, derivative_backend)


def _formulation(
    config: Mapping[str, Any],
    problem: PDEProblem,
    registry: ComponentRegistry,
) -> FormulationProtocol:
    loss_config = _mapping(config.get("loss"), "loss")
    name = _normalized_name(loss_config.get("formulation", "strong"))
    differentiator = _mapping(config.get("differentiator"), "differentiator")
    derivative_name = _normalized_name(
        differentiator.get("mode", differentiator.get("name", "autodiff"))
    )
    # mode/name choose the registry entry. Remaining fields are backend options,
    # so PR59's unknown-key checks still reject typos such as a bad spacing key.
    derivative_options = {
        key: value
        for key, value in differentiator.items()
        if key not in {"mode", "name"}
    }
    derivative = registry.create(
        "derivative", derivative_name, config=derivative_options
    )
    if not callable(getattr(derivative, "grad", None)) or not callable(
        getattr(derivative, "hessian", None)
    ):
        raise ValueError(
            "PINN strong-form training requires a coordinate derivative backend; "
            f"{derivative_name!r} is grid-only."
        )
    formulation_name = registry.canonical_name("formulation", name)
    factory_options: dict[str, Any] = {
        "config": loss_config,
        "problem": problem,
    }
    # Preserve the established two-key factory contract for third-party
    # formulations. Built-in strong forms opt into derivative routing here.
    if formulation_name in {"strong", "interface_strong"}:
        factory_options["derivative_backend"] = derivative
    return registry.create("formulation", formulation_name, **factory_options)


def _fixed_loss_factory(
    *,
    config: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
    reduction: str = "sum",
) -> LossComposer:
    values = _component_options(config)
    if values:
        raise ValueError(
            "Fixed loss weighting received unsupported option(s): "
            + ", ".join(sorted(values))
        )
    return LossComposer(weights, reduction=reduction)


def _self_adaptive_loss_factory(
    *,
    config: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
    reduction: str = "sum",
) -> AdaptiveLossComposer:
    values = _component_options(config)
    allowed = {"lr", "learning_rate", "min_weight", "max_weight"}
    _reject_unknown(values, allowed, "self-adaptive loss weighting")
    if "lr" in values and "learning_rate" in values:
        raise ValueError("Pass loss.weighting.lr or learning_rate, not both.")
    learning_rate = float(values.pop("lr", values.pop("learning_rate", 1e-3)))
    return AdaptiveLossComposer(
        weights or {},
        reduction=reduction,
        lr=learning_rate,
        **values,
    )


def _pointwise_self_adaptive_loss_factory(
    *,
    config: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
    reduction: str = "sum",
) -> PointwiseAdaptiveLossComposer:
    values = _component_options(config)
    allowed = {"lr", "learning_rate", "min_weight", "max_weight"}
    _reject_unknown(values, allowed, "pointwise self-adaptive loss weighting")
    if "lr" in values and "learning_rate" in values:
        raise ValueError("Pass loss.weighting.lr or learning_rate, not both.")
    learning_rate = float(values.pop("lr", values.pop("learning_rate", 1e-2)))
    return PointwiseAdaptiveLossComposer(
        weights or {},
        reduction=reduction,
        lr=learning_rate,
        **values,
    )


def _uncertainty_loss_factory(
    *,
    config: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
    reduction: str = "sum",
) -> UncertaintyLossComposer:
    values = _component_options(config)
    allowed = {
        "lr",
        "learning_rate",
        "min_log_variance",
        "max_log_variance",
    }
    _reject_unknown(values, allowed, "uncertainty loss weighting")
    if "lr" in values and "learning_rate" in values:
        raise ValueError("Pass loss.weighting.lr or learning_rate, not both.")
    learning_rate = float(values.pop("lr", values.pop("learning_rate", 1e-3)))
    return UncertaintyLossComposer(
        weights or {},
        reduction=reduction,
        lr=learning_rate,
        **values,
    )


def _relobralo_loss_factory(
    *,
    config: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
    reduction: str = "sum",
) -> ReLoBRaLoLossComposer:
    values = _component_options(config)
    allowed = {
        "alpha",
        "temperature",
        "expected_lookback",
        "epsilon",
        "seed",
    }
    _reject_unknown(values, allowed, "ReLoBRaLo loss weighting")
    return ReLoBRaLoLossComposer(weights or {}, reduction=reduction, **values)


def _magnitude_power_loss_factory(
    *,
    config: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
    reduction: str = "sum",
) -> MagnitudePowerLossComposer:
    values = _component_options(config)
    allowed = {"roots", "default_root", "epsilon"}
    _reject_unknown(values, allowed, "magnitude-power loss balancing")
    selected_weights = dict(weights or {})
    roots = _mapping(values.pop("roots", None), "loss.weighting.roots")
    applicable_roots = {
        name: root for name, root in roots.items() if name in selected_weights
    }
    return MagnitudePowerLossComposer(
        selected_weights,
        roots=applicable_roots,
        reduction=reduction,
        **values,
    )


def _autodiff_derivative_factory(**kwargs: Any) -> Any:
    del kwargs
    from .autodiff import AutodiffDerivative

    return AutodiffDerivative()


def _spectral_derivative_factory(
    *, config: Mapping[str, Any] | None = None, **kwargs: Any
) -> Any:
    from .operators.grid_derivative import SpectralGridDerivative

    values = _component_options(config or {})
    spacing = values.pop("spacing", kwargs.get("spacing", 1.0))
    if values:
        raise ValueError(
            "Spectral derivative received unsupported option(s): "
            + ", ".join(sorted(values))
        )
    return SpectralGridDerivative(spacing=spacing)


def _finite_difference_derivative_factory(
    *, config: Mapping[str, Any] | None = None, **kwargs: Any
) -> Any:
    from .operators.grid_derivative import FiniteDifferenceGridDerivative

    values = _component_options(config or {})
    spacing = values.pop("spacing", kwargs.get("spacing", 1.0))
    if values:
        raise ValueError(
            "Finite-difference derivative received unsupported option(s): "
            + ", ".join(sorted(values))
        )
    return FiniteDifferenceGridDerivative(spacing=spacing)


def _loss_composer(
    loss_config: Mapping[str, Any],
    registry: ComponentRegistry,
) -> (
    LossComposer
    | AdaptiveLossComposer
    | UncertaintyLossComposer
    | ReLoBRaLoLossComposer
    | MagnitudePowerLossComposer
    | PointwiseAdaptiveLossComposer
):
    weights = _mapping(loss_config.get("weights"), "loss.weights")
    reduction = str(loss_config.get("reduction", "sum"))
    weighting_name, weighting = _named_component_config(
        loss_config.get("weighting", "fixed"), "loss.weighting", "fixed"
    )
    return registry.create(
        "loss",
        weighting_name,
        config=weighting,
        weights=weights,
        reduction=reduction,
    )


def _uniform_sampler_factory(
    *,
    config: Mapping[str, Any],
    problem: PDEProblem,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> UniformSampler:
    values = _component_options(config)
    allowed = {
        "interior",
        "n_interior",
        "boundary",
        "n_boundary",
        "initial",
        "n_initial",
        "seed",
    }
    _reject_unknown(values, allowed, "uniform sampler")
    return UniformSampler(
        problem,
        n_interior=int(values.get("interior", values.get("n_interior", 128))),
        n_boundary=int(values.get("boundary", values.get("n_boundary", 32))),
        n_initial=int(values.get("initial", values.get("n_initial", 32))),
        seed=int(values.get("seed", seed)),
        device=device,
        dtype=dtype,
    )


def _residual_adaptive_sampler_factory(
    *,
    config: Mapping[str, Any],
    problem: PDEProblem,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ResidualAdaptiveSampler:
    values = _component_options(config)
    allowed = {
        "interior",
        "n_interior",
        "boundary",
        "n_boundary",
        "initial",
        "n_initial",
        "seed",
        "candidate_pool",
        "power",
        "uniform_mix",
        "epsilon",
    }
    _reject_unknown(values, allowed, "residual-adaptive sampler")
    return ResidualAdaptiveSampler(
        problem,
        n_interior=int(values.pop("interior", values.pop("n_interior", 128))),
        n_boundary=int(values.pop("boundary", values.pop("n_boundary", 32))),
        n_initial=int(values.pop("initial", values.pop("n_initial", 32))),
        seed=int(values.pop("seed", seed)),
        device=device,
        dtype=dtype,
        **values,
    )


def _adaptive_movement_sampler_factory(
    *,
    config: Mapping[str, Any],
    problem: PDEProblem,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> AdaptiveMovementSampler:
    values = _component_options(config)
    allowed = {
        "interior",
        "n_interior",
        "boundary",
        "n_boundary",
        "initial",
        "n_initial",
        "seed",
        "indicator",
        "candidate_pool",
        "movable_fraction",
        "power",
        "epsilon",
    }
    _reject_unknown(values, allowed, "adaptive-movement sampler")
    return AdaptiveMovementSampler(
        problem,
        n_interior=int(values.pop("interior", values.pop("n_interior", 128))),
        n_boundary=int(values.pop("boundary", values.pop("n_boundary", 32))),
        n_initial=int(values.pop("initial", values.pop("n_initial", 32))),
        seed=int(values.pop("seed", seed)),
        device=device,
        dtype=dtype,
        **values,
    )


def _importance_sampling_sampler_factory(
    *,
    config: Mapping[str, Any],
    problem: PDEProblem,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ImportanceSamplingSampler:
    values = _component_options(config)
    allowed = {
        "interior",
        "n_interior",
        "boundary",
        "n_boundary",
        "initial",
        "n_initial",
        "seed",
        "candidate_pool",
        "epsilon",
    }
    _reject_unknown(values, allowed, "importance sampler")
    return ImportanceSamplingSampler(
        problem,
        n_interior=int(values.pop("interior", values.pop("n_interior", 128))),
        n_boundary=int(values.pop("boundary", values.pop("n_boundary", 32))),
        n_initial=int(values.pop("initial", values.pop("n_initial", 32))),
        seed=int(values.pop("seed", seed)),
        device=device,
        dtype=dtype,
        **values,
    )


def _time_window_sampler_factory(
    *,
    config: Mapping[str, Any],
    problem: PDEProblem,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TimeWindowSampler:
    values = _component_options(config)
    allowed = {
        "interior",
        "n_interior",
        "boundary",
        "n_boundary",
        "initial",
        "n_initial",
        "seed",
        "window_ends",
    }
    _reject_unknown(values, allowed, "time-window sampler")
    window_ends = values.pop("window_ends", None)
    if not isinstance(window_ends, (list, tuple)):
        raise ValueError("Time-window sampling requires options.window_ends.")
    return TimeWindowSampler(
        problem,
        n_interior=int(values.pop("interior", values.pop("n_interior", 128))),
        n_boundary=int(values.pop("boundary", values.pop("n_boundary", 32))),
        n_initial=int(values.pop("initial", values.pop("n_initial", 32))),
        window_ends=window_ends,
        seed=int(values.pop("seed", seed)),
        device=device,
        dtype=dtype,
    )


def _sampler(
    config: Mapping[str, Any],
    problem: PDEProblem,
    registry: ComponentRegistry,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> SamplerProtocol:
    sampling = _mapping(
        config.get("sampling", config.get("sampler")), "sampling"
    )
    name = _normalized_name(sampling.get("name", sampling.get("kind", "uniform")))
    return registry.create(
        "sampler",
        name,
        config=sampling,
        problem=problem,
        seed=seed,
        device=device,
        dtype=dtype,
    )


def _optimizer_options(config: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    values = _component_options(config)
    if "lr" in values and "learning_rate" in values:
        raise ValueError("Pass optimizer.lr or optimizer.learning_rate, not both.")
    learning_rate = float(values.pop("lr", values.pop("learning_rate", 1e-3)))
    return learning_rate, values


def _adam_optimizer_factory(
    *, config: Mapping[str, Any], parameters: list[nn.Parameter]
) -> OptimizerAdapter:
    learning_rate, values = _optimizer_options(config)
    return AdamAdapter(parameters, lr=learning_rate, **values)


def _adamw_optimizer_factory(
    *, config: Mapping[str, Any], parameters: list[nn.Parameter]
) -> OptimizerAdapter:
    learning_rate, values = _optimizer_options(config)
    return AdamWAdapter(parameters, lr=learning_rate, **values)


def _lbfgs_optimizer_factory(
    *, config: Mapping[str, Any], parameters: list[nn.Parameter]
) -> OptimizerAdapter:
    learning_rate, values = _optimizer_options(config)
    return LBFGSAdapter(parameters, lr=learning_rate, **values)


def _staged_optimizer_factory(
    *, config: Mapping[str, Any], parameters: list[nn.Parameter]
) -> StagedOptimizerAdapter:
    values = _component_options(config)
    stages = values.pop("stages", None)
    if stages is None or values:
        raise ValueError("adam_lbfgs requires only optimizer.stages.")
    return StagedOptimizerAdapter(parameters, stages)


def _optimizer(
    config: Mapping[str, Any],
    model: nn.Module,
    registry: ComponentRegistry,
) -> tuple[str, OptimizerAdapter]:
    optimizer_config = _mapping(config.get("optimizer"), "optimizer")
    name = _normalized_name(
        optimizer_config.get("name", optimizer_config.get("kind", "adam"))
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("The model has no trainable parameters.")
    adapter = registry.create(
        "optimizer", name, config=optimizer_config, parameters=parameters
    )
    return registry.canonical_name("optimizer", name), adapter


def default_component_registry() -> ComponentRegistry:
    """Return a fresh registry containing the supported starter components."""

    registry = ComponentRegistry()
    registry.register(
        "problem",
        "poisson",
        _poisson_problem_factory,
        aliases=("poisson_1d", "static"),
    )
    registry.register(
        "problem",
        "heterogeneous_poisson",
        _heterogeneous_poisson_problem_factory,
        aliases=(
            "heterogeneous_poisson_1d",
            "piecewise_poisson",
            "piecewise_poisson_1d",
            "heterogeneous_static",
            "hetero",
        ),
    )
    registry.register(
        "problem",
        "burgers",
        _burgers_problem_factory,
        aliases=("burger", "burgers_1d", "dynamic"),
    )
    registry.register(
        "problem",
        "coupled_elliptic_manufactured",
        _coupled_manufactured_problem_factory,
        aliases=(
            "coupled_elliptic_manufactured_1d",
            "coupled_manufactured",
            "coupled_static",
        ),
    )
    registry.register(
        "problem",
        "tunnel_consolidation_model1",
        _tunnel_model1_problem_factory,
        aliases=("tunnel_model1", "model1", "twin_tunnel_consolidation"),
    )
    registry.register(
        "model", "mlp", _mlp_model_factory, aliases=("fnn", "fully_connected")
    )
    registry.register(
        "model",
        "modified_mlp",
        _modified_mlp_model_factory,
        aliases=("modifiedmlp", "gated_mlp"),
    )
    registry.register(
        "model",
        "domain_decomposition",
        _domain_decomposition_model_factory,
        aliases=("two_subdomain", "xpinn_minimal"),
    )
    registry.register(
        "feature",
        "identity",
        _identity_feature_factory,
        aliases=("none", "raw"),
    )
    registry.register(
        "feature",
        "fourier",
        _fourier_feature_factory,
        aliases=("fourier_features", "random_fourier"),
    )
    registry.register(
        "constraint",
        "identity",
        _identity_constraint_factory,
        aliases=("none", "soft"),
    )
    registry.register(
        "constraint",
        "hard",
        _hard_constraint_factory,
        aliases=("hard_constraint", "analytic_distance"),
    )
    registry.register(
        "formulation", "strong", _strong_formulation_factory, aliases=("strong_form",)
    )
    registry.register(
        "formulation",
        "interface_strong",
        _interface_strong_formulation_factory,
        aliases=("strong_interface",),
    )
    registry.register(
        "sampler", "uniform", _uniform_sampler_factory, aliases=("random_uniform",)
    )
    registry.register(
        "sampler",
        "residual_adaptive",
        _residual_adaptive_sampler_factory,
        aliases=("rad", "residual_weighted"),
    )
    registry.register(
        "sampler",
        "adaptive_movement",
        _adaptive_movement_sampler_factory,
        aliases=("ram", "wam", "collocation_movement"),
    )
    registry.register(
        "sampler",
        "importance_sampling",
        _importance_sampling_sampler_factory,
        aliases=("importance", "loss_importance"),
    )
    registry.register(
        "sampler",
        "time_window",
        _time_window_sampler_factory,
        aliases=("time_curriculum", "sequential_time_window"),
    )
    registry.register("optimizer", "adam", _adam_optimizer_factory)
    registry.register("optimizer", "adamw", _adamw_optimizer_factory)
    registry.register(
        "optimizer", "lbfgs", _lbfgs_optimizer_factory, aliases=("lbgfs",)
    )
    registry.register(
        "optimizer",
        "adam_lbfgs",
        _staged_optimizer_factory,
        aliases=("staged", "adam_then_lbfgs"),
    )
    registry.register("loss", "fixed", _fixed_loss_factory, aliases=("static",))
    registry.register(
        "loss",
        "self_adaptive",
        _self_adaptive_loss_factory,
        aliases=("adaptive",),
    )
    registry.register(
        "loss",
        "pointwise_self_adaptive",
        _pointwise_self_adaptive_loss_factory,
        aliases=("pointwise_sapinn", "soft_attention"),
    )
    registry.register(
        "loss",
        "uncertainty",
        _uncertainty_loss_factory,
        aliases=("uncertainty_weighting", "homoscedastic"),
    )
    registry.register(
        "loss",
        "relobralo",
        _relobralo_loss_factory,
        aliases=("relative_balancing",),
    )
    registry.register(
        "loss",
        "magnitude_power",
        _magnitude_power_loss_factory,
        aliases=("power_loss", "mmpinn_power"),
    )
    registry.register("derivative", "autodiff", _autodiff_derivative_factory)
    registry.register(
        "derivative",
        "grid",
        _spectral_derivative_factory,
        aliases=("spectral", "fourier"),
    )
    registry.register(
        "derivative",
        "finite_difference",
        _finite_difference_derivative_factory,
        aliases=("fd",),
    )
    from .ecosystem_components import register_method_ecosystem_builtins

    register_method_ecosystem_builtins(registry)
    return registry


def _evaluation_coordinates(
    problem: PDEProblem,
    evaluation: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    points = evaluation.get("points", 256)
    if isinstance(problem.domain, Interval):
        count = int(points if not isinstance(points, Mapping) else points.get("space", 256))
        if count < 2:
            raise ValueError("Poisson evaluation requires at least two points.")
        return torch.linspace(
            problem.domain.lower,
            problem.domain.upper,
            count,
            device=device,
            dtype=dtype,
        ).reshape(-1, 1)
    if isinstance(problem.domain, (Disk, Ellipse, RoundedRectangle, Polygon, ClosedSpline, RectangleWithCircularHoles)):
        if isinstance(points, Mapping):
            x_count = int(points.get("x", points.get("space", 25)))
            y_count = int(points.get("y", points.get("space", 25)))
        else:
            side = max(2, int(math.ceil(math.sqrt(max(4, int(points))))))
            x_count = side
            y_count = side
        if min(x_count, y_count) < 2:
            raise ValueError("2D evaluation requires at least two x/y points.")
        (x_lower, x_upper), (y_lower, y_upper) = problem.domain.bounds
        x_values = torch.linspace(
            x_lower, x_upper, x_count, device=device, dtype=dtype
        )
        y_values = torch.linspace(
            y_lower, y_upper, y_count, device=device, dtype=dtype
        )
        x_grid, y_grid = torch.meshgrid(x_values, y_values, indexing="ij")
        coordinates = torch.stack(
            (x_grid.reshape(-1), y_grid.reshape(-1)), dim=1
        )
        return coordinates[problem.domain.contains(coordinates, atol=1e-7)]
    if isinstance(problem.domain, SpaceTimeDomain) and isinstance(
        problem.domain.space, (Disk, Ellipse, RoundedRectangle, Polygon, ClosedSpline, RectangleWithCircularHoles)
    ):
        explicit_time_slices = evaluation.get("time_slices")
        if explicit_time_slices is not None:
            time_values = torch.as_tensor(
                explicit_time_slices, device=device, dtype=dtype
            ).reshape(-1)
            if time_values.numel() == 0:
                raise ValueError("evaluation.time_slices cannot be empty.")
            if not problem.domain.time.contains(time_values.reshape(-1, 1)).all():
                raise ValueError("An evaluation time slice lies outside the time domain.")
        else:
            time_count = int(points.get("time", points.get("t", 3))) if isinstance(
                points, Mapping
            ) else 3
            if time_count < 1:
                raise ValueError("2D space-time evaluation requires at least one time value.")
            time_values = torch.linspace(
                problem.domain.time.lower,
                problem.domain.time.upper,
                time_count,
                device=device,
                dtype=dtype,
            )
        if isinstance(points, Mapping):
            x_count = int(points.get("x", points.get("space", 25)))
            y_count = int(points.get("y", points.get("space", 25)))
        else:
            side = max(2, int(math.ceil(math.sqrt(max(4, int(points)) / time_values.numel()))))
            x_count = side
            y_count = side
        if min(x_count, y_count) < 2:
            raise ValueError("2D space-time evaluation requires at least two x/y points.")
        (x_lower, x_upper), (y_lower, y_upper) = problem.domain.space.bounds
        x_values = torch.linspace(x_lower, x_upper, x_count, device=device, dtype=dtype)
        y_values = torch.linspace(y_lower, y_upper, y_count, device=device, dtype=dtype)
        x_grid, y_grid = torch.meshgrid(x_values, y_values, indexing="ij")
        spatial = torch.stack((x_grid.reshape(-1), y_grid.reshape(-1)), dim=1)
        spatial = spatial[problem.domain.space.contains(spatial, atol=1e-7)]
        slices = [
            torch.cat(
                (spatial, torch.full((spatial.shape[0], 1), time, device=device, dtype=dtype)),
                dim=1,
            )
            for time in time_values
        ]
        return torch.cat(slices, dim=0)
    explicit_time_slices = evaluation.get("time_slices")
    if explicit_time_slices is not None or "x_points" in evaluation:
        configured_space = points.get("space", points.get("x", 64)) if isinstance(points, Mapping) else 64
        space_count = int(evaluation.get("x_points", configured_space))
        if explicit_time_slices is None:
            time_count = int(evaluation.get("time_points", 32))
            time_values = torch.linspace(
                problem.domain.time.lower,
                problem.domain.time.upper,
                time_count,
                device=device,
                dtype=dtype,
            )
        else:
            time_values = torch.as_tensor(
                explicit_time_slices, device=device, dtype=dtype
            ).reshape(-1)
            if time_values.numel() == 0:
                raise ValueError("evaluation.time_slices cannot be empty.")
            if not problem.domain.time.contains(time_values.reshape(-1, 1)).all():
                raise ValueError("An evaluation time slice lies outside the time domain.")
    elif isinstance(points, Mapping):
        space_count = int(points.get("space", points.get("x", 64)))
        time_count = int(points.get("time", points.get("t", 32)))
        time_values = torch.linspace(
            problem.domain.time.lower,
            problem.domain.time.upper,
            time_count,
            device=device,
            dtype=dtype,
        )
    else:
        requested = max(4, int(points))
        space_count = max(2, int(math.sqrt(2 * requested)))
        time_count = max(2, int(math.ceil(requested / space_count)))
        time_values = torch.linspace(
            problem.domain.time.lower,
            problem.domain.time.upper,
            time_count,
            device=device,
            dtype=dtype,
        )
    if space_count < 2:
        raise ValueError("Burgers evaluation requires at least two spatial points.")
    x = torch.linspace(
        problem.domain.space.lower,
        problem.domain.space.upper,
        space_count,
        device=device,
        dtype=dtype,
    )
    x_grid, t_grid = torch.meshgrid(x, time_values, indexing="ij")
    return torch.stack((x_grid.reshape(-1), t_grid.reshape(-1)), dim=1)


def _condition_metrics(
    problem: PDEProblem,
    model: nn.Module,
    evaluation: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Evaluate constraints on deterministic grids, not the final training batch."""

    boundary_count = max(2, int(evaluation.get("boundary_points", 64)))
    initial_count = max(2, int(evaluation.get("initial_points", 64)))
    if isinstance(problem.domain, Interval):
        boundary_points = torch.tensor(
            [[problem.domain.lower], [problem.domain.upper]],
            device=device,
            dtype=dtype,
        )
        initial_points = torch.empty((0, 1), device=device, dtype=dtype)
    elif isinstance(problem.domain, (Disk, Ellipse, RoundedRectangle, Polygon, ClosedSpline, RectangleWithCircularHoles)):
        generator = torch.Generator(device="cpu").manual_seed(24680)
        boundary_points = problem.domain.sample_boundary(
            boundary_count,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        initial_points = torch.empty((0, 2), device=device, dtype=dtype)
    elif isinstance(problem.domain, SpaceTimeDomain) and isinstance(
        problem.domain.space, (Disk, Ellipse, RoundedRectangle, Polygon, ClosedSpline, RectangleWithCircularHoles)
    ):
        generator = torch.Generator(device="cpu").manual_seed(24680)
        boundary_points = problem.domain.sample_boundary(
            boundary_count,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        initial_points = problem.domain.sample_initial(
            initial_count,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    else:
        times = torch.linspace(
            problem.domain.time.lower,
            problem.domain.time.upper,
            boundary_count,
            device=device,
            dtype=dtype,
        ).reshape(-1, 1)
        left = torch.cat(
            (torch.full_like(times, problem.domain.space.lower), times), dim=1
        )
        right = torch.cat(
            (torch.full_like(times, problem.domain.space.upper), times), dim=1
        )
        boundary_points = torch.cat((left, right), dim=0)
        initial_x = torch.linspace(
            problem.domain.space.lower,
            problem.domain.space.upper,
            initial_count,
            device=device,
            dtype=dtype,
        ).reshape(-1, 1)
        initial_points = torch.cat(
            (initial_x, torch.full_like(initial_x, problem.domain.time.lower)), dim=1
        )

    values: dict[str, float] = {}
    model.eval()
    for condition in problem.conditions:
        points = boundary_points if condition.sample_kind == "boundary" else initial_points
        if points.shape[0] == 0:
            continue
        condition_name = condition.name
        name = f"{condition_name}_rmse"
        if name in values:
            suffix = 2
            while f"{name}_{suffix}" in values:
                suffix += 1
            name = f"{name}_{suffix}"
        errors = condition.error(model, points)
        values[name] = rmse(errors)
        if errors.ndim == 2 and errors.shape[1] == problem.output_dim:
            for index, field_name in enumerate(problem.field_names):
                values[f"{condition_name}_{field_name}_rmse"] = rmse(
                    errors[:, index : index + 1]
                )
    return values


def _heterogeneous_interface_metrics(
    problem: PDEProblem,
    model: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Measure learned solution/flux jumps without claiming domain decomposition."""

    if problem.metadata.get("kind") != "heterogeneous_static":
        return {}
    interface = float(problem.metadata["interface"])
    lower, upper = problem.domain.bounds  # type: ignore[union-attr]
    epsilon = max(1e-7, (upper - lower) * 1e-6)
    points = torch.tensor(
        [[interface - epsilon], [interface + epsilon]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    prediction = model(points)
    slopes = torch.autograd.grad(
        prediction,
        points,
        grad_outputs=torch.ones_like(prediction),
        create_graph=False,
    )[0]
    left_flux = float(problem.metadata["left_diffusivity"]) * slopes[0, 0]
    right_flux = float(problem.metadata["right_diffusivity"]) * slopes[1, 0]
    return {
        "interface_solution_jump": float(
            torch.abs(prediction[1, 0] - prediction[0, 0]).detach().cpu()
        ),
        "interface_flux_jump": float(torch.abs(right_flux - left_flux).detach().cpu()),
    }


def _tunnel_consolidation_outputs(
    problem: PDEProblem,
    coordinates: Tensor,
    prediction: Tensor,
    metrics: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return physical Model-1 indicators without inventing a reference field."""

    if problem.name != "tunnel_consolidation_model1":
        return {}, {}
    parameters = problem.metadata.get("physical_parameters")
    if not isinstance(parameters, Mapping):
        return {}, {}

    width = float(parameters["width_m"])
    height = float(parameters["height_m"])
    duration = float(parameters["duration_years"])
    initial_pressure = float(parameters["initial_pressure_kpa"])
    radius = float(parameters["tunnel_radius_m"])
    centers = np.asarray(parameters["tunnel_centers_m"], dtype=float)
    normalized_coordinates = coordinates.detach().cpu().numpy()
    normalized_pressure = prediction[:, 0].detach().cpu().numpy()
    physical_coordinates = normalized_coordinates * np.asarray(
        (width, height, duration), dtype=normalized_coordinates.dtype
    )
    pressure = normalized_pressure * initial_pressure
    time_values = np.unique(physical_coordinates[:, 2])
    mean_pressure = np.asarray(
        [
            pressure[np.isclose(physical_coordinates[:, 2], time_value)].mean()
            for time_value in time_values
        ],
        dtype=float,
    )
    below_zero = normalized_pressure < 0.0
    above_initial = normalized_pressure > 1.0
    final_mean = float(mean_pressure[-1])

    values: dict[str, Any] = {
        "reference_solution_available": False,
        "pressure_min_kpa": float(pressure.min()),
        "pressure_max_kpa": float(pressure.max()),
        "mean_pressure_initial_time_kpa": float(mean_pressure[0]),
        "mean_pressure_final_time_kpa": final_mean,
        "mean_pressure_dissipation_percent": float(
            100.0 * (1.0 - final_mean / initial_pressure)
        ),
        "negative_pressure_fraction": float(np.mean(below_zero)),
        "above_initial_pressure_fraction": float(np.mean(above_initial)),
        "physical_range_violation_fraction": float(
            np.mean(below_zero | above_initial)
        ),
        "domain_width_m": width,
        "domain_height_m": height,
        "duration_years": duration,
        "initial_pressure_kpa": initial_pressure,
        "tunnel_radius_m": radius,
    }
    if "boundary_rmse" in metrics:
        values["boundary_pressure_rmse_kpa"] = (
            float(metrics["boundary_rmse"]) * initial_pressure
        )
    if "initial_rmse" in metrics:
        values["initial_pressure_rmse_kpa"] = (
            float(metrics["initial_rmse"]) * initial_pressure
        )
    arrays = {
        "physical_coordinates": physical_coordinates,
        "pressure_kpa": pressure[:, None],
        "evaluation_time_years": time_values,
        "mean_pressure_kpa": mean_pressure,
        "tunnel_centers_m": centers,
        "tunnel_radius_m": np.asarray([radius], dtype=float),
    }
    return values, arrays


@dataclass
class ExperimentResult:
    """In-memory result plus the path to the standard artifact directory."""

    metrics: dict[str, Any]
    history: list[dict[str, float]]
    predictions: dict[str, np.ndarray]
    output_dir: Path
    model: nn.Module
    compute_ledger: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        """Allow lightweight mapping-style access used by scripts."""

        return getattr(self, key)

    def as_dict(self) -> dict[str, Any]:
        """Return the serializable result summary."""

        return {
            "metrics": dict(self.metrics),
            "history": [dict(record) for record in self.history],
            "predictions": {key: value.copy() for key, value in self.predictions.items()},
            "output_dir": str(self.output_dir),
            "compute_ledger": dict(self.compute_ledger),
        }


class Experiment:
    """Build and run a starter Poisson or Burgers experiment from configuration."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        registry: ComponentRegistry | None = None,
    ) -> None:
        if registry is not None and not isinstance(registry, ComponentRegistry):
            raise TypeError("registry must be a ComponentRegistry.")
        self.registry = (
            default_component_registry() if registry is None else registry.copy()
        )
        _validate_config(config, self.registry)
        self.config = copy.deepcopy(dict(config))

    @classmethod
    def from_config(
        cls,
        source: str | Path | Mapping[str, Any],
        *,
        registry: ComponentRegistry | None = None,
    ) -> "Experiment":
        """Create an experiment from YAML/JSON or an existing mapping."""

        return cls(load_config(source), registry=registry)

    def _output_dir(self, override: str | Path | None, problem: PDEProblem, seed: int) -> Path:
        if override is not None:
            return Path(override).resolve()
        output = _mapping(self.config.get("output"), "output")
        explicit = self.config.get("output_dir", output.get("directory", output.get("dir")))
        if explicit is not None:
            return Path(explicit).resolve()
        root = Path(output.get("root", "runs"))
        experiment = _mapping(self.config.get("experiment"), "experiment")
        run_name = output.get(
            "run_name", experiment.get("name", f"{problem.name}_seed{seed}")
        )
        return (root / str(run_name)).resolve()

    def run(self, *, output_dir: str | Path | None = None) -> ExperimentResult:
        """Train, evaluate, and write the standard reproducibility artifacts."""

        experiment_config = _mapping(self.config.get("experiment"), "experiment")
        seed = int(self.config.get("seed", experiment_config.get("seed", 0)))
        torch.manual_seed(seed)
        np.random.seed(seed)
        selected_device = _device(
            self.config.get("device", experiment_config.get("device", "cpu"))
        )
        selected_dtype = _dtype(
            self.config.get("dtype", experiment_config.get("dtype", "float32"))
        )
        if selected_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats(selected_device)

        problem = _problem(self.config, self.registry)
        model = _model(self.config, problem, self.registry).to(
            device=selected_device, dtype=selected_dtype
        )
        training = _mapping(self.config.get("training"), "training")
        initial_checkpoint_value = training.get("initial_checkpoint")
        initial_checkpoint_path: Path | None = None
        initial_checkpoint_sha256: str | None = None
        if initial_checkpoint_value not in {None, ""}:
            initial_checkpoint_path = Path(str(initial_checkpoint_value)).resolve()
            if not initial_checkpoint_path.is_file():
                raise FileNotFoundError(initial_checkpoint_path)
            checkpoint = torch.load(
                initial_checkpoint_path,
                map_location=selected_device,
                weights_only=True,
            )
            if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
                raise ValueError(
                    "training.initial_checkpoint must be an OpenPINN Trainer checkpoint."
                )
            try:
                model.load_state_dict(checkpoint["model"], strict=True)
            except RuntimeError as exc:
                raise ValueError(
                    "Initial checkpoint model is incompatible with the configured architecture."
                ) from exc
            initial_checkpoint_sha256 = hashlib.sha256(
                initial_checkpoint_path.read_bytes()
            ).hexdigest()
        formulation = _formulation(self.config, problem, self.registry)
        sampler = _sampler(
            self.config,
            problem,
            self.registry,
            seed=seed,
            device=selected_device,
            dtype=selected_dtype,
        )
        loss = _mapping(self.config.get("loss"), "loss")
        composer = _loss_composer(loss, self.registry)
        optimizer_name, optimizer_adapter = _optimizer(
            self.config, model, self.registry
        )
        trainer = Trainer(
            model,
            formulation,
            sampler,
            optimizer=optimizer_adapter,
            loss_composer=composer,
            device=selected_device,
            dtype=selected_dtype,
        )
        steps = int(training.get("steps", training.get("epochs", 1000)))
        log_every = int(training.get("log_every", 0))
        resample_every = int(training.get("resample_every", 1))

        artifact_dir = self._output_dir(output_dir, problem, seed)
        if artifact_dir.exists() and any(artifact_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {artifact_dir}. Choose a fresh run directory."
            )
        writer = ArtifactWriter(artifact_dir)
        persisted_config = copy.deepcopy(self.config)
        persisted_config.setdefault("schema_version", 1)
        writer.write_config(persisted_config)
        started = time.perf_counter()
        history: TrainingHistory = trainer.fit(
            steps=steps,
            resample_every=resample_every,
            log_every=log_every,
        )
        elapsed = time.perf_counter() - started

        evaluation = _mapping(self.config.get("evaluation"), "evaluation")
        coordinates = _evaluation_coordinates(
            problem, evaluation, device=selected_device, dtype=selected_dtype
        ).detach().requires_grad_(True)
        model.eval()
        prediction = model(coordinates)
        residual_map = problem.evaluate_residual(coordinates, prediction)
        residual_columns = torch.cat(
            [value.reshape(value.shape[0], -1) for value in residual_map.values()], dim=1
        )
        metrics: dict[str, Any] = {
            "problem": problem.name,
            "method": _mapping(self.config.get("method"), "method").get(
                "name", "component_config"
            ),
            "variant": _mapping(self.config.get("method"), "method").get(
                "variant", "component_config"
            ),
            "steps": steps,
            "final_total_loss": history.last["total"] if history else None,
            "pde_residual_rmse": rmse(residual_columns),
            "reference_solution_available": problem.exact_solution is not None,
        }
        current_weights = getattr(composer, "current_weights", None)
        if callable(current_weights):
            metrics.update(
                {
                    f"final_weight_{name}": float(value.cpu())
                    for name, value in current_weights(detach=True).items()
                }
            )
        for residual_name, residual_value in residual_map.items():
            metrics[f"residual_{residual_name}_rmse"] = rmse(residual_value)
        metrics["residual_rmse"] = metrics["pde_residual_rmse"]
        if history:
            for name, value in history.last.items():
                if name in {
                    "step",
                    "elapsed_seconds",
                    "total",
                    "optimizer_stage",
                } or name.startswith("weight_"):
                    continue
                metrics[f"train_final_{name}_loss"] = value
            if "boundary" in history.last:
                metrics["train_final_boundary_rmse"] = math.sqrt(
                    max(0.0, history.last["boundary"])
                )
            if "initial" in history.last:
                metrics["train_final_initial_rmse"] = math.sqrt(
                    max(0.0, history.last["initial"])
                )
        predictions: dict[str, np.ndarray] = {
            "coordinates": coordinates.detach().cpu().numpy(),
            "prediction": prediction.detach().cpu().numpy(),
            "residual": residual_columns.detach().cpu().numpy(),
        }
        for residual_name, residual_value in residual_map.items():
            predictions[f"residual_{residual_name}"] = (
                residual_value.detach().cpu().numpy()
            )
        for index, field_name in enumerate(problem.field_names):
            predictions[f"field_{field_name}"] = (
                prediction[:, index : index + 1].detach().cpu().numpy()
            )
        if problem.exact_solution is not None:
            reference = problem.exact(coordinates)
            metrics["relative_l2_error"] = relative_l2_error(prediction, reference)
            metrics["max_absolute_error"] = max_absolute_error(prediction, reference)
            metrics["rmse"] = rmse(prediction, reference)
            metrics["mae"] = mean_absolute_error(prediction, reference)
            metrics["r2_score"] = r2_score(prediction, reference)
            metrics["relative_l2"] = metrics["relative_l2_error"]
            metrics["max_abs_error"] = metrics["max_absolute_error"]
            predictions["reference"] = reference.detach().cpu().numpy()
            for index, field_name in enumerate(problem.field_names):
                predicted_field = prediction[:, index : index + 1]
                reference_field = reference[:, index : index + 1]
                metrics[f"relative_l2_{field_name}"] = relative_l2_error(
                    predicted_field, reference_field
                )
                metrics[f"max_absolute_error_{field_name}"] = max_absolute_error(
                    predicted_field, reference_field
                )
                metrics[f"rmse_{field_name}"] = rmse(
                    predicted_field, reference_field
                )
                metrics[f"mae_{field_name}"] = mean_absolute_error(
                    predicted_field, reference_field
                )
                metrics[f"r2_score_{field_name}"] = r2_score(
                    predicted_field, reference_field
                )
                predictions[f"reference_{field_name}"] = (
                    reference_field.detach().cpu().numpy()
                )
            reference_metadata = problem.metadata.get("reference")
            if isinstance(reference_metadata, Mapping):
                for key, value in reference_metadata.items():
                    metrics[f"reference_{key}"] = value

        metrics.update(
            _condition_metrics(
                problem,
                model,
                evaluation,
                device=selected_device,
                dtype=selected_dtype,
            )
        )
        metrics.update(
            _heterogeneous_interface_metrics(
                problem,
                model,
                device=selected_device,
                dtype=selected_dtype,
            )
        )
        tunnel_metrics, tunnel_predictions = _tunnel_consolidation_outputs(
            problem,
            coordinates,
            prediction,
            metrics,
        )
        metrics.update(tunnel_metrics)
        predictions.update(tunnel_predictions)
        diagnostics_method = getattr(sampler, "diagnostics", None)
        sampler_diagnostics = (
            dict(diagnostics_method()) if callable(diagnostics_method) else {}
        )
        optimizer_diagnostics_method = getattr(trainer.optimizer, "diagnostics", None)
        optimizer_diagnostics = (
            dict(optimizer_diagnostics_method())
            if callable(optimizer_diagnostics_method)
            else {}
        )
        for name, value in sampler_diagnostics.items():
            if isinstance(value, Number):
                metrics[f"sampling_{name}"] = value

        try:
            _assert_finite(history.as_list(), "history")
            _assert_finite(metrics, "metrics")
            _assert_finite(predictions, "predictions")
        except FloatingPointError as exc:
            writer.write_json(
                "status.json",
                {
                    "status": "failed",
                    "problem": problem.name,
                    "steps": steps,
                    "finite_outputs": False,
                    "accuracy_claim": False,
                    "failure_reason": str(exc),
                },
            )
            raise

        model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
        model_trainable_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        adaptive_parameter_count = (
            sum(parameter.numel() for parameter in composer.parameters())
            if isinstance(composer, nn.Module)
            else 0
        )
        parameter_count = model_parameter_count + adaptive_parameter_count
        trainable_count = model_trainable_count + adaptive_parameter_count
        ledger: dict[str, Any] = {
            "status": "completed",
            "problem": problem.name,
            "method": _mapping(self.config.get("method"), "method").get(
                "name", "component_config"
            ),
            "variant": _mapping(self.config.get("method"), "method").get(
                "variant", "component_config"
            ),
            "seed": seed,
            "device": str(selected_device),
            "dtype": str(selected_dtype).replace("torch.", ""),
            "steps": steps,
            "optimizer": optimizer_name,
            "closure_calls": trainer.optimizer.closure_calls,
            "elapsed_seconds": elapsed,
            "wall_time_seconds": elapsed,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_count,
            "trainable_parameters": trainable_count,
            "model_parameter_count": model_parameter_count,
            "adaptive_weight_parameter_count": adaptive_parameter_count,
            "sampler": type(sampler).__name__,
            "sampler_diagnostics": sampler_diagnostics,
            "optimizer_diagnostics": optimizer_diagnostics,
            "warm_started": initial_checkpoint_path is not None,
            "initial_checkpoint": (
                str(initial_checkpoint_path) if initial_checkpoint_path is not None else None
            ),
            "initial_checkpoint_sha256": initial_checkpoint_sha256,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(selected_device))
                if selected_device.type == "cuda"
                else 0
            ),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "failure_reason": None,
        }

        writer.write_metrics(metrics)
        writer.write_history(history.as_list())
        writer.write_predictions(predictions)
        writer.write_compute_ledger(ledger)
        if sampler_diagnostics:
            writer.write_json("sampling_diagnostics.json", sampler_diagnostics)
        if optimizer_diagnostics:
            writer.write_json("optimizer_diagnostics.json", optimizer_diagnostics)
        writer.write_json(
            "status.json",
            {
                "status": "completed",
                "problem": problem.name,
                "steps": steps,
                "finite_outputs": True,
                "accuracy_claim": False,
            },
        )
        output_options = _mapping(self.config.get("output"), "output")
        if bool(output_options.get("save_checkpoint", False)):
            writer.write_checkpoint(trainer.state_dict())
        writer.write_manifest(
            metadata={"problem": problem.name, "seed": seed, "schema": "openpinn-starter-v1"}
        )
        return ExperimentResult(
            metrics=metrics,
            history=history.as_list(),
            predictions=predictions,
            output_dir=artifact_dir,
            model=model,
            compute_ledger=ledger,
        )
