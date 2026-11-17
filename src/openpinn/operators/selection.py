"""Shared operator selection for ``build_operator_config`` and ``train_operator``.

DeepONet (M1c), FNO (M1d), and PINO (M1e) are selected through this module.
``solve`` remains the PINN entry point.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import torch
from torch import nn

_METADATA_KEYS = frozenset({"operator", "kind", "derivative_mode", "title"})
TRAINING_MODES = ("data_only", "physics_only", "hybrid")


@dataclass(frozen=True)
class OperatorTrainResult:
    """One operator-training run, synthetic smoke or designed problem."""

    operator: str
    steps: int
    initial_loss: float
    final_loss: float
    history: tuple[float, ...]
    model: nn.Module
    config: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    evaluator: Callable[[torch.Tensor], torch.Tensor] | None = field(
        default=None, repr=False, compare=False
    )


def _require_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return int(value)


def _require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _widths(name: str, value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        parsed: list[Any] = []
        for part in parts:
            try:
                parsed.append(int(part))
            except ValueError as exc:
                raise ValueError(f"{name} entries must be positive integers.") from exc
        value = parsed
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list of positive integers.")
    widths: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or int(item) <= 0:
            raise ValueError(f"{name} entries must be positive integers.")
        widths.append(int(item))
    return tuple(widths)


def _validate_option(name: str, value: Any, schema: Mapping[str, Any]) -> Any:
    kind = schema["kind"]
    if kind == "integer":
        return _require_int(name, value, minimum=int(schema["minimum"]))
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number.")
        selected = float(value)
        if not math.isfinite(selected):
            raise ValueError(f"{name} must be finite.")
        minimum = schema.get("minimum")
        if minimum is not None and selected < float(minimum):
            raise ValueError(f"{name} must be >= {minimum}.")
        return selected
    if kind == "boolean":
        return _require_bool(name, value)
    if kind == "choice":
        selected = str(value).lower()
        if selected not in schema["choices"]:
            raise ValueError(f"{name} must be one of {list(schema['choices'])!r}.")
        return selected
    if kind == "integer_list":
        return list(_widths(name, value))
    raise ValueError(f"Unknown operator parameter kind {kind!r}.")


def _copy_default(value: Any) -> Any:
    if isinstance(value, list):
        return list(value)
    return value


def _network_config(name: str, options: Mapping[str, Any]) -> None:
    if name == "deeponet":
        from .deeponet import _network_config as build_config

        build_config(options)
        return
    if name == "fno":
        from .fno import _network_config as build_config

        build_config(options)
        return
    raise ValueError(f"No operator builder is registered for {name!r}.")


def build_operator_config(name: str = "deeponet", **options: Any) -> dict[str, Any]:
    """Resolve one selectable operator and its configurator knobs.

    Unknown names, unimplemented hooks, and unknown option names fail before
    a network is constructed. The returned mapping is the input to
    :func:`build_deeponet`, :func:`build_fno`, :func:`build_pino`, and
    :func:`train_operator`.
    """

    from ..operator_protocol import default_operator_registry

    registry = default_operator_registry()
    spec = registry.resolve(name)
    selected_options = dict(options)
    if spec.name == "pino":
        warnings.warn(
            "The 'pino' operator selection name is deprecated; use model='fno' "
            "with training_mode='hybrid' or 'physics_only'.",
            DeprecationWarning,
            stacklevel=2,
        )
        legacy_mode = selected_options.pop("mode", None)
        if legacy_mode is not None and "training_mode" in selected_options:
            raise ValueError("Pass mode or training_mode for the pino shim, not both.")
        selected_options["training_mode"] = (
            legacy_mode
            if legacy_mode is not None
            else selected_options.get("training_mode", "hybrid")
        )
        return build_operator_config("fno", **selected_options)
    if spec.name == "deeponet" and "derivative" in selected_options:
        legacy_derivative = str(selected_options.pop("derivative")).lower()
        if legacy_derivative not in ("spectral", "finite_difference"):
            raise ValueError(
                "The deprecated DeepONet derivative option must be 'spectral' "
                "or 'finite_difference'; DeepONet physics now always uses "
                "coordinate autodiff."
            )
        warnings.warn(
            "The DeepONet derivative option is deprecated and ignored; "
            "DeepONet physics now always uses coordinate autodiff.",
            DeprecationWarning,
            stacklevel=2,
        )
    if not spec.implemented or not spec.selectable:
        raise ValueError(
            f"Operator {spec.name!r} is not selectable. "
            "Only implemented operators can be built."
        )
    schema = dict(spec.parameters)
    if not schema:
        raise ValueError(f"No operator builder is registered for {spec.name!r}.")
    unknown = sorted(set(selected_options).difference(schema))
    if unknown:
        available = ", ".join(schema)
        raise ValueError(
            f"Unknown option(s) for operator {spec.name!r}: {', '.join(unknown)}; "
            f"available options: {available}."
        )
    resolved = {
        key: (
            _validate_option(key, selected_options[key], schema[key])
            if key in selected_options and selected_options[key] is not None
            else _copy_default(schema[key]["default"])
        )
        for key in schema
    }
    _network_config(spec.name, resolved)
    return {
        "operator": spec.name,
        "kind": spec.kind,
        "derivative_mode": spec.derivative_mode,
        "title": spec.title,
        **resolved,
    }


def _as_operator_config(
    source: str | Mapping[str, Any],
    options: Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(source, str):
        return build_operator_config(source, **dict(options))
    if not isinstance(source, Mapping):
        raise TypeError("train_operator expects an operator name or a config mapping.")
    if options:
        raise ValueError("Pass keyword options with an operator name, or pass a config mapping.")
    stored = {key: value for key, value in source.items() if key not in _METADATA_KEYS}
    name = str(source.get("operator", "deeponet"))
    return build_operator_config(name, **stored)


def train_operator(
    source: str | Mapping[str, Any] = "deeponet",
    *,
    steps: int | None = None,
    seed: int | None = None,
    device: str = "cpu",
    **options: Any,
) -> OperatorTrainResult:
    """Train one selectable operator on a tiny synthetic batch.

    ``training_mode='data_only'`` keeps the model's existing supervised smoke.
    ``physics_only`` and ``hybrid`` use each model's natural derivative:
    DeepONet differentiates at query coordinates, while FNO reuses the grid-PDE
    trainer. Each path reuses one fixed batch, so this is a smoke, not a benchmark.
    """

    config = _as_operator_config(source, options)
    if steps is not None:
        config["steps"] = _require_int("steps", steps, minimum=1)
    if seed is not None:
        config["seed"] = _require_int("seed", seed, minimum=0)
    selected_device = str(device)
    if selected_device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")

    name = str(config["operator"])
    training_mode = str(config["training_mode"])
    if training_mode not in TRAINING_MODES:
        raise ValueError(f"training_mode must be one of {list(TRAINING_MODES)!r}.")
    if name == "deeponet" and training_mode == "data_only":
        from .deeponet import train_deeponet

        return train_deeponet(config, device=selected_device)
    if name == "fno" and training_mode == "data_only":
        from .fno import train_fno

        return train_fno(config, device=selected_device)
    if name == "deeponet":
        from .deeponet import train_deeponet_physics

        return train_deeponet_physics(config, device=selected_device)
    if name == "fno":
        from .pino import train_pino

        physics_config = _physics_config(config)
        torch.manual_seed(int(config["seed"]))
        from .fno import build_fno

        model = build_fno(config)
        result = train_pino(physics_config, device=selected_device, model=model)
        return replace(result, operator=name, config=dict(config))
    raise ValueError(f"No trainer is registered for operator {name!r}.")


def _physics_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Translate an FNO config into the M10 grid-PDE smoke contract."""

    name = str(config["operator"])
    if name == "fno":
        if (
            int(config["n_dims"]) != 1
            or int(config["in_channels"]) != 1
            or int(config["out_channels"]) != 1
        ):
            raise ValueError(
                "physics training supports n_dims=1, in_channels=1, and "
                "out_channels=1 for the built-in grid problem."
            )
        grid_size = int(config["grid_size"])
        n_modes = int(config["n_modes"])
        n_harmonics = int(config["n_harmonics"])
    else:
        raise ValueError(f"No grid-physics trainer is registered for operator {name!r}.")
    return {
        "operator": name,
        "n_dims": 1,
        "in_channels": 1,
        "out_channels": 1,
        "grid_size": grid_size,
        "n_modes": n_modes,
        "mode": str(config["training_mode"]),
        "data_weight": float(config["data_weight"]),
        "physics_weight": float(config["physics_weight"]),
        "data_pretrain_steps": int(config["data_pretrain_steps"]),
        "lr": float(config["lr"]),
        "steps": int(config["steps"]),
        "batch_size": int(config["batch_size"]),
        "n_harmonics": n_harmonics,
        "seed": int(config["seed"]),
        "derivative": str(config["derivative"]),
    }
