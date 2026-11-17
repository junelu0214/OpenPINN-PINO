"""Short, user-facing OpenPINN problem → method → training entry points."""

from __future__ import annotations

import importlib
import importlib.util
import json
import copy
import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from .artifacts import ArtifactWriter, write_failure_artifacts, write_result_figures
from .config import load_config
from .experiment import Experiment, ExperimentResult, default_component_registry
from .registry import ComponentRegistry
from .method_registry import MethodSpec, apply_method, deep_merge
from .method_options import (
    MethodSelection,
    coerce_method_selection,
    method as select_method,
    validate_method_stack,
)
from .problems import PDEProblem


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


_DEFAULT_RUN_NAME_MAX_BYTES = 180


def _utf8_prefix(value: str, byte_limit: int) -> str:
    """Return a valid UTF-8 prefix that fits in one filesystem component."""

    return value.encode("utf-8")[:byte_limit].decode("utf-8", errors="ignore").rstrip("._-")


def _default_run_name(problem_label: str, method_slug: str) -> str:
    """Build a readable run name without exceeding common filename limits."""

    timestamp = _timestamp()
    requested = f"{problem_label}_{method_slug}_{timestamp}"
    if len(requested.encode("utf-8")) <= _DEFAULT_RUN_NAME_MAX_BYTES:
        return requested

    digest = hashlib.sha256(requested.encode("utf-8")).hexdigest()[:12]
    suffix = f"__{digest}_{timestamp}"
    prefix_limit = _DEFAULT_RUN_NAME_MAX_BYTES - len(suffix.encode("utf-8"))
    prefix = _utf8_prefix(f"{problem_label}_{method_slug}", prefix_limit)
    return f"{prefix}{suffix}"


def _default_problem_loss_weights(problem: PDEProblem) -> dict[str, float]:
    """Return named unit weights matching a user problem's declared terms."""

    declared = problem.metadata.get("residual_names", ("pde",))
    if isinstance(declared, str):
        residual_names = (declared,)
    else:
        residual_names = tuple(str(name) for name in declared)
    if not residual_names or any(not name.strip() for name in residual_names):
        raise ValueError("problem.metadata.residual_names must contain non-empty names.")

    term_names = list(residual_names)
    for condition in problem.conditions:
        name = str(condition.name)
        if name in term_names:
            suffix = 2
            while f"{name}_{suffix}" in term_names:
                suffix += 1
            name = f"{name}_{suffix}"
        term_names.append(name)
    return {name: 1.0 for name in term_names}


def _problem_defaults(problem: str | PDEProblem) -> tuple[str, dict[str, Any]]:
    if isinstance(problem, PDEProblem):
        return "user_problem", {
            "problem": {"name": "user_problem"},
            "model": {
                "name": "mlp",
                "hidden_layers": [64, 64, 64],
                "activation": "tanh",
                "feature": {"name": "identity"},
                "output_transform": {"name": "soft"},
            },
            "sampling": {"name": "uniform", "interior": 128, "boundary": 32, "initial": 32},
            "loss": {
                "formulation": "strong",
                "weights": _default_problem_loss_weights(problem),
            },
            "evaluation": {"points": 256},
        }

    normalized = "".join(character for character in problem.lower() if character.isalnum())
    if normalized in {"poisson", "poisson1d", "static"}:
        return "poisson", {
            "problem": {
                "name": "poisson",
                "domain": {"x": [-1.0, 1.0]},
                "parameters": {"wave_number": 1.0, "amplitude": 1.0},
            },
            "model": {
                "name": "mlp",
                "hidden_layers": [64, 64, 64],
                "activation": "tanh",
                "feature": {"name": "identity"},
                "output_transform": {"name": "soft"},
            },
            "sampling": {"name": "uniform", "interior": 128, "boundary": 32},
            "loss": {
                "formulation": "strong",
                "weights": {"pde": 1.0, "boundary": 1.0},
            },
            "evaluation": {"points": 257},
        }
    if normalized in {"burgers", "burgers1d", "burger", "dynamic"}:
        return "burgers", {
            "problem": {
                "name": "burgers",
                "domain": {"x": [-1.0, 1.0], "t": [0.0, 1.0]},
                "parameters": {"viscosity": 0.003183098861837907},
            },
            "model": {
                "name": "mlp",
                "hidden_layers": [64, 64, 64, 64],
                "activation": "tanh",
                "feature": {"name": "identity"},
                "output_transform": {"name": "soft"},
            },
            "sampling": {
                "name": "uniform",
                "interior": 256,
                "boundary": 64,
                "initial": 64,
            },
            "loss": {
                "formulation": "strong",
                "weights": {"pde": 1.0, "boundary": 1.0, "initial": 1.0},
            },
            "evaluation": {
                "points": {"space": 129, "time": 4},
                "time_slices": [0.0, 0.33, 0.67, 1.0],
            },
        }
    if normalized in {
        "tunnelmodel1",
        "model1",
        "tunnelconsolidationmodel1",
        "twintunnelconsolidation",
    }:
        return "tunnel_model1", {
            "problem": {
                "name": "tunnel_consolidation_model1",
                "parameters": {
                    "width_m": 40.0,
                    "height_m": 40.0,
                    "duration_years": 40.0,
                    "tunnel_radius_m": 4.0,
                    "tunnel_centers_m": [[10.0, 20.0], [30.0, 20.0]],
                    "initial_pressure_kpa": 20.0,
                    "consolidation_x_m2_per_year": 0.30,
                    "consolidation_y_m2_per_year": 0.60,
                },
            },
            "model": {
                "name": "mlp",
                "hidden_layers": [64, 64, 64, 64],
                "activation": "tanh",
                "feature": {"name": "identity"},
                "output_transform": {"name": "soft"},
            },
            "sampling": {
                "name": "uniform",
                "interior": 512,
                "boundary": 256,
                "initial": 256,
            },
            "loss": {
                "formulation": "strong",
                "weights": {"pde": 1.0, "boundary": 1.0, "initial": 1.0},
            },
            "evaluation": {
                "points": {"x": 61, "y": 61, "time": 3},
                "time_slices": [0.0, 0.5, 1.0],
            },
        }
    if normalized in {
        "heterogeneouspoisson",
        "heterogeneouspoisson1d",
        "piecewisepoisson",
        "hetero",
    }:
        return "heterogeneous_poisson", {
            "problem": {
                "name": "heterogeneous_poisson",
                "domain": {"x": [-1.0, 1.0]},
                "parameters": {
                    "interface": 0.0,
                    "left_diffusivity": 1.0,
                    "right_diffusivity": 4.0,
                },
            },
            "model": {
                "name": "mlp",
                "hidden_layers": [32, 32, 32],
                "activation": "tanh",
                "feature": {"name": "identity"},
                "output_transform": {"name": "soft"},
            },
            "sampling": {"name": "uniform", "interior": 128, "boundary": 32},
            "loss": {
                "formulation": "strong",
                "weights": {"pde": 1.0, "boundary": 1.0},
            },
            "evaluation": {"points": 257},
        }
    if normalized in {
        "coupledmanufactured",
        "coupledellipticmanufactured",
        "coupledellipticmanufactured1d",
    }:
        return "coupled_manufactured", {
            "problem": {
                "name": "coupled_elliptic_manufactured",
                "domain": {"x": [-1.0, 1.0]},
                "parameters": {"coupling_strength": 1.0},
            },
            "model": {
                "name": "mlp",
                "hidden_layers": [32, 32, 32],
                "activation": "tanh",
                "feature": {"name": "identity"},
                "output_transform": {"name": "soft"},
            },
            "sampling": {"name": "uniform", "interior": 128, "boundary": 32},
            "loss": {
                "formulation": "strong",
                "weights": {
                    "equation_u": 1.0,
                    "equation_v": 1.0,
                    "coupling": 1.0,
                    "boundary": 1.0,
                },
            },
            "evaluation": {"points": 257},
        }
    raise ValueError(
        f"Unknown built-in problem {problem!r}. Choose poisson, burgers, "
        "tunnel_model1, heterogeneous_poisson, coupled_manufactured, or pass "
        "a PDEProblem object."
    )


def build_config(
    problem: str | PDEProblem,
    *,
    method: str | None = None,
    variant: str | None = None,
    methods: Sequence[MethodSelection | Mapping[str, Any]] | None = None,
    steps: int = 1000,
    seed: int = 0,
    device: str = "cpu",
    dtype: str = "float32",
    method_options: Mapping[str, Any] | None = None,
    problem_options: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
    sampling: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    output_root: str | Path = "runs",
    run_name: str | None = None,
) -> tuple[dict[str, Any], MethodSpec]:
    """Build the resolved low-level configuration used by :class:`Experiment`."""

    if isinstance(steps, bool) or int(steps) <= 0:
        raise ValueError("steps must be a positive integer.")
    steps = int(steps)
    if methods is not None:
        if method is not None or variant is not None or method_options is not None:
            raise ValueError(
                "Pass methods=[...] or the legacy method/variant/method_options "
                "arguments, not both."
            )
        selections = validate_method_stack(
            tuple(coerce_method_selection(item) for item in methods)
        )
    else:
        selections = validate_method_stack(
            (
                select_method(
                    method or "baseline",
                    variant,
                    **dict(method_options or {}),
                ),
            )
        )
    problem_label, defaults = _problem_defaults(problem)
    method_slug = "__".join(
        f"{selection.method}-{selection.variant}" for selection in selections
    )
    selected_run_name = run_name or _default_run_name(problem_label, method_slug)
    config: dict[str, Any] = {
        "schema_version": 1,
        "experiment": {
            "name": selected_run_name,
            "description": "Resolved by the short OpenPINN problem + method API.",
            "seed": int(seed),
            "device": str(device),
            "dtype": str(dtype),
        },
        **defaults,
        "optimizer": {"name": "adam", "lr": 1e-3},
        "training": {"steps": steps, "log_every": max(1, steps // 10)},
        "output": {
            "root": str(output_root),
            "run_name": selected_run_name,
            "save_checkpoint": True,
        },
    }
    if problem_options:
        config["problem"] = deep_merge(config["problem"], problem_options)
    if model:
        config["model"] = deep_merge(config["model"], model)
    if sampling:
        config["sampling"] = deep_merge(config["sampling"], sampling)
    if training:
        config["training"] = deep_merge(config["training"], training)
        config["training"]["steps"] = steps
    specs: list[MethodSpec] = []
    method_records: list[dict[str, Any]] = []
    for selection in selections:
        config, selected_spec = apply_method(
            config,
            selection.method,
            variant=selection.variant,
            problem=problem,
            steps=steps,
            options=selection.options,
        )
        specs.append(selected_spec)
        method_records.append(copy.deepcopy(config["method"]))

    if len(specs) == 1:
        spec = specs[0]
    else:
        spec = MethodSpec(
            name="+".join(item.name for item in specs),
            title=" + ".join(item.title for item in specs),
            component_file=", ".join(item.component_file for item in specs),
            description="Composed OpenPINN method stack.",
            variant_file=", ".join(item.variant_file for item in specs),
            default_variant="+".join(item.variant for item in specs),
            variant="+".join(item.variant for item in specs),
            variant_title=" + ".join(item.variant_title for item in specs),
        )
        config["methods"] = method_records
        config["method"] = {
            "name": spec.name,
            "variant": spec.variant,
            "variant_title": spec.variant_title,
            "title": spec.title,
            "component_file": spec.component_file,
            "variant_file": spec.variant_file,
            "options": {
                f"{item['name']}/{item['variant']}": item["options"]
                for item in method_records
            },
            "resolved": True,
        }
    if overrides:
        config = deep_merge(config, overrides)
        config.setdefault("training", {})["steps"] = steps
    config["experiment"]["description"] = (
        f"{problem_label} with {spec.name}/{spec.variant}; selected through the "
        "unified OpenPINN method-stack API."
    )
    return config, spec


def _failure_target(output_dir: str | Path | None, config: Mapping[str, Any] | None) -> Path:
    if output_dir is not None:
        target = Path(output_dir).expanduser().resolve()
    elif config is not None:
        output = dict(config.get("output", {}))
        target = (Path(output.get("root", "runs")) / output.get("run_name", f"failed_{_timestamp()}"))
        target = target.expanduser().resolve()
    else:
        target = (Path("runs") / f"failed_{_timestamp()}").resolve()
    if target.exists() and any(target.iterdir()):
        target = target.parent / f"{target.name}_failure_{_timestamp()}"
    return target


def solve(
    problem: str | PDEProblem,
    *,
    method: str | None = None,
    variant: str | None = None,
    methods: Sequence[MethodSelection | Mapping[str, Any]] | None = None,
    steps: int = 1000,
    seed: int = 0,
    device: str = "cpu",
    dtype: str = "float32",
    method_options: Mapping[str, Any] | None = None,
    problem_options: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
    sampling: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    output_root: str | Path = "runs",
    run_name: str | None = None,
    plot: bool = True,
    registry: ComponentRegistry | None = None,
) -> ExperimentResult:
    """Train one built-in or user-defined PDE using method shorthands.

    Examples
    --------
    ``solve("poisson", method="fourier", variant="random_fourier", steps=500)``

    Independent improvements can be composed with
    ``methods=[method("fourier"), method("adaptive_weights", "relobralo")]``.
    A component-slot conflict is rejected before training starts.

    For a new equation, construct :class:`PDEProblem` once and pass that object
    in place of the built-in problem name.

    Project-specific model, sampler, loss, optimizer, and derivative factories
    can be registered on a :class:`ComponentRegistry` and supplied through
    ``registry``.  The registry is copied before the run, so ``solve`` does not
    mutate the caller's extension set.
    """

    config: dict[str, Any] | None = None
    problem_label = problem.name if isinstance(problem, PDEProblem) else str(problem)
    try:
        config, spec = build_config(
            problem,
            method=method,
            variant=variant,
            methods=methods,
            steps=steps,
            seed=seed,
            device=device,
            dtype=dtype,
            method_options=method_options,
            problem_options=problem_options,
            model=model,
            sampling=sampling,
            training=training,
            overrides=overrides,
            output_root=output_root,
            run_name=run_name,
        )
        selected_registry = (
            default_component_registry() if registry is None else registry.copy()
        )
        if isinstance(problem, PDEProblem):
            selected_registry.register(
                "problem",
                "user_problem",
                lambda *, config: problem,
                replace=True,
            )
        result = Experiment(config, registry=selected_registry).run(output_dir=output_dir)
        writer = ArtifactWriter(result.output_dir)
        if plot:
            write_result_figures(
                result,
                method_title=f"{spec.title} · {spec.variant_title}",
            )
        (writer.output_dir / "failure_notes.md").write_text(
            "# Failure and limitation notes\n\n"
            "- Training completed with finite saved outputs.\n"
            "- This run alone is not a statistical superiority claim.\n"
            "- The exact/reference solution, when available, is evaluation-only.\n",
            encoding="utf-8",
        )
        writer.write_manifest(
            metadata={
                "schema": "openpinn-unified-v0.6",
                "problem": result.metrics.get("problem"),
                "method": spec.name,
                "variant": spec.variant,
                "methods": config.get("methods", [config.get("method")]),
                "status": "completed",
            }
        )
        return result
    except Exception as error:
        failure_dir = _failure_target(output_dir, config)
        write_failure_artifacts(
            failure_dir,
            error,
            problem=problem_label,
            method=(
                ", ".join(
                    f"{item.method}/{item.variant}"
                    for item in (
                        coerce_method_selection(value) for value in methods
                    )
                )
                if methods is not None
                else f"{method or 'baseline'}/{variant}" if variant else method or "baseline"
            ),
        )
        raise


def _load_module_from_file(path: Path) -> ModuleType:
    module_name = f"openpinn_user_problem_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load user problem module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_problem_factory(
    reference: str,
    *,
    base_dir: str | Path | None = None,
) -> Callable[..., PDEProblem]:
    """Load ``module:function`` or ``relative/file.py:function`` safely."""

    if ":" not in reference:
        raise ValueError("Problem factory must use module:function syntax.")
    module_reference, attribute = reference.rsplit(":", 1)
    if module_reference.endswith(".py"):
        path = Path(module_reference)
        if not path.is_absolute():
            path = Path(base_dir or Path.cwd()) / path
        module = _load_module_from_file(path.resolve())
    else:
        module = importlib.import_module(module_reference)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise TypeError(f"Problem factory {reference!r} is not callable.")
    return factory


def solve_config(
    source: str | Path | Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    steps: int | None = None,
) -> ExperimentResult:
    """Run the compact problem/method YAML format through :func:`solve`."""

    values = load_config(source)
    schema_version = int(values.get("schema_version", 2))
    if schema_version != 2:
        raise ValueError(
            "The compact problem + method YAML entry requires schema_version: 2. "
            "Legacy schema_version: 1 files remain available under "
            "configs/reference_runs and run through Experiment directly."
        )
    source_path = Path(source).resolve() if isinstance(source, (str, Path)) else None
    problem_value = values.get("problem", "poisson")
    problem_options: dict[str, Any] = {}
    if isinstance(problem_value, Mapping):
        problem_mapping = dict(problem_value)
        factory_reference = problem_mapping.pop("factory", None)
        factory_options = dict(problem_mapping.pop("factory_options", {}))
        problem_name = problem_mapping.pop("name", "poisson")
        nested_options = problem_mapping.pop("options", {})
        if nested_options:
            problem_mapping = deep_merge(problem_mapping, nested_options)
        if factory_reference is not None:
            factory = load_problem_factory(
                str(factory_reference),
                base_dir=source_path.parent if source_path is not None else None,
            )
            problem = factory(**factory_options)
            if not isinstance(problem, PDEProblem):
                raise TypeError("A user problem factory must return openpinn.PDEProblem.")
        else:
            problem = str(problem_name)
            problem_options = problem_mapping
    else:
        problem = str(problem_value)

    method_value = values.get("method", "baseline")
    if isinstance(method_value, Mapping):
        method_mapping = dict(method_value)
        method_name = str(method_mapping.pop("name", "baseline"))
        method_variant = method_mapping.pop("variant", None)
        method_options = method_mapping.pop("options", {})
        if method_mapping:
            raise ValueError(
                "Compact method configuration supports only name, variant and options; got: "
                + ", ".join(sorted(method_mapping))
            )
    else:
        method_name = str(method_value)
        method_variant = None
        method_options = {}

    experiment = dict(values.get("experiment", {}))
    training = dict(values.get("training", {}))
    configured_steps = int(training.pop("steps", values.get("steps", 1000)))
    selected_steps = configured_steps if steps is None else int(steps)
    output = dict(values.get("output", {}))
    selected_output = output_dir or output.pop("directory", output.pop("dir", None))
    output_root = output.pop("root", "runs")
    run_name = output.pop("run_name", experiment.get("name"))
    overrides = {
        key: values[key]
        for key in ("loss", "optimizer", "evaluation")
        if key in values
    }
    if output:
        overrides["output"] = output
    unknown = sorted(
        set(values).difference(
            {
                "schema_version",
                "problem",
                "method",
                "experiment",
                "model",
                "sampling",
                "training",
                "loss",
                "optimizer",
                "evaluation",
                "output",
                "steps",
            }
        )
    )
    if unknown:
        raise ValueError("Unsupported compact configuration fields: " + ", ".join(unknown))
    return solve(
        problem,
        method=method_name,
        variant=None if method_variant is None else str(method_variant),
        steps=selected_steps,
        seed=int(experiment.get("seed", 0)),
        device=str(experiment.get("device", "cpu")),
        dtype=str(experiment.get("dtype", "float32")),
        method_options=method_options,
        problem_options=problem_options,
        model=values.get("model"),
        sampling=values.get("sampling"),
        training=training,
        overrides=overrides,
        output_dir=selected_output,
        output_root=output_root,
        run_name=run_name,
    )


def result_summary(result: ExperimentResult) -> str:
    """Return a compact JSON summary suitable for CLI output."""

    return json.dumps(
        {
            "output_dir": str(result.output_dir),
            "problem": result.metrics.get("problem"),
            "method": result.metrics.get("method"),
            "variant": result.metrics.get("variant"),
            "relative_l2_error": result.metrics.get("relative_l2_error"),
            "elapsed_seconds": result.compute_ledger.get("elapsed_seconds"),
        },
        ensure_ascii=False,
        indent=2,
    )
