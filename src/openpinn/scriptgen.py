"""Render local Python scripts for the documentation configurator.

The docs pages do not train or solve. They write a script that the user runs
with a local Python interpreter. Illegal combinations raise
:class:`ScriptgenError` instead of substituting another operator, physics
model, or method.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from typing import Any

from .method_registry import default_method_registry
from .operator_protocol import default_operator_registry, operator_hook_records


class ScriptgenError(ValueError):
    """The requested script is illegal. No fallback script is produced."""


_SAFE_TEXT = re.compile(r"^[A-Za-z0-9_./-]+$")
_OPERATOR_EXTENSION_DEFAULTS = {
    "derivative": "grid",
    "sampler": "uniform",
    "optimizer": "adam",
}

_TRAINER_LIMITS: dict[str, dict[str, Any]] = {
    "deeponet": {
        "equals": {"n_dims": 1, "n_fields": 1},
        "grid_size_at_least_twice_modes": False,
        "sensor_grid_at_least_twice_modes": True,
        "weight_rules": True,
        "messages": {
            "equals": (
                "train_operator supports the 1D scalar DeepONet smoke "
                "(n_dims=1, n_fields=1). This page will not switch to "
                "build_deeponet or another operator."
            ),
            "grid": (
                "n_sensors must be at least 2 * n_modes for the DeepONet "
                "coordinate-Poisson smoke. "
                "No smaller mode count was substituted."
            ),
            "hybrid_weights": (
                "hybrid mode needs a positive data_weight or physics_weight. "
                "The training mode was not changed."
            ),
            "physics_weights": (
                "physics_only mode needs a positive physics_weight. "
                "The training mode was not changed."
            ),
        },
    },
    "fno": {
        "equals": {"n_dims": 1, "in_channels": 1, "out_channels": 1},
        "grid_size_at_least_twice_modes": True,
        "weight_rules": True,
        "messages": {
            "equals": (
                "train_operator supports the 1D scalar periodic derivative "
                "(n_dims=1, in_channels=1, out_channels=1). This page will not "
                "switch to build_fno or another operator."
            ),
            "grid": (
                "grid_size must be at least 2 * n_modes for the built-in trainer. "
                "No smaller mode count was substituted."
            ),
            "hybrid_weights": (
                "hybrid mode needs a positive data_weight or physics_weight. "
                "The training mode was not changed."
            ),
            "physics_weights": (
                "physics_only mode needs a positive physics_weight. "
                "The training mode was not changed."
            ),
        },
    },
}

_PLATE_LIMITS = {
    "dims": [2],
    "physics": ["linear_elasticity"],
    "outer_types": ["rectangle"],
    "constitutive": ["isotropic_linear"],
    "structural_assumptions": ["plane_stress"],
    "field_kinds": ["displacement"],
    "solver_paths": ["pinn"],
}


def python_literal(value: Any, *, indent: int = 0) -> str:
    """Serialize a JSON-like value as a Python literal."""

    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ScriptgenError(
                "Non-finite numbers cannot be embedded in a generated script."
            )
        text = repr(float(value))
        if "e" not in text and "E" not in text and "." not in text:
            text += ".0"
        return text
    if isinstance(value, str):
        if not _SAFE_TEXT.fullmatch(value):
            raise ScriptgenError(
                "Generated scripts only embed strings made of letters, digits, "
                "underscore, dot, slash, and hyphen. "
                f"Rejected value: {value!r}."
            )
        return repr(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        parts = [python_literal(item, indent=indent) for item in value]
        return "[" + ", ".join(parts) + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        pad = " " * indent
        inner = " " * (indent + 4)
        lines = ["{"]
        for key, item in value.items():
            if not isinstance(key, str):
                raise ScriptgenError("Generated script mappings must use string keys.")
            rendered = python_literal(item, indent=indent + 4)
            lines.append(f"{inner}{key!r}: {rendered},")
        lines.append(f"{pad}}}")
        return "\n".join(lines)
    raise ScriptgenError(
        f"Cannot embed a {type(value).__name__} value in a generated script."
    )


def _header(title: str, filename: str) -> str:
    return (
        f'"""{title}\n'
        "\n"
        f"Save this file as examples/{filename} and run it from the repository root:\n"
        "\n"
        f"    python examples/{filename}\n"
        "\n"
        "Training or solving runs in this local Python process.\n"
        "The documentation page does not train or solve in the browser.\n"
        '"""\n'
        "\n"
        "from __future__ import annotations\n"
    )


def _download_stem(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", str(name)).strip("_")
    if not cleaned:
        raise ScriptgenError(
            "The script name is empty after sanitizing. No default name was substituted."
        )
    return f"{cleaned}_config"


def _copy_default(value: Any) -> Any:
    if isinstance(value, list):
        return list(value)
    return value


def _trainer_limits(canonical: str) -> dict[str, Any]:
    try:
        return _TRAINER_LIMITS[canonical]
    except KeyError as exc:
        raise ScriptgenError(
            f"No built-in trainer is registered for operator {canonical!r}. "
            "No other trainer was selected."
        ) from exc


def _check_trainer_limits(config: Mapping[str, Any]) -> None:
    canonical = str(config["operator"])
    limits = _trainer_limits(canonical)
    messages = limits["messages"]
    mismatched: list[str] = []
    for key, expected in limits["equals"].items():
        actual = config[key]
        if actual != expected:
            mismatched.append(f"{key}={actual!r} (built-in trainer requires {expected!r})")
    if mismatched:
        raise ScriptgenError(messages["equals"] + " " + "; ".join(mismatched))
    if limits.get("grid_size_at_least_twice_modes"):
        grid_size = int(config["grid_size"])
        n_modes = int(config["n_modes"])
        if grid_size < 2 * n_modes:
            raise ScriptgenError(messages["grid"])
    if limits.get("sensor_grid_at_least_twice_modes"):
        if (
            str(config["training_mode"]) != "data_only"
            and int(config["n_sensors"]) < 2 * int(config["n_modes"])
        ):
            raise ScriptgenError(messages["grid"])
    if limits.get("weight_rules"):
        mode = str(config["training_mode"])
        data_weight = float(config["data_weight"])
        physics_weight = float(config["physics_weight"])
        if mode == "hybrid" and data_weight == 0.0 and physics_weight == 0.0:
            raise ScriptgenError(messages["hybrid_weights"])
        if mode == "physics_only" and physics_weight == 0.0:
            raise ScriptgenError(messages["physics_weights"])
        if mode not in ("data_only", "hybrid", "physics_only"):
            raise ScriptgenError(
                "training_mode must be 'data_only', 'physics_only', or 'hybrid', "
                f"got {mode!r}. The training mode was not changed."
            )


def operator_extension_contract() -> dict[str, Any]:
    """Return registry-driven advanced choices for operator scripts."""

    from .experiment import default_component_registry

    registry = default_component_registry()
    choices = {
        kind: list(registry.available(kind))
        for kind in _OPERATOR_EXTENSION_DEFAULTS
    }
    return {
        "choices": choices,
        "defaults": copy.deepcopy(_OPERATOR_EXTENSION_DEFAULTS),
    }


def _resolve_operator_extensions(
    supplied: Mapping[str, Any] | None,
) -> dict[str, str]:
    contract = operator_extension_contract()
    selected = dict(contract["defaults"])
    incoming = dict(supplied or {})
    unknown = sorted(set(incoming).difference(contract["choices"]))
    if unknown:
        raise ScriptgenError(
            "Unknown operator extension kind(s): " + ", ".join(unknown) + "."
        )
    for kind, value in incoming.items():
        name = str(value)
        if name not in contract["choices"][kind]:
            available = ", ".join(contract["choices"][kind])
            raise ScriptgenError(
                f"Operator extension {kind}={name!r} is not registered; "
                f"available: {available}. No fallback was selected."
            )
        selected[kind] = name
    return selected


def render_operator_script(
    name: str,
    options: Mapping[str, Any] | None = None,
    *,
    device: str = "cpu",
    extensions: Mapping[str, Any] | None = None,
) -> str:
    """Return a local ``train_operator`` script, or raise if ``name`` is illegal.

    Omitted options use the registered schema defaults. Unknown options,
    non-selectable hooks, and built-in trainer mismatches raise
    :class:`ScriptgenError`. The selected name is kept, including aliases.
    """

    from .operators.selection import build_operator_config

    if not isinstance(device, str) or not device.strip():
        raise ScriptgenError(
            "device must be a non-empty string. It was not replaced with 'cpu'."
        )
    if not _SAFE_TEXT.fullmatch(device):
        raise ScriptgenError(
            f"device {device!r} contains characters this script emitter rejects. "
            "It was not replaced with 'cpu'."
        )
    registry = default_operator_registry()
    try:
        spec = registry.resolve(name)
    except ValueError as exc:
        raise ScriptgenError(
            f"{exc} No fallback operator was selected."
        ) from exc
    if not spec.implemented or not spec.selectable:
        raise ScriptgenError(
            f"Operator {spec.name!r} is not selectable. "
            "Choose an implemented operator; this page will not substitute another one."
        )
    supplied = dict(options or {})
    unknown = sorted(set(supplied).difference(spec.parameters))
    if unknown:
        available = ", ".join(spec.parameters)
        raise ScriptgenError(
            f"Unknown option(s) for operator {spec.name!r}: {', '.join(unknown)}; "
            f"available options: {available}. Unknown options were not dropped."
        )
    resolved = {
        key: (
            supplied[key]
            if key in supplied and supplied[key] is not None
            else _copy_default(spec.parameters[key]["default"])
        )
        for key in spec.parameters
    }
    try:
        config = build_operator_config(str(name), **resolved)
    except (TypeError, ValueError) as exc:
        raise ScriptgenError(str(exc)) from exc
    _check_trainer_limits(config)
    selected_extensions = _resolve_operator_extensions(extensions)
    stored = {key: config[key] for key in spec.parameters}
    filename = _download_stem(str(name)) + ".py"
    lines = ["import openpinn as op", ""]
    if stored.get("training_mode") in ("physics_only", "hybrid"):
        physics_note = (
            "# DeepONet physics differentiates predictions at query coordinates with autodiff."
            if spec.name == "deeponet"
            else "# FNO physics reuses GridPDEProblem and structured-grid derivatives."
        )
        lines.extend(
            [
                physics_note,
                "# The requested training_mode is not rewritten to another mode.",
                "",
            ]
        )
    lines.extend(
        [
            f"OPERATOR = {python_literal(str(name))}",
            f"OPTIONS = {python_literal(stored)}",
            f"EXTENSIONS = {python_literal(selected_extensions)}",
            "# EXTENSIONS records registry selections for local trainer customization.",
            "",
            "CONFIG = op.build_operator_config(OPERATOR, **OPTIONS)",
            f"RESULT = op.train_operator(CONFIG, device={python_literal(device)})",
            "print(",
            "    {",
            '        "operator": RESULT.operator,',
            '        "steps": RESULT.steps,',
            '        "initial_loss": RESULT.initial_loss,',
            '        "final_loss": RESULT.final_loss,',
            "    }",
            ")",
            "",
        ]
    )
    return _header(f"Local {spec.title} training script.", filename) + "\n" + "\n".join(lines)


def default_plate_problem() -> dict[str, Any]:
    """Canonical 2D linear-elastic plate with one circular hole."""

    return {
        "dim": 2,
        "physics": "linear_elasticity",
        "geometry": {
            "outer": {
                "type": "rectangle",
                "x_min": 0.0,
                "x_max": 2.0,
                "y_min": 0.0,
                "y_max": 2.0,
            },
            "holes": [
                {"type": "circle", "center": [1.0, 1.0], "radius": 0.25},
            ],
        },
        "field": {"kind": "displacement", "components": ["ux", "uy"]},
        "constitutive": {
            "model": "isotropic_linear",
            "assumption": "plane_stress",
            "E": 210.0e9,
            "nu": 0.3,
            "stress_voigt": True,
        },
        "boundaries": [
            {
                "id": "fix_left",
                "region": "left",
                "type": "dirichlet",
                "dofs": ["ux", "uy"],
                "value": [0.0, 0.0],
            },
            {
                "id": "traction_right",
                "region": "right",
                "type": "neumann",
                "dofs": ["ux"],
                "value": 1.0e6,
            },
            {
                "id": "shear_right",
                "region": "right",
                "type": "neumann",
                "dofs": ["uy"],
                "value": 0.0,
            },
            {
                "id": "free_top",
                "region": "top",
                "type": "neumann",
                "dofs": ["ux", "uy"],
                "value": [0.0, 0.0],
            },
            {
                "id": "free_bottom",
                "region": "bottom",
                "type": "neumann",
                "dofs": ["ux", "uy"],
                "value": [0.0, 0.0],
            },
            {
                "id": "free_hole",
                "region": "hole_0",
                "type": "neumann",
                "dofs": ["ux", "uy"],
                "value": [0.0, 0.0],
            },
        ],
        "units": {"length": "m", "force": "N", "stress": "Pa"},
    }


def default_plate_method() -> dict[str, Any]:
    """Short PINN method spec for a local plate-with-hole smoke run."""

    return {
        "method": "baseline",
        "variant": "plain_pinn",
        "path": "pinn",
        "sampling": {"name": "uniform", "interior": 32, "boundary": 16},
        "loss": {
            "formulation": "strong",
            "weights": {"pde": 1.0, "boundary": 10.0},
        },
        "optimizer": {"name": "adam", "lr": 0.001},
        "network": {
            "name": "mlp",
            "hidden_layers": [16, 16],
            "activation": "tanh",
        },
        "differentiator": {"mode": "autodiff"},
        "training": {"steps": 2},
    }


def pinn_method_choices() -> list[dict[str, str]]:
    """PINN method/variant pairs the plate form may select."""

    registry = default_method_registry()
    choices: list[dict[str, str]] = []
    for family in registry.specs():
        for variant in registry.variant_specs(family.name):
            choices.append(
                {
                    "method": family.name,
                    "variant": variant.name,
                    "title": f"{family.title} / {variant.title}",
                }
            )
    return choices


def _require_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ScriptgenError(
            f"{name} must be a positive integer, got {value!r}. "
            "The previous value was not kept."
        )
    return value


def _format_agent_errors(report: Any) -> str:
    details = "; ".join(
        f"[{item.code}] {item.path}: {item.message}" for item in report.errors
    )
    return (
        f"Specification Validation Agent failed with {len(report.errors)} error(s): "
        f"{details} No fallback spec was generated."
    )


def render_plate_with_hole_script(
    problem: Mapping[str, Any] | None = None,
    method: Mapping[str, Any] | None = None,
    *,
    steps: int | None = None,
    seed: int = 0,
    device: str = "cpu",
    output_dir: str = "runs",
) -> str:
    """Return a local plate-with-hole script that validates, then calls M3 codegen.

    The Specification Validation Agent (``openpinn.agents.validate_specs``)
    runs before the script is returned.
    ``ok=False`` raises :class:`ScriptgenError`. The script calls the same
    gate again and then :func:`openpinn.codegen.from_specs`.
    """

    problem_dict = copy.deepcopy(default_plate_problem() if problem is None else dict(problem))
    method_dict = copy.deepcopy(default_plate_method() if method is None else dict(method))
    if steps is not None:
        method_dict.setdefault("training", {})
        if not isinstance(method_dict["training"], dict):
            raise ScriptgenError(
                "method.training must be a mapping. It was not replaced."
            )
        method_dict["training"] = dict(method_dict["training"])
        method_dict["training"]["steps"] = _require_positive_int("steps", steps)
    training = method_dict.get("training")
    if not isinstance(training, dict) or "steps" not in training:
        raise ScriptgenError(
            "method.training.steps is required. A default step count was not inserted."
        )
    effective_steps = _require_positive_int("training.steps", training["steps"])
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ScriptgenError(
            f"seed must be a non-negative integer, got {seed!r}. It was not replaced with 0."
        )
    if not isinstance(device, str) or not _SAFE_TEXT.fullmatch(device):
        raise ScriptgenError(
            f"device {device!r} is not a usable script token. It was not replaced with 'cpu'."
        )
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ScriptgenError(
            "output_dir must be a non-empty string. It was not replaced with 'runs'."
        )
    if not _SAFE_TEXT.fullmatch(output_dir):
        raise ScriptgenError(
            f"output_dir {output_dir!r} contains characters this script emitter rejects. "
            "It was not replaced."
        )

    from .agents.validation import validate_specs

    report = validate_specs(problem_dict, method_dict)
    if not report.ok:
        raise ScriptgenError(_format_agent_errors(report))

    filename = "plate_with_hole_config.py"
    body = "\n".join(
        [
            "from datetime import datetime, timezone",
            "from pathlib import Path",
            "",
            "import openpinn as op",
            "from openpinn.agents import validate_specs",
            "from openpinn.codegen import from_specs",
            "",
            f"PROBLEM = {python_literal(problem_dict)}",
            f"METHOD = {python_literal(method_dict)}",
            f"STEPS = {python_literal(effective_steps)}",
            f"SEED = {python_literal(seed)}",
            f"DEVICE = {python_literal(device)}",
            f"OUTPUT_ROOT = Path({python_literal(output_dir)})",
            'OUTPUT_DIR = OUTPUT_ROOT / ("plate_with_hole_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))',
            "",
            "REPORT = validate_specs(PROBLEM, METHOD)",
            "if not REPORT.ok:",
            "    details = '; '.join(",
            '        f"[{item.code}] {item.path}: {item.message}" for item in REPORT.errors',
            "    )",
            "    raise SystemExit(",
            '        "Specification Validation Agent failed with "',
            '        f"{len(REPORT.errors)} error(s): {details} "',
            '        "No fallback spec was generated."',
            "    )",
            "",
            "PROBLEM_OBJ, KWARGS = from_specs(",
            "    PROBLEM,",
            "    METHOD,",
            "    steps=STEPS,",
            "    device=DEVICE,",
            "    pre_validate=True,",
            ")",
            "try:",
            "    RESULT = op.solve(",
            "        PROBLEM_OBJ,",
            "        **KWARGS,",
            '        run_name="plate_with_hole",',
            "        output_dir=OUTPUT_DIR,",
            "        seed=SEED,",
            "        plot=False,",
            "    )",
            "except BaseException as ERROR:",
            "    FAILED_ARTIFACT_PATH = op.write_diagnostic_run_artifact(",
            '        OUTPUT_DIR, status="interrupted" if isinstance(ERROR, KeyboardInterrupt) else "failed",',
            '        exported_pair_snapshot={"problem": PROBLEM, "method": METHOD},',
            '        script_path=__file__, code_origin="exported", source="configurator",',
            '        generator_id="openpinn.configurator.v1", budgets={"max_steps": STEPS, "max_wall_seconds": None},',
            "        steps_completed=0, device=DEVICE, failure=ERROR,",
            "    )",
            '    print("RunArtifact:", FAILED_ARTIFACT_PATH)',
            "    raise",
            "ARTIFACT_PATH = op.write_diagnostic_run_artifact(",
            "    RESULT.output_dir,",
            "    result=RESULT,",
            '    status="budget_exhausted",',
            '    exported_pair_snapshot={"problem": PROBLEM, "method": METHOD},',
            "    script_path=__file__,",
            '    code_origin="exported",',
            '    source="configurator",',
            '    generator_id="openpinn.configurator.v1",',
            '    budgets={"max_steps": STEPS, "max_wall_seconds": None},',
            "    steps_completed=len(RESULT.history),",
            "    device=DEVICE,",
            ")",
            'print(RESULT.metrics.get("final_total_loss"))',
            'print("RunArtifact:", ARTIFACT_PATH)',
            "",
        ]
    )
    return _header(
        "Local 2D linear-elastic plate-with-hole script.",
        filename,
    ) + "\n" + body


def annotate_operator_hooks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach built-in trainer limits used by the browser script form."""

    annotated: list[dict[str, Any]] = []
    for record in records:
        item = copy.deepcopy(record)
        item["trainer_limits"] = copy.deepcopy(_TRAINER_LIMITS.get(str(item.get("name", ""))))
        annotated.append(item)
    return annotated


def operator_choices(hooks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonical selectable model names for the configurator dropdown."""

    choices: list[dict[str, Any]] = []
    for hook in hooks:
        selectable = bool(hook.get("selectable") and hook.get("implemented"))
        if selectable:
            choices.append(
                {
                    "value": hook["name"],
                    "label": hook.get("title") or hook["name"],
                    "canonical": hook["name"],
                    "selectable": True,
                }
            )
    return choices


def plate_script_contract() -> dict[str, Any]:
    """Defaults and limits the plate form shares with the Validation Agent."""

    return {
        "problem": default_plate_problem(),
        "method": default_plate_method(),
        "limits": copy.deepcopy(_PLATE_LIMITS),
        "rejected_physics": ["poisson"],
        "rejected_solver_paths": ["operator"],
        "rejected_methods": ["not_a_method"],
        "pinn_methods": pinn_method_choices(),
        "download_name": "plate_with_hole_config",
        "operator_path_error": (
            "[codegen.operator_path] method.path: M3 plate-with-hole codegen uses the PINN "
            "solve path. Operator methods are trained with train_operator, not solve. "
            "Switch path to 'pinn' or generate an operator script instead. "
            "No PINN method was substituted."
        ),
    }


def browser_payload() -> dict[str, Any]:
    """JSON payload shared by the Sphinx configurator and the problem designer."""

    hooks = annotate_operator_hooks([dict(record) for record in operator_hook_records()])
    return {
        "operator_hooks": hooks,
        "operator_choices": operator_choices(hooks),
        "training_modes": ["data_only", "physics_only", "hybrid"],
        "operator_extensions": operator_extension_contract(),
        "plate_contract": plate_script_contract(),
    }
