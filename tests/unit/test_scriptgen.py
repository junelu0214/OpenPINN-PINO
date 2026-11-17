"""Local script generation for operator and plate-with-hole paths."""

from __future__ import annotations

import ast
import math
import subprocess
import sys
from pathlib import Path

import pytest

from openpinn.scriptgen import (
    ScriptgenError,
    browser_payload,
    default_plate_method,
    default_plate_problem,
    render_operator_script,
    render_plate_with_hole_script,
)


def test_operator_payload_is_model_by_mode_with_registry_extensions() -> None:
    payload = browser_payload()
    assert [item["value"] for item in payload["operator_choices"]] == [
        "deeponet",
        "fno",
    ]
    assert payload["training_modes"] == ["data_only", "physics_only", "hybrid"]
    assert payload["operator_extensions"]["defaults"] == {
        "derivative": "grid",
        "sampler": "uniform",
        "optimizer": "adam",
    }
    assert "adamw" in payload["operator_extensions"]["choices"]["optimizer"]


def _run_script(source: str, tmp_path: Path, name: str) -> subprocess.CompletedProcess[str]:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def test_operator_scripts_keep_api_alias_and_shared_physics_mode() -> None:
    alias = render_operator_script("deep_operator_network", {"steps": 1})
    assert "OPERATOR = 'deep_operator_network'" in alias
    assert "OPERATOR = 'deeponet'" not in alias
    assert "train_operator" in alias
    assert "op.solve" not in alias

    physics_only = render_operator_script(
        "fno", {"training_mode": "physics_only", "steps": 1}
    )
    assert "'training_mode': 'physics_only'" in physics_only
    assert "not rewritten to another mode" in physics_only
    assert "'training_mode': 'hybrid'" not in physics_only
    assert "GridPDEProblem" in physics_only

    coordinate_physics = render_operator_script(
        "deeponet", {"training_mode": "physics_only", "steps": 1}
    )
    assert "query coordinates with autodiff" in coordinate_physics
    assert "GridPDEProblem" not in coordinate_physics

    with pytest.raises(ScriptgenError, match="not selectable"):
        render_operator_script("pino")


def test_illegal_operator_combinations_do_not_fall_back() -> None:
    with pytest.raises(ScriptgenError, match="No fallback operator"):
        render_operator_script("not_an_operator")
    with pytest.raises(ScriptgenError, match="will not switch to build_fno"):
        render_operator_script("fno", {"n_dims": 2})
    with pytest.raises(ScriptgenError, match="No smaller mode count was substituted"):
        render_operator_script("fno", {"grid_size": 4, "n_modes": 4})
    with pytest.raises(ScriptgenError, match="Unknown option"):
        render_operator_script("deeponet", {"not_a_knob": 1})
    with pytest.raises(ScriptgenError, match="training mode was not changed"):
        render_operator_script(
            "fno",
            {"training_mode": "physics_only", "physics_weight": 0.0},
        )
    with pytest.raises(ScriptgenError, match="No fallback was selected"):
        render_operator_script("fno", extensions={"optimizer": "rmsprop"})


@pytest.mark.parametrize(
    ("name", "training_mode"),
    [
        ("deeponet", "data_only"),
        ("fno", "data_only"),
        ("deeponet", "physics_only"),
        ("fno", "physics_only"),
    ],
)
def test_operator_script_smoke(
    name: str, training_mode: str, tmp_path: Path
) -> None:
    overrides: dict[str, object] = {"steps": 2, "training_mode": training_mode}
    if name == "deeponet":
        overrides.update(
            {
                "n_sensors": 8,
                "latent_dim": 4,
                "branch_hidden": [8],
                "trunk_hidden": [8],
                "batch_size": 2,
                "n_query": 4,
            }
        )
    else:
        overrides.update(
            {
                "width": 4,
                "n_layers": 1,
                "n_modes": 2,
                "grid_size": 8,
                "batch_size": 1,
                "project_width": 4,
            }
        )
    source = render_operator_script(name, overrides)
    assert "build_operator_config" in source
    assert "train_operator" in source
    assert "EXTENSIONS =" in source
    completed = _run_script(source, tmp_path, f"{name}_{training_mode}_config.py")
    assert completed.returncode == 0, completed.stderr
    payload = ast.literal_eval(completed.stdout.strip())
    assert payload["operator"] == name
    assert math.isfinite(payload["final_loss"])


def test_plate_script_rejects_illegal_specs() -> None:
    poisson = default_plate_problem()
    poisson["physics"] = "poisson"
    with pytest.raises(ScriptgenError, match="linear_elasticity|physics"):
        render_plate_with_hole_script(poisson, default_plate_method())

    operator_method = default_plate_method()
    operator_method["path"] = "operator"
    operator_method["method"] = "deeponet"
    operator_method.pop("variant")
    with pytest.raises(ScriptgenError, match="codegen.operator_path"):
        render_plate_with_hole_script(default_plate_problem(), operator_method)

    unknown = default_plate_method()
    unknown["method"] = "not_a_method"
    with pytest.raises(ScriptgenError, match="not_a_method"):
        render_plate_with_hole_script(default_plate_problem(), unknown)

    outside = default_plate_problem()
    outside["geometry"]["holes"][0]["radius"] = 5.0
    with pytest.raises(ScriptgenError, match="No fallback spec was generated"):
        render_plate_with_hole_script(outside, default_plate_method())


def test_plate_script_smoke(tmp_path: Path) -> None:
    source = render_plate_with_hole_script()
    assert "from openpinn.agents import validate_specs" in source
    assert "pre_validate=True" in source
    assert "from_specs" in source
    assert "'assumption': 'plane_stress'" in source
    assert "op.solve" in source
    assert "write_diagnostic_run_artifact" in source
    assert "datetime.now(timezone.utc)" in source
    completed = _run_script(source, tmp_path, "plate_with_hole_config.py")
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.strip().splitlines()
    loss = float(lines[-2])
    assert math.isfinite(loss)
    artifact_path = Path(lines[-1].split("RunArtifact:", 1)[1].strip())
    assert artifact_path.is_file()
    artifact = __import__("json").loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["status"] == "budget_exhausted"
    assert artifact["metrics"]["loss_history"]


def test_configurator_sources_wire_local_paths() -> None:
    root = Path(__file__).resolve().parents[2]
    extension = (root / "docs/_ext/openpinn_method_catalog.py").read_text(encoding="utf-8")
    configurator = (root / "docs/_static/method_configurator.js").read_text(encoding="utf-8")
    designer = (root / "docs/_static/problem_designer.js").read_text(encoding="utf-8")
    paths = (root / "docs/_static/configurator_paths.js").read_text(encoding="utf-8")
    assert 'data-role="solution-path"' not in extension
    assert '"name": "plate_with_hole"' not in extension
    assert '"name": "custom_equation_template"' in extension
    assert 'data-role="operator-hooks"' in extension
    assert 'data-role="local-script"' in extension
    assert 'data-role="local-download"' in extension
    assert "configurator_paths.js" in extension
    assert "OpenPINNConfiguratorPaths.renderState" in configurator
    assert "OpenPINNConfiguratorPaths.prepare" in configurator
    assert "root.dataset.scriptName" in configurator
    assert "OpenPINNConfiguratorPaths.prepare" in designer
    assert "train_operator" in paths
    assert "pre_validate=True" in paths
    assert "未选择备用 Operator" in paths
    assert "DESIGNED_PROBLEM = PROBLEM" in paths
