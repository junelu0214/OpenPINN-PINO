"""Run designed-PDE operator scripts emitted by the real configurator DOM."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_dom_generated_operator_scripts_run(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js and jsdom are required for the DOM codegen smoke.")
    root = Path(__file__).resolve().parents[2]
    jsdom = subprocess.run(
        [node, "-e", "require.resolve('jsdom')"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if jsdom.returncode != 0:
        pytest.skip("jsdom is not installed; run npm ci to enable the DOM codegen smoke.")
    generated = subprocess.run(
        [node, "tests/js/operator_configurator.test.cjs", "--json"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert generated.returncode == 0, generated.stdout + generated.stderr
    programs = json.loads(generated.stdout)
    assert [item["name"] for item in programs] == ["deeponet", "fno"]

    for item in programs:
        source = item["code"]
        compile(source, item["name"], "exec")
        script = tmp_path / f"{item['name']}_config.py"
        script.write_text(source, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        payload = ast.literal_eval(completed.stdout.strip().splitlines()[-1])
        assert payload["operator"] == item["name"]
        assert payload["designed_problem"] == f"operator_{item['dimension']}d_problem"
        assert payload["designed_dimension"] == item["dimension"]
        assert payload["trainer_scope"] == "designed_problem"
        assert payload["training_mode"] in {"data_only", "physics_only", "hybrid"}
        assert math.isfinite(float(payload["final_loss"]))
        if item["name"] == "deeponet":
            assert "OBSERVATION_FILE" in source
            assert "evaluate_operator_reference_csv" not in source
            assert "REFERENCE_FILE" not in source
        if item["name"] == "fno":
            assert "OBSERVATION_FILE" not in source
            assert "REFERENCE_FILE" not in source
