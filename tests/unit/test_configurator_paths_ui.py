"""Browser checks for operator scripts and retained plate codegen."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_configurator_paths_dom_and_renderer_match() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the configurator path UI test.")
    root = Path(__file__).resolve().parents[2]
    probe = subprocess.run(
        [node, "-e", "require('jsdom')"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("jsdom is required for the configurator path UI test.")
    env = os.environ.copy()
    env["OPENPINN_PYTHON"] = sys.executable
    completed = subprocess.run(
        [node, str(root / "tests/js/configurator_paths.test.cjs")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CONFIGURATOR_PATHS_OK" in completed.stdout
    assert "PYTHON_COMPARE_OK" in completed.stdout
