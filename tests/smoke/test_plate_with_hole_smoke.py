"""Smoke test: plate-with-hole codegen + short training run."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest
import torch

import openpinn as op
from openpinn.codegen import from_specs


SPEC_DIR = Path(__file__).resolve().parents[2] / "examples" / "specs"
PROBLEM_YAML = str(SPEC_DIR / "plate_with_hole_problem.yaml")
METHOD_YAML = str(SPEC_DIR / "pinn_method.yaml")


class TestPlateWithHoleSmoke:

    def test_codegen_and_short_train(self, tmp_path):
        """Valid specs -> codegen -> 5-step train -> finite outputs."""
        problem, kwargs = from_specs(PROBLEM_YAML, METHOD_YAML, steps=5)
        kwargs["sampling"] = {"name": "uniform", "interior": 32, "boundary": 16}
        kwargs["model"] = {
            "name": "mlp",
            "hidden_layers": [16, 16],
            "activation": "tanh",
        }

        result = op.solve(
            problem,
            **kwargs,
            run_name="plate_smoke_test",
            output_dir=tmp_path / "plate_smoke",
            plot=False,
        )

        assert result.output_dir.exists()
        assert math.isfinite(result.metrics["final_total_loss"])
        assert result.metrics.get("pde_residual_rmse") is not None
        assert "prediction" in result.predictions
        assert result.predictions["prediction"].shape[1] == 2

    def test_kirsch_comparison_path_exists(self, tmp_path):
        """Kirsch comparison entrypoint runs (numeric gate deliberately loose)."""
        from openpinn.kirsch import compare_displacement_l2, kirsch_scf
        import numpy as np

        problem, kwargs = from_specs(PROBLEM_YAML, METHOD_YAML, steps=3)
        kwargs["sampling"] = {"name": "uniform", "interior": 32, "boundary": 16}
        kwargs["model"] = {
            "name": "mlp",
            "hidden_layers": [16, 16],
            "activation": "tanh",
        }

        result = op.solve(
            problem,
            **kwargs,
            run_name="kirsch_smoke",
            output_dir=tmp_path / "kirsch_smoke",
            plot=False,
        )

        coords = result.predictions["coordinates"]
        pred = result.predictions["prediction"]
        metrics = compare_displacement_l2(
            pred[:, 0], pred[:, 1], coords,
            sigma_inf=1e6, a=0.25, E=210e9, nu=0.3,
            center=(1.0, 1.0),
        )

        assert metrics["n_comparison_points"] > 0
        assert math.isfinite(metrics["relative_l2_total"])
        assert kirsch_scf() == 3.0
