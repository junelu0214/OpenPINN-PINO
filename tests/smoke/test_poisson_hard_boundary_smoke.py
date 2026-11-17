from __future__ import annotations

from pathlib import Path

import openpinn as op

from examples.poisson_hard_boundary import run_comparison


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "reference_runs"


def test_hard_boundary_uses_the_unified_output_transform_switch(tmp_path: Path) -> None:
    config = op.load_config(CONFIG_DIR / "poisson_hard_boundary.yaml")
    config["training"] = {"steps": 2, "log_every": 0}
    config["sampling"]["interior"] = 16
    config["sampling"]["boundary"] = 2
    config["evaluation"]["points"] = 33

    comparison = run_comparison(config, tmp_path / "hard_boundary_comparison")

    assert comparison["status"] == "completed"
    assert comparison["changed_component"] == "model.output_transform: soft -> hard"
    assert comparison["fairness"]["same_parameter_count"] is True
    hard_metrics = op.load_config(
        tmp_path / "hard_boundary_comparison" / "hard_boundary" / "metrics.json"
    )
    assert hard_metrics["boundary_rmse"] <= 1e-7
    hard_config = op.load_config(
        tmp_path / "hard_boundary_comparison" / "hard_boundary" / "config.yaml"
    )
    assert hard_config["model"]["output_transform"]["name"] == "hard"
    assert (tmp_path / "hard_boundary_comparison" / "comparison.svg").stat().st_size > 0
    assert (tmp_path / "hard_boundary_comparison" / "failure_notes.md").is_file()
