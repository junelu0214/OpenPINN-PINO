from __future__ import annotations

from pathlib import Path

import openpinn as op

from examples.burgers_adaptive_activation import run_comparison


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "reference_runs"


def test_adaptive_activation_uses_the_unified_activation_switch(tmp_path: Path) -> None:
    config = op.load_config(CONFIG_DIR / "burgers_adaptive_activation.yaml")
    config["training"] = {"steps": 2, "log_every": 0}
    config["sampling"]["interior"] = 16
    config["sampling"]["boundary"] = 4
    config["sampling"]["initial"] = 4
    config["evaluation"]["points"] = {"space": 17, "time": 4}

    comparison = run_comparison(config, tmp_path / "adaptive_activation_comparison")

    assert comparison["status"] == "completed"
    assert comparison["changed_component"] == "model.activation: tanh -> adaptive_tanh"
    assert comparison["fairness"]["same_steps"] is True
    assert "adaptive_minus_plain_relative_l2_error" in comparison["differences"]
    adaptive_config = op.load_config(
        tmp_path / "adaptive_activation_comparison" / "adaptive_tanh" / "config.yaml"
    )
    assert adaptive_config["model"]["activation"] == "adaptive_tanh"
    assert (tmp_path / "adaptive_activation_comparison" / "comparison.svg").stat().st_size > 0
    svg = (tmp_path / "adaptive_activation_comparison" / "comparison.svg").read_text(
        encoding="utf-8"
    )
    assert "Reference" in svg
    assert "Maximum absolute error by time slice" in svg
    assert (tmp_path / "adaptive_activation_comparison" / "failure_notes.md").is_file()
