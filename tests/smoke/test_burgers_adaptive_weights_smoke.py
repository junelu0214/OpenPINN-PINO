from __future__ import annotations

from pathlib import Path

import openpinn as op

from examples.burgers_adaptive_weights import run_comparison


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "reference_runs"


def test_adaptive_weights_use_the_unified_loss_weighting_switch(tmp_path: Path) -> None:
    config = op.load_config(CONFIG_DIR / "burgers_adaptive_weights.yaml")
    config["training"] = {"steps": 2, "log_every": 0}
    config["sampling"]["interior"] = 16
    config["sampling"]["boundary"] = 4
    config["sampling"]["initial"] = 4
    config["evaluation"]["points"] = {"space": 17, "time": 4}

    comparison = run_comparison(config, tmp_path / "adaptive_weights_comparison")

    assert comparison["status"] == "completed"
    assert comparison["changed_component"] == "loss.weighting: fixed -> self_adaptive"
    assert "adaptive_minus_plain_relative_l2_error" in comparison["differences"]
    assert set(comparison["learned_weights"]) == {"pde", "boundary", "initial"}
    assert all(value > 1.0 for value in comparison["learned_weights"].values())
    adaptive_ledger = op.load_config(
        tmp_path / "adaptive_weights_comparison" / "adaptive_weights" / "compute_ledger.json"
    )
    assert adaptive_ledger["adaptive_weight_parameter_count"] == 3
    assert (tmp_path / "adaptive_weights_comparison" / "comparison.svg").stat().st_size > 0
    svg = (tmp_path / "adaptive_weights_comparison" / "comparison.svg").read_text(
        encoding="utf-8"
    )
    assert "Reference" in svg
    assert "Maximum absolute error by time slice" in svg
    assert (tmp_path / "adaptive_weights_comparison" / "failure_notes.md").is_file()
