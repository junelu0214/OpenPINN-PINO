from __future__ import annotations

from pathlib import Path

import openpinn as op

from examples.burgers_transfer import run_comparison


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "reference_runs"


def test_transfer_uses_model_only_initial_checkpoint(tmp_path: Path) -> None:
    config = op.load_config(CONFIG_DIR / "burgers_transfer.yaml")
    config["training"] = {"steps": 2, "log_every": 0, "initial_checkpoint": None}
    config["sampling"]["interior"] = 16
    config["sampling"]["boundary"] = 4
    config["sampling"]["initial"] = 4
    config["evaluation"]["points"] = {"space": 17, "time": 4}

    comparison = run_comparison(config, tmp_path / "transfer_comparison")

    assert comparison["status"] == "completed"
    assert comparison["changed_component"].startswith("target initialization")
    source_checkpoint = tmp_path / "transfer_comparison" / "source_pretraining" / "checkpoint.pt"
    full_transfer_ledger = op.load_config(
        tmp_path / "transfer_comparison" / "full_transfer_target" / "compute_ledger.json"
    )
    partial_transfer_ledger = op.load_config(
        tmp_path / "transfer_comparison" / "partial_transfer_target" / "compute_ledger.json"
    )
    scratch_ledger = op.load_config(
        tmp_path / "transfer_comparison" / "scratch_target" / "compute_ledger.json"
    )
    assert source_checkpoint.is_file()
    assert full_transfer_ledger["warm_started"] is True
    assert full_transfer_ledger["initial_checkpoint"] == str(source_checkpoint.resolve())
    assert len(full_transfer_ledger["initial_checkpoint_sha256"]) == 64
    assert partial_transfer_ledger["warm_started"] is True
    assert partial_transfer_ledger["trainable_parameter_count"] < full_transfer_ledger[
        "trainable_parameter_count"
    ]
    assert partial_transfer_ledger["parameter_count"] == full_transfer_ledger[
        "parameter_count"
    ]
    assert scratch_ledger["warm_started"] is False
    assert (tmp_path / "transfer_comparison" / "comparison.svg").stat().st_size > 0
    assert (tmp_path / "transfer_comparison" / "partial_transfer_comparison.svg").stat().st_size > 0
    comparison_svg = (tmp_path / "transfer_comparison" / "comparison.svg").read_text(
        encoding="utf-8"
    )
    assert "Reference" in comparison_svg and "Maximum absolute error" in comparison_svg
    assert "full_transfer_minus_scratch_relative_l2_error" in comparison["differences"]
    assert "partial_transfer_minus_scratch_relative_l2_error" in comparison["differences"]
    assert (tmp_path / "transfer_comparison" / "failure_notes.md").is_file()
