from __future__ import annotations

from pathlib import Path

import openpinn as op

from examples.burgers_time_window import run_comparison


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "reference_runs"


def test_time_window_uses_the_unified_sampler_switch(tmp_path: Path) -> None:
    config = op.load_config(CONFIG_DIR / "burgers_time_window.yaml")
    config["training"] = {"steps": 4, "resample_every": 1, "log_every": 0}
    config["sampling"]["interior"] = 16
    config["sampling"]["boundary"] = 4
    config["sampling"]["initial"] = 4
    config["evaluation"]["points"] = {"space": 17, "time": 4}

    comparison = run_comparison(config, tmp_path / "time_window_comparison")

    assert comparison["status"] == "completed"
    assert comparison["changed_component"] == "sampling.name: uniform -> time_window"
    assert "window_minus_plain_relative_l2_error" in comparison["differences"]
    assert "not patented" in comparison["method_scope"]
    diagnostics = comparison["time_window_diagnostics"]
    assert diagnostics["update_count"] == 3
    assert diagnostics["current_window_index"] == 3
    assert diagnostics["current_window_end"] == 1.0
    assert diagnostics["window_ends"] == [0.25, 0.5, 0.75, 1.0]
    assert (
        tmp_path / "time_window_comparison" / "time_window" / "sampling_diagnostics.json"
    ).is_file()
    assert (tmp_path / "time_window_comparison" / "comparison.svg").stat().st_size > 0
    svg = (tmp_path / "time_window_comparison" / "comparison.svg").read_text(
        encoding="utf-8"
    )
    assert "Reference" in svg
    assert "Maximum absolute error by time slice" in svg
    assert (tmp_path / "time_window_comparison" / "failure_notes.md").is_file()
