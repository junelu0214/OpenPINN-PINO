from __future__ import annotations

from pathlib import Path

import openpinn as op

from examples.burgers_adaptive_sampling import run_comparison


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "reference_runs"


def test_burgers_adaptive_sampling_uses_the_unified_sampler_switch(
    tmp_path: Path,
) -> None:
    config = op.load_config(CONFIG_DIR / "burgers_adaptive_sampling.yaml")
    config["training"] = {"steps": 3, "resample_every": 1, "log_every": 0}
    config["sampling"]["interior"] = 8
    config["sampling"]["boundary"] = 4
    config["sampling"]["initial"] = 4
    config["sampling"]["options"]["candidate_pool"] = 32
    config["evaluation"]["points"] = {"space": 17, "time": 4}

    root = tmp_path / "burgers_adaptive_sampling_comparison"
    comparison = run_comparison(config, root)

    assert comparison["status"] == "completed"
    assert comparison["changed_component"] == (
        "sampling.name: uniform -> residual_adaptive"
    )
    assert comparison["sampling_diagnostics"]["update_count"] == 2
    assert comparison["fairness"]["same_interior_point_count"] is True
    assert "adaptive_minus_plain_relative_l2_error" in comparison["differences"]
    adaptive_config = op.load_config(root / "adaptive_sampling" / "config.yaml")
    assert adaptive_config["sampling"]["name"] == "residual_adaptive"
    assert (root / "adaptive_sampling" / "sampling_diagnostics.json").is_file()
    svg = (root / "comparison.svg").read_text(encoding="utf-8")
    assert "Reference" in svg
    assert "Maximum absolute error by time slice" in svg
    assert (root / "failure_notes.md").is_file()
