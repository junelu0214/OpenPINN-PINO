from __future__ import annotations

from pathlib import Path

import openpinn as op

from examples.poisson_adaptive_sampling import run_comparison


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "reference_runs"


def test_adaptive_sampling_uses_the_unified_sampler_switch(tmp_path: Path) -> None:
    config = op.load_config(CONFIG_DIR / "poisson_adaptive_sampling.yaml")
    config["training"] = {"steps": 3, "resample_every": 1, "log_every": 0}
    config["sampling"]["interior"] = 8
    config["sampling"]["boundary"] = 2
    config["sampling"]["options"]["candidate_pool"] = 32
    config["evaluation"]["points"] = 33

    comparison = run_comparison(config, tmp_path / "adaptive_sampling_comparison")

    assert comparison["status"] == "completed"
    assert comparison["changed_component"] == "sampling.name: uniform -> residual_adaptive"
    assert comparison["sampling_diagnostics"]["update_count"] == 2
    assert comparison["fairness"]["same_interior_point_count"] is True
    adaptive_config = op.load_config(
        tmp_path / "adaptive_sampling_comparison" / "adaptive_sampling" / "config.yaml"
    )
    assert adaptive_config["sampling"]["name"] == "residual_adaptive"
    assert (
        tmp_path
        / "adaptive_sampling_comparison"
        / "adaptive_sampling"
        / "sampling_diagnostics.json"
    ).is_file()
    assert (tmp_path / "adaptive_sampling_comparison" / "comparison.svg").stat().st_size > 0
    assert (tmp_path / "adaptive_sampling_comparison" / "failure_notes.md").is_file()
