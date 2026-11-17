from __future__ import annotations

from pathlib import Path

import openpinn as op


def test_residual_adaptive_variant_trains_through_family_selector(
    tmp_path: Path,
) -> None:
    result = op.solve(
        "burgers",
        method="adaptive_sampling",
        variant="residual_adaptive",
        steps=2,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 8, "boundary": 2, "initial": 2},
        method_options={"candidate_pool": 16, "resample_every": 1},
        overrides={"evaluation": {"points": {"space": 5, "time": 4}}},
        output_dir=tmp_path / "residual_adaptive",
    )
    assert result.metrics["method"] == "adaptive_sampling"
    assert result.metrics["variant"] == "residual_adaptive"
    assert result.compute_ledger["variant"] == "residual_adaptive"
    assert (result.output_dir / "config.yaml").is_file()
    assert (result.output_dir / "result.svg").is_file()
