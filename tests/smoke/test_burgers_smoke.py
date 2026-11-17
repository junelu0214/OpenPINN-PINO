from __future__ import annotations

from pathlib import Path
from typing import Any

from tests._assertions import assert_finite_tree


def test_dynamic_burgers_runs_end_to_end_on_cpu(burgers_smoke_result: Any) -> None:
    result = burgers_smoke_result

    assert Path(result.output_dir).is_dir()
    assert result.model is not None
    assert all(parameter.device.type == "cpu" for parameter in result.model.parameters())
    assert_finite_tree(result.metrics)
    assert_finite_tree(result.predictions)
    assert result.history
    assert result.predictions["coordinates"].shape == (9 * 4, 2)
    assert result.predictions["reference"].shape == (9 * 4, 1)
    assert "relative_l2_error" in result.metrics
    assert "max_absolute_error" in result.metrics
    assert "boundary_rmse" in result.metrics
    assert "initial_rmse" in result.metrics
    assert (Path(result.output_dir) / "reference_comparison.svg").is_file()
    assert (Path(result.output_dir) / "reference_validation.json").is_file()
    assert (Path(result.output_dir) / "failure_notes.md").is_file()
