from __future__ import annotations

from pathlib import Path
from typing import Any

from tests._assertions import assert_finite_tree


def test_heterogeneous_poisson_runs_end_to_end_on_cpu(
    heterogeneous_poisson_smoke_result: Any,
) -> None:
    result = heterogeneous_poisson_smoke_result

    assert Path(result.output_dir).is_dir()
    assert result.metrics["problem"] == "piecewise_poisson_1d"
    assert all(parameter.device.type == "cpu" for parameter in result.model.parameters())
    assert_finite_tree(result.metrics)
    assert_finite_tree(result.predictions)
    assert result.history
    assert "interface_solution_jump" in result.metrics
    assert "interface_flux_jump" in result.metrics
