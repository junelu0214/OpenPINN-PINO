from __future__ import annotations

from pathlib import Path
from typing import Any

from tests._assertions import assert_finite_tree


def test_static_poisson_runs_end_to_end_on_cpu(poisson_smoke_result: Any) -> None:
    result = poisson_smoke_result

    assert Path(result.output_dir).is_dir()
    assert result.model is not None
    assert all(parameter.device.type == "cpu" for parameter in result.model.parameters())
    assert_finite_tree(result.metrics)
    assert_finite_tree(result.predictions)
    assert result.history
    assert (Path(result.output_dir) / "reference_comparison.svg").is_file()
    assert (Path(result.output_dir) / "failure_notes.md").is_file()
