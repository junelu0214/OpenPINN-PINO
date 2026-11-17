from __future__ import annotations

from pathlib import Path
from typing import Any

from tests._assertions import assert_finite_tree


def test_coupled_manufactured_runs_without_a_trainer_special_case(
    coupled_manufactured_smoke_result: Any,
) -> None:
    result = coupled_manufactured_smoke_result

    assert Path(result.output_dir).is_dir()
    assert result.metrics["problem"] == "coupled_elliptic_manufactured_1d"
    assert_finite_tree(result.metrics)
    assert_finite_tree(result.predictions)
    assert result.predictions["prediction"].shape == (33, 2)
    assert result.predictions["reference"].shape == (33, 2)
    assert result.predictions["residual"].shape == (33, 3)
    assert result.predictions["field_u"].shape == (33, 1)
    assert result.predictions["field_v"].shape == (33, 1)
    assert result.predictions["residual_coupling"].shape == (33, 1)
    assert {
        "equation_u",
        "equation_v",
        "coupling",
        "boundary",
        "total",
    } <= set(result.history[-1])
    assert {
        "relative_l2_u",
        "relative_l2_v",
        "residual_equation_u_rmse",
        "residual_equation_v_rmse",
        "residual_coupling_rmse",
        "boundary_u_rmse",
        "boundary_v_rmse",
    } <= set(result.metrics)
