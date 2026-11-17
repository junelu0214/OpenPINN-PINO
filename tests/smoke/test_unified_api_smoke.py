from __future__ import annotations

from pathlib import Path

import openpinn as op
import pytest


def _assert_unified_output(output_dir: Path) -> None:
    required = {
        "config.yaml",
        "status.json",
        "metrics.json",
        "history.csv",
        "predictions.npz",
        "compute_ledger.json",
        "checkpoint.pt",
        "result.svg",
        "loss.svg",
        "failure_notes.md",
        "manifest.json",
    }
    assert required.issubset(path.name for path in output_dir.iterdir())


def test_short_problem_and_method_call_runs(tmp_path: Path) -> None:
    result = op.solve(
        "poisson",
        method="fourier",
        steps=2,
        seed=7,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 12, "boundary": 2},
        output_dir=tmp_path / "poisson_fourier",
    )
    assert result.metrics["method"] == "fourier"
    assert result.metrics["variant"] == "random_fourier"
    _assert_unified_output(result.output_dir)


def test_custom_problem_factory_yaml_runs(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    result = op.solve_config(
        project_root / "configs" / "experiments" / "custom_equation.yaml",
        output_dir=tmp_path / "custom_equation",
        steps=2,
    )
    assert result.metrics["problem"] == "my_poisson"
    assert result.metrics["method"] == "adaptive_sampling"
    assert result.metrics["variant"] == "residual_adaptive"
    _assert_unified_output(result.output_dir)


def test_tunnel_model1_runs_from_short_name_and_saves_2d_figure(tmp_path: Path) -> None:
    result = op.solve(
        "tunnel_model1",
        method="baseline",
        steps=1,
        seed=5,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 8, "boundary": 8, "initial": 8},
        overrides={
            "evaluation": {
                "points": {"x": 7, "y": 7, "time": 2},
                "time_slices": [0.0, 1.0],
                "boundary_points": 8,
                "initial_points": 8,
            }
        },
        output_dir=tmp_path / "tunnel_model1",
    )

    assert result.metrics["problem"] == "tunnel_consolidation_model1"
    assert result.predictions["coordinates"].shape[1] == 3
    assert "relative_l2_error" not in result.metrics
    assert result.metrics["reference_solution_available"] is False
    assert result.metrics["initial_pressure_kpa"] == 20.0
    assert "physical_range_violation_fraction" in result.metrics
    assert result.predictions["physical_coordinates"].shape[1] == 3
    assert result.predictions["pressure_kpa"].shape[1] == 1
    _assert_unified_output(result.output_dir)
    result_svg = (result.output_dir / "result.svg").read_text(encoding="utf-8")
    diagnostics_svg = (result.output_dir / "diagnostics.svg").read_text(
        encoding="utf-8"
    )
    assert "Twin-tunnel consolidation" in result_svg
    assert "Pressure (kPa)" in result_svg
    assert "data:image/png;base64" in result_svg
    assert "bicubic interpolation" in result_svg
    assert result_svg.count("<ellipse") == 4
    assert "PDE residual" in diagnostics_svg
    assert "data:image/png;base64" in diagnostics_svg
    assert "L2/RMSE/R2" in diagnostics_svg


@pytest.mark.parametrize(
    ("problem", "method", "steps"),
    (
        ("poisson", "modified_mlp", 1),
        ("burgers", "adaptive_activation", 1),
        ("poisson", "hard_boundary", 1),
        ("burgers", "adaptive_weights", 1),
        ("burgers", "adaptive_sampling", 1),
        ("poisson", "adam_lbfgs", 2),
        ("burgers", "time_window", 1),
        ("heterogeneous_poisson", "domain_decomposition", 1),
    ),
)
def test_every_non_transfer_method_runs_through_short_api(
    tmp_path: Path,
    problem: str,
    method: str,
    steps: int,
) -> None:
    dynamic = problem == "burgers"
    result = op.solve(
        problem,
        method=method,
        steps=steps,
        seed=3,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 8, "boundary": 2, "initial": 2},
        overrides={
            "evaluation": (
                {"points": {"space": 5, "time": 4}}
                if dynamic
                else {"points": 9}
            )
        },
        output_dir=tmp_path / method,
    )
    assert result.metrics["method"] == method
    _assert_unified_output(result.output_dir)


def test_transfer_method_runs_with_a_source_checkpoint(tmp_path: Path) -> None:
    shared_model = {"hidden_layers": [8, 8]}
    shared_sampling = {"interior": 8, "boundary": 2, "initial": 2}
    small_evaluation = {"evaluation": {"points": {"space": 5, "time": 4}}}
    source = op.solve(
        "burgers",
        method="baseline",
        steps=1,
        model=shared_model,
        sampling=shared_sampling,
        overrides=small_evaluation,
        output_dir=tmp_path / "source",
    )
    target = op.solve(
        "burgers",
        method="transfer",
        method_options={"checkpoint": source.output_dir / "checkpoint.pt"},
        steps=1,
        model=shared_model,
        sampling=shared_sampling,
        overrides=small_evaluation,
        output_dir=tmp_path / "target",
    )
    assert target.metrics["method"] == "transfer"
    assert target.compute_ledger["warm_started"] is True
    _assert_unified_output(target.output_dir)


def test_long_method_stack_uses_a_filesystem_safe_default_run_name(
    tmp_path: Path,
) -> None:
    shared_model = {"hidden_layers": [8, 8]}
    shared_sampling = {"interior": 8, "boundary": 4}
    small_evaluation = {"evaluation": {"points": 9}}
    source = op.solve(
        "poisson",
        methods=[
            op.method("modified_mlp"),
            op.method("adaptive_activation"),
            op.method("fourier"),
            op.method("hard_boundary"),
        ],
        steps=1,
        model=shared_model,
        sampling=shared_sampling,
        overrides=small_evaluation,
        output_root=tmp_path / "source",
    )
    target = op.solve(
        "poisson",
        methods=[
            op.method("modified_mlp"),
            op.method("adaptive_activation"),
            op.method("fourier"),
            op.method("hard_boundary"),
            op.method("transfer", checkpoint=source.output_dir / "checkpoint.pt"),
            op.method("adaptive_weights", "relobralo"),
            op.method(
                "adaptive_sampling",
                "residual_adaptive",
                candidate_pool=16,
                resample_every=1,
            ),
            op.method(
                "adam_lbfgs",
                "adam_then_lbfgs",
                lbfgs_max_steps=1,
                lbfgs_max_iter=1,
            ),
        ],
        steps=2,
        model=shared_model,
        sampling=shared_sampling,
        overrides=small_evaluation,
        output_root=tmp_path / "target",
    )

    assert len(target.output_dir.name.encode("utf-8")) <= 180
    assert target.metrics["method"] == (
        "modified_mlp+adaptive_activation+fourier+hard_boundary+transfer+"
        "adaptive_weights+adaptive_sampling+adam_lbfgs"
    )
    _assert_unified_output(target.output_dir)


def test_incompatible_method_saves_failure_reason(tmp_path: Path) -> None:
    output_dir = tmp_path / "incompatible"
    with pytest.raises(ValueError, match="not compatible"):
        op.solve(
            "poisson",
            method="time_window",
            steps=1,
            output_dir=output_dir,
        )
    assert (output_dir / "failure.json").is_file()
    assert "not compatible" in (output_dir / "failure_notes.md").read_text(
        encoding="utf-8"
    )
