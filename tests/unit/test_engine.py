from __future__ import annotations

from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import time

import pytest
import torch

import openpinn as op


def fixture_problem(kind: str = "quadratic") -> op.PDEProblem:
    domain = op.Interval(-1.0, 1.0)

    def residual(points: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        if kind == "quadratic":
            return prediction - points.square()
        if kind == "shifted":
            return prediction + 0.25 * points
        raise AssertionError("unknown test fixture")

    return op.PDEProblem(
        domain,
        residual,
        [op.DirichletBC(0.0, name="boundary")],
        name=f"fixture_{kind}",
    )


def engine_parts(problem: op.PDEProblem) -> tuple[op.MLP, op.UniformSampler]:
    torch.manual_seed(7)
    model = op.MLP(problem.domain.dim, problem.output_dim, hidden_layers=(8,))
    sampler = op.UniformSampler(
        problem, n_interior=8, n_boundary=4, n_initial=0, seed=11
    )
    return model, sampler


def run_fixture(tmp_path: Path, *, kind: str = "quadratic", **kwargs: object):
    problem = fixture_problem(kind)
    model, sampler = engine_parts(problem)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    result = op.run_pinn(
        problem,
        model=model,
        sampler=sampler,
        output_dir=tmp_path,
        max_steps=int(kwargs.pop("max_steps", 3)),
        max_wall_seconds=float(kwargs.pop("max_wall_seconds", 10.0)),
        **kwargs,
    )
    return problem, model, sampler, before, result


def test_real_cpu_steps_write_finite_history_and_budget_status(tmp_path: Path) -> None:
    _, model, _, before, result = run_fixture(tmp_path)

    assert result.status == "budget_exhausted"
    assert result.stop_reason == "max_steps"
    assert result.steps_completed == result.max_steps == 3
    assert result.device == "cpu"
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.parameters(), strict=True)
    )
    rows = list(csv.DictReader((tmp_path / result.history_path).open(encoding="utf-8-sig")))
    assert len(rows) == 3
    assert all(math.isfinite(float(row["total"])) for row in rows)
    persisted = json.loads((tmp_path / "engine_run.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "budget_exhausted"
    assert persisted["max_wall_seconds"] == 10.0


def test_wall_clock_budget_stops_without_completed_status(tmp_path: Path) -> None:
    def slow_step(_record: object) -> None:
        time.sleep(0.02)

    *_, result = run_fixture(
        tmp_path,
        max_steps=20,
        max_wall_seconds=0.005,
        callback=slow_step,
    )

    assert result.status == "budget_exhausted"
    assert result.stop_reason == "max_wall_seconds"
    assert result.steps_completed < 20


def test_keyboard_interrupt_keeps_history_and_loadable_checkpoint(tmp_path: Path) -> None:
    def interrupt(record: dict[str, float]) -> None:
        if record["step"] >= 2:
            raise KeyboardInterrupt

    problem, _, sampler, _, result = run_fixture(
        tmp_path, max_steps=5, callback=interrupt
    )

    assert result.status == "interrupted"
    assert result.steps_completed == 2
    assert result.checkpoint_path is not None
    checkpoint = tmp_path / result.checkpoint_path
    assert checkpoint.is_file()
    replacement = op.MLP(1, 1, hidden_layers=(8,))
    restored = op.Trainer(replacement, problem, sampler, optimizer="adam", lr=1e-3)
    restored.load(checkpoint, map_location="cpu")
    assert restored.steps_completed == 2
    assert len(restored.history) == 2


def test_cooperative_cancel_maps_to_interrupted(tmp_path: Path) -> None:
    calls = 0

    def cancel() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 3

    *_, result = run_fixture(tmp_path, max_steps=6, cancel_requested=cancel)

    assert result.status == "interrupted"
    assert result.stop_reason == "cancel_requested"
    assert result.steps_completed == 2


@pytest.mark.parametrize("failure_kind", ["nan", "exception"])
def test_nonfinite_or_exception_is_persisted_as_failed(
    tmp_path: Path, failure_kind: str
) -> None:
    def residual(points: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        if failure_kind == "exception":
            raise RuntimeError("forced fixture failure")
        return prediction * points.new_tensor(float("nan"))

    problem = op.PDEProblem(op.Interval(0.0, 1.0), residual)
    model, sampler = engine_parts(problem)
    result = op.run_pinn(
        problem,
        model=model,
        sampler=sampler,
        output_dir=tmp_path,
        max_steps=2,
        max_wall_seconds=10.0,
    )

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure["type"] in {"NonFiniteLoss", "RuntimeError"}
    assert len(result.failure["traceback"]) <= 4000
    persisted = json.loads((tmp_path / "engine_run.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["failure"]["type"] == result.failure["type"]


def test_two_different_residual_hooks_use_the_same_entry(tmp_path: Path) -> None:
    first = run_fixture(tmp_path / "one", kind="quadratic", max_steps=1)[-1]
    second = run_fixture(tmp_path / "two", kind="shifted", max_steps=1)[-1]

    assert first.status == second.status == "budget_exhausted"
    assert first.steps_completed == second.steps_completed == 1


def test_engine_accepts_formulation_protocol_directly(tmp_path: Path) -> None:
    problem = fixture_problem()
    model, sampler = engine_parts(problem)

    result = op.run_pinn(
        op.StrongFormulation(problem),
        model=model,
        sampler=sampler,
        output_dir=tmp_path,
        max_steps=1,
        max_wall_seconds=10.0,
    )

    assert result.status == "budget_exhausted"
    assert result.steps_completed == 1


def test_stop_condition_controls_completed_and_not_converged(tmp_path: Path) -> None:
    completed = run_fixture(
        tmp_path / "met",
        max_steps=5,
        stop_condition=lambda record: record["step"] >= 2,
    )[-1]
    missed = run_fixture(
        tmp_path / "missed",
        max_steps=2,
        stop_condition=lambda _record: False,
    )[-1]

    assert completed.status == "completed"
    assert completed.stop_reason == "stop_condition_met"
    assert completed.steps_completed == 2
    assert missed.status == "not_converged"
    assert missed.stop_reason == "stop_condition_not_met_before_max_steps"


def compiled_snapshot() -> dict[str, object]:
    return {
        "coordinates": [
            {"name": "x", "role": "space", "unit": "m", "bounds": {"min": 0, "max": 1}},
            {"name": "t", "role": "time", "unit": "s", "bounds": {"min": 0, "max": 1}},
        ],
        "fields": [{"name": "u", "components": ["u"], "unit": "K"}],
        "parameters": [{"name": "alpha", "value": 0.01, "unit": "m^2/s"}],
        "equations": [{"id": "compiled_fixture", "expression": "dt(u)-alpha*dx(dx(u))=0"}],
        "boundaries": [
            {"location": "x=0", "type": "dirichlet", "value": 0},
            {"location": "x=1", "type": "neumann", "value": 0},
        ],
        "initial_conditions": [{"field": "u", "expression": "sin(pi*x/2)"}],
        "trainable_contract": {"mapping_complete": True},
    }


def test_stage_b_compiled_residual_attaches_for_one_finite_engine_step(
    tmp_path: Path,
) -> None:
    compiled = op.compile_problem_expressions(deepcopy(compiled_snapshot()))
    domain = op.SpaceTimeDomain(op.Interval(0.0, 1.0), 0.0, 1.0)
    left = lambda points: torch.isclose(points[:, 0], points.new_tensor(0.0))
    right = lambda points: torch.isclose(points[:, 0], points.new_tensor(1.0))
    conditions = [
        op.DirichletBC(0.0, boundary=left, name="left"),
        op.NeumannBC(0.0, normal=[1.0, 0.0], boundary=right, name="right"),
        op.InitialCondition(
            lambda points: torch.sin(math.pi * points[:, :1] / 2), name="initial"
        ),
    ]
    problem = op.problem_from_compiled_expressions(compiled, domain, conditions)
    torch.manual_seed(17)
    model = op.MLP(2, 1, hidden_layers=(8,))
    sampler = op.UniformSampler(
        problem, n_interior=6, n_boundary=4, n_initial=4, seed=19
    )

    result = op.run_pinn(
        problem,
        model=model,
        sampler=sampler,
        output_dir=tmp_path,
        max_steps=1,
        max_wall_seconds=10.0,
    )

    assert result.status == "budget_exhausted"
    assert result.steps_completed == 1
    row = list(csv.DictReader((tmp_path / "history.csv").open(encoding="utf-8-sig")))[0]
    assert math.isfinite(float(row["total"]))


def test_codebuilder_keeps_honest_skeleton_and_registered_engine_path() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "docs/_static/codebuilder_agent.js").read_text(encoding="utf-8")

    assert 'var GENERATOR_ID = "openpinn.generic-training-skeleton.v1"' in source
    assert 'var RUNNABLE_GENERATOR_ID = "openpinn.shared-pinn-engine.v1"' in source
    assert "backend_runnable: runnable" in source
    assert "NotImplementedError" in source
    assert "op.run_registered_pinn" in source


@pytest.mark.parametrize(
    ("max_steps", "max_wall_seconds"),
    [(0, 1.0), (1, 0.0), (True, 1.0), (1, float("inf"))],
)
def test_engine_requires_explicit_positive_budgets(
    tmp_path: Path, max_steps: object, max_wall_seconds: float
) -> None:
    problem = fixture_problem()
    model, sampler = engine_parts(problem)
    with pytest.raises(ValueError):
        op.run_pinn(
            problem,
            model=model,
            sampler=sampler,
            output_dir=tmp_path,
            max_steps=max_steps,  # type: ignore[arg-type]
            max_wall_seconds=max_wall_seconds,
        )
