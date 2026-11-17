from __future__ import annotations

import csv

import pytest
import torch

from openpinn.inverse import (
    consolidation_pressure,
    darcy_pressure,
    inverse_consolidation_operator,
    inverse_darcy_pinn,
    load_consolidation_observations_csv,
    load_inverse_observations_csv,
    train_consolidation_family_operator,
)


def observations(permeability: float, source: float = 1.0):
    x = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]).reshape(-1, 1)
    return x, darcy_pressure(x, permeability, source)


def test_pinn_jointly_identifies_one_darcy_instance() -> None:
    x, pressure = observations(2.0, 1.2)
    result = inverse_darcy_pinn(
        x, pressure, source=1.2, initial_permeability=0.7,
        hidden_layers=(16, 16), collocation_points=32, steps=1200,
        learning_rate=3e-3, seed=0,
    )
    assert result.estimated_permeability == pytest.approx(2.0, rel=0.08)
    assert result.history[-1]["total_loss"] < result.history[0]["total_loss"]
    assert 0.2 < result.estimated_permeability < 5.0


def test_operator_learns_consolidation_family_then_inverts_a_new_member() -> None:
    family = train_consolidation_family_operator(
        steps=600, family_batch_size=32, query_points=32, seed=2,
    )
    points = torch.tensor([
        [0.2, 0.05], [0.2, 0.2], [0.2, 0.5],
        [0.5, 0.05], [0.5, 0.2], [0.5, 0.5],
        [0.8, 0.05], [0.8, 0.2], [0.8, 0.5],
    ])
    pressure = consolidation_pressure(points, 0.3, 1.2)
    result = inverse_consolidation_operator(
        family, points, pressure, amplitude=1.2,
        initial_coefficient=0.15, steps=250,
    )
    assert result.estimated_coefficient == pytest.approx(0.3, rel=0.2)
    assert result.history[-1]["observation_loss"] < result.history[0]["observation_loss"]
    assert result.predictions.shape == pressure.shape


def test_inverse_csv_requires_named_numeric_columns(tmp_path) -> None:
    source = tmp_path / "observations.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "p"])
        writer.writerows([[0.25, 0.1], [0.75, 0.1]])
    x, p = load_inverse_observations_csv(source)
    assert x.shape == p.shape == (2, 1)

    source.write_text("x,value\n0.5,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain"):
        load_inverse_observations_csv(source)


def test_consolidation_csv_loads_named_spacetime_columns(tmp_path) -> None:
    source = tmp_path / "consolidation.csv"
    source.write_text("z,t,u\n0.2,0.1,0.5\n0.5,0.2,0.4\n0.8,0.3,0.2\n", encoding="utf-8")
    points, values = load_consolidation_observations_csv(source)
    assert points.shape == (3, 2)
    assert values.shape == (3, 1)
