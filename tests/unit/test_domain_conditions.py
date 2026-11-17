from __future__ import annotations

import torch
from torch import nn

import openpinn as op


def test_interval_sampling_and_membership_are_reproducible() -> None:
    domain = op.Interval(-1.0, 2.0)
    first_generator = torch.Generator().manual_seed(19)
    second_generator = torch.Generator().manual_seed(19)

    first = domain.sample_interior(8, generator=first_generator)
    second = domain.sample_interior(8, generator=second_generator)

    torch.testing.assert_close(first, second)
    assert first.shape == (8, 1)
    assert domain.contains(first).all()
    boundary = domain.sample_boundary(4)
    assert domain.on_boundary(boundary).all()
    torch.testing.assert_close(boundary[:, 0], torch.tensor([-1.0, 2.0, -1.0, 2.0]))


def test_space_time_domain_separates_spatial_boundary_and_initial_plane() -> None:
    domain = op.SpaceTimeDomain(op.Interval(-1.0, 1.0), op.Interval(0.0, 1.0))
    generator = torch.Generator().manual_seed(23)

    interior = domain.sample_interior(7, generator=generator)
    boundary = domain.sample_boundary(6, generator=generator)
    initial = domain.sample_initial(5, generator=generator)

    assert interior.shape == (7, 2)
    assert domain.contains(interior).all()
    assert domain.on_spatial_boundary(boundary).all()
    assert domain.on_initial(initial).all()


def test_conditions_return_named_pointwise_errors() -> None:
    model = nn.Sequential(nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 2.0]]))
    points = torch.tensor([[-1.0, 0.0], [1.0, 0.25], [0.5, 0.0]])

    boundary = op.DirichletBC(
        0.0, boundary=lambda values: values[:, 0].abs() == 1.0, name="boundary"
    )
    initial = op.InitialCondition(
        lambda values: values[:, :1], selector=lambda values: values[:, 1] == 0.0
    )

    boundary_error = boundary.error(model, points)
    initial_error = initial.error(model, points)
    assert boundary.name == "boundary"
    assert initial.name == "initial"
    assert boundary_error.shape == (2, 1)
    assert initial_error.shape == (2, 1)
    torch.testing.assert_close(initial_error, torch.zeros_like(initial_error))


def test_neumann_and_robin_conditions_use_outward_normal_derivative() -> None:
    model = nn.Sequential(nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        model[0].weight.fill_(2.0)
    points = torch.tensor([[-1.0], [1.0]], requires_grad=True)

    left_flux = op.NeumannBC(
        -2.0,
        normal=(-1.0,),
        boundary=lambda values: values[:, 0] < 0,
        name="left_flux",
    )
    right_robin = op.RobinBC(
        5.0,
        normal=(1.0,),
        boundary=lambda values: values[:, 0] > 0,
        alpha=0.5,
        beta=2.0,
        name="right_robin",
    )

    torch.testing.assert_close(left_flux.error(model, points), torch.zeros((1, 1)))
    torch.testing.assert_close(right_robin.error(model, points), torch.zeros((1, 1)))


def test_derivative_conditions_accept_a_constant_model() -> None:
    class ConstantModel(nn.Module):
        def forward(self, points: torch.Tensor) -> torch.Tensor:
            return torch.full_like(points[:, :1], 3.0, requires_grad=False)

    model = ConstantModel()
    points = torch.tensor([[-1.0], [1.0]])
    zero_flux = op.NeumannBC(0.0, normal=(1.0,))
    constant_robin = op.RobinBC(6.0, normal=(1.0,), alpha=2.0, beta=1.0)

    torch.testing.assert_close(zero_flux.error(model, points), torch.zeros((2, 1)))
    torch.testing.assert_close(constant_robin.error(model, points), torch.zeros((2, 1)))


def test_uniform_sampler_groups_static_and_dynamic_points() -> None:
    poisson = op.make_poisson_problem()
    poisson_sampler = op.UniformSampler(
        poisson, interior=9, boundary=4, initial=3, seed=31
    )
    first = poisson_sampler.sample()
    poisson_sampler.reset()
    second = poisson_sampler.sample()
    torch.testing.assert_close(first.interior, second.interior)
    assert first.interior.shape == (9, 1)
    assert first.boundary.shape == (4, 1)
    assert first.initial.shape == (0, 1)

    burgers = op.make_burgers_problem()
    dynamic = op.UniformSampler(
        burgers, interior=9, boundary=4, initial=3, seed=37
    ).sample()
    assert dynamic.interior.shape == (9, 2)
    assert dynamic.boundary.shape == (4, 2)
    assert dynamic.initial.shape == (3, 2)


def test_residual_adaptive_sampler_updates_fixed_size_interior_batch() -> None:
    torch.manual_seed(71)
    problem = op.make_poisson_problem(wave_number=4.0)
    model = op.PINNModel(op.MLP(1, 1, hidden_layers=(8, 8)))
    sampler = op.ResidualAdaptiveSampler(
        problem,
        n_interior=8,
        n_boundary=2,
        n_initial=0,
        candidate_pool=64,
        uniform_mix=0.0,
        seed=73,
    )

    sampler.update(model)
    batch = sampler.sample()
    diagnostics = sampler.diagnostics()

    assert batch.interior.shape == (8, 1)
    assert batch.boundary.shape == (2, 1)
    assert diagnostics["update_count"] == 1
    assert diagnostics["selected_mean_residual"] >= diagnostics["candidate_mean_residual"]
    assert torch.isfinite(batch.interior).all()


def test_time_window_sampler_opens_time_prefix_in_order() -> None:
    problem = op.make_burgers_problem()
    model = op.PINNModel(op.MLP(2, 1, hidden_layers=(8,)))
    sampler = op.TimeWindowSampler(
        problem,
        n_interior=32,
        n_boundary=8,
        n_initial=8,
        window_ends=(0.25, 0.5, 0.75, 1.0),
        seed=79,
    )

    first = sampler.sample()
    sampler.update(model)
    second = sampler.sample()

    assert float(first.interior[:, 1].max().detach()) <= 0.25
    assert float(second.interior[:, 1].max().detach()) <= 0.5
    assert sampler.current_window_index == 1
    assert sampler.diagnostics()["current_window_end"] == 0.5
    assert torch.all(first.initial[:, 1] == 0.0)
    assert torch.all(second.initial[:, 1] == 0.0)
