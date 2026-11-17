from __future__ import annotations

import torch

import openpinn as op


def test_static_polygon_problem_short_run(tmp_path) -> None:
    domain = op.Polygon(((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)))

    def residual(points: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return (
            -op.hessian(values, points, i=0, j=0)
            - op.hessian(values, points, i=1, j=1)
            - torch.ones_like(values)
        )

    problem = op.PDEProblem(
        domain,
        residual,
        [op.DirichletBC(0.0, name="boundary_polygon")],
        name="polygon_static_smoke",
        metadata={
            "kind": "custom_static",
            "geometry": "polygon",
            "residual_names": ("pde",),
        },
    )

    result = op.solve(
        problem,
        methods=[op.method("baseline", "plain_pinn")],
        steps=1,
        seed=0,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 16, "boundary": 12},
        output_root=tmp_path,
        plot=False,
    )

    assert result.metrics["method"] == "baseline"
    assert "boundary_polygon_rmse" in result.metrics


def test_dynamic_polygon_neumann_normal_short_run(tmp_path) -> None:
    space = op.Polygon(((0, 0), (2, 0), (1, 1.5)))
    domain = op.SpaceTimeDomain(space, 0.0, 1.0)

    def residual(points: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        gradient = op.grad(values, points)
        return gradient[:, 2:3] - op.hessian(values, points, i=0, j=0) - op.hessian(
            values, points, i=1, j=1
        )

    def boundary_normal(points: torch.Tensor) -> torch.Tensor:
        spatial = space.boundary_normal(points[:, :2])
        return torch.cat((spatial, torch.zeros_like(points[:, 2:])), dim=1)

    problem = op.PDEProblem(
        domain,
        residual,
        [
            op.NeumannBC(0.0, normal=boundary_normal, name="boundary_polygon"),
            op.InitialCondition(0.0, name="initial"),
        ],
        name="polygon_dynamic_smoke",
        metadata={
            "kind": "custom_dynamic",
            "geometry": "polygon",
            "residual_names": ("pde",),
        },
    )

    result = op.solve(
        problem,
        methods=[op.method("baseline", "plain_pinn")],
        steps=1,
        seed=0,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 12, "boundary": 12, "initial": 12},
        output_root=tmp_path,
        plot=False,
    )

    assert result.metrics["method"] == "baseline"
    assert "boundary_polygon_rmse" in result.metrics
    assert "initial_rmse" in result.metrics


def test_seven_edge_polygon_accepts_independent_mixed_conditions(tmp_path) -> None:
    vertices = tuple(
        (
            float(torch.cos(torch.tensor(2.0 * torch.pi * index / 7))),
            float(torch.sin(torch.tensor(2.0 * torch.pi * index / 7))),
        )
        for index in range(7)
    )
    space = op.Polygon(vertices)

    def edge_selector(ring_index: int, edge_index: int):
        start_coordinates = space.rings[ring_index][edge_index]
        ring = space.rings[ring_index]
        end_coordinates = ring[(edge_index + 1) % len(ring)]

        def selector(points: torch.Tensor) -> torch.Tensor:
            spatial = points[:, :2]
            start = spatial.new_tensor(start_coordinates)
            end = spatial.new_tensor(end_coordinates)
            edge = end - start
            projection = ((spatial - start) * edge).sum(dim=1) / (edge * edge).sum()
            closest = start + projection.clamp(0.0, 1.0).unsqueeze(1) * edge
            tolerance = 1e-6 * torch.maximum(
                torch.linalg.vector_norm(edge), spatial.new_tensor(1.0)
            )
            return torch.linalg.vector_norm(spatial - closest, dim=1) <= tolerance

        return selector

    def normal(points: torch.Tensor) -> torch.Tensor:
        return space.boundary_normal(points[:, :2])

    conditions: list[op.Condition] = []
    for edge_index in range(7):
        selector = edge_selector(0, edge_index)
        name = f"boundary_polygon_r1_e{edge_index + 1}"
        if edge_index % 3 == 0:
            condition = op.DirichletBC(float(edge_index), boundary=selector, name=name)
        elif edge_index % 3 == 1:
            condition = op.NeumannBC(0.0, normal=normal, boundary=selector, name=name)
        else:
            condition = op.RobinBC(
                0.0,
                normal=normal,
                boundary=selector,
                alpha=1.0,
                beta=0.5,
                name=name,
            )
        conditions.append(condition)

    def residual(points: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return -op.hessian(values, points, i=0, j=0) - op.hessian(
            values, points, i=1, j=1
        )

    problem = op.PDEProblem(
        space,
        residual,
        conditions,
        name="polygon_seven_edge_mixed_smoke",
        metadata={"kind": "custom_static", "geometry": "polygon"},
    )
    result = op.solve(
        problem,
        methods=[op.method("baseline", "plain_pinn")],
        steps=1,
        seed=0,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 16, "boundary": 28},
        output_root=tmp_path,
        plot=False,
    )

    for edge_index in range(7):
        assert f"boundary_polygon_r1_e{edge_index + 1}_rmse" in result.metrics
