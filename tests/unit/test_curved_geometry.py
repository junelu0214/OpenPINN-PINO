from __future__ import annotations

import math

import pytest
import torch

import openpinn as op


def test_disk_samples_and_normals_are_analytic() -> None:
    disk = op.Disk(center=(1.5, -0.5), radius=2.0)
    interior = disk.sample(256, seed=7)
    boundary = disk.boundary(64, seed=11)

    assert torch.all(disk.contains(interior))
    assert torch.all(disk.on_boundary(boundary, atol=1e-5))
    normals = disk.boundary_normal(boundary)
    assert torch.allclose(
        torch.linalg.vector_norm(normals, dim=1), torch.ones(64), atol=1e-6
    )
    assert disk.area == pytest.approx(4.0 * math.pi)


def test_rotated_ellipse_samples_bounds_and_normals_are_analytic() -> None:
    ellipse = op.Ellipse(
        center=(1.5, -0.5),
        semi_axis_a=3.0,
        semi_axis_b=1.25,
        rotation_deg=37.0,
    )
    interior = ellipse.sample(512, seed=7, dtype=torch.float64)
    boundary = ellipse.boundary(256, seed=11, dtype=torch.float64)

    assert torch.all(ellipse.contains(interior))
    assert torch.all(ellipse.on_boundary(boundary, atol=1e-10))
    normals = ellipse.boundary_normal(boundary, atol=1e-10)
    assert torch.allclose(
        torch.linalg.vector_norm(normals, dim=1),
        torch.ones(256, dtype=torch.float64),
        atol=1e-10,
    )
    assert torch.all(torch.sum(normals * (boundary - boundary.new_tensor(ellipse.center)), dim=1) > 0)
    assert ellipse.area == pytest.approx(math.pi * 3.0 * 1.25)
    (x_min, x_max), (y_min, y_max) = ellipse.bounds
    assert torch.all((boundary[:, 0] >= x_min) & (boundary[:, 0] <= x_max))
    assert torch.all((boundary[:, 1] >= y_min) & (boundary[:, 1] <= y_max))


def test_ellipse_with_equal_axes_matches_a_disk() -> None:
    ellipse = op.Ellipse(center=(0.5, -2.0), semi_axis_a=2.0, semi_axis_b=2.0, rotation_deg=63.0)
    boundary = ellipse.boundary(128, seed=13, dtype=torch.float64)
    radii = torch.linalg.vector_norm(boundary - boundary.new_tensor(ellipse.center), dim=1)

    assert torch.allclose(radii, torch.full_like(radii, 2.0), atol=1e-10)
    assert ellipse.bounds[0] == pytest.approx((-1.5, 2.5))
    assert ellipse.bounds[1] == pytest.approx((-4.0, 0.0))


@pytest.mark.parametrize("radius", [0.0, 0.4, 1.0])
def test_rounded_rectangle_preserves_curved_boundary(radius: float) -> None:
    geometry = op.RoundedRectangle(lower=(-2.0, -1.0), upper=(2.0, 1.0), corner_radius=radius)
    interior = geometry.sample(256, seed=3)
    boundary = geometry.boundary(128, seed=5)

    assert torch.all(geometry.contains(interior))
    assert torch.all(geometry.on_boundary(boundary, atol=1e-5))
    normals = geometry.boundary_normal(boundary)
    assert torch.allclose(
        torch.linalg.vector_norm(normals, dim=1), torch.ones(128), atol=1e-5
    )


def test_square_maximum_rounding_is_a_disk() -> None:
    rounded = op.RoundedRectangle(lower=(-1.0, -1.0), upper=(1.0, 1.0), corner_radius=1.0)
    boundary = rounded.boundary(96, seed=13)
    radii = torch.linalg.vector_norm(boundary, dim=1)

    assert torch.allclose(radii, torch.ones_like(radii), atol=1e-6)
    assert rounded.area == pytest.approx(math.pi)


def test_closed_spline_samples_true_curve_and_analytic_normals() -> None:
    controls = tuple(
        (math.cos(index * math.pi / 4), math.sin(index * math.pi / 4))
        for index in range(8)
    )
    geometry = op.ClosedSpline(controls, tension=0.2)
    interior = geometry.sample(128, seed=17, dtype=torch.float64)
    boundary = geometry.boundary(96, seed=19, dtype=torch.float64)

    assert torch.all(geometry.contains(interior))
    assert torch.all(geometry.on_boundary(boundary, atol=1e-8))
    normals = geometry.boundary_normal(boundary, atol=1e-8)
    assert torch.allclose(
        torch.linalg.vector_norm(normals, dim=1), torch.ones(96, dtype=torch.float64), atol=1e-8
    )
    assert torch.all(torch.sum(normals * boundary, dim=1) > 0)


def test_closed_spline_tension_changes_curvature_without_moving_controls() -> None:
    controls = ((-2.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-2.0, 1.0))
    smooth = op.ClosedSpline(controls, tension=0.0)
    tight = op.ClosedSpline(controls, tension=0.8)

    assert smooth.control_points == tight.control_points
    assert smooth.area > tight.area


def test_curved_geometry_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="radius"):
        op.Disk(center=(0.0, 0.0), radius=0.0)
    with pytest.raises(ValueError, match="semi_axis_a"):
        op.Ellipse(center=(0.0, 0.0), semi_axis_a=0.0, semi_axis_b=1.0)
    with pytest.raises(ValueError, match="corner_radius"):
        op.RoundedRectangle(lower=(0.0, 0.0), upper=(2.0, 1.0), corner_radius=0.6)
    with pytest.raises(ValueError, match="tension"):
        op.ClosedSpline(((0, 0), (1, 0), (1, 1), (0, 1)), tension=1.1)
