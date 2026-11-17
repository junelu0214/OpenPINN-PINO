"""Analytic circular holes: exact sampling, domain membership and normals."""
import math

import pytest
import torch

from openpinn import NeumannBC, RectangleWithCircularHoles, RobinBC


def geometry(**kwargs):
    return RectangleWithCircularHoles((0, 0), (40, 40), ((10, 20, 4), (30, 20, 4)), **kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stratified_boundary_covers_each_component(dtype):
    domain = geometry(stratified_boundary=True)
    for seed in range(20):
        points = domain.sample_boundary(6, dtype=dtype, generator=torch.Generator().manual_seed(seed))
        assert points.shape == (6, 2)
        assert domain.on_boundary(points, atol=1e-5).all()
        masks = [torch.isclose(points[:, axis], points.new_tensor(value)) for axis, value in [(0, 0), (0, 40), (1, 0), (1, 40)]]
        for cx, cy, radius in domain.holes:
            radial = points - points.new_tensor((cx, cy))
            masks.append(torch.isclose(torch.linalg.vector_norm(radial, dim=1), points.new_tensor(radius), atol=1e-5))
        assert all(mask.sum() == 1 for mask in masks)
        assert torch.stack(masks).sum(dim=0).eq(1).all()
        normals = domain.boundary_normal(points)
        torch.testing.assert_close(torch.linalg.vector_norm(normals, dim=1), torch.ones(6, dtype=dtype))
        for index, (cx, cy, _) in enumerate(domain.holes):
            inward = points.new_tensor((cx, cy)) - points[masks[4 + index]]
            assert (normals[masks[4 + index]] * inward).sum() > 0


def test_exact_hole_normals_and_no_extended_rectangle_edges():
    domain = geometry()
    points = torch.tensor([[0., 10.], [40., 10.], [5., 0.], [5., 40.], [14., 20.], [10., 24.], [26., 20.]])
    expected = torch.tensor([[-1., 0.], [1., 0.], [0., -1.], [0., 1.], [-1., 0.], [0., -1.], [1., 0.]])
    torch.testing.assert_close(domain.boundary_normal(points), expected)
    assert not domain.on_boundary(torch.tensor([[0., 50.]])).item()
    with pytest.raises(ValueError, match="requires points"):
        domain.boundary_normal(torch.tensor([[0., 50.]]))
    torch.testing.assert_close(domain.boundary_normal(torch.tensor([[0., 0.]])), torch.tensor([[-1 / math.sqrt(2), -1 / math.sqrt(2)]]))
    assert domain.boundary_normal(torch.empty((0, 2))).shape == (0, 2)


def test_exact_circles_not_inscribed_polygons_and_reproducible_sampling():
    domain = geometry(stratified_boundary=True)
    angle = math.pi / 128
    # This lies outside an inscribed 128-gon but inside the removed true disk.
    radius = 4 * (1 + math.cos(angle)) / 2
    point = torch.tensor([[10 + radius * math.cos(angle), 20 + radius * math.sin(angle)]], dtype=torch.float64)
    assert not domain.contains(point).item()
    for n in [6, 19, 256]:
        left = domain.boundary(n, seed=19, dtype=torch.float64)
        right = domain.boundary(n, seed=19, dtype=torch.float64)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert domain.contains(domain.sample(1000, seed=3)).all()
    with pytest.raises(ValueError, match="at least 6"):
        domain.sample_boundary(5)
    with pytest.raises(ValueError, match="strictly inside"):
        RectangleWithCircularHoles((0, 0), (40, 40), ((4, 20, 4),))
    with pytest.raises(ValueError, match="overlap or touch"):
        RectangleWithCircularHoles((0, 0), (40, 40), ((10, 20, 4), (18, 20, 4)))
    with pytest.raises(ValueError, match="finite"):
        RectangleWithCircularHoles((0, 0), (float('inf'), 40))


def test_hole_neumann_and_robin_have_correct_sign():
    domain = geometry()
    angle = torch.linspace(0, 2 * math.pi, 129, dtype=torch.float64)[:-1]
    points = torch.stack((10 + 4 * torch.cos(angle), 20 + 4 * torch.sin(angle)), dim=1)

    class Quadratic(torch.nn.Module):
        def forward(self, inputs):
            return (inputs - inputs.new_tensor((10, 20))).square().sum(dim=1, keepdim=True)

    # On hole 1, u=16 and du/dn=-8 because the normal points into the disk.
    for condition in [NeumannBC(-8, domain.boundary_normal),
                      RobinBC(28, domain.boundary_normal, alpha=2, beta=0.5)]:
        torch.testing.assert_close(condition.error(Quadratic(), points), torch.zeros((128, 1), dtype=torch.float64), atol=1e-12, rtol=0)
