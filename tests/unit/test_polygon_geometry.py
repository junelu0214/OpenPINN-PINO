from __future__ import annotations

import json

import pytest
import torch

import openpinn as op


def test_concave_polygon_sampling_stays_inside_and_is_reproducible() -> None:
    polygon = op.Polygon(((0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)))

    first = polygon.sample(500, seed=17)
    second = polygon.sample(500, seed=17)

    torch.testing.assert_close(first, second)
    assert bool(torch.all(polygon.contains(first)))
    assert not bool(polygon.contains(torch.tensor([[2.0, 2.0]]))[0])
    assert polygon.area == pytest.approx(5.0)
    assert polygon.bounds == ((0.0, 3.0), (0.0, 3.0))


def test_polygon_hole_is_excluded_and_both_boundaries_are_sampled() -> None:
    polygon = op.Polygon(
        ((0, 0), (4, 0), (4, 4), (0, 4)),
        holes=(((1, 1), (1, 3), (3, 3), (3, 1)),),
    )

    assert polygon.area == pytest.approx(12.0)
    query = torch.tensor([[0.5, 0.5], [2.0, 2.0], [1.0, 2.0], [4.0, 2.0]])
    assert polygon.contains(query).tolist() == [True, False, False, True]
    assert polygon.on_boundary(query).tolist() == [False, False, True, True]
    boundary = polygon.boundary(2000, seed=8)
    assert bool(torch.all(polygon.on_boundary(boundary, atol=2e-5)))
    assert bool(torch.any(torch.isclose(boundary[:, 0], boundary.new_tensor(1.0), atol=2e-5)))


def test_polygon_boundary_normals_point_out_of_domain() -> None:
    polygon = op.Polygon(
        ((0, 0), (4, 0), (4, 4), (0, 4)),
        holes=(((1, 1), (1, 3), (3, 3), (3, 1)),),
    )
    points = torch.tensor([[2.0, 0.0], [4.0, 2.0], [1.0, 2.0], [2.0, 3.0]])

    normals = polygon.boundary_normal(points)

    torch.testing.assert_close(
        normals,
        torch.tensor([[0.0, -1.0], [1.0, 0.0], [1.0, 0.0], [0.0, -1.0]]),
    )


def test_polygon_boundary_sampling_covers_every_edge_when_count_allows() -> None:
    polygon = op.Polygon(
        ((0, 0), (4, 0), (4, 0.2), (1, 0.2), (1, 2), (0, 2))
    )

    points = polygon.boundary(6, seed=23)

    assert points.shape == (6, 2)
    for start, end in zip(
        polygon.vertices, polygon.vertices[1:] + polygon.vertices[:1]
    ):
        start_tensor = points.new_tensor(start)
        edge = points.new_tensor(end) - start_tensor
        projection = ((points - start_tensor) * edge).sum(dim=1) / (edge * edge).sum()
        closest = start_tensor + projection.clamp(0.0, 1.0).unsqueeze(1) * edge
        distance = torch.linalg.vector_norm(points - closest, dim=1)
        assert bool(torch.any(distance <= 1e-6))


@pytest.mark.parametrize("suffix", [".json", ".geojson"])
def test_polygon_from_json_and_geojson(tmp_path, suffix: str) -> None:
    path = tmp_path / f"domain{suffix}"
    if suffix == ".json":
        payload = {
            "outer": [[0, 0], [3, 0], [3, 2], [0, 2]],
            "holes": [[[1, 0.5], [2, 0.5], [2, 1.5], [1, 1.5]]],
        }
    else:
        payload = {
            "type": "Feature",
            "properties": {"name": "test"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[0, 0], [3, 0], [3, 2], [0, 2], [0, 0]],
                    [[1, 0.5], [1, 1.5], [2, 1.5], [2, 0.5], [1, 0.5]],
                ],
            },
        }
    path.write_text(json.dumps(payload), encoding="utf-8")

    polygon = op.Polygon.from_file(path)

    assert polygon.area == pytest.approx(5.0)
    assert not bool(polygon.contains(torch.tensor([[1.5, 1.0]]))[0])


def test_polygon_from_grouped_csv(tmp_path) -> None:
    path = tmp_path / "domain.csv"
    path.write_text(
        "ring,x,y\n"
        "outer,0,0\nouter,4,0\nouter,4,4\nouter,0,4\n"
        "hole_1,1,1\nhole_1,1,2\nhole_1,2,2\nhole_1,2,1\n",
        encoding="utf-8",
    )

    polygon = op.Polygon.from_file(path)

    assert polygon.area == pytest.approx(15.0)
    assert polygon.bounds == ((0.0, 4.0), (0.0, 4.0))


def test_polygon_rejects_self_intersection_and_invalid_holes() -> None:
    with pytest.raises(ValueError, match="self-intersecting"):
        op.Polygon(((0, 0), (2, 2), (0, 2), (2, 0)))
    with pytest.raises(ValueError, match="strictly inside"):
        op.Polygon(
            ((0, 0), (2, 0), (2, 2), (0, 2)),
            holes=(((1, 1), (3, 1), (3, 3), (1, 3)),),
        )


def test_polygon_supports_space_time_sampling() -> None:
    domain = op.SpaceTimeDomain(
        op.Polygon(((0, 0), (2, 0), (1, 1))),
        0.0,
        2.0,
    )

    interior = domain.sample(50, seed=2)
    boundary = domain.boundary(50, seed=3)

    assert interior.shape == (50, 3)
    assert boundary.shape == (50, 3)
    assert bool(torch.all(domain.contains(interior)))
    assert bool(torch.all(domain.on_spatial_boundary(boundary, atol=2e-5)))
