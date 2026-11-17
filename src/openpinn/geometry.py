"""Small, inspectable geometries used by OpenPINN problems."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def _rand(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator | None,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> Tensor:
    """Draw reproducible CPU samples, then move them to the target device."""

    values = torch.rand(shape, generator=generator, dtype=dtype, device="cpu")
    return values.to(device=device) if device is not None else values


Point2D = tuple[float, float]
Ring = tuple[Point2D, ...]


def _signed_ring_area(ring: Ring) -> float:
    return 0.5 * sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1])
    )


def _point_on_segment(
    point: Point2D,
    start: Point2D,
    end: Point2D,
    *,
    atol: float = 1e-10,
) -> bool:
    px, py = point
    ax, ay = start
    bx, by = end
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    scale = max(1.0, abs(bx - ax), abs(by - ay))
    if abs(cross) > atol * scale:
        return False
    return (
        min(ax, bx) - atol <= px <= max(ax, bx) + atol
        and min(ay, by) - atol <= py <= max(ay, by) + atol
    )


def _orientation(a: Point2D, b: Point2D, c: Point2D) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(
    a: Point2D,
    b: Point2D,
    c: Point2D,
    d: Point2D,
    *,
    atol: float = 1e-10,
) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if ((o1 > atol and o2 < -atol) or (o1 < -atol and o2 > atol)) and (
        (o3 > atol and o4 < -atol) or (o3 < -atol and o4 > atol)
    ):
        return True
    return (
        (abs(o1) <= atol and _point_on_segment(c, a, b, atol=atol))
        or (abs(o2) <= atol and _point_on_segment(d, a, b, atol=atol))
        or (abs(o3) <= atol and _point_on_segment(a, c, d, atol=atol))
        or (abs(o4) <= atol and _point_on_segment(b, c, d, atol=atol))
    )


def _point_in_ring(point: Point2D, ring: Ring, *, include_boundary: bool) -> bool:
    if any(
        _point_on_segment(point, start, end)
        for start, end in zip(ring, ring[1:] + ring[:1])
    ):
        return include_boundary
    x, y = point
    inside = False
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        if (y0 > y) != (y1 > y):
            crossing_x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < crossing_x:
                inside = not inside
    return inside


def _normalise_ring(values: Any, *, name: str, clockwise: bool) -> Ring:
    try:
        ring = tuple((float(point[0]), float(point[1])) for point in values)
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"{name} must be a sequence of [x, y] coordinates.") from exc
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) < 3:
        raise ValueError(f"{name} requires at least three distinct vertices.")
    if len(ring) > 5_000:
        raise ValueError(f"{name} exceeds the 5,000-vertex safety limit.")
    if not all(math.isfinite(value) for point in ring for value in point):
        raise ValueError(f"{name} coordinates must be finite.")
    if any(start == end for start, end in zip(ring, ring[1:] + ring[:1])):
        raise ValueError(f"{name} contains repeated consecutive vertices.")
    edge_count = len(ring)
    for left in range(edge_count):
        left_start = ring[left]
        left_end = ring[(left + 1) % edge_count]
        for right in range(left + 1, edge_count):
            if right in {left, (left + 1) % edge_count}:
                continue
            if left == 0 and right == edge_count - 1:
                continue
            if _segments_intersect(
                left_start,
                left_end,
                ring[right],
                ring[(right + 1) % edge_count],
            ):
                raise ValueError(f"{name} is self-intersecting.")
    area = _signed_ring_area(ring)
    if abs(area) <= 1e-12:
        raise ValueError(f"{name} has zero area.")
    if clockwise == (area > 0.0):
        ring = tuple(reversed(ring))
    return ring


def _rings_intersect(left: Ring, right: Ring) -> bool:
    return any(
        _segments_intersect(a, b, c, d)
        for a, b in zip(left, left[1:] + left[:1])
        for c, d in zip(right, right[1:] + right[:1])
    )


def _polygon_coordinates_from_json(data: Any) -> tuple[Any, Any]:
    if isinstance(data, dict) and "outer" in data:
        return data["outer"], data.get("holes", [])
    if isinstance(data, list):
        return data, []
    if not isinstance(data, dict):
        raise ValueError("JSON geometry must be a coordinate list or object.")
    geometry = data
    if data.get("type") == "Feature":
        geometry = data.get("geometry")
    elif data.get("type") == "FeatureCollection":
        geometries = [
            feature.get("geometry")
            for feature in data.get("features", [])
            if isinstance(feature, dict)
            and isinstance(feature.get("geometry"), dict)
            and feature["geometry"].get("type") == "Polygon"
        ]
        if len(geometries) != 1:
            raise ValueError("GeoJSON FeatureCollection must contain exactly one Polygon.")
        geometry = geometries[0]
    if not isinstance(geometry, dict) or geometry.get("type") != "Polygon":
        raise ValueError("Only GeoJSON Polygon geometry is supported; MultiPolygon is not yet supported.")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or not coordinates:
        raise ValueError("GeoJSON Polygon coordinates are empty.")
    return coordinates[0], coordinates[1:]


def _polygon_coordinates_from_csv(path: Path) -> tuple[Any, Any]:
    groups: dict[str, list[list[float]]] = {}
    current = "outer"
    blank_index = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            cells = [cell.strip() for cell in row]
            if not cells or not any(cells):
                blank_index += 1
                current = f"hole_{blank_index}"
                continue
            if cells[0].startswith("#"):
                continue
            lower = [cell.lower() for cell in cells]
            if line_number == 1 and "x" in lower and "y" in lower:
                continue
            try:
                if len(cells) >= 3:
                    label = cells[0] or current
                    x_value, y_value = float(cells[-2]), float(cells[-1])
                    current = label
                elif len(cells) == 2:
                    x_value, y_value = float(cells[0]), float(cells[1])
                else:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(
                    f"CSV line {line_number} must be x,y or ring,x,y."
                ) from exc
            groups.setdefault(current, []).append([x_value, y_value])
    if not groups:
        raise ValueError("CSV geometry contains no coordinates.")
    keys = list(groups)
    outer_key = "outer" if "outer" in groups else keys[0]
    holes = [groups[key] for key in keys if key != outer_key]
    return groups[outer_key], holes


@dataclass(frozen=True)
class Interval:
    """A closed one-dimensional interval ``[lower, upper]``."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not self.upper > self.lower:
            raise ValueError("Interval requires upper > lower.")

    @property
    def dim(self) -> int:
        """Coordinate dimension."""

        return 1

    @property
    def bounds(self) -> tuple[float, float]:
        """Return ``(lower, upper)``."""

        return self.lower, self.upper

    def sample_interior(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Uniformly sample points in the interval."""

        if n < 0:
            raise ValueError("n must be non-negative.")
        unit = _rand((n, 1), generator=generator, dtype=dtype, device=device)
        return self.lower + (self.upper - self.lower) * unit

    def sample(
        self,
        n: int,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Convenience alias for :meth:`sample_interior` with an optional seed."""

        if seed is not None:
            if generator is not None:
                raise ValueError("Pass seed or generator, not both.")
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        return self.sample_interior(
            n, generator=generator, dtype=dtype, device=device
        )

    def sample_boundary(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Return alternating copies of both interval endpoints."""

        del generator  # Kept for a uniform geometry method signature.
        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0:
            return torch.empty((0, 1), dtype=dtype, device=device)
        values = torch.empty((n, 1), dtype=dtype, device=device)
        values[0::2, 0] = self.lower
        values[1::2, 0] = self.upper
        return values

    def boundary(
        self,
        n: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Convenience alias for :meth:`sample_boundary`."""

        return self.sample_boundary(n, dtype=dtype, device=device)

    def contains(self, points: Tensor, *, atol: float = 0.0) -> Tensor:
        """Return a Boolean mask indicating whether each point lies inside."""

        if points.ndim != 2 or points.shape[1] != 1:
            raise ValueError("Interval points must have shape (N, 1).")
        return ((points[:, 0] >= self.lower - atol) & (points[:, 0] <= self.upper + atol))

    def on_boundary(self, points: Tensor, *, atol: float = 1e-7) -> Tensor:
        """Return a mask selecting either endpoint."""

        if points.ndim != 2 or points.shape[1] != 1:
            raise ValueError("Interval points must have shape (N, 1).")
        lower = torch.isclose(points[:, 0], points.new_tensor(self.lower), atol=atol, rtol=0.0)
        upper = torch.isclose(points[:, 0], points.new_tensor(self.upper), atol=atol, rtol=0.0)
        return lower | upper


@dataclass(frozen=True)
class Disk:
    """A two-dimensional disk represented analytically, not by line segments."""

    center: tuple[float, float]
    radius: float

    def __post_init__(self) -> None:
        center = tuple(float(value) for value in self.center)
        radius = float(self.radius)
        if len(center) != 2 or not all(math.isfinite(value) for value in center):
            raise ValueError("Disk center must contain two finite coordinates.")
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("Disk radius must be a positive finite number.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)

    @property
    def dim(self) -> int:
        return 2

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        center_x, center_y = self.center
        return (
            (center_x - self.radius, center_x + self.radius),
            (center_y - self.radius, center_y + self.radius),
        )

    @property
    def area(self) -> float:
        return math.pi * self.radius**2

    def sample_interior(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if n < 0:
            raise ValueError("n must be non-negative.")
        unit = _rand((n, 2), generator=generator, dtype=dtype, device=device)
        radius = self.radius * torch.sqrt(unit[:, 0])
        angle = 2.0 * math.pi * unit[:, 1]
        center = unit.new_tensor(self.center)
        return center + torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=1)

    def sample(
        self,
        n: int,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if seed is not None:
            if generator is not None:
                raise ValueError("Pass seed or generator, not both.")
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        return self.sample_interior(n, generator=generator, dtype=dtype, device=device)

    def sample_boundary(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if n < 0:
            raise ValueError("n must be non-negative.")
        unit = _rand((n,), generator=generator, dtype=dtype, device=device)
        angle = 2.0 * math.pi * unit
        center = unit.new_tensor(self.center)
        return center + self.radius * torch.stack((torch.cos(angle), torch.sin(angle)), dim=1)

    def boundary(
        self,
        n: int,
        *,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        return self.sample_boundary(n, generator=generator, dtype=dtype, device=device)

    def contains(self, points: Tensor, *, atol: float = 0.0) -> Tensor:
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Disk points must have shape (N, 2).")
        center = points.new_tensor(self.center)
        return torch.linalg.vector_norm(points - center, dim=1) <= self.radius + atol

    def on_boundary(self, points: Tensor, *, atol: float = 1e-7) -> Tensor:
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Disk points must have shape (N, 2).")
        center = points.new_tensor(self.center)
        distance = torch.linalg.vector_norm(points - center, dim=1)
        return torch.isclose(distance, distance.new_tensor(self.radius), atol=atol, rtol=0.0)

    def boundary_normal(self, points: Tensor, *, atol: float = 1e-5) -> Tensor:
        if not torch.all(self.on_boundary(points, atol=atol)):
            raise ValueError("boundary_normal requires points on the disk boundary.")
        center = points.new_tensor(self.center)
        direction = points - center
        return direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True)


@dataclass(frozen=True)
class Ellipse:
    """A rotated two-dimensional ellipse represented analytically.

    ``semi_axis_a`` and ``semi_axis_b`` are the two semi-axis lengths.  They
    are intentionally not reordered, so editing ``a`` and ``b`` in the
    configurator has the same meaning as in
    ``x_local**2 / a**2 + y_local**2 / b**2 = 1``.  ``rotation_deg`` rotates
    the local ``a`` axis counter-clockwise in physical coordinates.
    """

    center: tuple[float, float]
    semi_axis_a: float
    semi_axis_b: float
    rotation_deg: float = 0.0

    def __post_init__(self) -> None:
        center = tuple(float(value) for value in self.center)
        semi_axis_a = float(self.semi_axis_a)
        semi_axis_b = float(self.semi_axis_b)
        rotation_deg = float(self.rotation_deg)
        if len(center) != 2 or not all(math.isfinite(value) for value in center):
            raise ValueError("Ellipse center must contain two finite coordinates.")
        if not math.isfinite(semi_axis_a) or semi_axis_a <= 0.0:
            raise ValueError("Ellipse semi_axis_a must be a positive finite number.")
        if not math.isfinite(semi_axis_b) or semi_axis_b <= 0.0:
            raise ValueError("Ellipse semi_axis_b must be a positive finite number.")
        if not math.isfinite(rotation_deg):
            raise ValueError("Ellipse rotation_deg must be finite.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "semi_axis_a", semi_axis_a)
        object.__setattr__(self, "semi_axis_b", semi_axis_b)
        object.__setattr__(self, "rotation_deg", rotation_deg)

    @property
    def dim(self) -> int:
        return 2

    @property
    def _rotation(self) -> tuple[float, float]:
        angle = math.radians(self.rotation_deg)
        return math.cos(angle), math.sin(angle)

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        cosine, sine = self._rotation
        half_width = math.hypot(self.semi_axis_a * cosine, self.semi_axis_b * sine)
        half_height = math.hypot(self.semi_axis_a * sine, self.semi_axis_b * cosine)
        return (
            (self.center[0] - half_width, self.center[0] + half_width),
            (self.center[1] - half_height, self.center[1] + half_height),
        )

    @property
    def area(self) -> float:
        return math.pi * self.semi_axis_a * self.semi_axis_b

    def _to_local(self, points: Tensor) -> Tensor:
        centered = points - points.new_tensor(self.center)
        cosine, sine = self._rotation
        return torch.stack(
            (
                cosine * centered[:, 0] + sine * centered[:, 1],
                -sine * centered[:, 0] + cosine * centered[:, 1],
            ),
            dim=1,
        )

    def _from_local(self, points: Tensor) -> Tensor:
        cosine, sine = self._rotation
        rotated = torch.stack(
            (
                cosine * points[:, 0] - sine * points[:, 1],
                sine * points[:, 0] + cosine * points[:, 1],
            ),
            dim=1,
        )
        return rotated + points.new_tensor(self.center)

    def sample_interior(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if n < 0:
            raise ValueError("n must be non-negative.")
        unit = _rand((n, 2), generator=generator, dtype=dtype, device=device)
        radius = torch.sqrt(unit[:, 0])
        angle = 2.0 * math.pi * unit[:, 1]
        local = torch.stack(
            (
                self.semi_axis_a * radius * torch.cos(angle),
                self.semi_axis_b * radius * torch.sin(angle),
            ),
            dim=1,
        )
        return self._from_local(local)

    def sample(
        self,
        n: int,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if seed is not None:
            if generator is not None:
                raise ValueError("Pass seed or generator, not both.")
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        return self.sample_interior(n, generator=generator, dtype=dtype, device=device)

    def sample_boundary(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0:
            return torch.empty((0, 2), dtype=dtype, device=device)
        # Rejection weighting by parametric speed produces boundary samples
        # that are uniform in arc length rather than clustered by angle.
        accepted: list[Tensor] = []
        accepted_count = 0
        maximum_speed = max(self.semi_axis_a, self.semi_axis_b)
        while accepted_count < n:
            remaining = n - accepted_count
            unit = _rand((max(32, 2 * remaining), 2), generator=generator, dtype=dtype, device=device)
            angle = 2.0 * math.pi * unit[:, 0]
            speed = torch.sqrt(
                (self.semi_axis_a * torch.sin(angle)) ** 2
                + (self.semi_axis_b * torch.cos(angle)) ** 2
            )
            angle = angle[unit[:, 1] <= speed / maximum_speed][:remaining]
            local = torch.stack(
                (self.semi_axis_a * torch.cos(angle), self.semi_axis_b * torch.sin(angle)),
                dim=1,
            )
            accepted.append(local)
            accepted_count += local.shape[0]
        return self._from_local(torch.cat(accepted, dim=0))

    def boundary(
        self,
        n: int,
        *,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        return self.sample_boundary(n, generator=generator, dtype=dtype, device=device)

    def _implicit_value(self, points: Tensor) -> Tensor:
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Ellipse points must have shape (N, 2).")
        local = self._to_local(points)
        return (local[:, 0] / self.semi_axis_a) ** 2 + (local[:, 1] / self.semi_axis_b) ** 2

    def contains(self, points: Tensor, *, atol: float = 0.0) -> Tensor:
        return self._implicit_value(points) <= 1.0 + atol

    def on_boundary(self, points: Tensor, *, atol: float = 1e-7) -> Tensor:
        implicit = self._implicit_value(points)
        return torch.isclose(implicit, implicit.new_tensor(1.0), atol=atol, rtol=0.0)

    def boundary_normal(self, points: Tensor, *, atol: float = 1e-5) -> Tensor:
        if not torch.all(self.on_boundary(points, atol=atol)):
            raise ValueError("boundary_normal requires points on the ellipse boundary.")
        local = self._to_local(points)
        local_gradient = torch.stack(
            (
                local[:, 0] / self.semi_axis_a**2,
                local[:, 1] / self.semi_axis_b**2,
            ),
            dim=1,
        )
        cosine, sine = self._rotation
        global_gradient = torch.stack(
            (
                cosine * local_gradient[:, 0] - sine * local_gradient[:, 1],
                sine * local_gradient[:, 0] + cosine * local_gradient[:, 1],
            ),
            dim=1,
        )
        return global_gradient / torch.linalg.vector_norm(global_gradient, dim=1, keepdim=True)


@dataclass(frozen=True)
class RoundedRectangle:
    """An axis-aligned rectangle whose four corners are exact circular arcs.

    ``corner_radius=0`` gives a rectangle.  A square with radius equal to half
    its side is a disk, while intermediate values preserve a constant corner
    curvature when the width or height is edited.
    """

    lower: tuple[float, float]
    upper: tuple[float, float]
    corner_radius: float

    def __post_init__(self) -> None:
        lower = tuple(float(value) for value in self.lower)
        upper = tuple(float(value) for value in self.upper)
        radius = float(self.corner_radius)
        if len(lower) != 2 or len(upper) != 2:
            raise ValueError("Rounded-rectangle bounds must contain two coordinates.")
        if not all(math.isfinite(value) for value in lower + upper + (radius,)):
            raise ValueError("Rounded-rectangle parameters must be finite.")
        if not (upper[0] > lower[0] and upper[1] > lower[1]):
            raise ValueError("Rounded-rectangle upper bounds must exceed lower bounds.")
        maximum = 0.5 * min(upper[0] - lower[0], upper[1] - lower[1])
        if radius < 0.0 or radius > maximum:
            raise ValueError(f"corner_radius must be between 0 and {maximum}.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "corner_radius", radius)

    @property
    def dim(self) -> int:
        return 2

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((self.lower[0], self.upper[0]), (self.lower[1], self.upper[1]))

    @property
    def area(self) -> float:
        width = self.upper[0] - self.lower[0]
        height = self.upper[1] - self.lower[1]
        return width * height - (4.0 - math.pi) * self.corner_radius**2

    @property
    def _center_half_inner(self) -> tuple[float, float, float, float]:
        center_x = 0.5 * (self.lower[0] + self.upper[0])
        center_y = 0.5 * (self.lower[1] + self.upper[1])
        half_x = 0.5 * (self.upper[0] - self.lower[0])
        half_y = 0.5 * (self.upper[1] - self.lower[1])
        return center_x, center_y, half_x - self.corner_radius, half_y - self.corner_radius

    def _signed_distance(self, points: Tensor) -> Tensor:
        center_x, center_y, inner_x, inner_y = self._center_half_inner
        local = torch.abs(points - points.new_tensor((center_x, center_y)))
        q = local - points.new_tensor((inner_x, inner_y))
        outside = torch.linalg.vector_norm(torch.clamp(q, min=0.0), dim=1)
        inside = torch.minimum(torch.maximum(q[:, 0], q[:, 1]), q.new_tensor(0.0))
        return outside + inside - self.corner_radius

    def sample_interior(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0:
            return torch.empty((0, 2), dtype=dtype, device=device)
        accepted: list[Tensor] = []
        accepted_count = 0
        while accepted_count < n:
            remaining = n - accepted_count
            unit = _rand((max(32, 2 * remaining), 2), generator=generator, dtype=dtype, device=device)
            lower = unit.new_tensor(self.lower)
            upper = unit.new_tensor(self.upper)
            candidates = lower + (upper - lower) * unit
            selected = candidates[self.contains(candidates)][:remaining]
            accepted.append(selected)
            accepted_count += selected.shape[0]
        return torch.cat(accepted, dim=0)

    def sample(
        self,
        n: int,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if seed is not None:
            if generator is not None:
                raise ValueError("Pass seed or generator, not both.")
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        return self.sample_interior(n, generator=generator, dtype=dtype, device=device)

    def sample_boundary(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0:
            return torch.empty((0, 2), dtype=dtype, device=device)
        radius = self.corner_radius
        width = self.upper[0] - self.lower[0]
        height = self.upper[1] - self.lower[1]
        straight_x = max(0.0, width - 2.0 * radius)
        straight_y = max(0.0, height - 2.0 * radius)
        components: list[tuple[str, float, float]] = []
        for name, length, start_angle in (
            ("bottom", straight_x, 0.0), ("arc_br", 0.5 * math.pi * radius, -0.5 * math.pi),
            ("right", straight_y, 0.0), ("arc_tr", 0.5 * math.pi * radius, 0.0),
            ("top", straight_x, 0.0), ("arc_tl", 0.5 * math.pi * radius, 0.5 * math.pi),
            ("left", straight_y, 0.0), ("arc_bl", 0.5 * math.pi * radius, math.pi),
        ):
            if length > 0.0:
                components.append((name, length, start_angle))
        total = sum(component[1] for component in components)
        distance = _rand((n,), generator=generator, dtype=dtype, device=device) * total
        points = torch.empty((n, 2), dtype=dtype, device=device)
        previous = 0.0
        corner_centers = {
            "arc_br": (self.upper[0] - radius, self.lower[1] + radius),
            "arc_tr": (self.upper[0] - radius, self.upper[1] - radius),
            "arc_tl": (self.lower[0] + radius, self.upper[1] - radius),
            "arc_bl": (self.lower[0] + radius, self.lower[1] + radius),
        }
        for component_index, (name, length, start_angle) in enumerate(components):
            upper = previous + length
            mask = (distance >= previous) & ((distance < upper) if component_index + 1 < len(components) else (distance <= upper))
            local = distance[mask] - previous
            if name == "bottom":
                points[mask, 0] = self.lower[0] + radius + local
                points[mask, 1] = self.lower[1]
            elif name == "right":
                points[mask, 0] = self.upper[0]
                points[mask, 1] = self.lower[1] + radius + local
            elif name == "top":
                points[mask, 0] = self.upper[0] - radius - local
                points[mask, 1] = self.upper[1]
            elif name == "left":
                points[mask, 0] = self.lower[0]
                points[mask, 1] = self.upper[1] - radius - local
            else:
                center_x, center_y = corner_centers[name]
                angle = start_angle + local / radius
                points[mask, 0] = center_x + radius * torch.cos(angle)
                points[mask, 1] = center_y + radius * torch.sin(angle)
            previous = upper
        return points

    def boundary(
        self,
        n: int,
        *,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        return self.sample_boundary(n, generator=generator, dtype=dtype, device=device)

    def contains(self, points: Tensor, *, atol: float = 0.0) -> Tensor:
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("RoundedRectangle points must have shape (N, 2).")
        return self._signed_distance(points) <= atol

    def on_boundary(self, points: Tensor, *, atol: float = 1e-7) -> Tensor:
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("RoundedRectangle points must have shape (N, 2).")
        return torch.abs(self._signed_distance(points)) <= atol

    def boundary_normal(self, points: Tensor, *, atol: float = 1e-5) -> Tensor:
        if not torch.all(self.on_boundary(points, atol=atol)):
            raise ValueError("boundary_normal requires points on the rounded-rectangle boundary.")
        center_x, center_y, inner_x, inner_y = self._center_half_inner
        local = points - points.new_tensor((center_x, center_y))
        absolute = torch.abs(local)
        signs = torch.sign(local)
        signs[signs == 0] = 1
        corner = (absolute[:, 0] > inner_x + atol) & (absolute[:, 1] > inner_y + atol)
        normals = torch.zeros_like(points)
        corner_center = signs * points.new_tensor((inner_x, inner_y))
        direction = local - corner_center
        if torch.any(corner):
            normals[corner] = direction[corner] / torch.linalg.vector_norm(
                direction[corner], dim=1, keepdim=True
            )
        side = ~corner
        if torch.any(side):
            x_distance = torch.abs(absolute[:, 0] - (inner_x + self.corner_radius))
            y_distance = torch.abs(absolute[:, 1] - (inner_y + self.corner_radius))
            use_x = side & (x_distance <= y_distance)
            use_y = side & ~use_x
            normals[use_x, 0] = signs[use_x, 0]
            normals[use_y, 1] = signs[use_y, 1]
        return normals


@dataclass(frozen=True)
class RectangleWithCircularHoles:
    """An axis-aligned rectangle with non-overlapping circular holes.

    Interior points are sampled uniformly by rejection sampling. Boundary
    points are sampled uniformly by arc length across the four outer edges and
    every circular hole.  This geometry is sufficient for the twin-tunnel
    consolidation example while remaining independent of that PDE.
    """

    lower: tuple[float, float]
    upper: tuple[float, float]
    holes: tuple[tuple[float, float, float], ...] = ()
    stratified_boundary: bool = False

    def __post_init__(self) -> None:
        lower = tuple(float(value) for value in self.lower)
        upper = tuple(float(value) for value in self.upper)
        holes = tuple(tuple(float(value) for value in hole) for hole in self.holes)
        if len(lower) != 2 or len(upper) != 2:
            raise ValueError("Rectangle bounds must each contain two coordinates.")
        if not all(math.isfinite(value) for value in lower + upper):
            raise ValueError("Rectangle bounds must be finite.")
        if not isinstance(self.stratified_boundary, bool):
            raise TypeError("stratified_boundary must be bool.")
        if not (upper[0] > lower[0] and upper[1] > lower[1]):
            raise ValueError("Rectangle upper bounds must exceed lower bounds.")
        for index, hole in enumerate(holes):
            if len(hole) != 3:
                raise ValueError("Each circular hole must be (center_x, center_y, radius).")
            center_x, center_y, radius = hole
            if radius <= 0.0 or not all(math.isfinite(value) for value in hole):
                raise ValueError("Circular-hole centers must be finite and radius positive.")
            if not (
                lower[0] < center_x - radius
                and center_x + radius < upper[0]
                and lower[1] < center_y - radius
                and center_y + radius < upper[1]
            ):
                raise ValueError(f"Circular hole {index} must lie strictly inside the rectangle.")
        for left_index, left in enumerate(holes):
            for right in holes[left_index + 1 :]:
                separation = math.hypot(left[0] - right[0], left[1] - right[1])
                if separation <= left[2] + right[2]:
                    raise ValueError("Circular holes must not overlap or touch.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "holes", holes)

    @property
    def dim(self) -> int:
        """Coordinate dimension."""

        return 2

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ``((x_min, x_max), (y_min, y_max))``."""

        return ((self.lower[0], self.upper[0]), (self.lower[1], self.upper[1]))

    @property
    def area(self) -> float:
        """Area of the rectangle after subtracting the circular holes."""

        rectangle = (self.upper[0] - self.lower[0]) * (self.upper[1] - self.lower[1])
        return rectangle - sum(math.pi * hole[2] ** 2 for hole in self.holes)

    def sample_interior(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Uniformly sample soil points while rejecting points inside holes."""

        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0:
            return torch.empty((0, 2), dtype=dtype, device=device)
        accepted: list[Tensor] = []
        accepted_count = 0
        while accepted_count < n:
            remaining = n - accepted_count
            unit = _rand(
                (max(32, 2 * remaining), 2),
                generator=generator,
                dtype=dtype,
                device=device,
            )
            lower = unit.new_tensor(self.lower)
            upper = unit.new_tensor(self.upper)
            candidates = lower + (upper - lower) * unit
            mask = torch.ones(candidates.shape[0], dtype=torch.bool, device=candidates.device)
            for center_x, center_y, radius in self.holes:
                center = candidates.new_tensor((center_x, center_y))
                mask &= (candidates - center).square().sum(dim=1) > radius**2
            selected = candidates[mask][:remaining]
            accepted.append(selected)
            accepted_count += selected.shape[0]
        return torch.cat(accepted, dim=0)

    def sample(
        self,
        n: int,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Convenience alias for :meth:`sample_interior`."""

        if seed is not None:
            if generator is not None:
                raise ValueError("Pass seed or generator, not both.")
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        return self.sample_interior(n, generator=generator, dtype=dtype, device=device)

    def sample_boundary(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Sample the outer rectangle and all circular perimeters.

        By default use the historical arc-length random sampling. With
        ``stratified_boundary=True``, reserve one point for each of the four
        edges and each hole, then allocate the remainder by perimeter length.
        This prevents empty selectors for separately prescribed boundaries.
        """

        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0:
            return torch.empty((0, 2), dtype=dtype, device=device)
        width = self.upper[0] - self.lower[0]
        height = self.upper[1] - self.lower[1]
        segment_lengths = (width, height, width, height) + tuple(
            2.0 * math.pi * hole[2] for hole in self.holes
        )
        cumulative: list[float] = []
        total = 0.0
        for length in segment_lengths:
            total += length
            cumulative.append(total)
        if self.stratified_boundary:
            if n < len(segment_lengths):
                raise ValueError(
                    f"Stratified boundary sampling needs at least {len(segment_lengths)} "
                    "points (four rectangle edges plus one per hole)."
                )
            shares = [(n - len(segment_lengths)) * length / total for length in segment_lengths]
            counts = [1 + math.floor(share) for share in shares]
            order = sorted(range(len(counts)), key=lambda i: shares[i] - math.floor(shares[i]), reverse=True)
            for index in order[:n - sum(counts)]:
                counts[index] += 1
            distances = []
            offset = 0.0
            for length, count in zip(segment_lengths, counts):
                unit = _rand((count,), generator=generator, dtype=dtype, device=device)
                # Avoid exact corners and prevent rounding into the next component.
                eps = torch.finfo(dtype).eps
                local = unit.clamp(eps, 1.0 - eps) * length + offset
                start = local.new_tensor(offset)
                end = local.new_tensor(offset + length)
                distances.append(local.clamp(torch.nextafter(start, end), torch.nextafter(end, start)))
                offset += length
            distance = torch.cat(distances)
            permutation = torch.randperm(n, generator=generator, device="cpu").to(distance.device)
            distance = distance[permutation]
        else:
            distance = _rand((n,), generator=generator, dtype=dtype, device=device) * total
        points = torch.empty((n, 2), dtype=dtype, device=device)
        previous = 0.0
        for index, upper_distance in enumerate(cumulative):
            mask = (distance >= previous) & (distance < upper_distance)
            local = distance[mask] - previous
            if index == 0:  # bottom, left to right
                points[mask, 0] = self.lower[0] + local
                points[mask, 1] = self.lower[1]
            elif index == 1:  # right, bottom to top
                points[mask, 0] = self.upper[0]
                points[mask, 1] = self.lower[1] + local
            elif index == 2:  # top, right to left
                points[mask, 0] = self.upper[0] - local
                points[mask, 1] = self.upper[1]
            elif index == 3:  # left, top to bottom
                points[mask, 0] = self.lower[0]
                points[mask, 1] = self.upper[1] - local
            else:
                center_x, center_y, radius = self.holes[index - 4]
                angle = local / radius
                points[mask, 0] = center_x + radius * torch.cos(angle)
                points[mask, 1] = center_y + radius * torch.sin(angle)
            previous = upper_distance
        return points

    def boundary(
        self,
        n: int,
        *,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Convenience alias for :meth:`sample_boundary`."""

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        return self.sample_boundary(n, generator=generator, dtype=dtype, device=device)

    def contains(self, points: Tensor, *, atol: float = 0.0) -> Tensor:
        """Return a mask for points in the rectangle but outside every hole."""

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Rectangle points must have shape (N, 2).")
        inside = (
            (points[:, 0] >= self.lower[0] - atol)
            & (points[:, 0] <= self.upper[0] + atol)
            & (points[:, 1] >= self.lower[1] - atol)
            & (points[:, 1] <= self.upper[1] + atol)
        )
        for center_x, center_y, radius in self.holes:
            center = points.new_tensor((center_x, center_y))
            inside &= (points - center).square().sum(dim=1) >= max(0.0, radius - atol) ** 2
        return inside

    def on_boundary(self, points: Tensor, *, atol: float = 1e-7) -> Tensor:
        """Return a mask for outer edges or circular-hole perimeters."""

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Rectangle points must have shape (N, 2).")
        on_outer = (
            torch.isclose(points[:, 0], points.new_tensor(self.lower[0]), atol=atol, rtol=0.0)
            | torch.isclose(points[:, 0], points.new_tensor(self.upper[0]), atol=atol, rtol=0.0)
            | torch.isclose(points[:, 1], points.new_tensor(self.lower[1]), atol=atol, rtol=0.0)
            | torch.isclose(points[:, 1], points.new_tensor(self.upper[1]), atol=atol, rtol=0.0)
        )
        on_hole = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        for center_x, center_y, radius in self.holes:
            center = points.new_tensor((center_x, center_y))
            distance = torch.linalg.vector_norm(points - center, dim=1)
            on_hole |= torch.isclose(
                distance, distance.new_tensor(radius), atol=atol, rtol=0.0
            )
        return (on_outer | on_hole) & self.contains(points, atol=atol)

    def boundary_normal(self, points: Tensor, *, atol: float = 1e-5) -> Tensor:
        """Outward unit normals; hole normals point into the removed disks.

        At rectangle corners use the normalized sum of adjacent edge normals.
        Callers prescribing different edge conditions should sample away from
        corners, where the smooth-boundary normal is not uniquely defined.
        """
        if not torch.all(self.on_boundary(points, atol=atol)):
            raise ValueError("boundary_normal requires points on the rectangle/hole boundary.")
        normal = torch.zeros_like(points)
        for axis in range(2):
            low = torch.isclose(points[:, axis], points.new_tensor(self.lower[axis]), atol=atol, rtol=0.0)
            high = torch.isclose(points[:, axis], points.new_tensor(self.upper[axis]), atol=atol, rtol=0.0)
            normal[low, axis] -= 1.0
            normal[high, axis] += 1.0
        for center_x, center_y, radius in self.holes:
            radial = points - points.new_tensor((center_x, center_y))
            length = torch.linalg.vector_norm(radial, dim=1)
            mask = torch.isclose(length, length.new_tensor(radius), atol=atol, rtol=0.0)
            normal[mask] = -radial[mask] / length[mask, None]
        lengths = torch.linalg.vector_norm(normal, dim=1, keepdim=True)
        if torch.any(lengths == 0):
            raise ValueError("Could not determine boundary normal; check geometry scale and tolerance.")
        return normal / lengths


@dataclass(frozen=True)
class Polygon:
    """A validated two-dimensional polygon with optional polygonal holes.

    The outer ring may be convex or concave.  Rings are automatically
    normalised to counter-clockwise orientation for the outer boundary and
    clockwise orientation for holes, so :meth:`boundary_normal` consistently
    returns the outward normal of the computational domain.  The class has no
    optional geometry dependency and can also read CSV, JSON, and GeoJSON
    polygon files via :meth:`from_file`.
    """

    vertices: Ring
    holes: tuple[Ring, ...] = ()

    def __post_init__(self) -> None:
        outer = _normalise_ring(self.vertices, name="Outer polygon ring", clockwise=False)
        holes = tuple(
            _normalise_ring(hole, name=f"Polygon hole {index}", clockwise=True)
            for index, hole in enumerate(self.holes, start=1)
        )
        for index, hole in enumerate(holes, start=1):
            if _rings_intersect(outer, hole) or not all(
                _point_in_ring(point, outer, include_boundary=False) for point in hole
            ):
                raise ValueError(f"Polygon hole {index} must lie strictly inside the outer ring.")
        for left_index, left in enumerate(holes):
            for right_index, right in enumerate(holes[left_index + 1 :], start=left_index + 2):
                if (
                    _rings_intersect(left, right)
                    or _point_in_ring(left[0], right, include_boundary=True)
                    or _point_in_ring(right[0], left, include_boundary=True)
                ):
                    raise ValueError(
                        f"Polygon holes {left_index + 1} and {right_index} overlap or touch."
                    )
        if abs(_signed_ring_area(outer)) <= sum(
            abs(_signed_ring_area(hole)) for hole in holes
        ):
            raise ValueError("Polygon holes consume the entire outer area.")
        object.__setattr__(self, "vertices", outer)
        object.__setattr__(self, "holes", holes)

    @classmethod
    def from_file(cls, path: str | Path) -> "Polygon":
        """Load a polygon from CSV, JSON, or GeoJSON.

        JSON accepts ``{"outer": [[x, y], ...], "holes": [...]}`` and
        standard GeoJSON ``Polygon``/``Feature`` objects.  CSV accepts either
        ``x,y`` rows for one outer ring or ``ring,x,y`` rows where the ring is
        named ``outer``, ``hole_1``, and so on.  Blank lines in a two-column
        CSV start a new hole ring.
        """

        source = Path(path)
        suffix = source.suffix.lower()
        if suffix == ".csv":
            outer, holes = _polygon_coordinates_from_csv(source)
        elif suffix in {".json", ".geojson"}:
            with source.open("r", encoding="utf-8-sig") as stream:
                outer, holes = _polygon_coordinates_from_json(json.load(stream))
        else:
            raise ValueError("Polygon files must use .csv, .json, or .geojson.")
        return cls(outer, tuple(holes))

    @property
    def dim(self) -> int:
        """Coordinate dimension."""

        return 2

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ``((x_min, x_max), (y_min, y_max))``."""

        x_values = [point[0] for point in self.vertices]
        y_values = [point[1] for point in self.vertices]
        return ((min(x_values), max(x_values)), (min(y_values), max(y_values)))

    @property
    def area(self) -> float:
        """Area after subtracting all holes."""

        return abs(_signed_ring_area(self.vertices)) - sum(
            abs(_signed_ring_area(hole)) for hole in self.holes
        )

    @property
    def rings(self) -> tuple[Ring, ...]:
        """Outer ring followed by hole rings."""

        return (self.vertices,) + self.holes

    def _segments(self) -> tuple[tuple[Point2D, Point2D], ...]:
        return tuple(
            (start, end)
            for ring in self.rings
            for start, end in zip(ring, ring[1:] + ring[:1])
        )

    def sample_interior(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Uniformly sample interior points by bounding-box rejection."""

        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0:
            return torch.empty((0, 2), dtype=dtype, device=device)
        (x_min, x_max), (y_min, y_max) = self.bounds
        box_area = (x_max - x_min) * (y_max - y_min)
        acceptance = max(min(self.area / box_area, 1.0), 1e-4)
        accepted: list[Tensor] = []
        accepted_count = 0
        attempts = 0
        candidates_drawn = 0
        candidate_limit = max(5_000_000, 100 * n)
        while accepted_count < n:
            attempts += 1
            if attempts > 1000:
                raise RuntimeError("Unable to sample polygon interior; check polygon validity.")
            remaining = n - accepted_count
            batch = max(32, min(250_000, math.ceil(1.25 * remaining / acceptance)))
            candidates_drawn += batch
            if candidates_drawn > candidate_limit:
                raise RuntimeError(
                    "Polygon occupies too little of its bounding box for safe rejection "
                    "sampling; simplify or partition the geometry."
                )
            unit = _rand(
                (batch, 2), generator=generator, dtype=dtype, device=device
            )
            lower = unit.new_tensor((x_min, y_min))
            upper = unit.new_tensor((x_max, y_max))
            candidates = lower + (upper - lower) * unit
            selected = candidates[self.contains(candidates)][:remaining]
            if selected.numel():
                accepted.append(selected)
                accepted_count += selected.shape[0]
        return torch.cat(accepted, dim=0)

    def sample(
        self,
        n: int,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Convenience alias for :meth:`sample_interior`."""

        if seed is not None:
            if generator is not None:
                raise ValueError("Pass seed or generator, not both.")
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        return self.sample_interior(
            n, generator=generator, dtype=dtype, device=device
        )

    def sample_boundary(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Sample outer and hole edges approximately uniformly by arc length.

        When the requested count is at least the number of polygon edges, every
        edge receives at least one point.  This makes independently selected
        per-edge boundary conditions reliable even for short edges.
        """

        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0:
            return torch.empty((0, 2), dtype=dtype, device=device)
        segments = self._segments()
        lengths = [math.dist(start, end) for start, end in segments]
        total = sum(lengths)
        if n >= len(segments):
            remaining = n - len(segments)
            shares = [remaining * length / total for length in lengths]
            counts = [1 + math.floor(share) for share in shares]
            leftover = n - sum(counts)
            order = sorted(
                range(len(segments)),
                key=lambda index: shares[index] - math.floor(shares[index]),
                reverse=True,
            )
            for index in order[:leftover]:
                counts[index] += 1
            samples: list[Tensor] = []
            for (start, end), count in zip(segments, counts):
                fraction = _rand(
                    (count, 1), generator=generator, dtype=dtype, device=device
                )
                start_tensor = fraction.new_tensor(start)
                end_tensor = fraction.new_tensor(end)
                samples.append(start_tensor + fraction * (end_tensor - start_tensor))
            points = torch.cat(samples, dim=0)
            permutation = torch.randperm(n, generator=generator, device="cpu").to(
                device=points.device
            )
            return points[permutation]
        distance = _rand((n,), generator=generator, dtype=dtype, device=device) * total
        points = torch.empty((n, 2), dtype=dtype, device=device)
        previous = 0.0
        for (start, end), length in zip(segments, lengths):
            upper = previous + length
            mask = (distance >= previous) & (distance < upper)
            fraction = ((distance[mask] - previous) / length).unsqueeze(1)
            start_tensor = points.new_tensor(start)
            end_tensor = points.new_tensor(end)
            points[mask] = start_tensor + fraction * (end_tensor - start_tensor)
            previous = upper
        return points

    def boundary(
        self,
        n: int,
        *,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Convenience alias for :meth:`sample_boundary`."""

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        return self.sample_boundary(
            n, generator=generator, dtype=dtype, device=device
        )

    @staticmethod
    def _ring_masks(points: Tensor, ring: Ring, *, atol: float) -> tuple[Tensor, Tensor]:
        x = points[:, 0]
        y = points[:, 1]
        inside = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        boundary = torch.zeros_like(inside)
        for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
            dx = x1 - x0
            dy = y1 - y0
            length_squared = dx * dx + dy * dy
            projection = ((x - x0) * dx + (y - y0) * dy) / length_squared
            projection_clamped = projection.clamp(0.0, 1.0)
            closest_x = x0 + projection_clamped * dx
            closest_y = y0 + projection_clamped * dy
            boundary |= torch.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2) <= atol
            crosses = (y0 > y) != (y1 > y)
            denominator = dy if dy != 0.0 else 1.0
            crossing_x = x0 + (y - y0) * dx / denominator
            inside ^= crosses & (x < crossing_x)
        return inside, boundary

    def contains(self, points: Tensor, *, atol: float = 0.0) -> Tensor:
        """Return a mask for points inside the outer ring and outside holes."""

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Polygon points must have shape (N, 2).")
        outer_inside, outer_boundary = self._ring_masks(
            points, self.vertices, atol=max(atol, 0.0)
        )
        inside = outer_inside | outer_boundary
        for hole in self.holes:
            hole_inside, hole_boundary = self._ring_masks(
                points, hole, atol=max(atol, 0.0)
            )
            inside &= ~(hole_inside | hole_boundary)
        return inside

    def on_boundary(self, points: Tensor, *, atol: float = 1e-7) -> Tensor:
        """Return a mask for outer or hole boundary segments."""

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Polygon points must have shape (N, 2).")
        mask = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        for ring in self.rings:
            _, ring_boundary = self._ring_masks(points, ring, atol=atol)
            mask |= ring_boundary
        return mask

    def boundary_normal(self, points: Tensor, *, atol: float = 1e-5) -> Tensor:
        """Return the outward unit normal at points on polygon edges."""

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Polygon points must have shape (N, 2).")
        if points.shape[0] == 0:
            return torch.empty_like(points)
        segments = self._segments()
        distances: list[Tensor] = []
        normals: list[Point2D] = []
        for start, end in segments:
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length_squared = dx * dx + dy * dy
            projection = (
                (points[:, 0] - start[0]) * dx
                + (points[:, 1] - start[1]) * dy
            ) / length_squared
            projection = projection.clamp(0.0, 1.0)
            closest_x = start[0] + projection * dx
            closest_y = start[1] + projection * dy
            distances.append(
                (points[:, 0] - closest_x) ** 2
                + (points[:, 1] - closest_y) ** 2
            )
            length = math.sqrt(length_squared)
            normals.append((dy / length, -dx / length))
        distance_matrix = torch.stack(distances, dim=1)
        closest_distance, closest_index = distance_matrix.min(dim=1)
        if torch.any(closest_distance > atol**2):
            raise ValueError("boundary_normal requires points on the polygon boundary.")
        normal_table = points.new_tensor(normals)
        return normal_table[closest_index]


@dataclass(frozen=True)
class ClosedSpline:
    """A smooth closed cardinal spline controlled by editable points.

    Consecutive control points are joined with cubic Bezier segments.  The
    ``tension`` parameter lies in ``[0, 1]``: zero gives the usual smooth
    Catmull--Rom curve and increasing it shortens the tangent handles.  The
    boundary is sampled on the cubic segments and its normal is obtained from
    the analytic first derivative; a polyline is used only for inside/outside
    tests and never as the represented boundary.
    """

    control_points: Ring
    tension: float = 0.2
    resolution: int = 32

    def __post_init__(self) -> None:
        points = _normalise_ring(
            self.control_points, name="Closed-spline control ring", clockwise=False
        )
        tension = float(self.tension)
        resolution = int(self.resolution)
        if len(points) < 4:
            raise ValueError("ClosedSpline requires at least four control points.")
        if not math.isfinite(tension) or not 0.0 <= tension <= 1.0:
            raise ValueError("ClosedSpline tension must lie between 0 and 1.")
        if not 8 <= resolution <= 256:
            raise ValueError("ClosedSpline resolution must lie between 8 and 256.")
        object.__setattr__(self, "control_points", points)
        object.__setattr__(self, "tension", tension)
        object.__setattr__(self, "resolution", resolution)
        # Catch strong overshoot or a bad control ring early.  Eight samples
        # per segment are enough for validation while keeping the check small.
        _normalise_ring(
            self._dense_ring(per_segment=8),
            name="Closed-spline boundary",
            clockwise=False,
        )

    @property
    def dim(self) -> int:
        return 2

    def _bezier_controls(self, index: int) -> tuple[Point2D, Point2D, Point2D, Point2D]:
        points = self.control_points
        count = len(points)
        previous = points[(index - 1) % count]
        start = points[index]
        end = points[(index + 1) % count]
        following = points[(index + 2) % count]
        scale = (1.0 - self.tension) / 6.0
        first = (
            start[0] + scale * (end[0] - previous[0]),
            start[1] + scale * (end[1] - previous[1]),
        )
        second = (
            end[0] - scale * (following[0] - start[0]),
            end[1] - scale * (following[1] - start[1]),
        )
        return start, first, second, end

    def _point(self, index: int, parameter: float) -> Point2D:
        start, first, second, end = self._bezier_controls(index)
        t = float(parameter)
        u = 1.0 - t
        return (
            u**3 * start[0] + 3.0 * u**2 * t * first[0]
            + 3.0 * u * t**2 * second[0] + t**3 * end[0],
            u**3 * start[1] + 3.0 * u**2 * t * first[1]
            + 3.0 * u * t**2 * second[1] + t**3 * end[1],
        )

    def _dense_ring(self, *, per_segment: int | None = None) -> Ring:
        steps = int(per_segment or self.resolution)
        return tuple(
            self._point(index, step / steps)
            for index in range(len(self.control_points))
            for step in range(steps)
        )

    def _evaluate_tensor(
        self,
        indices: Tensor,
        parameters: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        table = parameters.new_tensor(self.control_points)
        count = table.shape[0]
        previous = table[(indices - 1) % count]
        start = table[indices]
        end = table[(indices + 1) % count]
        following = table[(indices + 2) % count]
        scale = (1.0 - self.tension) / 6.0
        first = start + scale * (end - previous)
        second = end - scale * (following - start)
        t = parameters.unsqueeze(1)
        u = 1.0 - t
        point = u**3 * start + 3.0 * u**2 * t * first + 3.0 * u * t**2 * second + t**3 * end
        derivative = (
            3.0 * u**2 * (first - start)
            + 6.0 * u * t * (second - first)
            + 3.0 * t**2 * (end - second)
        )
        second_derivative = 6.0 * u * (second - 2.0 * first + start) + 6.0 * t * (
            end - 2.0 * second + first
        )
        return point, derivative, second_derivative

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        ring = self._dense_ring()
        x_values = [point[0] for point in ring]
        y_values = [point[1] for point in ring]
        return ((min(x_values), max(x_values)), (min(y_values), max(y_values)))

    @property
    def area(self) -> float:
        return abs(_signed_ring_area(self._dense_ring()))

    def contains(self, points: Tensor, *, atol: float = 0.0) -> Tensor:
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("ClosedSpline points must have shape (N, 2).")
        inside, boundary = Polygon._ring_masks(
            points, self._dense_ring(), atol=max(float(atol), 0.0)
        )
        return inside | boundary

    def sample_interior(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0:
            return torch.empty((0, 2), dtype=dtype, device=device)
        (x_min, x_max), (y_min, y_max) = self.bounds
        box_area = (x_max - x_min) * (y_max - y_min)
        acceptance = max(min(self.area / box_area, 1.0), 1e-4)
        accepted: list[Tensor] = []
        accepted_count = 0
        attempts = 0
        while accepted_count < n:
            attempts += 1
            if attempts > 1000:
                raise RuntimeError("Unable to sample spline interior; check curve validity.")
            remaining = n - accepted_count
            batch = max(32, min(250_000, math.ceil(1.25 * remaining / acceptance)))
            unit = _rand((batch, 2), generator=generator, dtype=dtype, device=device)
            lower = unit.new_tensor((x_min, y_min))
            upper = unit.new_tensor((x_max, y_max))
            selected = (lower + (upper - lower) * unit)
            selected = selected[self.contains(selected)][:remaining]
            if selected.numel():
                accepted.append(selected)
                accepted_count += selected.shape[0]
        return torch.cat(accepted, dim=0)

    def sample(
        self,
        n: int,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if seed is not None:
            if generator is not None:
                raise ValueError("Pass seed or generator, not both.")
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        return self.sample_interior(n, generator=generator, dtype=dtype, device=device)

    def _arc_table(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        count = len(self.control_points)
        steps = self.resolution
        indices = torch.arange(count, device=device).repeat_interleave(steps)
        parameters = torch.arange(steps, dtype=dtype, device=device).repeat(count) / steps
        points, _, _ = self._evaluate_tensor(indices, parameters)
        lengths = torch.linalg.vector_norm(torch.roll(points, -1, dims=0) - points, dim=1)
        cumulative = torch.cumsum(lengths, dim=0)
        return indices, parameters, points, cumulative

    def sample_boundary(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0:
            return torch.empty((0, 2), dtype=dtype, device=device)
        indices, parameters, _, cumulative = self._arc_table(dtype=dtype, device=device)
        total = cumulative[-1]
        distance = _rand((n,), generator=generator, dtype=dtype, device=device) * total
        table_index = torch.searchsorted(cumulative, distance, right=True).clamp_max(
            cumulative.numel() - 1
        )
        previous = torch.where(
            table_index > 0,
            cumulative[(table_index - 1).clamp_min(0)],
            torch.zeros_like(distance),
        )
        segment_length = cumulative[table_index] - previous
        fraction = (distance - previous) / segment_length.clamp_min(torch.finfo(dtype).eps)
        selected_indices = indices[table_index]
        selected_parameters = parameters[table_index] + fraction / self.resolution
        points, _, _ = self._evaluate_tensor(selected_indices, selected_parameters)
        return points

    def boundary(
        self,
        n: int,
        *,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(int(seed))
        return self.sample_boundary(n, generator=generator, dtype=dtype, device=device)

    def _nearest(self, points: Tensor) -> tuple[Tensor, Tensor]:
        indices, parameters, samples, _ = self._arc_table(
            dtype=points.dtype, device=points.device
        )
        nearest = torch.cdist(points, samples).argmin(dim=1)
        base_segment = indices[nearest]
        base_parameter = parameters[nearest]
        count = len(self.control_points)
        segment = torch.stack(
            ((base_segment - 1) % count, base_segment, (base_segment + 1) % count), dim=1
        ).reshape(-1)
        parameter = torch.stack(
            (torch.ones_like(base_parameter), base_parameter, torch.zeros_like(base_parameter)),
            dim=1,
        ).reshape(-1)
        repeated_points = points.repeat_interleave(3, dim=0)
        for _ in range(8):
            curve, derivative, second = self._evaluate_tensor(segment, parameter)
            delta = curve - repeated_points
            numerator = torch.sum(delta * derivative, dim=1)
            denominator = torch.sum(derivative * derivative + delta * second, dim=1)
            safe = torch.where(
                torch.abs(denominator) > torch.finfo(points.dtype).eps,
                denominator,
                torch.ones_like(denominator),
            )
            parameter = (parameter - numerator / safe).clamp(0.0, 1.0)
        curve, derivative, _ = self._evaluate_tensor(segment, parameter)
        distances = torch.linalg.vector_norm(curve - repeated_points, dim=1).reshape(-1, 3)
        choice = distances.argmin(dim=1)
        offsets = torch.arange(points.shape[0], device=points.device) * 3 + choice
        return curve[offsets], derivative[offsets]

    def on_boundary(self, points: Tensor, *, atol: float = 1e-7) -> Tensor:
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("ClosedSpline points must have shape (N, 2).")
        nearest, _ = self._nearest(points)
        return torch.linalg.vector_norm(points - nearest, dim=1) <= atol

    def boundary_normal(self, points: Tensor, *, atol: float = 1e-5) -> Tensor:
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("ClosedSpline points must have shape (N, 2).")
        nearest, tangent = self._nearest(points)
        if torch.any(torch.linalg.vector_norm(points - nearest, dim=1) > atol):
            raise ValueError("boundary_normal requires points on the spline boundary.")
        normal = torch.stack((tangent[:, 1], -tangent[:, 0]), dim=1)
        return normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True)


@dataclass(frozen=True, init=False)
class SpaceTimeDomain:
    """A Cartesian product of a spatial geometry and a time interval.

    Spatial coordinates come first and time is always the last column.  Thus
    the order is ``(x, t)`` for an interval and ``(x, y, t)`` for a 2D space.
    """

    space: Interval | Disk | Ellipse | RoundedRectangle | RectangleWithCircularHoles | Polygon | ClosedSpline
    time: Interval

    def __init__(
        self,
        space: Interval | Disk | Ellipse | RoundedRectangle | RectangleWithCircularHoles | Polygon | ClosedSpline,
        time: Interval | float = 0.0,
        time_upper: float | None = None,
    ) -> None:
        if not isinstance(
            space, (Interval, Disk, Ellipse, RoundedRectangle, RectangleWithCircularHoles, Polygon, ClosedSpline)
        ):
            raise TypeError("SpaceTimeDomain requires a supported spatial geometry.")
        if isinstance(time, Interval):
            if time_upper is not None:
                raise ValueError("Do not pass time_upper when time is already an Interval.")
            time_interval = time
        else:
            if time_upper is None:
                raise ValueError("Pass both time lower and time_upper, or an Interval.")
            time_interval = Interval(float(time), float(time_upper))
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "time", time_interval)

    @property
    def dim(self) -> int:
        """Total coordinate dimension, spatial dimensions plus one time."""

        return self.space.dim + 1

    @property
    def time_index(self) -> int:
        """Column index of time in sampled coordinates."""

        return self.space.dim

    def sample_interior(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Uniformly sample the open space-time volume."""

        x = self.space.sample_interior(
            n, generator=generator, dtype=dtype, device=device
        )
        t = self.time.sample_interior(
            n, generator=generator, dtype=dtype, device=device
        )
        return torch.cat((x, t), dim=1)

    def sample(
        self,
        n: int,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Convenience alias for :meth:`sample_interior` with an optional seed."""

        if seed is not None:
            if generator is not None:
                raise ValueError("Pass seed or generator, not both.")
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        return self.sample_interior(
            n, generator=generator, dtype=dtype, device=device
        )

    def sample_boundary(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Sample the spatial boundary across the time interval."""

        x = self.space.sample_boundary(
            n, generator=generator, dtype=dtype, device=device
        )
        t = self.time.sample_interior(
            n, generator=generator, dtype=dtype, device=device
        )
        return torch.cat((x, t), dim=1)

    def boundary(
        self,
        n: int,
        *,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Convenience alias for spatial-boundary sampling."""

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        return self.sample_boundary(
            n, generator=generator, dtype=dtype, device=device
        )

    def sample_initial(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Sample spatial points at the initial time."""

        x = self.space.sample_interior(
            n, generator=generator, dtype=dtype, device=device
        )
        t = torch.full((n, 1), self.time.lower, dtype=dtype, device=device)
        return torch.cat((x, t), dim=1)

    def contains(self, points: Tensor, *, atol: float = 0.0) -> Tensor:
        """Return a mask for points inside both space and time bounds."""

        if points.ndim != 2 or points.shape[1] != self.dim:
            raise ValueError(f"Space-time points must have shape (N, {self.dim}).")
        in_space = self.space.contains(points[:, : self.time_index], atol=atol)
        in_time = self.time.contains(points[:, self.time_index :], atol=atol)
        return in_space & in_time

    def on_spatial_boundary(self, points: Tensor, *, atol: float = 1e-7) -> Tensor:
        """Return a mask for the spatial boundary at any valid time."""

        if points.ndim != 2 or points.shape[1] != self.dim:
            raise ValueError(f"Space-time points must have shape (N, {self.dim}).")
        return self.space.on_boundary(points[:, : self.time_index], atol=atol)

    def on_initial(self, points: Tensor, *, atol: float = 1e-7) -> Tensor:
        """Return a mask for the initial-time plane."""

        if points.ndim != 2 or points.shape[1] != self.dim:
            raise ValueError(f"Space-time points must have shape (N, {self.dim}).")
        return torch.isclose(
            points[:, self.time_index],
            points.new_tensor(self.time.lower),
            atol=atol,
            rtol=0.0,
        )
