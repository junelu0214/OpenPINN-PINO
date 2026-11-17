"""Standard, inspectable experiment artifact writing."""

from __future__ import annotations

import base64
import csv
import hashlib
import html
import json
import math
import struct
import traceback
import uuid
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class ArtifactWriter:
    """Write OpenPINN's minimum transparent output contract."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_config(self, config: Mapping[str, Any]) -> Path:
        """Write ``config.yaml``."""

        path = self.output_dir / "config.yaml"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(_jsonable(config), handle, allow_unicode=True, sort_keys=False)
        return path

    def write_json(self, filename: str, values: Mapping[str, Any]) -> Path:
        """Write an indented UTF-8 JSON mapping."""

        path = self.output_dir / filename
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                _jsonable(values),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        return path

    def write_metrics(self, metrics: Mapping[str, Any]) -> Path:
        """Write ``metrics.json``."""

        return self.write_json("metrics.json", metrics)

    def write_compute_ledger(self, ledger: Mapping[str, Any]) -> Path:
        """Write ``compute_ledger.json``."""

        return self.write_json("compute_ledger.json", ledger)

    def write_history(self, history: Sequence[Mapping[str, Any]]) -> Path:
        """Write named scalar training records to ``history.csv``."""

        path = self.output_dir / "history.csv"
        fieldnames: list[str] = []
        for record in history:
            for key in record:
                if str(key) not in fieldnames:
                    fieldnames.append(str(key))
        if not fieldnames:
            fieldnames = ["step", "total"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for record in history:
                writer.writerow({str(key): _jsonable(value) for key, value in record.items()})
        return path

    def write_predictions(self, predictions: Mapping[str, Any]) -> Path:
        """Write array predictions to compressed ``predictions.npz``."""

        path = self.output_dir / "predictions.npz"
        arrays = {
            str(key): (
                value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            )
            for key, value in predictions.items()
        }
        np.savez_compressed(path, **arrays)
        return path

    def write_checkpoint(self, state: Mapping[str, Any], filename: str = "checkpoint.pt") -> Path:
        """Write a PyTorch checkpoint."""

        path = self.output_dir / filename
        torch.save(dict(state), path)
        return path

    def write_manifest(self, *, metadata: Mapping[str, Any] | None = None) -> Path:
        """Hash all current files and write ``manifest.json`` last."""

        files: dict[str, dict[str, Any]] = {}
        for path in sorted(self.output_dir.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.name == "manifest.json":
                continue
            content = path.read_bytes()
            files[path.name] = {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        return self.write_json(
            "manifest.json",
            {"schema_version": 1, "files": files, "metadata": dict(metadata or {})},
        )


def _finite_range(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lower, upper = float(finite.min()), float(finite.max())
    if math.isclose(lower, upper):
        padding = max(1.0, abs(lower)) * 0.05
    else:
        padding = (upper - lower) * 0.06
    return lower - padding, upper + padding


def _format_number(value: float) -> str:
    absolute = abs(float(value))
    if absolute != 0.0 and (absolute < 1e-3 or absolute >= 1e4):
        return f"{value:.2e}"
    return f"{value:.4g}"


def _polyline(
    x: np.ndarray,
    y: np.ndarray,
    *,
    panel: tuple[float, float, float, float],
    color: str,
    dash: str | None = None,
) -> str:
    px, py, width, height = panel
    x0, x1 = _finite_range(x)
    y0, y1 = _finite_range(y)
    points: list[str] = []
    for x_value, y_value in zip(x, y, strict=True):
        if not (math.isfinite(float(x_value)) and math.isfinite(float(y_value))):
            continue
        sx = px + (float(x_value) - x0) / max(x1 - x0, 1e-12) * width
        sy = py + height - (float(y_value) - y0) / max(y1 - y0, 1e-12) * height
        points.append(f"{sx:.2f},{sy:.2f}")
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
        f'stroke-width="2" stroke-linejoin="round"{dash_attr}/>'
    )


def _plot_panel(
    *,
    title: str,
    x: np.ndarray,
    curves: Sequence[tuple[str, np.ndarray, str, str | None]],
    panel: tuple[float, float, float, float],
    x_label: str = "",
    y_label: str = "",
) -> str:
    px, py, width, height = panel
    values = np.concatenate([np.asarray(curve[1]).reshape(-1) for curve in curves])
    y0, y1 = _finite_range(values)
    x0, x1 = _finite_range(x)
    parts = [
        f'<text x="{px}" y="{py - 16}" class="panel-title">{html.escape(title)}</text>',
        f'<rect x="{px}" y="{py}" width="{width}" height="{height}" class="plot"/>',
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        sx, sy = px + fraction * width, py + fraction * height
        parts.append(
            f'<line x1="{sx}" y1="{py}" x2="{sx}" y2="{py + height}" class="grid"/>'
        )
        parts.append(
            f'<line x1="{px}" y1="{sy}" x2="{px + width}" y2="{sy}" class="grid"/>'
        )
    # Solid predictions are drawn first and dashed references last.  The
    # original sequence is retained for the legend, so a near-perfect
    # prediction cannot visually erase its reference curve.
    drawing_curves = sorted(curves, key=lambda item: item[3] is not None)
    for label, curve, color, dash in drawing_curves:
        points: list[str] = []
        for x_value, y_value in zip(x, curve, strict=True):
            sx = px + (float(x_value) - x0) / max(x1 - x0, 1e-12) * width
            sy = py + height - (float(y_value) - y0) / max(y1 - y0, 1e-12) * height
            if math.isfinite(sx) and math.isfinite(sy):
                points.append(f"{sx:.2f},{sy:.2f}")
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linejoin="round"{dash_attr}/>'
        )
    for index, (label, curve, color, dash) in enumerate(curves):
        legend_y = py + 18 + index * 20
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<line x1="{px + 10}" y1="{legend_y}" x2="{px + 34}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="2"{dash_attr}/>'
        )
        parts.append(
            f'<text x="{px + 41}" y="{legend_y + 4}" class="legend">'
            f'{html.escape(label)}</text>'
        )
    parts.extend(
        (
            f'<text x="{px - 8}" y="{py + 4}" class="tick" text-anchor="end">'
            f'{html.escape(_format_number(y1))}</text>',
            f'<text x="{px - 8}" y="{py + height + 4}" class="tick" text-anchor="end">'
            f'{html.escape(_format_number(y0))}</text>',
            f'<text x="{px}" y="{py + height + 18}" class="tick">'
            f'{html.escape(_format_number(x0))}</text>',
            f'<text x="{px + width}" y="{py + height + 18}" class="tick" text-anchor="end">'
            f'{html.escape(_format_number(x1))}</text>',
        )
    )
    if x_label:
        parts.append(
            f'<text x="{px + width / 2}" y="{py + height + 35}" class="axis" '
            f'text-anchor="middle">{html.escape(x_label)}</text>'
        )
    if y_label:
        label_x = px - 49
        label_y = py + height / 2
        parts.append(
            f'<text x="{label_x}" y="{label_y}" class="axis" text-anchor="middle" '
            f'transform="rotate(-90 {label_x} {label_y})">{html.escape(y_label)}</text>'
        )
    return "\n".join(parts)


def _svg_document(
    title: str,
    body: str,
    *,
    width: int,
    height: int,
    subtitle: str = "OpenPINN saved result; the reference, when present, is evaluation-only.",
) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: Arial, "Noto Sans CJK SC", sans-serif; fill: #172b4d; }}
.title {{ font-size: 26px; font-weight: 700; }}
.subtitle {{ font-size: 13px; fill: #52657a; }}
.panel-title {{ font-size: 17px; font-weight: 700; }}
.plot {{ fill: #f8fafc; stroke: #94a3b8; }}
.plot-border {{ fill: none; stroke: #94a3b8; }}
.grid {{ stroke: #e2e8f0; stroke-width: 1; }}
.legend {{ font-size: 11px; fill: #334155; }}
.tick {{ font-size: 10px; fill: #52657a; }}
.axis {{ font-size: 11px; fill: #334155; font-weight: 600; }}
.metric {{ font-size: 13px; fill: #172b4d; }}
.note {{ font-size: 11px; fill: #64748b; }}
</style>
<rect width="{width}" height="{height}" fill="#ffffff"/>
<text x="50" y="44" class="title">{html.escape(title)}</text>
<text x="50" y="69" class="subtitle">{html.escape(subtitle)}</text>
{body}
</svg>
'''


def _interpolated_color(
    fraction: float,
    low: tuple[int, int, int],
    high: tuple[int, int, int],
) -> str:
    fraction = min(1.0, max(0.0, float(fraction)))
    values = tuple(
        round(start + fraction * (end - start))
        for start, end in zip(low, high, strict=True)
    )
    return f"rgb({values[0]},{values[1]},{values[2]})"


def _nearest_fill_missing(grid: np.ndarray) -> np.ndarray:
    """Fill excluded geometry cells from their nearest saved grid neighbour."""

    filled = np.asarray(grid, dtype=float).copy()
    valid = np.argwhere(np.isfinite(filled))
    missing = np.argwhere(~np.isfinite(filled))
    if not valid.size:
        return np.zeros_like(filled)
    for row, column in missing:
        distances = np.square(valid[:, 0] - row) + np.square(valid[:, 1] - column)
        nearest_row, nearest_column = valid[int(np.argmin(distances))]
        filled[row, column] = filled[nearest_row, nearest_column]
    return filled


def _smooth_grid_rgb(
    *,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    value_range: tuple[float, float],
    palette: tuple[tuple[int, int, int], tuple[int, int, int]],
    pixels: int = 480,
) -> np.ndarray:
    """Create a smooth RGB view without changing the saved numerical values."""

    x_values = np.unique(np.asarray(x, dtype=float))
    y_values = np.unique(np.asarray(y, dtype=float))
    grid = np.full((y_values.size, x_values.size), np.nan, dtype=float)
    x_indices = np.searchsorted(x_values, np.asarray(x, dtype=float))
    y_indices = np.searchsorted(y_values, np.asarray(y, dtype=float))
    grid[y_indices, x_indices] = np.asarray(values, dtype=float)
    grid = _nearest_fill_missing(grid)

    tensor = torch.as_tensor(grid, dtype=torch.float64).reshape(1, 1, *grid.shape)
    dense = torch.nn.functional.interpolate(
        tensor,
        size=(pixels, pixels),
        mode="bicubic",
        align_corners=True,
    )[0, 0].numpy()
    value_min, value_max = value_range
    fractions = np.clip(
        (dense - value_min) / max(value_max - value_min, 1e-12),
        0.0,
        1.0,
    )
    low = np.asarray(palette[0], dtype=float)
    high = np.asarray(palette[1], dtype=float)
    rgb = np.rint(low + fractions[..., None] * (high - low)).astype(np.uint8)
    return np.ascontiguousarray(rgb[::-1])


def _png_data_uri(rgb: np.ndarray) -> str:
    """Encode one RGB array as an embedded PNG using only the standard library."""

    height, width, channels = rgb.shape
    if channels != 3 or rgb.dtype != np.uint8:
        raise ValueError("PNG input must be an H x W x 3 uint8 RGB array.")

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", checksum)
        )

    scanlines = b"".join(b"\x00" + row.tobytes() for row in rgb)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _grid_field_panel(
    *,
    title: str,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    panel: tuple[float, float, float, float],
    bounds: tuple[tuple[float, float], tuple[float, float]],
    holes: np.ndarray,
    hole_radius: float,
    value_range: tuple[float, float],
    palette: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> str:
    """Render a physical 2D grid while leaving the two tunnel voids visible."""

    px, py, width, height = panel
    (x0, x1), (y0, y1) = bounds
    raster = _smooth_grid_rgb(
        x=x,
        y=y,
        values=values,
        value_range=value_range,
        palette=palette,
    )
    image_uri = _png_data_uri(raster)
    parts = [
        f'<text x="{px}" y="{py - 16}" class="panel-title">{html.escape(title)}</text>',
        f'<rect x="{px}" y="{py}" width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<image x="{px}" y="{py}" width="{width}" height="{height}" '
        f'preserveAspectRatio="none" image-rendering="auto" href="{image_uri}"/>',
    ]
    for center_x, center_y in holes:
        sx = px + (float(center_x) - x0) / max(x1 - x0, 1e-12) * width
        sy = py + height - (float(center_y) - y0) / max(y1 - y0, 1e-12) * height
        radius_x = hole_radius / max(x1 - x0, 1e-12) * width
        radius_y = hole_radius / max(y1 - y0, 1e-12) * height
        parts.append(
            f'<ellipse cx="{sx:.2f}" cy="{sy:.2f}" rx="{radius_x:.2f}" '
            f'ry="{radius_y:.2f}" fill="#ffffff" stroke="#111827" stroke-width="2"/>'
        )
    parts.extend(
        (
            f'<rect x="{px}" y="{py}" width="{width}" height="{height}" class="plot-border"/>',
            f'<text x="{px}" y="{py + height + 18}" class="tick">{_format_number(x0)}</text>',
            f'<text x="{px + width}" y="{py + height + 18}" class="tick" text-anchor="end">{_format_number(x1)}</text>',
            f'<text x="{px - 8}" y="{py + height + 4}" class="tick" text-anchor="end">{_format_number(y0)}</text>',
            f'<text x="{px - 8}" y="{py + 4}" class="tick" text-anchor="end">{_format_number(y1)}</text>',
            f'<text x="{px + width / 2}" y="{py + height + 34}" class="axis" text-anchor="middle">x (m)</text>',
            f'<text x="{px - 42}" y="{py + height / 2}" class="axis" text-anchor="middle" transform="rotate(-90 {px - 42} {py + height / 2})">y (m)</text>',
        )
    )
    return "\n".join(parts)


def _colorbar(
    *,
    x: float,
    y: float,
    height: float,
    value_range: tuple[float, float],
    label: str,
    palette: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> str:
    value_min, value_max = value_range
    steps = 48
    parts = []
    for index in range(steps):
        fraction = index / max(1, steps - 1)
        top = y + (steps - 1 - index) / steps * height
        parts.append(
            f'<rect x="{x}" y="{top:.2f}" width="18" height="{height / steps + 0.4:.2f}" '
            f'fill="{_interpolated_color(fraction, *palette)}"/>'
        )
    parts.extend(
        (
            f'<rect x="{x}" y="{y}" width="18" height="{height}" fill="none" stroke="#64748b"/>',
            f'<text x="{x + 25}" y="{y + 4}" class="tick">{_format_number(value_max)}</text>',
            f'<text x="{x + 25}" y="{y + height + 4}" class="tick">{_format_number(value_min)}</text>',
            f'<text x="{x - 4}" y="{y - 14}" class="axis">{html.escape(label)}</text>',
        )
    )
    return "\n".join(parts)


def _metric_line(metrics: Mapping[str, Any], names: Sequence[tuple[str, str]]) -> str:
    values = []
    for key, label in names:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(f"{label}={_format_number(float(value))}")
    return "    ".join(values)


def _field_panel(
    *,
    title: str,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    panel: tuple[float, float, float, float],
    value_range: tuple[float, float],
) -> str:
    """Render one inspectable 2D point-field panel without plotting dependencies."""

    px, py, width, height = panel
    x0, x1 = _finite_range(x)
    y0, y1 = _finite_range(y)
    value_min, value_max = value_range
    parts = [
        f'<text x="{px}" y="{py - 16}" class="panel-title">{html.escape(title)}</text>',
        f'<rect x="{px}" y="{py}" width="{width}" height="{height}" class="plot"/>',
    ]
    radius = max(1.4, min(4.5, 0.42 * width / max(2.0, math.sqrt(x.size))))
    for x_value, y_value, field_value in zip(x, y, values, strict=True):
        if not all(math.isfinite(float(value)) for value in (x_value, y_value, field_value)):
            continue
        fraction = (float(field_value) - value_min) / max(value_max - value_min, 1e-12)
        fraction = min(1.0, max(0.0, fraction))
        red = round(37 + 202 * fraction)
        green = round(99 + 33 * (1.0 - abs(2.0 * fraction - 1.0)))
        blue = round(235 - 167 * fraction)
        sx = px + (float(x_value) - x0) / max(x1 - x0, 1e-12) * width
        sy = py + height - (float(y_value) - y0) / max(y1 - y0, 1e-12) * height
        parts.append(
            f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="{radius:.2f}" '
            f'fill="rgb({red},{green},{blue})"/>'
        )
    parts.extend(
        (
            f'<text x="{px}" y="{py + height + 24}" class="subtitle">X</text>',
            f'<text x="{px + width - 138}" y="{py + height + 24}" class="subtitle">'
            f'U range: {value_min:.3g} … {value_max:.3g}</text>',
        )
    )
    return "\n".join(parts)


def _write_tunnel_result_figures(result: Any, *, method_title: str) -> tuple[Path, Path]:
    """Write physical, problem-specific views for twin-tunnel consolidation."""

    output_dir = Path(result.output_dir)
    metrics = result.metrics
    coordinates = np.asarray(result.predictions["physical_coordinates"], dtype=float)
    pressure = np.asarray(result.predictions["pressure_kpa"], dtype=float).reshape(-1)
    residual = np.abs(np.asarray(result.predictions["residual"], dtype=float)[:, 0])
    centers = np.asarray(result.predictions["tunnel_centers_m"], dtype=float)
    radius = float(np.asarray(result.predictions["tunnel_radius_m"]).reshape(-1)[0])
    width = float(metrics["domain_width_m"])
    height = float(metrics["domain_height_m"])
    initial_pressure = float(metrics["initial_pressure_kpa"])
    bounds = ((0.0, width), (0.0, height))
    pressure_palette = ((239, 246, 255), (8, 81, 156))
    residual_palette = ((255, 247, 236), (179, 0, 0))

    time_values = np.unique(coordinates[:, 2])
    if time_values.size > 3:
        selected = np.linspace(0, time_values.size - 1, 3).round().astype(int)
        time_values = time_values[selected]
    resolution_mask = np.isclose(coordinates[:, 2], time_values[0])
    source_x_count = np.unique(coordinates[resolution_mask, 0]).size
    source_y_count = np.unique(coordinates[resolution_mask, 1]).size
    interpolation_note = (
        f"Display uses bicubic interpolation from the saved {source_x_count} x "
        f"{source_y_count} evaluation grid; saved values and metrics are unchanged."
    )
    finite_pressure = pressure[np.isfinite(pressure)]
    pressure_range = (
        min(0.0, float(finite_pressure.min())),
        max(initial_pressure, float(finite_pressure.max())),
    )
    if math.isclose(*pressure_range):
        pressure_range = (pressure_range[0] - 0.05, pressure_range[1] + 0.05)

    panels: list[str] = []
    positions = (60.0, 420.0, 780.0)
    for time_value, px in zip(time_values, positions, strict=False):
        mask = np.isclose(coordinates[:, 2], time_value)
        panels.append(
            _grid_field_panel(
                title=f"Predicted pressure at t={time_value:.3g} years",
                x=coordinates[mask, 0],
                y=coordinates[mask, 1],
                values=pressure[mask],
                panel=(px, 125.0, 300.0, 300.0),
                bounds=bounds,
                holes=centers,
                hole_radius=radius,
                value_range=pressure_range,
                palette=pressure_palette,
            )
        )
    panels.append(
        _colorbar(
            x=1125.0,
            y=125.0,
            height=300.0,
            value_range=pressure_range,
            label="Pressure (kPa)",
            palette=pressure_palette,
        )
    )
    primary_metrics = _metric_line(
        metrics,
        (
            ("pde_residual_rmse", "PDE residual RMSE"),
            ("boundary_pressure_rmse_kpa", "boundary RMSE (kPa)"),
            ("initial_pressure_rmse_kpa", "initial RMSE (kPa)"),
        ),
    )
    physical_metrics = _metric_line(
        metrics,
        (
            ("mean_pressure_final_time_kpa", "final mean pressure (kPa)"),
            ("mean_pressure_dissipation_percent", "mean dissipation (%)"),
            ("physical_range_violation_fraction", "outside [0,p0] fraction"),
        ),
    )
    panels.extend(
        (
            f'<text x="60" y="500" class="metric">{html.escape(primary_metrics)}</text>',
            f'<text x="60" y="528" class="metric">{html.escape(physical_metrics)}</text>',
            '<text x="60" y="558" class="note">'
            'The two white circles are drained tunnel walls. One shared colour scale is used for all time slices.</text>',
            '<text x="60" y="579" class="note">'
            f'{html.escape(interpolation_note)}</text>',
            '<text x="60" y="600" class="note">'
            'Solution L2/RMSE/R2 are not reported because no independent reference field was supplied.</text>',
        )
    )
    result_path = output_dir / "result.svg"
    result_path.write_text(
        _svg_document(
            f"Twin-tunnel consolidation · {method_title}",
            "\n".join(panels),
            width=1280,
            height=630,
            subtitle=(
                "Problem-specific view in physical coordinates; predicted excess pore "
                "pressure is shown in kPa."
            ),
        ),
        encoding="utf-8",
    )

    final_time = float(np.max(coordinates[:, 2]))
    final_mask = np.isclose(coordinates[:, 2], final_time)
    final_residual = residual[final_mask]
    residual_range = (0.0, max(1e-12, float(np.max(final_residual))))
    diagnostic_parts = [
        _grid_field_panel(
            title=f"|PDE residual| at t={final_time:.3g} years",
            x=coordinates[final_mask, 0],
            y=coordinates[final_mask, 1],
            values=final_residual,
            panel=(60.0, 135.0, 300.0, 300.0),
            bounds=bounds,
            holes=centers,
            hole_radius=radius,
            value_range=residual_range,
            palette=residual_palette,
        ),
        _colorbar(
            x=375.0,
            y=135.0,
            height=300.0,
            value_range=residual_range,
            label="Dimensionless",
            palette=residual_palette,
        ),
    ]

    available_y = np.unique(coordinates[final_mask, 1])
    target_y = 0.25 * height
    section_y = float(available_y[np.argmin(np.abs(available_y - target_y))])
    section_mask = final_mask & np.isclose(coordinates[:, 1], section_y)
    section_order = np.argsort(coordinates[section_mask, 0])
    diagnostic_parts.append(
        _plot_panel(
            title=f"Final pressure profile at y={section_y:.3g} m",
            x=coordinates[section_mask, 0][section_order],
            curves=(("Prediction", pressure[section_mask][section_order], "#2563eb", None),),
            panel=(485.0, 155.0, 300.0, 245.0),
            x_label="x (m)",
            y_label="Pressure (kPa)",
        )
    )
    evaluation_times = np.asarray(
        result.predictions["evaluation_time_years"], dtype=float
    ).reshape(-1)
    mean_pressure = np.asarray(
        result.predictions["mean_pressure_kpa"], dtype=float
    ).reshape(-1)
    diagnostic_parts.append(
        _plot_panel(
            title="Mean pressure dissipation",
            x=evaluation_times,
            curves=(("Spatial mean", mean_pressure, "#0f766e", None),),
            panel=(890.0, 155.0, 300.0, 245.0),
            x_label="Time (years)",
            y_label="Mean pressure (kPa)",
        )
    )
    diagnostic_parts.extend(
        (
            '<text x="485" y="470" class="metric">'
            f'{html.escape(primary_metrics)}</text>',
            '<text x="485" y="498" class="metric">'
            f'{html.escape(physical_metrics)}</text>',
            '<text x="485" y="528" class="note">'
            'Use residual and condition RMSE to check physics consistency; they are not substitutes for reference-solution errors.</text>',
            '<text x="485" y="549" class="note">'
            f'{html.escape(interpolation_note)}</text>',
            '<text x="485" y="570" class="note">'
            'L2/RMSE/R2 against a solution become available only after an independent COMSOL or other reference field is supplied.</text>',
        )
    )
    diagnostics_path = output_dir / "diagnostics.svg"
    diagnostics_path.write_text(
        _svg_document(
            "Twin-tunnel diagnostics",
            "\n".join(diagnostic_parts),
            width=1280,
            height=605,
            subtitle="Physics diagnostics, a physical pressure profile, and the predicted dissipation trend.",
        ),
        encoding="utf-8",
    )
    return result_path, diagnostics_path


def write_result_figures(result: Any, *, method_title: str) -> tuple[Path, ...]:
    """Write inspectable SVG result and loss figures for a completed run."""

    output_dir = Path(result.output_dir)
    coordinates = np.asarray(result.predictions["coordinates"], dtype=float)
    prediction = np.asarray(result.predictions["prediction"], dtype=float)[:, 0]
    reference_value = result.predictions.get("reference")
    reference = (
        np.asarray(reference_value, dtype=float)[:, 0]
        if reference_value is not None
        else None
    )
    figure_path = output_dir / "result.svg"
    additional_paths: list[Path] = []

    if coordinates.shape[1] == 1:
        order = np.argsort(coordinates[:, 0])
        x = coordinates[order, 0]
        curves: list[tuple[str, np.ndarray, str, str | None]] = [
            (method_title, prediction[order], "#2563eb", None)
        ]
        if reference is not None:
            curves.insert(0, ("Reference", reference[order], "#111827", "7 5"))
        panels = [
            _plot_panel(
                title="Solution",
                x=x,
                curves=curves,
                panel=(70, 120, 520, 300),
                x_label="x",
                y_label="Solution",
            )
        ]
        if reference is not None:
            panels.append(
                _plot_panel(
                    title="Absolute error",
                    x=x,
                    curves=(("Absolute error", np.abs(prediction[order] - reference[order]), "#dc2626", None),),
                    panel=(690, 120, 520, 300),
                    x_label="x",
                    y_label="Absolute error",
                )
            )
        summary = _metric_line(
            result.metrics,
            (
                ("relative_l2_error", "relative L2"),
                ("rmse", "RMSE"),
                ("mae", "MAE"),
                ("max_absolute_error", "max AE"),
                ("r2_score", "R2"),
            ),
        )
        if summary:
            panels.append(
                f'<text x="70" y="472" class="metric">{html.escape(summary)}</text>'
            )
        figure_path.write_text(
            _svg_document(
                f"{result.metrics.get('problem', 'PDE')} · {method_title}",
                "\n".join(panels),
                width=1280,
                height=505,
            ),
            encoding="utf-8",
        )
    elif coordinates.shape[1] == 2:
        times = np.unique(coordinates[:, 1])
        if times.size > 4:
            indices = np.linspace(0, times.size - 1, 4).round().astype(int)
            times = times[indices]
        panels: list[str] = []
        positions = ((70, 130), (680, 130), (70, 510), (680, 510))
        for time_value, (px, py) in zip(times[:4], positions, strict=False):
            mask = np.isclose(coordinates[:, 1], time_value)
            order = np.argsort(coordinates[mask, 0])
            x = coordinates[mask, 0][order]
            curves = [(method_title, prediction[mask][order], "#2563eb", None)]
            if reference is not None:
                curves.insert(
                    0,
                    ("Reference", reference[mask][order], "#111827", "7 5"),
                )
            panels.append(
                _plot_panel(
                    title=f"t = {time_value:.3g}",
                    x=x,
                    curves=curves,
                    panel=(px, py, 520, 280),
                    x_label="x",
                    y_label="Solution",
                )
            )
        figure_path.write_text(
            _svg_document(
                f"{result.metrics.get('problem', 'PDE')} · four time slices",
                "\n".join(panels),
                width=1280,
                height=900,
            ),
            encoding="utf-8",
        )
    elif coordinates.shape[1] == 3:
        if (
            result.metrics.get("problem") == "tunnel_consolidation_model1"
            and "physical_coordinates" in result.predictions
            and "pressure_kpa" in result.predictions
        ):
            figure_path, diagnostics_path = _write_tunnel_result_figures(
                result,
                method_title=method_title,
            )
            additional_paths.append(diagnostics_path)
        else:
            times = np.unique(coordinates[:, 2])
            if times.size > 4:
                indices = np.linspace(0, times.size - 1, 4).round().astype(int)
                times = times[indices]
            finite_prediction = prediction[np.isfinite(prediction)]
            value_range = (
                (float(finite_prediction.min()), float(finite_prediction.max()))
                if finite_prediction.size
                else (0.0, 1.0)
            )
            if math.isclose(*value_range):
                value_range = (value_range[0] - 0.05, value_range[1] + 0.05)
            positions = ((70, 130), (680, 130), (70, 560), (680, 560))
            panels = []
            for time_value, (px, py) in zip(times[:4], positions, strict=False):
                mask = np.isclose(coordinates[:, 2], time_value)
                panels.append(
                    _field_panel(
                        title=f"tau = {time_value:.3g}",
                        x=coordinates[mask, 0],
                        y=coordinates[mask, 1],
                        values=prediction[mask],
                        panel=(px, py, 500, 330),
                        value_range=value_range,
                    )
                )
            figure_path.write_text(
                _svg_document(
                    f"{result.metrics.get('problem', 'PDE')} · 2D time slices",
                    "\n".join(panels),
                    width=1280,
                    height=980,
                ),
                encoding="utf-8",
            )
    else:
        raise ValueError(
            "Automatic result figures currently support 1D, 1D+time and 2D+time coordinates."
        )

    history = list(result.history)
    loss_path = output_dir / "loss.svg"
    steps = np.asarray([record["step"] for record in history], dtype=float)
    totals = np.log10(
        np.maximum(np.asarray([record["total"] for record in history], dtype=float), 1e-30)
    )
    loss_panel = _plot_panel(
        title="Training loss (log10)",
        x=steps,
        curves=((method_title, totals, "#2563eb", None),),
        panel=(70, 115, 760, 300),
    )
    loss_path.write_text(
        _svg_document("OpenPINN training history", loss_panel, width=900, height=470),
        encoding="utf-8",
    )
    return figure_path, *additional_paths, loss_path


def write_diagnostic_run_artifact(
    output_dir: str | Path,
    *,
    result: Any | None = None,
    status: str | None = None,
    exported_pair_snapshot: Mapping[str, Any] | None = None,
    export_snapshot_fingerprint: str | None = None,
    script_path: str | Path | None = None,
    code_origin: str = "unknown",
    actual_config_source: str | None = None,
    generator_id: str = "openpinn.configurator.v1",
    source: str = "configurator",
    budgets: Mapping[str, Any] | None = None,
    steps_completed: int | None = None,
    elapsed_seconds: float | None = None,
    device: str | None = None,
    failure: BaseException | Mapping[str, Any] | None = None,
    issues_at_export: Sequence[Mapping[str, Any]] = (),
    semantic_concerns: Sequence[Mapping[str, Any]] = (),
    optional_references: Sequence[Mapping[str, Any]] = (),
    example: bool = False,
    fixture: bool = False,
    smoke_only: bool = False,
) -> Path:
    """Write one self-contained RunArtifact for post-run Diagnostic evaluation.

    This public writer is deliberately independent of CodeBuilder. Configurator
    scripts and user-edited scripts can call it after a successful run or from
    an exception handler. It records available evidence without inventing loss,
    timing, accuracy, or convergence claims.
    """

    allowed = {
        "completed", "budget_exhausted", "not_converged", "failed",
        "interrupted", "not_runnable", "unknown",
    }
    history = list(getattr(result, "history", ()) or ())
    result_metrics = dict(getattr(result, "metrics", {}) or {})
    completed_steps = steps_completed
    if completed_steps is None and history:
        completed_steps = int(history[-1].get("step", len(history)))
    selected_budgets = dict(budgets or {})
    if status is None:
        if isinstance(failure, KeyboardInterrupt):
            status = "interrupted"
        elif failure is not None:
            status = "failed"
        elif result is not None and selected_budgets.get("max_steps") is not None:
            status = "budget_exhausted"
        elif result is not None:
            status = "completed"
        else:
            status = "unknown"
    if status not in allowed:
        raise ValueError(f"Unsupported diagnostic RunArtifact status: {status!r}.")

    snapshot = _jsonable(dict(exported_pair_snapshot or {}))
    if export_snapshot_fingerprint is None and snapshot:
        canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        export_snapshot_fingerprint = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    script_hash: str | None = None
    script_status = "unavailable"
    if script_path is not None:
        candidate = Path(script_path)
        if candidate.is_file():
            script_hash = "sha256:" + hashlib.sha256(candidate.read_bytes()).hexdigest()
            script_status = "available"

    losses = [
        float(record["total"])
        for record in history
        if isinstance(record, Mapping) and isinstance(record.get("total"), (int, float))
    ]
    error_metrics = {
        str(key): float(value)
        for key, value in result_metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
        and key not in {"steps", "final_total_loss"}
    }
    failure_record = {"type": None, "message": None, "stack": None, "truncated": False}
    if isinstance(failure, BaseException):
        stack = "".join(traceback.format_exception(type(failure), failure, failure.__traceback__))
        failure_record = {"type": type(failure).__name__, "message": str(failure),
                          "stack": stack[-16000:], "truncated": len(stack) > 16000}
    elif isinstance(failure, Mapping):
        failure_record.update({key: _jsonable(value) for key, value in failure.items() if key in failure_record})

    artifact = {
        "schema_version": 1,
        "kind": "RunArtifact",
        "run_id": str(uuid.uuid4()),
        "status": status,
        "exported_pair_snapshot": snapshot,
        "export_snapshot_fingerprint": export_snapshot_fingerprint,
        # pair_fingerprint remains for old Diagnostic and CodeBuilder readers.
        "pair_fingerprint": export_snapshot_fingerprint,
        "script_hash": script_hash,
        "script_hash_status": script_status,
        "code_origin": code_origin,
        "actual_config_source": actual_config_source,
        "generator_id": generator_id,
        "source": source,
        "artifact_origin": "local_generated_run" if result is not None else "local_run_failure",
        "backend_runnable": status != "not_runnable",
        "budgets": selected_budgets,
        "steps_completed": completed_steps,
        "elapsed_seconds": elapsed_seconds,
        "device": device,
        "metrics": {
            "loss_history": losses,
            "final_loss": losses[-1] if losses else None,
            "error_metrics": error_metrics,
        },
        "optional_references": _jsonable(list(optional_references)),
        "failure": failure_record,
        "issues_at_export": _jsonable(list(issues_at_export)),
        "semantic_concerns": _jsonable(list(semantic_concerns)),
        "accuracy_threshold": None,
        "accuracy_verified": False,
        "example": bool(example),
        "fixture": bool(fixture),
        "smoke_only": bool(smoke_only),
    }
    return ArtifactWriter(output_dir).write_json("run_artifact.json", artifact)


def write_failure_artifacts(
    output_dir: str | Path,
    error: BaseException,
    *,
    problem: str,
    method: str,
) -> Path:
    """Persist a failed high-level run before the original exception is raised."""

    writer = ArtifactWriter(output_dir)
    reason = f"{type(error).__name__}: {error}"
    path = writer.write_json(
        "failure.json",
        {
            "status": "failed",
            "problem": problem,
            "method": method,
            "failure_reason": reason,
            "traceback": traceback.format_exc(),
        },
    )
    (writer.output_dir / "failure_notes.md").write_text(
        "# Failure notes\n\n"
        f"- Problem: {problem}\n"
        f"- Method: {method}\n"
        f"- Reason: {reason}\n",
        encoding="utf-8",
    )
    writer.write_manifest(
        metadata={"status": "failed", "problem": problem, "method": method}
    )
    return path
