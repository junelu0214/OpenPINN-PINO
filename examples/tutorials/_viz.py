"""Small dependency-free SVG helpers for committed tutorial outputs."""

from __future__ import annotations

import html
import math
from collections.abc import Sequence

import numpy as np


def _color(value: float, lower: float, upper: float) -> str:
    ratio = 0.5 if upper <= lower else min(1.0, max(0.0, (value - lower) / (upper - lower)))
    red = int(35 + 220 * ratio)
    blue = int(245 - 210 * ratio)
    green = int(95 + 70 * (1.0 - abs(2.0 * ratio - 1.0)))
    return f"#{red:02x}{green:02x}{blue:02x}"


def loss_svg(values: Sequence[float], title: str) -> str:
    data = np.asarray(values, dtype=float)
    logged = np.log10(np.maximum(data, np.finfo(float).tiny))
    low, high = float(logged.min()), float(logged.max())
    span = max(high - low, 1e-12)
    xs = np.linspace(55.0, 575.0, len(data))
    ys = 225.0 - 170.0 * (logged - low) / span
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="270" viewBox="0 0 640 270">
<rect width="640" height="270" fill="white"/><text x="320" y="25" text-anchor="middle" font-size="17">{html.escape(title)}</text>
<line x1="55" y1="225" x2="585" y2="225" stroke="#333"/><line x1="55" y1="45" x2="55" y2="225" stroke="#333"/>
<polyline points="{points}" fill="none" stroke="#2369bd" stroke-width="3"/>
<text x="320" y="255" text-anchor="middle" font-size="13">training step</text>
<text x="15" y="140" transform="rotate(-90 15 140)" text-anchor="middle" font-size="13">log10(loss)</text></svg>'''


def field_svg(
    coordinates: np.ndarray,
    values: np.ndarray,
    title: str,
    *,
    hole_radius: float | None = None,
    hole_center: tuple[float, float] = (0.0, 0.0),
) -> str:
    points = np.asarray(coordinates, dtype=float)
    field = np.asarray(values, dtype=float).reshape(-1)
    x, y = points[:, 0], points[:, 1]
    xmin, xmax, ymin, ymax = float(x.min()), float(x.max()), float(y.min()), float(y.max())
    sx = 500.0 / max(xmax - xmin, 1e-12)
    sy = 310.0 / max(ymax - ymin, 1e-12)
    low, high = float(field.min()), float(field.max())
    circles = []
    for px, py, value in zip(x, y, field):
        cx = 70.0 + (px - xmin) * sx
        cy = 365.0 - (py - ymin) * sy
        circles.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{_color(float(value), low, high)}"/>')
    hole = ""
    if hole_radius is not None:
        radius = hole_radius * min(sx, sy)
        cx = 70.0 + (hole_center[0] - xmin) * sx
        cy = 365.0 - (hole_center[1] - ymin) * sy
        hole = f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="white" stroke="#111" stroke-width="2"/>'
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="430" viewBox="0 0 640 430">
<rect width="640" height="430" fill="white"/><text x="320" y="28" text-anchor="middle" font-size="17">{html.escape(title)}</text>
<rect x="70" y="55" width="500" height="310" fill="#f5f5f5" stroke="#333"/>{''.join(circles)}{hole}
<text x="320" y="405" text-anchor="middle" font-size="13">range [{low:.3g}, {high:.3g}]</text></svg>'''


def comparison_svg(coordinates: np.ndarray, prediction: np.ndarray, reference: np.ndarray, title: str) -> str:
    left = field_svg(coordinates, prediction, "prediction").replace('width="640" height="430"', 'width="440" height="330"').replace('viewBox="0 0 640 430"', 'viewBox="0 0 640 430"')
    right = field_svg(coordinates, reference, "reference").replace('width="640" height="430"', 'width="440" height="330"').replace('viewBox="0 0 640 430"', 'viewBox="0 0 640 430"')
    return f'''<div><h4>{html.escape(title)}</h4><div style="display:flex;flex-wrap:wrap">{left}{right}</div></div>'''
