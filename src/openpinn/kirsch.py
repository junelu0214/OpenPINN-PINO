"""Kirsch analytical solution for an infinite plate with a circular hole.

The classical Kirsch solution gives closed-form stress and displacement
fields for an infinite elastic plate with a single circular hole under
uniaxial far-field tension.  A finite plate with the same geometry is an
approximation whose accuracy depends on the ratio of plate width to hole
diameter; the comparison is most useful near the hole where the stress
concentration factor (SCF) dominates.

Reference
---------
Kirsch, E.G. (1898). "Die Theorie der Elastizitat und die Bedurfnisse der
Festigkeitslehre." Zeitschrift des Vereines deutscher Ingenieure, 42, 797-807.

Timoshenko, S.P. & Goodier, J.N. (1970). Theory of Elasticity, 3rd ed.
McGraw-Hill. Section 4.4 (stress) and Problem 80 (displacement).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def kirsch_stress_polar(
    r: np.ndarray,
    theta: np.ndarray,
    sigma_inf: float,
    a: float,
) -> dict[str, np.ndarray]:
    """Kirsch stress field in polar coordinates (sigma_rr, sigma_tt, sigma_rt).

    Parameters
    ----------
    r : array
        Radial coordinate from hole center.
    theta : array
        Angle from loading direction (x-axis), in radians.
    sigma_inf : float
        Far-field uniaxial stress (tension positive).
    a : float
        Hole radius.
    """
    rho = a / r
    rho2 = rho ** 2
    rho4 = rho ** 4
    c2 = np.cos(2.0 * theta)
    s2 = np.sin(2.0 * theta)

    sigma_rr = 0.5 * sigma_inf * (
        (1.0 - rho2) + (1.0 - 4.0 * rho2 + 3.0 * rho4) * c2
    )
    sigma_tt = 0.5 * sigma_inf * (
        (1.0 + rho2) - (1.0 + 3.0 * rho4) * c2
    )
    sigma_rt = -0.5 * sigma_inf * (
        (1.0 + 2.0 * rho2 - 3.0 * rho4) * s2
    )
    return {"sigma_rr": sigma_rr, "sigma_tt": sigma_tt, "sigma_rt": sigma_rt}


def kirsch_stress_cartesian(
    x: np.ndarray,
    y: np.ndarray,
    sigma_inf: float,
    a: float,
    *,
    center: tuple[float, float] = (0.0, 0.0),
) -> dict[str, np.ndarray]:
    """Kirsch stress in Cartesian coordinates at points (x, y)."""
    dx = x - center[0]
    dy = y - center[1]
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    polar = kirsch_stress_polar(r, theta, sigma_inf, a)
    s_rr = polar["sigma_rr"]
    s_tt = polar["sigma_tt"]
    s_rt = polar["sigma_rt"]

    c = np.cos(theta)
    s = np.sin(theta)
    sigma_xx = s_rr * c**2 + s_tt * s**2 - 2.0 * s_rt * s * c
    sigma_yy = s_rr * s**2 + s_tt * c**2 + 2.0 * s_rt * s * c
    sigma_xy = (s_rr - s_tt) * s * c + s_rt * (c**2 - s**2)
    return {"sigma_xx": sigma_xx, "sigma_yy": sigma_yy, "sigma_xy": sigma_xy}


def kirsch_displacement(
    x: np.ndarray,
    y: np.ndarray,
    sigma_inf: float,
    a: float,
    E: float,
    nu: float,
    *,
    center: tuple[float, float] = (0.0, 0.0),
    plane: str = "stress",
) -> dict[str, np.ndarray]:
    """Kirsch displacement field in Cartesian coordinates.

    For plane stress, kappa = (3 - nu) / (1 + nu).
    For plane strain, kappa = 3 - 4*nu.
    """
    if plane == "stress":
        kappa = (3.0 - nu) / (1.0 + nu)
    elif plane == "strain":
        kappa = 3.0 - 4.0 * nu
    else:
        raise ValueError(f"plane must be 'stress' or 'strain', got {plane!r}")

    mu = E / (2.0 * (1.0 + nu))
    dx = x - center[0]
    dy = y - center[1]
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    c = np.cos(theta)
    s = np.sin(theta)
    c2 = np.cos(2.0 * theta)
    s2 = np.sin(2.0 * theta)

    rho = a / r
    rho2 = rho**2
    rho3 = rho**3

    # Radial and tangential displacements (Timoshenko & Goodier, eq. 80)
    u_r = (sigma_inf * r / (4.0 * mu)) * (
        (kappa - 1.0) / 2.0
        + c2
        + rho2 * (1.0 + (1.0 + kappa) * c2) / 2.0
        - rho2 * rho2 * c2
    )

    u_theta = (sigma_inf * r / (4.0 * mu)) * (
        -(kappa + 1.0) / 2.0 * s2
        + rho2 * (kappa - 1.0) * s2 / 2.0
        + rho2 * rho2 * s2
    )

    ux = u_r * c - u_theta * s
    uy = u_r * s + u_theta * c

    return {"ux": ux, "uy": uy}


def kirsch_scf() -> float:
    """Stress concentration factor for an infinite plate: SCF = 3."""
    return 3.0


def kirsch_hoop_stress_at_hole(
    theta: np.ndarray,
    sigma_inf: float,
) -> np.ndarray:
    """Hoop stress at hole boundary (r = a): sigma_tt = sigma_inf * (1 - 2*cos(2*theta))."""
    return sigma_inf * (1.0 - 2.0 * np.cos(2.0 * theta))


def compare_displacement_l2(
    predicted_ux: np.ndarray,
    predicted_uy: np.ndarray,
    coords: np.ndarray,
    sigma_inf: float,
    a: float,
    E: float,
    nu: float,
    *,
    center: tuple[float, float] = (0.0, 0.0),
    min_r_ratio: float = 0.0,
) -> dict[str, float]:
    """Compute relative L2 error of predicted displacement vs Kirsch.

    Parameters
    ----------
    predicted_ux, predicted_uy : (N,) arrays
        PINN predicted displacements.
    coords : (N, 2) array
        Evaluation coordinates.
    sigma_inf, a, E, nu : float
        Problem parameters.
    center : tuple
        Hole center.
    min_r_ratio : float
        Minimum r/a ratio for filtering points.  Points inside this radius
        are excluded (they are inside the hole anyway).

    Returns
    -------
    dict with ``relative_l2_ux``, ``relative_l2_uy``, ``relative_l2_total``.

    Notes
    -----
    The Kirsch solution assumes an infinite plate; mismatch near the outer
    boundary of a finite plate is expected and does NOT indicate a PINN bug.
    Points very close to the hole (within ~2 radii) give the most meaningful
    comparison.  The comparison is scaffolded here even when numeric thresholds
    cannot be set, so M3 smoke tests can exercise the path.
    """
    dx = coords[:, 0] - center[0]
    dy = coords[:, 1] - center[1]
    r = np.sqrt(dx**2 + dy**2)

    mask = r >= a * max(min_r_ratio, 1.0 + 1e-6)
    if not np.any(mask):
        return {
            "relative_l2_ux": float("nan"),
            "relative_l2_uy": float("nan"),
            "relative_l2_total": float("nan"),
            "n_comparison_points": 0,
        }

    ref = kirsch_displacement(
        coords[mask, 0], coords[mask, 1], sigma_inf, a, E, nu,
        center=center, plane="stress",
    )
    ref_ux = np.asarray(ref["ux"]).ravel()
    ref_uy = np.asarray(ref["uy"]).ravel()
    pred_ux = np.asarray(predicted_ux).ravel()[mask]
    pred_uy = np.asarray(predicted_uy).ravel()[mask]

    def _rel_l2(pred, ref_vals):
        diff_norm = np.sqrt(np.mean((pred - ref_vals) ** 2))
        ref_norm = np.sqrt(np.mean(ref_vals ** 2))
        if ref_norm < 1e-30:
            return float("inf") if diff_norm > 1e-30 else 0.0
        return float(diff_norm / ref_norm)

    return {
        "relative_l2_ux": _rel_l2(pred_ux, ref_ux),
        "relative_l2_uy": _rel_l2(pred_uy, ref_uy),
        "relative_l2_total": _rel_l2(
            np.stack([pred_ux, pred_uy], axis=-1).ravel(),
            np.stack([ref_ux, ref_uy], axis=-1).ravel(),
        ),
        "n_comparison_points": int(np.sum(mask)),
    }
