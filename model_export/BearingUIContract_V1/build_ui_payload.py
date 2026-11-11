"""Bearing Layer UI JSON contract V1.

Builds UI-ready data from frozen Bearing Layer predictions.
No training, inference checkpoint edits, or Core ML conversion.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

RELIABILITY_FILE = (
    ROOT / "BearingReliability_V1"
    / "bearing_reliability_ui.py"
)


def load_reliability():
    spec = importlib.util.spec_from_file_location(
        "bearing_reliability_portable",
        RELIABILITY_FILE,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def json_map(array):
    """Convert NaN to JSON null; retain Y,X orientation."""
    values = np.asarray(array, dtype=np.float64)

    return [
        [
            float(v) if np.isfinite(v) else None
            for v in row
        ]
        for row in values
    ]


def build_payload(
    depth_yx,
    cpt_xy,
    dx_m,
    dy_m,
    valid_xy=None,
    qc_min_mpa=10.0,
    ic_max=2.2,
    min_thickness_m=1.0,
):
    depth = np.asarray(depth_yx, dtype=np.float32)

    if depth.ndim != 2:
        raise ValueError("Depth map must be [Y,X]")

    if np.isinf(depth).any():
        raise ValueError("Infinite depth values unsupported")

    if np.any(np.isfinite(depth) & (depth < 0)):
        raise ValueError("Depth cannot be negative")

    if valid_xy is None:
        valid = np.ones(depth.shape, dtype=bool)
    else:
        valid = np.asarray(valid_xy, dtype=bool)

    if valid.shape != depth.shape or not valid.any():
        raise ValueError("Invalid valid_xy mask")

    if np.isfinite(depth[~valid]).any():
        raise ValueError(
            "Depth outside valid area must be NaN"
        )

    candidates = valid & np.isfinite(depth)

    finite_depth = depth[candidates]

    shallowest = (
        float(finite_depth.min())
        if len(finite_depth) else None
    )
    deepest = (
        float(finite_depth.max())
        if len(finite_depth) else None
    )
    variation = (
        deepest - shallowest
        if shallowest is not None else None
    )

    reliability_mod = load_reliability()

    try:
        result = reliability_mod.build_reliability(
            depth_yx=depth,
            cpt_xy=cpt_xy,
            dx_m=dx_m,
            dy_m=dy_m,
            valid_xy=valid,
            qc_min_mpa=qc_min_mpa,
            ic_max=ic_max,
            min_thickness_m=min_thickness_m,
        )

        reliability = result["summary"]
        risk_map = result["risk_map_yx"]
        p90_map = result["depth_uncertainty_map_yx"]
        supported = result["supported_mask_yx"]

    except ValueError as exc:
        # Unsupported calibration is not a prediction failure.
        # Other input errors must still be raised.
        message = str(exc)

        if not (
            "has no calibrated lookup" in message
            or "Thresholds not calibrated" in message
        ):
            raise

        reliability = {
            "status": "UNCALIBRATED",
            "depth_uncertainty_p90_m": None,
            "high_risk_area_pct": None,
            "supported_area_pct": 0.0,
            "unsupported_area_pct": 100.0,
            "warning": message,
            "field_validated": False,
        }

        risk_map = np.full(depth.shape, np.nan)
        p90_map = np.full(depth.shape, np.nan)
        supported = np.zeros(depth.shape, dtype=bool)

    ny, nx = depth.shape

    statuses = []

    for y in range(ny):
        row = []

        for x in range(nx):
            if not valid[y, x]:
                status = "OUTSIDE_VALID_AREA"
            elif not supported[y, x]:
                status = "INSUFFICIENT_EVIDENCE"
            elif candidates[y, x]:
                status = "CANDIDATE_DETECTED"
            else:
                status = "NO_CANDIDATE_DETECTED"

            row.append(status)

        statuses.append(row)

    payload = {
        "schema_version": "BearingUIContract_V1",
        "grid": {
            "nx": int(nx),
            "ny": int(ny),
            "dx_m": float(dx_m),
            "dy_m": float(dy_m),
            "orientation": "YX",
            "depth_unit": "m",
            "coordinates": "local_grid",
        },
        "criteria": {
            "qc_min_mpa": float(qc_min_mpa),
            "ic_max": float(ic_max),
            "min_thickness_m": float(min_thickness_m),
        },
        "bearing_layer": {
            "status": (
                "CANDIDATES_FOUND"
                if candidates.any()
                else "NO_CANDIDATE_DETECTED"
            ),
            "shallowest_depth_m": shallowest,
            "deepest_depth_m": deepest,
            "variation_m": variation,
            "candidate_cells": int(candidates.sum()),
            "valid_cells": int(valid.sum()),
            "candidate_area_pct": round(
                100 * candidates.sum() / valid.sum(), 2
            ),
            "depth_map_yx": json_map(depth),
            "location_status_yx": statuses,
        },
        "reliability": {
            **reliability,
            "risk_map_yx": json_map(risk_map),
            "depth_uncertainty_map_yx": json_map(p90_map),
        },
        "limitations": {
            "field_validated": False,
            "automatic_stop_allowed": False,
            "bearing_capacity_available": False,
            "note": (
                "Potential bearing layer only. "
                "No candidate detected does not prove "
                "the absence of a bearing stratum. "
                "Uncertainty estimates are based on "
                "synthetic calibration."
            ),
        },
    }

    return payload


def save_payload(payload, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            allow_nan=False,
        ) + "\n"
    )

    return path
