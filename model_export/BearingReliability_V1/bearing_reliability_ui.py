"""Experimental Bearing Reliability UI adapter.

Inputs:
  depth_yx: candidate depth map (Y,X), NaN if undetected
  cpt_xy: CPT grid coordinates (X,Y)
  dx_m, dy_m: grid cell dimensions in metres

Outputs are synthetic-calibration estimates, NOT field confidence.
"""

import json
from pathlib import Path

import numpy as np


DEFAULT_LOOKUP = (
    Path(__file__).resolve().parent
    / "reliability_lookup.json"
)


def build_reliability(
    depth_yx,
    cpt_xy,
    dx_m,
    dy_m,
    lookup_path=DEFAULT_LOOKUP,
    valid_xy=None,
    qc_min_mpa=10.0,
    ic_max=2.2,
    min_thickness_m=1.0,
):
    lookup = json.loads(Path(lookup_path).read_text())

    depth = np.asarray(depth_yx, dtype=np.float32)
    cpts = np.asarray(cpt_xy, dtype=np.int32)

    if depth.ndim != 2:
        raise ValueError("Depth map must have shape (Y,X)")

    if cpts.ndim != 2 or cpts.shape[1] != 2 or len(cpts) == 0:
        raise ValueError("CPT coordinates must have shape (N,2)")

    if len(np.unique(cpts, axis=0)) != len(cpts):
        raise ValueError("Duplicate CPT coordinates")

    if not (
        np.isfinite(dx_m) and np.isfinite(dy_m)
        and dx_m > 0 and dy_m > 0
    ):
        raise ValueError("Invalid grid resolution")

    ny, nx = depth.shape

    if (
        np.any(cpts[:, 0] < 0)
        or np.any(cpts[:, 0] >= nx)
        or np.any(cpts[:, 1] < 0)
        or np.any(cpts[:, 1] >= ny)
    ):
        raise ValueError("CPT coordinates outside grid")

    if valid_xy is None:
        valid = np.ones(depth.shape, dtype=bool)
    else:
        valid = np.asarray(valid_xy, dtype=bool)
        if valid.shape != depth.shape:
            raise ValueError("Invalid XY mask shape")

    if not valid.any():
        raise ValueError("No valid XY area")

    criteria = lookup["criteria"]
    requested = (
        qc_min_mpa, ic_max, min_thickness_m
    )
    calibrated = (
        criteria["qc_min_mpa"],
        criteria["ic_max"],
        criteria["min_thickness_m"],
    )

    if not np.allclose(requested, calibrated, rtol=0, atol=1e-9):
        raise ValueError(
            "Thresholds not calibrated; display uncertainty as unavailable"
        )

    n_cpt = len(cpts)
    rows = [
        r for r in lookup["lookup"]
        if r["cpt"] == n_cpt
    ]

    if not rows:
        raise ValueError(
            f"CPT count {n_cpt} has no calibrated lookup"
        )

    yy, xx = np.indices(depth.shape)

    dist2 = (
        ((xx[None] - cpts[:, 0, None, None]) * dx_m) ** 2
        + ((yy[None] - cpts[:, 1, None, None]) * dy_m) ** 2
    )

    nearest_m = np.sqrt(dist2.min(axis=0))

    risk = np.full(depth.shape, np.nan, np.float32)
    p90 = np.full(depth.shape, np.nan, np.float32)

    for row in rows:
        lo = float(row["distance_min_m"])
        hi = row["distance_max_m"]
        hi = float("inf") if hi is None else float(hi)

        selected = valid & (nearest_m >= lo) & (nearest_m < hi)

        if (
            selected.any()
            and row["supported"]
            and row["estimated_detection_risk"] is not None
            and row["conditional_depth_p90_m"] is not None
        ):
            risk[selected] = row["estimated_detection_risk"]
            p90[selected] = row["conditional_depth_p90_m"]

    supported = valid & np.isfinite(risk)
    predicted = supported & np.isfinite(depth)

    total_area = int(valid.sum())
    supported_area = int(supported.sum())

    high_risk_threshold = float(
        lookup["high_risk_threshold"]
    )

    high_risk = supported & (risk >= high_risk_threshold)

    # Conservative summary of conditional P90:
    # maximum across cells with detected candidate depths.
    # It is NOT an unconditional depth confidence interval.
    uncertainty = (
        float(np.max(p90[predicted]))
        if predicted.any() else None
    )

    result = {
        "status": (
            "EXPERIMENTAL"
            if supported_area == total_area
            else "PARTIALLY_SUPPORTED"
            if supported_area > 0
            else "INSUFFICIENT_EVIDENCE"
        ),
        "cpt_count": n_cpt,
        "depth_uncertainty_p90_m": uncertainty,
        "depth_uncertainty_type": "conditional_on_true_positive",
        "high_risk_area_pct": (
            round(100 * high_risk.sum() / supported_area, 2)
            if supported_area else None
        ),
        "high_risk_denominator": "supported_area_only",
        "supported_area_pct": round(
            100 * supported_area / total_area, 2
        ),
        "unsupported_area_pct": round(
            100 * (total_area - supported_area) / total_area, 2
        ),
        "high_risk_threshold": high_risk_threshold,
        "field_validated": False,
        "warning": (
            "Synthetic calibration only; estimated detection "
            "risk underestimated holdout error by approximately "
            "3-4 percentage points. No automatic engineering decisions."
        ),
    }

    # Do not claim depth uncertainty where no candidate was detected.
    p90_display = np.where(predicted, p90, np.nan)

    return {
        "summary": result,
        "risk_map_yx": risk,
        "depth_uncertainty_map_yx": p90_display,
        "supported_mask_yx": supported,
        "high_risk_mask_yx": high_risk,
    }


if __name__ == "__main__":
    # Synthetic smoke test; not an actual project prediction.
    dummy_depth = np.full((24, 24), np.nan, np.float32)
    dummy_depth[5:20, 5:20] = 3.0

    output = build_reliability(
        depth_yx=dummy_depth,
        cpt_xy=[[4, 4], [12, 12], [20, 20]],
        dx_m=1.0,
        dy_m=1.0,
    )

    print("=== BEARING RELIABILITY UI SMOKE TEST ===")
    print(json.dumps(output["summary"], indent=2))
    print("Risk map:", output["risk_map_yx"].shape)
    print("P90 map:", output["depth_uncertainty_map_yx"].shape)
    print("PASS: UI adapter")
