"""Spatial outputs for potential bearing candidate screening.

Axis convention: X,Y,Z
Depth convention: z=0 is the top grid boundary.
No engineering bearing-capacity certification.
"""

import numpy as np
from scipy import ndimage as ndi

from bearing_rule_engine_v1 import detect_candidates


STRUCT_3D = ndi.generate_binary_structure(3, 1)


def spatial_output(
    qc,
    valid,
    spacing_m,
    cpt_sites,
    qc_min_mpa=15.0,
    min_thickness_m=2.0,
    min_lateral_fraction=0.05,
    ic=None,
    ic_range=None,
):
    qc = np.asarray(qc, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    sites = np.asarray(cpt_sites, dtype=np.int64)

    dx, dy, dz = map(float, spacing_m)
    nx, ny, nz = qc.shape

    if sites.ndim != 2 or sites.shape[1] != 2:
        raise ValueError("CPT sites must have shape (N,2)")
    if len(sites) == 0:
        raise ValueError("At least one CPT required")
    if (
        (sites[:, 0] < 0).any()
        or (sites[:, 0] >= nx).any()
        or (sites[:, 1] < 0).any()
        or (sites[:, 1] >= ny).any()
    ):
        raise ValueError("CPT site outside grid")

    result = detect_candidates(
        qc=qc,
        valid=valid,
        spacing_m=spacing_m,
        qc_min_mpa=qc_min_mpa,
        min_thickness_m=min_thickness_m,
        min_lateral_fraction=min_lateral_fraction,
        ic=ic,
        ic_range=ic_range,
    )

    mask = result["mask"]

    labels, n_components = ndi.label(
        mask, structure=STRUCT_3D
    )

    # Horizontal nearest CPT distance in metres.
    xx, yy = np.indices((nx, ny))
    nearest = np.full(
        (nx, ny), np.inf, dtype=np.float32
    )

    for sx, sy in sites:
        d = np.sqrt(
            ((xx - sx) * dx)**2
            + ((yy - sy) * dy)**2
        )
        nearest = np.minimum(nearest, d)

    top = np.full(
        (nx, ny), np.nan, dtype=np.float32
    )
    bottom = np.full(
        (nx, ny), np.nan, dtype=np.float32
    )
    total_thickness = np.zeros(
        (nx, ny), dtype=np.float32
    )
    interval_count = np.zeros(
        (nx, ny), dtype=np.int16
    )

    intervals = []

    for x in range(nx):
        for y in range(ny):
            column = mask[x, y]

            if not column.any():
                continue

            padded = np.concatenate((
                [False], column, [False]
            ))

            starts = np.flatnonzero(
                ~padded[:-1] & padded[1:]
            )
            stops = np.flatnonzero(
                padded[:-1] & ~padded[1:]
            )

            top[x, y] = starts[0] * dz
            bottom[x, y] = stops[-1] * dz
            interval_count[x, y] = len(starts)

            for start, stop in zip(starts, stops):
                thickness = (stop - start) * dz

                total_thickness[x, y] += thickness

                component_ids = np.unique(
                    labels[x, y, start:stop]
                )
                component_ids = component_ids[
                    component_ids > 0
                ]

                intervals.append({
                    "x_index": int(x),
                    "y_index": int(y),
                    "x_center_m": float(
                        (x + 0.5) * dx
                    ),
                    "y_center_m": float(
                        (y + 0.5) * dy
                    ),
                    "top_depth_m": float(
                        start * dz
                    ),
                    "bottom_depth_m": float(
                        stop * dz
                    ),
                    "thickness_m": float(
                        thickness
                    ),
                    "component_ids": [
                        int(v) for v in component_ids
                    ],
                    "nearest_cpt_m": float(
                        nearest[x, y]
                    ),
                })

    components = []

    for cid in range(1, n_components + 1):
        active = labels == cid

        if not active.any():
            continue

        coordinates = np.argwhere(active)
        footprint = active.any(axis=2)

        components.append({
            "component_id": cid,
            "voxel_count": int(active.sum()),
            "footprint_columns": int(
                footprint.sum()
            ),
            "footprint_fraction": float(
                footprint.mean()
            ),
            "horizontal_area_m2": float(
                footprint.sum() * dx * dy
            ),
            "top_depth_m": float(
                coordinates[:, 2].min() * dz
            ),
            "bottom_depth_m": float(
                (coordinates[:, 2].max() + 1) * dz
            ),
        })

    covered = mask.any(axis=2)
    valid_columns = valid.any(axis=2)

    if intervals:
        shallowest = min(
            v["top_depth_m"] for v in intervals
        )
        deepest = max(
            v["bottom_depth_m"] for v in intervals
        )
        max_layer_thickness = max(
            v["thickness_m"] for v in intervals
        )
    else:
        shallowest = None
        deepest = None
        max_layer_thickness = None

    summary = {
        "status": (
            "CANDIDATE_DETECTED"
            if intervals
            else "NO_CANDIDATE_DETECTED"
        ),
        "cpt_count": int(len(sites)),
        "grid_shape_xyz": [nx, ny, nz],
        "spacing_m": [dx, dy, dz],
        "candidate_voxels": int(mask.sum()),
        "candidate_columns": int(covered.sum()),
        "valid_columns": int(valid_columns.sum()),
        "component_count": len(components),
        "column_interval_count": len(intervals),
        "shallowest_top_depth_m": shallowest,
        "deepest_bottom_depth_m": deepest,
        "max_contiguous_thickness_m": (
            max_layer_thickness
        ),
        "criteria": result["criteria"],
        "limitations": (
            "Candidate screening only; not a "
            "bearing capacity or foundation design result. "
            "No candidate detected does not establish "
            "absence of a bearing layer. "
            "Nearest CPT distance is not calibrated "
            "uncertainty."
        ),
    }

    return {
        "mask": mask,
        "component_ids": labels,
        "top_depth_m": top,
        "bottom_depth_m": bottom,
        "total_thickness_m": total_thickness,
        "interval_count": interval_count,
        "nearest_cpt_distance_m": nearest,
        "intervals": intervals,
        "components": components,
        "summary": summary,
    }


def inspect_xy(result, x_index, y_index):
    """Return all candidate intervals at one grid column."""
    nx, ny = result["top_depth_m"].shape

    if not (0 <= x_index < nx and 0 <= y_index < ny):
        raise IndexError("XY outside grid")

    matches = [
        row for row in result["intervals"]
        if row["x_index"] == x_index
        and row["y_index"] == y_index
    ]

    return {
        "x_index": int(x_index),
        "y_index": int(y_index),
        "status": (
            "CANDIDATE_DETECTED"
            if matches
            else "NO_CANDIDATE_DETECTED"
        ),
        "nearest_cpt_m": float(
            result["nearest_cpt_distance_m"][
                x_index, y_index
            ]
        ),
        "intervals": matches,
    }
