"""Deterministic 3D potential bearing candidate rule engine.

Volume axis order: (X, Y, Z).
Outputs are screening candidates, not verified bearing strata.
"""

import math
import numpy as np
from scipy import ndimage as ndi


STRUCT_3D = ndi.generate_binary_structure(3, 1)

STRUCT_Z = np.zeros((3, 3, 3), dtype=np.uint8)
STRUCT_Z[1, 1, :] = 1


def detect_candidates(
    qc,
    valid,
    spacing_m,
    qc_min_mpa=15.0,
    min_thickness_m=2.0,
    min_lateral_fraction=0.05,
    ic=None,
    ic_range=None,
):
    qc = np.asarray(qc, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)

    if qc.ndim != 3 or qc.shape != valid.shape:
        raise ValueError(
            "qc and valid must have identical 3D (X,Y,Z) shape"
        )

    dx, dy, dz = map(float, spacing_m)

    if not all(
        np.isfinite(x) and x > 0
        for x in (dx, dy, dz)
    ):
        raise ValueError("Invalid grid spacing")

    if not (
        np.isfinite(qc_min_mpa)
        and np.isfinite(min_thickness_m)
        and min_thickness_m > 0
        and 0 <= min_lateral_fraction <= 1
    ):
        raise ValueError("Invalid identification criteria")

    eligible = (
        valid
        & np.isfinite(qc)
        & (qc >= qc_min_mpa)
    )

    # Optional, experimental Ic filter.
    # Disabled by default and not part of legacy labels.
    if ic_range is not None:
        if ic is None:
            raise ValueError("ic required when ic_range is set")

        ic = np.asarray(ic, dtype=np.float32)

        if ic.shape != qc.shape:
            raise ValueError("Ic grid shape mismatch")

        low, high = map(float, ic_range)

        if not (
            np.isfinite(low)
            and np.isfinite(high)
            and low <= high
        ):
            raise ValueError("Invalid Ic range")

        eligible &= (
            np.isfinite(ic)
            & (ic >= low)
            & (ic <= high)
        )

    min_cells = max(
        1,
        math.ceil(min_thickness_m / dz - 1e-9)
    )

    vertical_ids, n_vertical = ndi.label(
        eligible,
        structure=STRUCT_Z
    )

    run_lengths = np.bincount(
        vertical_ids.ravel(),
        minlength=n_vertical + 1
    )
    run_lengths[0] = 0

    thick = eligible & (
        run_lengths[vertical_ids] >= min_cells
    )

    connected_ids, n_components = ndi.label(
        thick,
        structure=STRUCT_3D
    )

    nx, ny, _ = qc.shape

    min_columns = max(
        1,
        math.ceil(
            min_lateral_fraction * nx * ny
        )
    )

    accepted_ids = np.zeros(
        n_components + 1,
        dtype=bool
    )

    kept_components = 0

    for component_id, slices in enumerate(
        ndi.find_objects(connected_ids), 1
    ):
        if slices is None:
            continue

        local = (
            connected_ids[slices] == component_id
        )

        lateral_columns = int(
            np.count_nonzero(
                np.any(local, axis=2)
            )
        )

        if lateral_columns >= min_columns:
            accepted_ids[component_id] = True
            kept_components += 1

    mask = thick & accepted_ids[connected_ids]

    return {
        "mask": mask,
        "component_count": kept_components,
        "candidate_voxels": int(mask.sum()),
        "candidate_fraction": (
            float(mask.sum()) / int(valid.sum())
            if valid.any() else 0.0
        ),
        "minimum_vertical_cells": min_cells,
        "minimum_lateral_columns": min_columns,
        "criteria": {
            "qc_min_mpa": float(qc_min_mpa),
            "min_thickness_m": float(min_thickness_m),
            "min_lateral_fraction": float(
                min_lateral_fraction
            ),
            "ic_range": (
                list(ic_range)
                if ic_range is not None else None
            ),
        },
    }
