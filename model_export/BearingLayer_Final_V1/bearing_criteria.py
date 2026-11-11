"""Strict threshold-based candidate bearing layer detector.

Input arrays use Z,Y,X orientation.
This is a candidate indicator, not a certified
foundation bearing-stratum determination.
"""

import math
import numpy as np


def potential_bearing_depth(
    qc_mpa,
    ic,
    valid_mask,
    dz_m,
    qc_min_mpa=10.0,
    ic_max=2.2,
    min_thickness_m=1.0,
):
    qc = np.asarray(qc_mpa, dtype=np.float32)
    ic = np.asarray(ic, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)

    if qc.ndim != 3:
        raise ValueError("Expected Z,Y,X volumes")

    if qc.shape != ic.shape or qc.shape != valid.shape:
        raise ValueError("Input shapes must match")

    if not all(
        np.isfinite(v) for v in (
            dz_m, qc_min_mpa, ic_max, min_thickness_m
        )
    ):
        raise ValueError("Parameters must be finite")

    if dz_m <= 0 or min_thickness_m <= 0:
        raise ValueError("Depth spacing and thickness must be > 0")

    if not (
        np.isfinite(qc[valid]).all()
        and np.isfinite(ic[valid]).all()
    ):
        raise ValueError("Nonfinite values in valid voxels")

    qualifies = (
        (qc >= qc_min_mpa)
        & (ic <= ic_max)
        & valid
    )

    nz, ny, nx = qualifies.shape

    needed = max(
        1,
        math.ceil(min_thickness_m / dz_m - 1e-9),
    )

    depths = np.full(
        (ny, nx),
        np.nan,
        dtype=np.float32,
    )

    if needed > nz:
        return depths

    run = np.zeros((ny, nx), dtype=np.int32)

    for z in range(nz):
        run = np.where(qualifies[z], run + 1, 0)

        found = (
            (run >= needed)
            & ~np.isfinite(depths)
        )

        depths[found] = (
            z - needed + 1
        ) * dz_m

    return depths
