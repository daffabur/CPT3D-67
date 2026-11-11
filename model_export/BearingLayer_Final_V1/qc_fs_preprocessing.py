"""Build the frozen QC-FS V3-D nine-channel input.

Arrays are XYZ. Output is [1,9,Z,Y,X].
CPT profiles must be provided at selected XY locations.
"""

import numpy as np


def prepare_cpt_features(
    qc_mpa,
    fs_kpa,
    observed_mask,
    valid_mask,
    dx_m,
    dy_m,
    dz_m,
    qc_norm,
    fs_norm,
):
    qc = np.asarray(qc_mpa, dtype=np.float32)
    fs = np.asarray(fs_kpa, dtype=np.float32)
    observed = np.asarray(observed_mask, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)

    if qc.ndim != 3 or any(
        a.shape != qc.shape for a in (fs, observed, valid)
    ):
        raise ValueError("Expected matching XYZ volumes")

    if not all(
        np.isfinite(v) and v > 0
        for v in (dx_m, dy_m, dz_m)
    ):
        raise ValueError("Invalid grid spacing")

    if np.any(observed & ~valid):
        raise ValueError("Observed voxels must be valid")

    selected = np.argwhere(observed.any(axis=2))
    if len(selected) == 0:
        raise ValueError("At least one CPT is required")

    px, py = selected[:, 0], selected[:, 1]
    nx, ny, nz = qc.shape

    # Match original synthetic-pilot preprocessing:
    # selected CPTs have complete valid depth profiles.
    for x, y in selected:
        if not np.all(observed[x, y, :]):
            raise ValueError(
                "V3-D preprocessing requires full-depth CPT profiles"
            )

    if not (
        np.isfinite(qc[observed]).all()
        and np.isfinite(fs[observed]).all()
    ):
        raise ValueError("Nonfinite CPT measurements")

    if np.any(qc[observed] < 0) or np.any(fs[observed] < 0):
        raise ValueError("Negative CPT measurements unsupported")

    qc_profile = (
        np.log1p(qc[px, py, :])
        - qc_norm["mean"]
    ) / qc_norm["std"]

    fs_profile = (
        np.log1p(fs[px, py, :])
        - fs_norm["mean"]
    ) / fs_norm["std"]

    sparse_qc = np.zeros((nx, ny, nz), np.float32)
    sparse_fs = np.zeros_like(sparse_qc)
    mask = observed.astype(np.float32)

    sparse_qc[px, py, :] = qc_profile
    sparse_fs[px, py, :] = fs_profile

    xx, yy = np.meshgrid(
        np.arange(nx, dtype=np.float32) * dx_m,
        np.arange(ny, dtype=np.float32) * dy_m,
        indexing="ij",
    )

    dist = np.sqrt(
        (xx[None] - px[:, None, None] * dx_m) ** 2
        + (yy[None] - py[:, None, None] * dy_m) ** 2
    )

    qc_weights = 1.0 / np.maximum(dist, 1e-5) ** 2
    qc_weights /= qc_weights.sum(axis=0, keepdims=True)

    qc_idw = np.einsum(
        "kxy,kz->xyz",
        qc_weights,
        qc_profile,
        optimize=True,
    ).astype(np.float32)

    qc_idw = np.where(mask > 0, sparse_qc, qc_idw)

    distance_scale = max(nx * dx_m, ny * dy_m, 1.0)

    distance = np.broadcast_to(
        dist.min(axis=0)[:, :, None] / distance_scale,
        qc.shape,
    ).astype(np.float32)

    # Original FS IDW uses a separate distance calculation.
    fxx, fyy = np.meshgrid(
        np.arange(nx) * dx_m,
        np.arange(ny) * dy_m,
        indexing="ij",
    )
    fs_dist2 = (
        (fxx[None] - px[:, None, None] * dx_m) ** 2
        + (fyy[None] - py[:, None, None] * dy_m) ** 2
    )

    fs_weights = 1.0 / np.maximum(fs_dist2, 1e-10)
    fs_weights /= fs_weights.sum(axis=0, keepdims=True)

    fs_idw = np.einsum(
        "nxy,nz->xyz",
        fs_weights,
        fs_profile,
    ).astype(np.float32)

    features_xyz = np.stack([
        sparse_qc,
        mask,
        qc_idw,
        distance,
        np.full_like(qc, np.log1p(dx_m)),
        np.full_like(qc, np.log1p(dy_m)),
        np.full_like(qc, np.log1p(dz_m)),
        sparse_fs,
        fs_idw,
    ]).astype(np.float32)

    return np.ascontiguousarray(
        features_xyz.transpose(0, 3, 2, 1)[None]
    )
