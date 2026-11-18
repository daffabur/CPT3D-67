from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from synthetic3d.v2.core import Grid
from dqn3d.v45.adapter import FrozenV4Reconstructor


ROOT = Path.home() / "Developer" / "cpt_super_learn"

DATA_PATH = (
    ROOT
    / "data"
    / "synthetic3d_v4_geosyn"
    / "full_24k"
    / "test"
    / "v4_test_03345.npz"
)

CHECKPOINT = (
    ROOT
    / "model_export"
    / "SchemaGAN_V4_2"
    / "SchemaGAN_V4_2.pt"
)

NORMALIZATION = (
    ROOT
    / "model_export"
    / "SchemaGAN_V4_2"
    / "normalization.json"
)

OUT_DIR = ROOT / "results" / "spatial_representativeness"


def build_grid(meta: dict) -> Grid:
    g = meta["grid"]

    return Grid(
        nx=int(g["nx"]),
        ny=int(g["ny"]),
        nz=int(g["nz"]),
        dx_m=float(g["dx_m"]),
        dy_m=float(g["dy_m"]),
        dz_m=float(g["dz_m"]),
    )


def choose_center_valid_cpt(valid: np.ndarray) -> tuple[int, int]:
    """
    Untuk smoke test:
    pilih kolom CPT full-depth yang paling dekat pusat grid.
    """

    nx, ny, nz = valid.shape

    full_depth = valid.all(axis=2)
    candidates = np.argwhere(full_depth)

    if len(candidates) == 0:
        raise RuntimeError(
            "Tidak ada kolom CPT yang valid sepanjang seluruh depth."
        )

    center = np.array([
        (nx - 1) / 2.0,
        (ny - 1) / 2.0,
    ])

    distances = np.linalg.norm(
        candidates.astype(float) - center[None, :],
        axis=1,
    )

    best = candidates[np.argmin(distances)]

    return int(best[0]), int(best[1])


def horizontal_distance_field(
    grid: Grid,
    cpt_x: int,
    cpt_y: int,
) -> np.ndarray:

    x_m = (
        np.arange(grid.nx, dtype=np.float32) - cpt_x
    ) * grid.dx_m

    y_m = (
        np.arange(grid.ny, dtype=np.float32) - cpt_y
    ) * grid.dy_m

    xx, yy = np.meshgrid(
        x_m,
        y_m,
        indexing="ij",
    )

    return np.sqrt(xx**2 + yy**2)


def main():

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== SR-01 SINGLE CPT SMOKE TEST ===")
    print("Data       :", DATA_PATH)
    print("Checkpoint :", CHECKPOINT)
    print()

    with np.load(DATA_PATH, allow_pickle=False) as d:
        ic = d["Ic"].astype(np.float32)
        valid = d["valid_mask"].astype(bool)
        unit_id = d["unit_id"].astype(np.int16)
        meta = json.loads(str(d["metadata_json"]))

    grid = build_grid(meta)

    print(
        f"Grid        : {grid.nx} x {grid.ny} x {grid.nz}"
    )
    print(
        f"Resolution  : "
        f"{grid.dx_m} x {grid.dy_m} x {grid.dz_m} m"
    )
    print(
        f"Physical    : "
        f"{grid.nx * grid.dx_m:.1f} x "
        f"{grid.ny * grid.dy_m:.1f} x "
        f"{grid.nz * grid.dz_m:.1f} m"
    )

    # ---------------------------------------------------------
    # 1. Pilih satu CPT untuk smoke test
    # ---------------------------------------------------------

    cpt_x, cpt_y = choose_center_valid_cpt(valid)

    print()
    print(
        f"Selected CPT: grid ({cpt_x}, {cpt_y})"
    )
    print(
        f"Physical XY : "
        f"x={(cpt_x + 0.5) * grid.dx_m:.1f} m, "
        f"y={(cpt_y + 0.5) * grid.dy_m:.1f} m"
    )

    # ---------------------------------------------------------
    # 2. Buat observed CPT profile
    # ---------------------------------------------------------

    observed_ic = np.zeros_like(ic, dtype=np.float32)
    observed_mask = np.zeros_like(valid, dtype=bool)

    observed_mask[cpt_x, cpt_y, :] = valid[cpt_x, cpt_y, :]
    observed_ic[observed_mask] = ic[observed_mask]

    print(
        "Measured CPT voxels:",
        int(observed_mask.sum()),
    )

    # ---------------------------------------------------------
    # 3. Load frozen SchemaGAN V4.2
    # ---------------------------------------------------------

    reconstructor = FrozenV4Reconstructor(
        checkpoint=CHECKPOINT,
        normalization_path=NORMALIZATION,
    )

    print("Device      :", reconstructor.device)

    # ---------------------------------------------------------
    # 4. Full reconstruction
    # ---------------------------------------------------------

    prediction, elapsed = reconstructor.predict(
        observed_ic=observed_ic,
        observed_mask=observed_mask,
        valid_mask=valid,
        grid=grid,
    )

    print(
        f"Inference   : {elapsed:.4f} s"
    )

    # ---------------------------------------------------------
    # 5. Error hanya pada voxel yang TIDAK diukur
    # ---------------------------------------------------------

    hidden = valid & (~observed_mask)

    abs_error = np.abs(prediction - ic)
    sq_error = (prediction - ic) ** 2

    overall_mae = float(
        abs_error[hidden].mean()
    )

    overall_rmse = float(
        np.sqrt(sq_error[hidden].mean())
    )

    overall_p90 = float(
        np.quantile(abs_error[hidden], 0.90)
    )

    print()
    print("=== OVERALL HIDDEN ERROR ===")
    print(f"MAE       : {overall_mae:.6f} Ic")
    print(f"RMSE      : {overall_rmse:.6f} Ic")
    print(f"P90 error : {overall_p90:.6f} Ic")

    # ---------------------------------------------------------
    # 6. Horizontal distance dari CPT
    # ---------------------------------------------------------

    distance_xy = horizontal_distance_field(
        grid,
        cpt_x,
        cpt_y,
    )

    distance_3d = np.broadcast_to(
        distance_xy[:, :, None],
        ic.shape,
    )

    # Gunakan physical distance bins.
    # Untuk smoke test, bin width = resolusi XY terkecil.
    bin_width = float(
        min(grid.dx_m, grid.dy_m)
    )

    max_distance = float(
        distance_xy.max()
    )

    upper = (
        math.ceil(max_distance / bin_width)
        * bin_width
    )

    bin_edges = np.arange(
        0.0,
        upper + bin_width,
        bin_width,
        dtype=float,
    )

    rows = []

    for lo, hi in zip(
        bin_edges[:-1],
        bin_edges[1:],
    ):

        selected = (
            hidden
            & (distance_3d >= lo)
            & (distance_3d < hi)
        )

        n = int(selected.sum())

        if n == 0:
            continue

        errors = abs_error[selected]

        rows.append({
            "distance_from_m": lo,
            "distance_to_m": hi,
            "distance_mid_m": (lo + hi) / 2.0,
            "n_voxels": n,
            "mae_ic": float(errors.mean()),
            "rmse_ic": float(
                np.sqrt(
                    sq_error[selected].mean()
                )
            ),
            "p90_abs_error_ic": float(
                np.quantile(errors, 0.90)
            ),
            "fraction_error_le_0_10": float(
                np.mean(errors <= 0.10)
            ),
            "fraction_error_le_0_15": float(
                np.mean(errors <= 0.15)
            ),
            "fraction_error_le_0_20": float(
                np.mean(errors <= 0.20)
            ),
        })

    # ---------------------------------------------------------
    # 7. Save result
    # ---------------------------------------------------------

    csv_path = OUT_DIR / "single_cpt_distance_error.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "study": "SR-01",
        "mode": "single_cpt_smoke_test",
        "volume": DATA_PATH.name,
        "scenario": meta.get("scenario"),
        "grid": {
            "nx": grid.nx,
            "ny": grid.ny,
            "nz": grid.nz,
            "dx_m": grid.dx_m,
            "dy_m": grid.dy_m,
            "dz_m": grid.dz_m,
        },
        "cpt": {
            "grid_x": cpt_x,
            "grid_y": cpt_y,
            "x_m": (cpt_x + 0.5) * grid.dx_m,
            "y_m": (cpt_y + 0.5) * grid.dy_m,
            "measured_voxels": int(observed_mask.sum()),
        },
        "overall_hidden": {
            "mae_ic": overall_mae,
            "rmse_ic": overall_rmse,
            "p90_abs_error_ic": overall_p90,
        },
        "distance_bin_width_m": bin_width,
        "inference_seconds": elapsed,
    }

    json_path = OUT_DIR / "single_cpt_summary.json"

    json_path.write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    npz_path = OUT_DIR / "single_cpt_smoke.npz"

    np.savez_compressed(
        npz_path,
        ground_truth_ic=ic,
        prediction_ic=prediction,
        absolute_error_ic=abs_error,
        valid_mask=valid,
        observed_mask=observed_mask,
        unit_id=unit_id,
        distance_xy_m=distance_xy,
    )

    # ---------------------------------------------------------
    # 8. Print distance-error curve
    # ---------------------------------------------------------

    print()
    print("=== DISTANCE vs ERROR ===")
    print(
        f"{'Distance':>15} "
        f"{'N':>8} "
        f"{'MAE':>10} "
        f"{'RMSE':>10} "
        f"{'P90':>10} "
        f"{'<=0.15':>10}"
    )

    for row in rows:
        label = (
            f"{row['distance_from_m']:.0f}-"
            f"{row['distance_to_m']:.0f}m"
        )

        print(
            f"{label:>15} "
            f"{row['n_voxels']:8d} "
            f"{row['mae_ic']:10.5f} "
            f"{row['rmse_ic']:10.5f} "
            f"{row['p90_abs_error_ic']:10.5f} "
            f"{row['fraction_error_le_0_15']*100:9.1f}%"
        )

    print()
    print("Saved:")
    print(" ", csv_path)
    print(" ", json_path)
    print(" ", npz_path)

    print()
    print(
        "NOTE: Ini belum menentukan representative radius. "
        "Ini hanya mengukur hubungan distance-error untuk "
        "1 CPT pada 1 volume."
    )


if __name__ == "__main__":
    main()
