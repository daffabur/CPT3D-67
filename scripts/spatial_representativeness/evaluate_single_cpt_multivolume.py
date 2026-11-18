from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from synthetic3d.v2.core import Grid
from dqn3d.v45.adapter import FrozenV4Reconstructor


ROOT = Path.home() / "Developer" / "cpt_super_learn"

DATA_DIR = (
    ROOT
    / "data"
    / "synthetic3d_v4_geosyn"
    / "full_24k"
    / "test"
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

OUT_DIR = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "single_cpt_multivolume"
)


# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Pilih 5 posisi CPT yang tersebar
# ------------------------------------------------------------

ANCHORS = [
    (0.50, 0.50),  # center
    (0.25, 0.25),
    (0.25, 0.75),
    (0.75, 0.25),
    (0.75, 0.75),
]


def choose_cpt_positions(
    valid: np.ndarray,
    n_positions: int,
) -> list[tuple[int, int]]:

    nx, ny, _ = valid.shape

    # Untuk pilot ini gunakan CPT yang valid sepanjang depth.
    full_depth = valid.all(axis=2)

    candidates = np.argwhere(full_depth)

    if len(candidates) < n_positions:
        raise RuntimeError(
            f"Hanya ada {len(candidates)} full-depth CPT candidates"
        )

    selected: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()

    anchors = ANCHORS[:n_positions]

    for ax, ay in anchors:

        target = np.array([
            ax * (nx - 1),
            ay * (ny - 1),
        ])

        distances = np.linalg.norm(
            candidates.astype(float) - target[None, :],
            axis=1,
        )

        for idx in np.argsort(distances):
            x, y = candidates[idx]
            pos = (int(x), int(y))

            if pos not in used:
                selected.append(pos)
                used.add(pos)
                break

    if len(selected) != n_positions:
        raise RuntimeError(
            f"Gagal memilih {n_positions} posisi CPT unik"
        )

    return selected


# ------------------------------------------------------------
# Distance field
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# IDW baseline
# ------------------------------------------------------------

def single_cpt_idw(
    ic: np.ndarray,
    cpt_x: int,
    cpt_y: int,
) -> np.ndarray:
    """
    Untuk 1 CPT full-depth, IDW secara lateral sama dengan
    menyebarkan profil CPT tersebut ke seluruh XY.
    """

    profile = ic[cpt_x, cpt_y, :]

    return np.broadcast_to(
        profile[None, None, :],
        ic.shape,
    ).astype(np.float32).copy()


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def metric_row(
    errors: np.ndarray,
    squared_errors: np.ndarray,
) -> dict:

    return {
        "mae_ic": float(errors.mean()),
        "rmse_ic": float(
            np.sqrt(squared_errors.mean())
        ),
        "p90_abs_error_ic": float(
            np.quantile(errors, 0.90)
        ),
        "frac_le_0_10": float(
            np.mean(errors <= 0.10)
        ),
        "frac_le_0_15": float(
            np.mean(errors <= 0.15)
        ),
        "frac_le_0_20": float(
            np.mean(errors <= 0.20)
        ),
    }


# ------------------------------------------------------------
# CSV helper
# ------------------------------------------------------------

def write_csv(
    path: Path,
    rows: list[dict],
):
    if not rows:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# Per-volume aggregation
# ------------------------------------------------------------

METRIC_NAMES = [
    "sgan_mae_ic",
    "sgan_rmse_ic",
    "sgan_p90_abs_error_ic",
    "sgan_frac_le_0_10",
    "sgan_frac_le_0_15",
    "sgan_frac_le_0_20",
    "idw_mae_ic",
    "idw_rmse_ic",
    "idw_p90_abs_error_ic",
    "idw_frac_le_0_10",
    "idw_frac_le_0_15",
    "idw_frac_le_0_20",
]


def aggregate_positions_within_volume(
    raw_rows: list[dict],
) -> list[dict]:

    groups = defaultdict(list)

    for row in raw_rows:
        key = (
            row["volume"],
            float(row["distance_from_m"]),
            float(row["distance_to_m"]),
        )
        groups[key].append(row)

    result = []

    for (
        volume,
        lo,
        hi,
    ), rows in sorted(groups.items()):

        out = {
            "volume": volume,
            "scenario": rows[0]["scenario"],
            "family": rows[0]["family"],
            "distance_from_m": lo,
            "distance_to_m": hi,
            "distance_mid_m": (lo + hi) / 2.0,
            "n_positions": len(rows),
        }

        for metric in METRIC_NAMES:
            vals = np.array(
                [float(r[metric]) for r in rows],
                dtype=float,
            )
            out[metric] = float(vals.mean())

        result.append(out)

    return result


# ------------------------------------------------------------
# Bootstrap across VOLUMES
# ------------------------------------------------------------

def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float]:

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")

    mean = float(values.mean())

    if len(values) == 1:
        return mean, mean, mean

    samples = rng.choice(
        values,
        size=(n_bootstrap, len(values)),
        replace=True,
    )

    boot_means = samples.mean(axis=1)

    low, high = np.quantile(
        boot_means,
        [0.025, 0.975],
    )

    return mean, float(low), float(high)


def aggregate_across_volumes(
    per_volume_rows: list[dict],
    n_bootstrap: int,
    seed: int,
) -> list[dict]:

    rng = np.random.default_rng(seed)

    groups = defaultdict(list)

    for row in per_volume_rows:
        key = (
            float(row["distance_from_m"]),
            float(row["distance_to_m"]),
        )
        groups[key].append(row)

    result = []

    for (lo, hi), rows in sorted(groups.items()):

        out = {
            "distance_from_m": lo,
            "distance_to_m": hi,
            "distance_mid_m": (lo + hi) / 2.0,
            "n_volumes": len(rows),
        }

        for metric in METRIC_NAMES:

            vals = np.array(
                [float(r[metric]) for r in rows],
                dtype=float,
            )

            mean, ci_low, ci_high = bootstrap_mean_ci(
                vals,
                rng,
                n_bootstrap,
            )

            out[f"{metric}_mean"] = mean
            out[f"{metric}_ci95_low"] = ci_low
            out[f"{metric}_ci95_high"] = ci_high

        # Improvement > 0 berarti SchemaGAN lebih baik.
        sgan_mae = np.array(
            [float(r["sgan_mae_ic"]) for r in rows]
        )
        idw_mae = np.array(
            [float(r["idw_mae_ic"]) for r in rows]
        )

        improvement = idw_mae - sgan_mae

        mean, ci_low, ci_high = bootstrap_mean_ci(
            improvement,
            rng,
            n_bootstrap,
        )

        out["mae_improvement_vs_idw_mean"] = mean
        out["mae_improvement_vs_idw_ci95_low"] = ci_low
        out["mae_improvement_vs_idw_ci95_high"] = ci_high

        result.append(out)

    return result


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-volumes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--positions-per-volume",
        type=int,
        default=5,
        choices=(1, 2, 3, 4, 5),
    )

    parser.add_argument(
        "--bin-width-m",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=67,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.n_volumes < 1:
        raise ValueError("--n-volumes must be >= 1")

    if args.bin_width_m <= 0:
        raise ValueError("--bin-width-m must be > 0")

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_paths = sorted(DATA_DIR.glob("*.npz"))

    if len(all_paths) < args.n_volumes:
        raise RuntimeError(
            f"Hanya ada {len(all_paths)} volumes"
        )

    # --------------------------------------------------------
    # Freeze pilot volume selection
    # --------------------------------------------------------

    rng = np.random.default_rng(args.seed)

    selected_indices = rng.choice(
        len(all_paths),
        size=args.n_volumes,
        replace=False,
    )

    selected_paths = [
        all_paths[int(i)]
        for i in sorted(selected_indices)
    ]

    manifest = {
        "study": "SR-01",
        "stage": "single_cpt_multivolume_pilot",
        "seed": args.seed,
        "n_volumes": args.n_volumes,
        "positions_per_volume": args.positions_per_volume,
        "bin_width_m": args.bin_width_m,
        "volumes": [
            p.name for p in selected_paths
        ],
        "note": (
            "Pilot subset only. Do not reuse these volumes "
            "as final independent confirmatory data."
        ),
    }

    (
        OUT_DIR / "pilot_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("=== SR-01 SINGLE CPT MULTI-VOLUME ===")
    print("Volumes              :", args.n_volumes)
    print(
        "CPT positions/volume :",
        args.positions_per_volume,
    )
    print("Distance bin          :", args.bin_width_m, "m")
    print("Checkpoint            :", CHECKPOINT)
    print()

    # --------------------------------------------------------
    # Load frozen model ONCE
    # --------------------------------------------------------

    reconstructor = FrozenV4Reconstructor(
        checkpoint=CHECKPOINT,
        normalization_path=NORMALIZATION,
        device=args.device,
    )

    print("Device:", reconstructor.device)
    print()

    raw_rows: list[dict] = []

    total_inferences = 0
    total_elapsed = 0.0
    skipped_volumes = []

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    for volume_index, path in enumerate(
        selected_paths,
        start=1,
    ):

        with np.load(
            path,
            allow_pickle=False,
        ) as d:

            ic = d["Ic"].astype(np.float32)
            valid = d["valid_mask"].astype(bool)
            meta = json.loads(str(d["metadata_json"]))

        grid = build_grid(meta)

        try:
            cpt_positions = choose_cpt_positions(
                valid,
                args.positions_per_volume,
            )
        except RuntimeError as exc:
            skipped_volumes.append({
                "volume": path.name,
                "reason": str(exc),
            })
            continue

        for position_index, (
            cpt_x,
            cpt_y,
        ) in enumerate(cpt_positions):

            observed_ic = np.zeros_like(
                ic,
                dtype=np.float32,
            )

            observed_mask = np.zeros_like(
                valid,
                dtype=bool,
            )

            observed_mask[
                cpt_x,
                cpt_y,
                :
            ] = valid[
                cpt_x,
                cpt_y,
                :
            ]

            observed_ic[observed_mask] = (
                ic[observed_mask]
            )

            prediction, elapsed = reconstructor.predict(
                observed_ic=observed_ic,
                observed_mask=observed_mask,
                valid_mask=valid,
                grid=grid,
            )

            total_inferences += 1
            total_elapsed += elapsed

            idw_prediction = single_cpt_idw(
                ic,
                cpt_x,
                cpt_y,
            )

            hidden = (
                valid
                & (~observed_mask)
            )

            sgan_abs = np.abs(
                prediction - ic
            )

            sgan_sq = (
                prediction - ic
            ) ** 2

            idw_abs = np.abs(
                idw_prediction - ic
            )

            idw_sq = (
                idw_prediction - ic
            ) ** 2

            distance_xy = horizontal_distance_field(
                grid,
                cpt_x,
                cpt_y,
            )

            distance_3d = np.broadcast_to(
                distance_xy[:, :, None],
                ic.shape,
            )

            max_distance = float(
                distance_xy.max()
            )

            max_edge = (
                np.ceil(
                    max_distance
                    / args.bin_width_m
                )
                * args.bin_width_m
            )

            edges = np.arange(
                0.0,
                max_edge + args.bin_width_m,
                args.bin_width_m,
            )

            for lo, hi in zip(
                edges[:-1],
                edges[1:],
            ):

                selected = (
                    hidden
                    & (distance_3d >= lo)
                    & (distance_3d < hi)
                )

                n_voxels = int(
                    selected.sum()
                )

                if n_voxels == 0:
                    continue

                sgan_metrics = metric_row(
                    sgan_abs[selected],
                    sgan_sq[selected],
                )

                idw_metrics = metric_row(
                    idw_abs[selected],
                    idw_sq[selected],
                )

                row = {
                    "volume": path.name,
                    "scenario": meta.get(
                        "scenario",
                        "",
                    ),
                    "family": meta.get(
                        "family",
                        "",
                    ),
                    "grid_nx": grid.nx,
                    "grid_ny": grid.ny,
                    "grid_nz": grid.nz,
                    "dx_m": grid.dx_m,
                    "dy_m": grid.dy_m,
                    "dz_m": grid.dz_m,
                    "position_index": position_index,
                    "cpt_x": cpt_x,
                    "cpt_y": cpt_y,
                    "cpt_x_m": (
                        cpt_x + 0.5
                    ) * grid.dx_m,
                    "cpt_y_m": (
                        cpt_y + 0.5
                    ) * grid.dy_m,
                    "distance_from_m": float(lo),
                    "distance_to_m": float(hi),
                    "distance_mid_m": float(
                        (lo + hi) / 2.0
                    ),
                    "n_voxels": n_voxels,
                }

                for key, value in sgan_metrics.items():
                    row[f"sgan_{key}"] = value

                for key, value in idw_metrics.items():
                    row[f"idw_{key}"] = value

                raw_rows.append(row)

        if (
            volume_index % 10 == 0
            or volume_index == len(selected_paths)
        ):
            print(
                f"Processed "
                f"{volume_index}/{len(selected_paths)} "
                f"volumes | "
                f"inferences={total_inferences}",
                flush=True,
            )

    if not raw_rows:
        raise RuntimeError(
            "No evaluation rows were produced."
        )

    # --------------------------------------------------------
    # Save raw rows
    # --------------------------------------------------------

    raw_path = OUT_DIR / "raw_position_bins.csv"

    write_csv(
        raw_path,
        raw_rows,
    )

    # --------------------------------------------------------
    # Average 5 CPT positions WITHIN each volume first
    # --------------------------------------------------------

    per_volume_rows = (
        aggregate_positions_within_volume(
            raw_rows
        )
    )

    per_volume_path = (
        OUT_DIR
        / "per_volume_distance_bins.csv"
    )

    write_csv(
        per_volume_path,
        per_volume_rows,
    )

    # --------------------------------------------------------
    # Aggregate across independent volumes + bootstrap CI
    # --------------------------------------------------------

    aggregate_rows = (
        aggregate_across_volumes(
            per_volume_rows,
            n_bootstrap=args.bootstrap,
            seed=args.seed + 1000,
        )
    )

    aggregate_path = (
        OUT_DIR
        / "aggregate_distance_error.csv"
    )

    write_csv(
        aggregate_path,
        aggregate_rows,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "study": "SR-01",
        "stage": "single_cpt_multivolume_pilot",
        "requested_volumes": args.n_volumes,
        "successful_volumes": len(
            {
                r["volume"]
                for r in per_volume_rows
            }
        ),
        "skipped_volumes": skipped_volumes,
        "positions_per_volume": (
            args.positions_per_volume
        ),
        "total_inferences": total_inferences,
        "model_inference_seconds": total_elapsed,
        "distance_bin_width_m": (
            args.bin_width_m
        ),
        "bootstrap_replicates": (
            args.bootstrap
        ),
        "bootstrap_unit": "volume",
        "checkpoint": str(CHECKPOINT),
        "important_note": (
            "This pilot does NOT define a universal "
            "CPT representative radius."
        ),
    }

    (
        OUT_DIR / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ) + "\n"
    )

    # --------------------------------------------------------
    # Print concise table
    # --------------------------------------------------------

    print()
    print("=== AGGREGATED DISTANCE vs ERROR ===")
    print(
        f"{'Distance':>12} "
        f"{'Nvol':>6} "
        f"{'SGAN MAE':>10} "
        f"{'IDW MAE':>10} "
        f"{'Δ vs IDW':>10} "
        f"{'SGAN <=.15':>11}"
    )

    for row in aggregate_rows:

        label = (
            f"{row['distance_from_m']:.0f}-"
            f"{row['distance_to_m']:.0f}m"
        )

        print(
            f"{label:>12} "
            f"{row['n_volumes']:6d} "
            f"{row['sgan_mae_ic_mean']:10.5f} "
            f"{row['idw_mae_ic_mean']:10.5f} "
            f"{row['mae_improvement_vs_idw_mean']:10.5f} "
            f"{row['sgan_frac_le_0_15_mean']*100:10.1f}%"
        )

    print()
    print("Saved:")
    print(" ", raw_path)
    print(" ", per_volume_path)
    print(" ", aggregate_path)
    print(" ", OUT_DIR / "pilot_manifest.json")
    print(" ", OUT_DIR / "summary.json")

    print()
    print(
        "Interpretation rule: Δ vs IDW > 0 means "
        "SchemaGAN has lower MAE than IDW."
    )

    print(
        "Do NOT interpret this pilot as the final "
        "representative radius."
    )


if __name__ == "__main__":
    main()
