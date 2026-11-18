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

SR02_MANIFEST = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "single_cpt_by_grid"
    / "pilot_manifest.json"
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
    / "multicpt_support"
)

CPT_COUNTS = [1, 2, 3, 4, 5, 8, 12, 16, 24, 32]


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


def farthest_sequence(
    valid: np.ndarray,
    grid: Grid,
    max_count: int,
) -> list[tuple[int, int]]:

    candidates = np.argwhere(
        valid.all(axis=2)
    )

    if len(candidates) < max_count:
        raise RuntimeError(
            f"Only {len(candidates)} valid CPT columns"
        )

    nx, ny, _ = valid.shape

    center = np.array([
        (nx - 1) / 2.0,
        (ny - 1) / 2.0,
    ])

    # CPT pertama = kandidat terdekat pusat.
    first_idx = np.argmin(
        np.linalg.norm(
            candidates.astype(float)
            - center[None, :],
            axis=1,
        )
    )

    first = tuple(
        int(v)
        for v in candidates[first_idx]
    )

    selected = [first]
    selected_set = {first}

    while len(selected) < max_count:

        best = None
        best_distance = -1.0

        for x, y in candidates:

            pos = (int(x), int(y))

            if pos in selected_set:
                continue

            min_distance = min(
                np.sqrt(
                    ((x - sx) * grid.dx_m) ** 2
                    +
                    ((y - sy) * grid.dy_m) ** 2
                )
                for sx, sy in selected
            )

            if min_distance > best_distance:
                best_distance = float(
                    min_distance
                )
                best = pos

        selected.append(best)
        selected_set.add(best)

    return selected


def nearest_distance_field(
    grid: Grid,
    positions: list[tuple[int, int]],
) -> np.ndarray:

    xx = np.arange(
        grid.nx,
        dtype=np.float32,
    )[:, None]

    yy = np.arange(
        grid.ny,
        dtype=np.float32,
    )[None, :]

    result = np.full(
        (grid.nx, grid.ny),
        np.inf,
        dtype=np.float32,
    )

    for x, y in positions:

        d = np.sqrt(
            ((xx - x) * grid.dx_m) ** 2
            +
            ((yy - y) * grid.dy_m) ** 2
        )

        result = np.minimum(
            result,
            d,
        )

    return result


def multicpt_idw(
    ic: np.ndarray,
    positions: list[tuple[int, int]],
    grid: Grid,
) -> np.ndarray:

    nx, ny, nz = ic.shape
    k = len(positions)

    profiles = np.stack(
        [
            ic[x, y, :]
            for x, y in positions
        ],
        axis=0,
    )

    gx = np.arange(
        nx,
        dtype=np.float32,
    )[:, None, None]

    gy = np.arange(
        ny,
        dtype=np.float32,
    )[None, :, None]

    px = np.array(
        [p[0] for p in positions],
        dtype=np.float32,
    )[None, None, :]

    py = np.array(
        [p[1] for p in positions],
        dtype=np.float32,
    )[None, None, :]

    d2 = (
        ((gx - px) * grid.dx_m) ** 2
        +
        ((gy - py) * grid.dy_m) ** 2
    )

    weights = 1.0 / np.maximum(
        d2,
        1e-6,
    )

    weights /= weights.sum(
        axis=2,
        keepdims=True,
    )

    pred = np.einsum(
        "xyk,kz->xyz",
        weights,
        profiles,
    ).astype(np.float32)

    # Preserve measured CPT exactly.
    for x, y in positions:
        pred[x, y, :] = ic[x, y, :]

    return pred


def calculate_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> dict:

    diff = pred[mask] - truth[mask]
    abs_err = np.abs(diff)

    return {
        "mae": float(
            abs_err.mean()
        ),
        "rmse": float(
            np.sqrt(
                np.mean(diff ** 2)
            )
        ),
        "p90": float(
            np.quantile(
                abs_err,
                0.90,
            )
        ),
        "frac_le_010": float(
            np.mean(
                abs_err <= 0.10
            )
        ),
        "frac_le_015": float(
            np.mean(
                abs_err <= 0.15
            )
        ),
        "frac_le_020": float(
            np.mean(
                abs_err <= 0.20
            )
        ),
    }


def write_csv(
    path: Path,
    rows: list[dict],
):

    if not rows:
        return

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    mean = float(
        values.mean()
    )

    if len(values) <= 1:
        return mean, mean, mean

    samples = rng.choice(
        values,
        size=(
            n_boot,
            len(values),
        ),
        replace=True,
    )

    means = samples.mean(axis=1)

    lo, hi = np.quantile(
        means,
        [0.025, 0.975],
    )

    return (
        mean,
        float(lo),
        float(hi),
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=67,
    )

    args = parser.parse_args()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = json.loads(
        SR02_MANIFEST.read_text()
    )

    reconstructor = (
        FrozenV4Reconstructor(
            checkpoint=CHECKPOINT,
            normalization_path=(
                NORMALIZATION
            ),
        )
    )

    print(
        "=== SR-03 MULTI-CPT SUPPORT ==="
    )

    print(
        "Sampling: sequential farthest-point"
    )

    print(
        "CPT counts:",
        CPT_COUNTS,
    )

    print(
        "Device:",
        reconstructor.device,
    )

    raw_rows = []
    distance_rows = []

    total_inferences = 0

    # --------------------------------------------------
    # Evaluate same SR-02 pilot volumes
    # --------------------------------------------------

    for grid_label, entries in (
        manifest["groups"].items()
    ):

        print()
        print(
            "=== GRID",
            grid_label,
            "===",
        )

        for vi, entry in enumerate(
            entries,
            start=1,
        ):

            path = (
                DATA_DIR
                / entry["file"]
            )

            with np.load(
                path,
                allow_pickle=False,
            ) as d:

                ic = d[
                    "Ic"
                ].astype(np.float32)

                valid = d[
                    "valid_mask"
                ].astype(bool)

                meta = json.loads(
                    str(
                        d[
                            "metadata_json"
                        ]
                    )
                )

            grid = build_grid(meta)

            sequence = farthest_sequence(
                valid,
                grid,
                max(CPT_COUNTS),
            )

            for n_cpt in CPT_COUNTS:

                positions = (
                    sequence[:n_cpt]
                )

                observed_ic = (
                    np.zeros_like(
                        ic,
                        dtype=np.float32,
                    )
                )

                observed_mask = (
                    np.zeros_like(
                        valid,
                        dtype=bool,
                    )
                )

                for x, y in positions:

                    observed_mask[
                        x,
                        y,
                        :
                    ] = valid[
                        x,
                        y,
                        :
                    ]

                observed_ic[
                    observed_mask
                ] = ic[
                    observed_mask
                ]

                hidden = (
                    valid
                    & (~observed_mask)
                )

                pred, elapsed = (
                    reconstructor.predict(
                        observed_ic=(
                            observed_ic
                        ),
                        observed_mask=(
                            observed_mask
                        ),
                        valid_mask=valid,
                        grid=grid,
                    )
                )

                total_inferences += 1

                idw = multicpt_idw(
                    ic,
                    positions,
                    grid,
                )

                sgan = calculate_metrics(
                    pred,
                    ic,
                    hidden,
                )

                idw_metrics = (
                    calculate_metrics(
                        idw,
                        ic,
                        hidden,
                    )
                )

                nearest = (
                    nearest_distance_field(
                        grid,
                        positions,
                    )
                )

                nearest_3d = (
                    np.broadcast_to(
                        nearest[
                            :,
                            :,
                            None,
                        ],
                        ic.shape,
                    )
                )

                raw_rows.append({
                    "grid": grid_label,
                    "volume": path.name,
                    "family": meta.get(
                        "family",
                        "",
                    ),
                    "n_cpt": n_cpt,
                    "sgan_mae": (
                        sgan["mae"]
                    ),
                    "sgan_rmse": (
                        sgan["rmse"]
                    ),
                    "sgan_p90": (
                        sgan["p90"]
                    ),
                    "sgan_frac_le_010": (
                        sgan[
                            "frac_le_010"
                        ]
                    ),
                    "sgan_frac_le_015": (
                        sgan[
                            "frac_le_015"
                        ]
                    ),
                    "sgan_frac_le_020": (
                        sgan[
                            "frac_le_020"
                        ]
                    ),
                    "idw_mae": (
                        idw_metrics[
                            "mae"
                        ]
                    ),
                    "idw_rmse": (
                        idw_metrics[
                            "rmse"
                        ]
                    ),
                    "idw_p90": (
                        idw_metrics[
                            "p90"
                        ]
                    ),
                    "inference_seconds": (
                        elapsed
                    ),
                    "max_nearest_distance_m": (
                        float(
                            nearest[
                                valid.any(
                                    axis=2
                                )
                            ].max()
                        )
                    ),
                })

                # --------------------------------------
                # Distance-error bins
                # --------------------------------------

                bin_width = float(
                    min(
                        grid.dx_m,
                        grid.dy_m,
                    )
                )

                max_d = float(
                    nearest.max()
                )

                edges = np.arange(
                    0.0,
                    max_d
                    + 2 * bin_width,
                    bin_width,
                )

                abs_err = np.abs(
                    pred - ic
                )

                for lo, hi in zip(
                    edges[:-1],
                    edges[1:],
                ):

                    mask = (
                        hidden
                        & (
                            nearest_3d
                            >= lo
                        )
                        & (
                            nearest_3d
                            < hi
                        )
                    )

                    if not mask.any():
                        continue

                    errors = (
                        abs_err[mask]
                    )

                    distance_rows.append({
                        "grid": (
                            grid_label
                        ),
                        "volume": (
                            path.name
                        ),
                        "family": (
                            meta.get(
                                "family",
                                "",
                            )
                        ),
                        "n_cpt": (
                            n_cpt
                        ),
                        "distance_from_m": (
                            float(lo)
                        ),
                        "distance_to_m": (
                            float(hi)
                        ),
                        "n_voxels": (
                            int(
                                mask.sum()
                            )
                        ),
                        "mae": float(
                            errors.mean()
                        ),
                        "p90": float(
                            np.quantile(
                                errors,
                                0.90,
                            )
                        ),
                        "frac_le_015": (
                            float(
                                np.mean(
                                    errors
                                    <= 0.15
                                )
                            )
                        ),
                    })

            if (
                vi % 20 == 0
                or vi == len(entries)
            ):
                print(
                    f"{vi}/{len(entries)} "
                    "volumes"
                )

    # --------------------------------------------------
    # Save raw
    # --------------------------------------------------

    write_csv(
        OUT_DIR
        / "per_volume_count.csv",
        raw_rows,
    )

    write_csv(
        OUT_DIR
        / "distance_bins.csv",
        distance_rows,
    )

    # --------------------------------------------------
    # Aggregate by grid + CPT count
    # --------------------------------------------------

    groups = defaultdict(list)

    for row in raw_rows:

        groups[
            (
                row["grid"],
                int(
                    row["n_cpt"]
                ),
            )
        ].append(row)

    rng = np.random.default_rng(
        args.seed
    )

    aggregate = []

    metrics = [
        "sgan_mae",
        "sgan_rmse",
        "sgan_p90",
        "sgan_frac_le_010",
        "sgan_frac_le_015",
        "sgan_frac_le_020",
        "idw_mae",
        "idw_rmse",
        "idw_p90",
        "max_nearest_distance_m",
    ]

    for (
        grid_label,
        n_cpt,
    ), rows in sorted(
        groups.items()
    ):

        out = {
            "grid": grid_label,
            "n_cpt": n_cpt,
            "n_volumes": len(rows),
        }

        for metric in metrics:

            values = np.array([
                r[metric]
                for r in rows
            ])

            mean, low, high = (
                bootstrap_ci(
                    values,
                    rng,
                    args.bootstrap,
                )
            )

            out[
                metric + "_mean"
            ] = mean

            out[
                metric + "_ci95_low"
            ] = low

            out[
                metric + "_ci95_high"
            ] = high

        improvement = np.array([
            r["idw_mae"]
            - r["sgan_mae"]
            for r in rows
        ])

        mean, low, high = (
            bootstrap_ci(
                improvement,
                rng,
                args.bootstrap,
            )
        )

        out[
            "mae_improvement_vs_idw"
        ] = mean

        out[
            "mae_improvement_ci95_low"
        ] = low

        out[
            "mae_improvement_ci95_high"
        ] = high

        aggregate.append(out)

    write_csv(
        OUT_DIR
        / "aggregate_by_count.csv",
        aggregate,
    )

    # --------------------------------------------------
    # Print
    # --------------------------------------------------

    print()
    print(
        "=== MULTI-CPT GLOBAL ACCURACY ==="
    )

    current_grid = None

    for row in aggregate:

        if row["grid"] != current_grid:

            current_grid = row["grid"]

            print()
            print(
                current_grid
            )

            print(
                f"{'CPT':>4} "
                f"{'SGAN MAE':>10} "
                f"{'IDW MAE':>10} "
                f"{'<=0.15':>10} "
                f"{'MaxDist':>10}"
            )

        print(
            f"{row['n_cpt']:4d} "
            f"{row['sgan_mae_mean']:10.5f} "
            f"{row['idw_mae_mean']:10.5f} "
            f"{row['sgan_frac_le_015_mean']*100:9.1f}% "
            f"{row['max_nearest_distance_m_mean']:9.1f}m"
        )

    print()
    print(
        "Total SchemaGAN inferences:",
        total_inferences,
    )

    print()
    print("Saved:")
    print(
        OUT_DIR
        / "per_volume_count.csv"
    )
    print(
        OUT_DIR
        / "distance_bins.csv"
    )
    print(
        OUT_DIR
        / "aggregate_by_count.csv"
    )

    print()
    print(
        "NOTE: Farthest-point sampling is "
        "a spatial baseline, not DQN."
    )


if __name__ == "__main__":
    main()
