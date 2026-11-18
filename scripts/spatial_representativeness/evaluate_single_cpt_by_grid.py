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
    / "single_cpt_by_grid"
)


ANCHORS = [
    (0.50, 0.50),
    (0.25, 0.25),
    (0.25, 0.75),
    (0.75, 0.25),
    (0.75, 0.75),
]


def load_meta(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return json.loads(str(d["metadata_json"]))


def grid_key(meta: dict) -> tuple:
    g = meta["grid"]

    return (
        int(g["nx"]),
        int(g["ny"]),
        int(g["nz"]),
        float(g["dx_m"]),
        float(g["dy_m"]),
        float(g["dz_m"]),
    )


def grid_label(key: tuple) -> str:
    nx, ny, nz, dx, dy, dz = key

    return (
        f"{nx}x{ny}x{nz}"
        f"_r{dx:g}x{dy:g}x{dz:g}m"
    )


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


def choose_positions(
    valid: np.ndarray,
    n_positions: int,
) -> list[tuple[int, int]]:

    nx, ny, _ = valid.shape

    candidates = np.argwhere(
        valid.all(axis=2)
    )

    if len(candidates) < n_positions:
        raise RuntimeError(
            "Not enough full-depth CPT columns"
        )

    selected = []
    used = set()

    for ax, ay in ANCHORS[:n_positions]:

        target = np.array([
            ax * (nx - 1),
            ay * (ny - 1),
        ])

        distance = np.linalg.norm(
            candidates.astype(float)
            - target[None, :],
            axis=1,
        )

        for idx in np.argsort(distance):
            x, y = candidates[idx]
            pos = (int(x), int(y))

            if pos not in used:
                selected.append(pos)
                used.add(pos)
                break

    return selected


def distance_field(
    grid: Grid,
    cpt_x: int,
    cpt_y: int,
) -> np.ndarray:

    x = (
        np.arange(grid.nx) - cpt_x
    ) * grid.dx_m

    y = (
        np.arange(grid.ny) - cpt_y
    ) * grid.dy_m

    xx, yy = np.meshgrid(
        x,
        y,
        indexing="ij",
    )

    return np.sqrt(
        xx**2 + yy**2
    ).astype(np.float32)


def idw_single_cpt(
    ic: np.ndarray,
    cpt_x: int,
    cpt_y: int,
) -> np.ndarray:

    profile = ic[cpt_x, cpt_y, :]

    return np.broadcast_to(
        profile[None, None, :],
        ic.shape,
    ).astype(np.float32).copy()


def metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    selected: np.ndarray,
) -> dict:

    err = np.abs(
        prediction[selected]
        - truth[selected]
    )

    sq = (
        prediction[selected]
        - truth[selected]
    ) ** 2

    return {
        "mae": float(err.mean()),
        "rmse": float(
            np.sqrt(sq.mean())
        ),
        "p90": float(
            np.quantile(err, 0.90)
        ),
        "frac_le_010": float(
            np.mean(err <= 0.10)
        ),
        "frac_le_015": float(
            np.mean(err <= 0.15)
        ),
        "frac_le_020": float(
            np.mean(err <= 0.20)
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


def balanced_sample(
    entries: list[tuple[Path, dict]],
    n: int,
    rng: np.random.Generator,
) -> list[tuple[Path, dict]]:

    groups = defaultdict(list)

    for entry in entries:
        family = entry[1].get(
            "family",
            "unknown",
        )
        groups[family].append(entry)

    families = sorted(groups)

    base = n // len(families)
    remainder = n % len(families)

    result = []

    for i, family in enumerate(families):

        quota = base + (
            1 if i < remainder else 0
        )

        pool = groups[family]

        idx = rng.choice(
            len(pool),
            size=quota,
            replace=False,
        )

        result.extend(
            pool[int(j)]
            for j in idx
        )

    return result


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

    if len(values) == 1:
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
        "--volumes-per-grid",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--positions-per-volume",
        type=int,
        default=5,
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

    args = parser.parse_args()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        args.seed
    )

    # --------------------------------------------------
    # Scan all test data
    # --------------------------------------------------

    groups = defaultdict(list)

    paths = sorted(
        DATA_DIR.glob("*.npz")
    )

    print(
        f"Scanning {len(paths)} "
        "test volumes..."
    )

    for path in paths:

        meta = load_meta(path)

        groups[
            grid_key(meta)
        ].append(
            (path, meta)
        )

    # --------------------------------------------------
    # Balanced sample per grid
    # --------------------------------------------------

    selected = {}

    manifest = {
        "study": "SR-02",
        "seed": args.seed,
        "volumes_per_grid": (
            args.volumes_per_grid
        ),
        "positions_per_volume": (
            args.positions_per_volume
        ),
        "groups": {},
    }

    for key in sorted(groups):

        sample = balanced_sample(
            groups[key],
            args.volumes_per_grid,
            rng,
        )

        selected[key] = sample

        label = grid_label(key)

        manifest["groups"][label] = [
            {
                "file": p.name,
                "family": m.get(
                    "family"
                ),
            }
            for p, m in sample
        ]

    (
        OUT_DIR
        / "pilot_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )

    # --------------------------------------------------
    # Load SchemaGAN once
    # --------------------------------------------------

    reconstructor = (
        FrozenV4Reconstructor(
            checkpoint=CHECKPOINT,
            normalization_path=(
                NORMALIZATION
            ),
        )
    )

    print(
        "Device:",
        reconstructor.device,
    )

    raw_rows = []

    inference_count = 0

    # --------------------------------------------------
    # Evaluate each grid separately
    # --------------------------------------------------

    for key in sorted(selected):

        label = grid_label(key)

        print()
        print(
            "=== GRID",
            label,
            "===",
        )

        sample = selected[key]

        for vi, (
            path,
            meta,
        ) in enumerate(
            sample,
            start=1,
        ):

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

            grid = build_grid(meta)

            positions = choose_positions(
                valid,
                args.positions_per_volume,
            )

            # Bin width mengikuti
            # resolusi horizontal grid.
            bin_width = float(
                min(
                    grid.dx_m,
                    grid.dy_m,
                )
            )

            for pi, (
                cpt_x,
                cpt_y,
            ) in enumerate(
                positions
            ):

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

                observed_mask[
                    cpt_x,
                    cpt_y,
                    :
                ] = valid[
                    cpt_x,
                    cpt_y,
                    :
                ]

                observed_ic[
                    observed_mask
                ] = ic[
                    observed_mask
                ]

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

                inference_count += 1

                idw = idw_single_cpt(
                    ic,
                    cpt_x,
                    cpt_y,
                )

                hidden = (
                    valid
                    & (~observed_mask)
                )

                dist2d = distance_field(
                    grid,
                    cpt_x,
                    cpt_y,
                )

                dist3d = np.broadcast_to(
                    dist2d[
                        :,
                        :,
                        None,
                    ],
                    ic.shape,
                )

                max_d = float(
                    dist2d.max()
                )

                edges = np.arange(
                    0,
                    max_d
                    + 2 * bin_width,
                    bin_width,
                )

                for lo, hi in zip(
                    edges[:-1],
                    edges[1:],
                ):

                    mask = (
                        hidden
                        & (dist3d >= lo)
                        & (dist3d < hi)
                    )

                    if not mask.any():
                        continue

                    sm = metrics(
                        pred,
                        ic,
                        mask,
                    )

                    im = metrics(
                        idw,
                        ic,
                        mask,
                    )

                    raw_rows.append({
                        "grid": label,
                        "volume": path.name,
                        "family": meta.get(
                            "family",
                            "",
                        ),
                        "position": pi,
                        "cpt_x": cpt_x,
                        "cpt_y": cpt_y,
                        "dx_m": grid.dx_m,
                        "dy_m": grid.dy_m,
                        "dz_m": grid.dz_m,
                        "distance_from_m": (
                            float(lo)
                        ),
                        "distance_to_m": (
                            float(hi)
                        ),
                        "distance_mid_m": (
                            float(
                                (
                                    lo
                                    + hi
                                )
                                / 2
                            )
                        ),
                        "n_voxels": int(
                            mask.sum()
                        ),
                        "sgan_mae": sm[
                            "mae"
                        ],
                        "sgan_rmse": sm[
                            "rmse"
                        ],
                        "sgan_p90": sm[
                            "p90"
                        ],
                        "sgan_frac_le_010": (
                            sm[
                                "frac_le_010"
                            ]
                        ),
                        "sgan_frac_le_015": (
                            sm[
                                "frac_le_015"
                            ]
                        ),
                        "sgan_frac_le_020": (
                            sm[
                                "frac_le_020"
                            ]
                        ),
                        "idw_mae": im[
                            "mae"
                        ],
                        "idw_rmse": im[
                            "rmse"
                        ],
                        "idw_p90": im[
                            "p90"
                        ],
                        "inference_seconds": (
                            elapsed
                        ),
                    })

            if (
                vi % 20 == 0
                or vi == len(sample)
            ):
                print(
                    f"{vi}/{len(sample)} "
                    "volumes"
                )

    # --------------------------------------------------
    # Save raw
    # --------------------------------------------------

    write_csv(
        OUT_DIR
        / "raw_bins.csv",
        raw_rows,
    )

    # --------------------------------------------------
    # Average positions within each volume
    # --------------------------------------------------

    per_volume_groups = defaultdict(
        list
    )

    for row in raw_rows:

        key = (
            row["grid"],
            row["volume"],
            row["family"],
            row["distance_from_m"],
            row["distance_to_m"],
        )

        per_volume_groups[
            key
        ].append(row)

    per_volume = []

    metric_names = [
        "sgan_mae",
        "sgan_rmse",
        "sgan_p90",
        "sgan_frac_le_010",
        "sgan_frac_le_015",
        "sgan_frac_le_020",
        "idw_mae",
        "idw_rmse",
        "idw_p90",
    ]

    for (
        grid_name,
        volume,
        family,
        lo,
        hi,
    ), rows in sorted(
        per_volume_groups.items()
    ):

        out = {
            "grid": grid_name,
            "volume": volume,
            "family": family,
            "distance_from_m": lo,
            "distance_to_m": hi,
            "distance_mid_m": (
                lo + hi
            ) / 2,
            "n_positions": len(rows),
        }

        for metric in metric_names:

            out[metric] = float(
                np.mean([
                    r[metric]
                    for r in rows
                ])
            )

        per_volume.append(out)

    write_csv(
        OUT_DIR
        / "per_volume_bins.csv",
        per_volume,
    )

    # --------------------------------------------------
    # Aggregate per GRID
    # --------------------------------------------------

    aggregate_groups = defaultdict(
        list
    )

    for row in per_volume:

        key = (
            row["grid"],
            row["distance_from_m"],
            row["distance_to_m"],
        )

        aggregate_groups[
            key
        ].append(row)

    boot_rng = np.random.default_rng(
        args.seed + 1000
    )

    aggregate = []

    for (
        grid_name,
        lo,
        hi,
    ), rows in sorted(
        aggregate_groups.items()
    ):

        out = {
            "grid": grid_name,
            "distance_from_m": lo,
            "distance_to_m": hi,
            "distance_mid_m": (
                lo + hi
            ) / 2,
            "n_volumes": len(rows),
        }

        for metric in metric_names:

            vals = np.array([
                r[metric]
                for r in rows
            ])

            mean, low, high = (
                bootstrap_ci(
                    vals,
                    boot_rng,
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

        mean, low, high = bootstrap_ci(
            improvement,
            boot_rng,
            args.bootstrap,
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
        / "aggregate_by_grid.csv",
        aggregate,
    )

    # --------------------------------------------------
    # Print pilot criterion summary
    # --------------------------------------------------

    print()
    print(
        "=== PILOT 90% / |ERROR| <= 0.15 ==="
    )

    for key in sorted(selected):

        label = grid_label(key)

        rows = [
            r
            for r in aggregate
            if r["grid"] == label
        ]

        passed = [
            r
            for r in rows
            if (
                r[
                    "sgan_frac_le_015_mean"
                ]
                >= 0.90
            )
        ]

        print()
        print(label)

        if passed:

            last = max(
                passed,
                key=lambda r:
                    r[
                        "distance_to_m"
                    ],
            )

            print(
                "  Last bin satisfying "
                "pilot criterion:",
                f"{last['distance_from_m']:.0f}"
                f"-"
                f"{last['distance_to_m']:.0f} m",
            )

            print(
                "  Fraction <= 0.15:",
                f"{last['sgan_frac_le_015_mean']*100:.1f}%",
            )

            print(
                "  SchemaGAN MAE:",
                f"{last['sgan_mae_mean']:.5f}",
            )

        else:
            print(
                "  No distance bin "
                "satisfies criterion."
            )

    print()
    print(
        "Total SchemaGAN inferences:",
        inference_count,
    )

    print()
    print("Saved:")
    print(
        OUT_DIR
        / "aggregate_by_grid.csv"
    )
    print(
        OUT_DIR
        / "per_volume_bins.csv"
    )
    print(
        OUT_DIR
        / "raw_bins.csv"
    )
    print(
        OUT_DIR
        / "pilot_manifest.json"
    )

    print()
    print(
        "IMPORTANT: the printed distance "
        "is a pilot criterion result, "
        "NOT a universal CPT radius."
    )


if __name__ == "__main__":
    main()
