from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from synthetic3d.v2.core import Grid
from dqn3d.v46.adapter import FrozenV4Reconstructor
from dqn3d.v46.agent import CandidateDQNAgent
from dqn3d.v46.simulator import SyntheticCPTSimulator


ROOT = Path.home() / "Developer" / "cpt_super_learn"

DATA_DIR = (
    ROOT
    / "data/synthetic3d_v4_geosyn/full_24k/test"
)

MANIFEST_PATH = (
    ROOT
    / "results/spatial_representativeness"
    / "single_cpt_by_grid"
    / "pilot_manifest.json"
)

SCHEMAGAN_CHECKPOINT = (
    ROOT
    / "model_export/SchemaGAN_V4_2/SchemaGAN_V4_2.pt"
)

NORMALIZATION = (
    ROOT
    / "model_export/SchemaGAN_V4_2/normalization.json"
)

DQN_CHECKPOINT = (
    ROOT
    / "results/dqn_v46a_experimental"
    / "training_5000/best.pt"
)

OUT_DIR = (
    ROOT
    / "results/spatial_representativeness"
    / "v46a_distance_fraction_sweep"
)

ACTUAL_THRESHOLD = 0.90
PERSISTENCE = 3

# DEVELOPMENT candidates only.
DISTANCE_FRACTIONS = (
    0.04,
    0.06,
    0.08,
    0.10,
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


def site_geometry(grid: Grid):
    width = max(
        (grid.nx - 1) * float(grid.dx_m),
        0.0,
    )

    height = max(
        (grid.ny - 1) * float(grid.dy_m),
        0.0,
    )

    diagonal = float(
        np.hypot(width, height)
    )

    return width, height, diagonal


def choose_initial_center(
    valid: np.ndarray,
) -> tuple[int, int]:

    nx, ny, _ = valid.shape

    candidates = np.argwhere(
        valid.all(axis=2)
    )

    if len(candidates) == 0:
        candidates = np.argwhere(
            valid.any(axis=2)
        )

    if len(candidates) == 0:
        raise RuntimeError(
            "No valid CPT columns."
        )

    center = np.array([
        (nx - 1) / 2.0,
        (ny - 1) / 2.0,
    ])

    distances = np.linalg.norm(
        candidates.astype(float)
        - center[None, :],
        axis=1,
    )

    x, y = candidates[
        np.argmin(distances)
    ]

    return int(x), int(y)


def select_balanced_entries(
    entries: list[dict],
    n: int,
    rng: np.random.Generator,
) -> list[dict]:

    groups = defaultdict(list)

    for entry in entries:
        groups[
            entry.get("family", "unknown")
        ].append(entry)

    families = sorted(groups)

    base = n // len(families)
    remainder = n % len(families)

    selected = []

    for i, family in enumerate(families):
        quota = base + (
            1 if i < remainder else 0
        )

        pool = groups[family]

        if quota > len(pool):
            raise RuntimeError(
                f"Not enough volumes for {family}"
            )

        idx = rng.choice(
            len(pool),
            size=quota,
            replace=False,
        )

        selected.extend(
            pool[int(j)]
            for j in idx
        )

    return selected


def current_observed_xy(
    sim: SyntheticCPTSimulator,
) -> np.ndarray:

    return np.argwhere(
        sim.observed_mask.any(axis=2)
    )


def nearest_candidate_distances(
    sim: SyntheticCPTSimulator,
    candidates: np.ndarray,
) -> np.ndarray:

    observed = current_observed_xy(sim)

    if len(observed) == 0:
        raise RuntimeError(
            "No observed CPT."
        )

    cx = candidates[:, 0:1].astype(float)
    cy = candidates[:, 1:2].astype(float)

    ox = observed[:, 0][None, :].astype(float)
    oy = observed[:, 1][None, :].astype(float)

    dx = (
        cx - ox
    ) * float(sim.grid.dx_m)

    dy = (
        cy - oy
    ) * float(sim.grid.dy_m)

    distances = np.sqrt(
        dx * dx + dy * dy
    )

    return distances.min(axis=1)


def max_nearest_distance(
    sim: SyntheticCPTSimulator,
) -> float:

    valid_xy = np.argwhere(
        sim.valid.any(axis=2)
    )

    if len(valid_xy) == 0:
        return 0.0

    nearest = nearest_candidate_distances(
        sim,
        valid_xy,
    )

    return float(nearest.max())


def reconstruction_metrics(
    sim: SyntheticCPTSimulator,
) -> dict:

    hidden = (
        sim.valid
        & (~sim.observed_mask)
    )

    if not hidden.any():
        return {
            "actual_coverage": 1.0,
            "mae": 0.0,
            "rmse": 0.0,
        }

    diff = (
        sim.prediction[hidden]
        - sim._truth[hidden]
    )

    abs_err = np.abs(diff)

    return {
        "actual_coverage": float(
            np.mean(abs_err <= 0.15)
        ),
        "mae": float(
            abs_err.mean()
        ),
        "rmse": float(
            np.sqrt(
                np.mean(diff ** 2)
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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--volumes-per-grid",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-cpt",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=6708,
    )

    args = parser.parse_args()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        args.seed
    )

    manifest = json.loads(
        MANIFEST_PATH.read_text()
    )

    reconstructor = FrozenV4Reconstructor(
        checkpoint=SCHEMAGAN_CHECKPOINT,
        normalization_path=NORMALIZATION,
    )

    agent, _ = CandidateDQNAgent.load(
        DQN_CHECKPOINT,
        map_location="cpu",
    )

    selected_by_grid = {}

    # Freeze the exact same volumes for every
    # distance fraction.
    for grid_label, entries in (
        manifest["groups"].items()
    ):
        selected_by_grid[grid_label] = (
            select_balanced_entries(
                entries,
                args.volumes_per_grid,
                rng,
            )
        )

    step_rows = []
    volume_rows = []

    print(
        "=== DQN V4.6-A SCALE-AWARE "
        "MIN-DISTANCE SWEEP ==="
    )

    print(
        "Fractions:",
        DISTANCE_FRACTIONS,
    )

    print(
        "Actual sufficiency:",
        f"coverage >= {ACTUAL_THRESHOLD}",
    )

    print(
        "Actual persistence:",
        PERSISTENCE,
    )

    print(
        "NOTE: development evaluation only; "
        "no model training."
    )

    for fraction in DISTANCE_FRACTIONS:

        print()
        print(
            "================================"
        )
        print(
            f"MIN DISTANCE = "
            f"{fraction*100:.0f}% diagonal"
        )
        print(
            "================================"
        )

        for grid_label, entries in (
            selected_by_grid.items()
        ):

            print()
            print(
                f"GRID {grid_label}"
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

                    truth = (
                        d["Ic"]
                        .astype(np.float32)
                    )

                    valid = (
                        d["valid_mask"]
                        .astype(bool)
                    )

                    meta = json.loads(
                        str(
                            d["metadata_json"]
                        )
                    )

                grid = build_grid(meta)

                family = meta.get(
                    "family",
                    "",
                )

                width, height, diagonal = (
                    site_geometry(grid)
                )

                min_distance_m = (
                    fraction * diagonal
                )

                initial_xy = (
                    choose_initial_center(
                        valid
                    )
                )

                sim = SyntheticCPTSimulator(
                    reconstructor,
                    truth,
                    valid,
                    grid,
                )

                state = sim.reset(
                    [initial_xy]
                )

                positive_run = 0
                first_actual = None
                first_persistent = None
                blocked_at = None

                # -----------------------------
                # CPT 1
                # -----------------------------

                metrics = (
                    reconstruction_metrics(
                        sim
                    )
                )

                actual_sufficient = (
                    metrics[
                        "actual_coverage"
                    ]
                    >= ACTUAL_THRESHOLD
                )

                positive_run = (
                    1
                    if actual_sufficient
                    else 0
                )

                if actual_sufficient:
                    first_actual = 1

                step_rows.append({
                    "distance_fraction":
                        fraction,
                    "distance_percent":
                        fraction * 100.0,
                    "grid":
                        grid_label,
                    "volume":
                        path.name,
                    "family":
                        family,
                    "n_cpt":
                        1,
                    "site_width_m":
                        width,
                    "site_height_m":
                        height,
                    "site_diagonal_m":
                        diagonal,
                    "minimum_distance_m":
                        min_distance_m,
                    "selected_x":
                        initial_xy[0],
                    "selected_y":
                        initial_xy[1],
                    "selected_nearest_cpt_m":
                        "",
                    "q_value":
                        "",
                    "actual_coverage":
                        metrics[
                            "actual_coverage"
                        ],
                    "actual_sufficient":
                        actual_sufficient,
                    "actual_positive_run":
                        positive_run,
                    "max_nearest_distance_m":
                        max_nearest_distance(
                            sim
                        ),
                    "mae":
                        metrics["mae"],
                    "rmse":
                        metrics["rmse"],
                })

                # -----------------------------
                # CPT 2 ... max CPT
                # -----------------------------

                for n_cpt in range(
                    2,
                    args.max_cpt + 1,
                ):

                    if len(
                        state.candidate_xy
                    ) == 0:
                        blocked_at = n_cpt
                        break

                    candidates = (
                        state.candidate_xy
                    )

                    nearest = (
                        nearest_candidate_distances(
                            sim,
                            candidates,
                        )
                    )

                    eligible = np.where(
                        nearest + 1e-9
                        >= min_distance_m
                    )[0]

                    if len(eligible) == 0:
                        blocked_at = n_cpt
                        break

                    scores = agent.scores(
                        state
                    )

                    # Highest Q only among
                    # candidates passing the
                    # scale-aware distance rule.
                    local_best = int(
                        np.argmax(
                            scores[eligible]
                        )
                    )

                    action = int(
                        eligible[local_best]
                    )

                    q_value = float(
                        scores[action]
                    )

                    selected_distance = float(
                        nearest[action]
                    )

                    selected_x, selected_y = (
                        map(
                            int,
                            candidates[action],
                        )
                    )

                    (
                        state,
                        _reward,
                        _done,
                        _info,
                    ) = sim.step(
                        action
                    )

                    metrics = (
                        reconstruction_metrics(
                            sim
                        )
                    )

                    actual_sufficient = (
                        metrics[
                            "actual_coverage"
                        ]
                        >= ACTUAL_THRESHOLD
                    )

                    if actual_sufficient:
                        positive_run += 1

                        if first_actual is None:
                            first_actual = n_cpt
                    else:
                        positive_run = 0

                    if (
                        positive_run
                        >= PERSISTENCE
                        and first_persistent
                        is None
                    ):
                        first_persistent = (
                            n_cpt
                        )

                    step_rows.append({
                        "distance_fraction":
                            fraction,
                        "distance_percent":
                            fraction * 100.0,
                        "grid":
                            grid_label,
                        "volume":
                            path.name,
                        "family":
                            family,
                        "n_cpt":
                            n_cpt,
                        "site_width_m":
                            width,
                        "site_height_m":
                            height,
                        "site_diagonal_m":
                            diagonal,
                        "minimum_distance_m":
                            min_distance_m,
                        "selected_x":
                            selected_x,
                        "selected_y":
                            selected_y,
                        "selected_nearest_cpt_m":
                            selected_distance,
                        "q_value":
                            q_value,
                        "actual_coverage":
                            metrics[
                                "actual_coverage"
                            ],
                        "actual_sufficient":
                            actual_sufficient,
                        "actual_positive_run":
                            positive_run,
                        "max_nearest_distance_m":
                            max_nearest_distance(
                                sim
                            ),
                        "mae":
                            metrics["mae"],
                        "rmse":
                            metrics["rmse"],
                    })

                    # Development efficiency:
                    # once actual sufficiency is
                    # stable 3 times, no benefit
                    # in adding more CPTs.
                    if (
                        first_persistent
                        is not None
                    ):
                        break

                volume_rows.append({
                    "distance_fraction":
                        fraction,
                    "distance_percent":
                        fraction * 100.0,
                    "grid":
                        grid_label,
                    "volume":
                        path.name,
                    "family":
                        family,
                    "site_diagonal_m":
                        diagonal,
                    "minimum_distance_m":
                        min_distance_m,
                    "first_actual_sufficient_cpt":
                        (
                            first_actual
                            if first_actual
                            is not None
                            else ""
                        ),
                    "first_actual_persistent3_cpt":
                        (
                            first_persistent
                            if first_persistent
                            is not None
                            else ""
                        ),
                    "resolved_by_32":
                        (
                            first_persistent
                            is not None
                        ),
                    "blocked_at_cpt":
                        (
                            blocked_at
                            if blocked_at
                            is not None
                            else ""
                        ),
                })

                if (
                    vi % 5 == 0
                    or vi == len(entries)
                ):
                    print(
                        f"  {vi}/{len(entries)}"
                    )

    # =====================================================
    # SUMMARY
    # =====================================================

    summary_rows = []

    print()
    print(
        "=== DEVELOPMENT EFFICIENCY SUMMARY ==="
    )

    for fraction in DISTANCE_FRACTIONS:

        rows = [
            r
            for r in volume_rows
            if np.isclose(
                r["distance_fraction"],
                fraction,
            )
        ]

        stops = np.array([
            float(
                r[
                    "first_actual_persistent3_cpt"
                ]
            )
            for r in rows
            if (
                r[
                    "first_actual_persistent3_cpt"
                ]
                != ""
            )
        ])

        resolved = len(stops)
        total = len(rows)

        blocked = sum(
            r["blocked_at_cpt"] != ""
            for r in rows
        )

        median_stop = (
            float(
                np.median(stops)
            )
            if len(stops)
            else None
        )

        mean_stop = (
            float(
                stops.mean()
            )
            if len(stops)
            else None
        )

        summary = {
            "distance_fraction":
                fraction,
            "distance_percent":
                fraction * 100.0,
            "volumes":
                total,
            "resolved_actual_persistent3":
                resolved,
            "resolved_fraction":
                resolved / total,
            "median_cpt_to_actual_persistent3":
                (
                    median_stop
                    if median_stop
                    is not None
                    else ""
                ),
            "mean_cpt_to_actual_persistent3":
                (
                    mean_stop
                    if mean_stop
                    is not None
                    else ""
                ),
            "blocked_before_32":
                blocked,
        }

        summary_rows.append(
            summary
        )

        print(
            f"{fraction*100:>4.0f}%"
            f" | resolved "
            f"{resolved:>3}/{total}"
            f" ({resolved/total*100:5.1f}%)"
            f" | median CPT="
            f"{median_stop}"
            f" | mean CPT="
            f"{mean_stop}"
            f" | blocked={blocked}"
        )

    write_csv(
        OUT_DIR / "per_step.csv",
        step_rows,
    )

    write_csv(
        OUT_DIR / "per_volume.csv",
        volume_rows,
    )

    write_csv(
        OUT_DIR / "summary.csv",
        summary_rows,
    )

    manifest_out = {
        "purpose":
            "Development-only scale-aware "
            "DQN V4.6-A minimum-distance sweep",
        "training":
            False,
        "distance_fractions":
            list(DISTANCE_FRACTIONS),
        "actual_coverage_threshold":
            ACTUAL_THRESHOLD,
        "actual_persistence":
            PERSISTENCE,
        "volumes_per_grid":
            args.volumes_per_grid,
        "max_cpt":
            args.max_cpt,
        "seed":
            args.seed,
        "selection_rule":
            "argmax frozen DQN Q among "
            "candidates satisfying "
            "nearest-CPT distance >= "
            "fraction * site diagonal",
        "site_diagonal_definition":
            "hypot((nx-1)*dx_m, "
            "(ny-1)*dy_m)",
        "warning":
            "Ground truth is used only for "
            "offline development evaluation. "
            "It is not available in the app.",
    }

    (
        OUT_DIR
        / "evaluation_manifest.json"
    ).write_text(
        json.dumps(
            manifest_out,
            indent=2,
        )
        + "\n"
    )

    print()
    print("Saved:")
    print(
        OUT_DIR / "summary.csv"
    )
    print(
        OUT_DIR / "per_volume.csv"
    )
    print(
        OUT_DIR / "per_step.csv"
    )


if __name__ == "__main__":
    main()
