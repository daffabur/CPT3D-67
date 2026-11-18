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
    / "data"
    / "synthetic3d_v4_geosyn"
    / "v3_gate_calibration_100"
    / "test"
)

MANIFEST_PATH = (
    ROOT
    / "data"
    / "synthetic3d_v4_geosyn"
    / "v3_gate_calibration_100"
    / "evaluator_manifest.json"
)

SCHEMAGAN_CHECKPOINT = (
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

DQN_CHECKPOINT = (
    ROOT
    / "results"
    / "dqn_v46a_experimental"
    / "training_5000"
    / "best.pt"
)

OUT_DIR = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "v3_gate_calibration_v46a"
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


def choose_initial_center(
    valid: np.ndarray,
) -> tuple[int, int]:

    nx, ny, _ = valid.shape

    candidates = np.argwhere(
        valid.all(axis=2)
    )

    if len(candidates) == 0:
        raise RuntimeError(
            "No full-depth valid CPT columns."
        )

    center = np.array([
        (nx - 1) / 2.0,
        (ny - 1) / 2.0,
    ])

    distance = np.linalg.norm(
        candidates.astype(float)
        - center[None, :],
        axis=1,
    )

    x, y = candidates[
        np.argmin(distance)
    ]

    return int(x), int(y)


def current_observed_xy(
    sim: SyntheticCPTSimulator,
) -> list[tuple[int, int]]:

    mask2d = sim.observed_mask.any(
        axis=2
    )

    coords = np.argwhere(mask2d)

    return [
        (int(x), int(y))
        for x, y in coords
    ]


def max_nearest_distance(
    sim: SyntheticCPTSimulator,
) -> float:

    observed = current_observed_xy(sim)

    valid2d = sim.valid.any(axis=2)

    candidates = np.argwhere(valid2d)

    max_distance = 0.0

    for x, y in candidates:

        d = min(
            np.sqrt(
                ((int(x) - ox) * sim.grid.dx_m) ** 2
                +
                ((int(y) - oy) * sim.grid.dy_m) ** 2
            )
            for ox, oy in observed
        )

        max_distance = max(
            max_distance,
            float(d),
        )

    return max_distance


def reconstruction_metrics(
    sim: SyntheticCPTSimulator,
) -> dict:

    hidden = (
        sim.valid
        & (~sim.observed_mask)
    )

    if not hidden.any():
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "p90": 0.0,
            "frac_le_010": 1.0,
            "frac_le_015": 1.0,
            "frac_le_020": 1.0,
        }

    diff = (
        sim.prediction[hidden]
        - sim._truth[hidden]
    )

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


def farthest_action_index(
    sim: SyntheticCPTSimulator,
) -> int:

    state = sim.state

    if state is None:
        raise RuntimeError(
            "Simulator has no state."
        )

    observed = current_observed_xy(sim)

    candidates = state.candidate_xy

    if len(candidates) == 0:
        raise RuntimeError(
            "No candidates available."
        )

    rows = []

    for index, xy in enumerate(candidates):

        x, y = map(int, xy)

        nearest = min(
            np.sqrt(
                ((x - ox) * sim.grid.dx_m) ** 2
                +
                ((y - oy) * sim.grid.dy_m) ** 2
            )
            for ox, oy in observed
        )

        rows.append(
            (
                -float(nearest),
                x,
                y,
                index,
            )
        )

    # max distance first,
    # deterministic tie break by x/y.
    rows.sort()

    return int(rows[0][3])


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

    mean = float(values.mean())

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


def select_balanced_entries(
    entries: list[dict],
    n: int,
    rng: np.random.Generator,
) -> list[dict]:

    groups = defaultdict(list)

    for entry in entries:
        groups[
            entry.get(
                "family",
                "unknown",
            )
        ].append(entry)

    families = sorted(groups)

    base = n // len(families)
    remainder = n % len(families)

    selected = []

    for i, family in enumerate(families):

        quota = (
            base
            + (
                1
                if i < remainder
                else 0
            )
        )

        pool = groups[family]

        if quota > len(pool):
            raise RuntimeError(
                f"Not enough {family} volumes."
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


def record_state(
    rows: list[dict],
    *,
    policy: str,
    grid_label: str,
    path: Path,
    family: str,
    sim: SyntheticCPTSimulator,
    n_cpt: int,
):

    m = reconstruction_metrics(sim)

    rows.append({
        "policy": policy,
        "grid": grid_label,
        "volume": path.name,
        "family": family,
        "n_cpt": n_cpt,
        "mae": m["mae"],
        "rmse": m["rmse"],
        "p90": m["p90"],
        "frac_le_010": (
            m["frac_le_010"]
        ),
        "frac_le_015": (
            m["frac_le_015"]
        ),
        "frac_le_020": (
            m["frac_le_020"]
        ),
        "max_nearest_distance_m": (
            max_nearest_distance(sim)
        ),
    })


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

    manifest = json.loads(
        MANIFEST_PATH.read_text()
    )

    reconstructor = (
        FrozenV4Reconstructor(
            checkpoint=(
                SCHEMAGAN_CHECKPOINT
            ),
            normalization_path=(
                NORMALIZATION
            ),
        )
    )

    agent, dqn_metadata = (
        CandidateDQNAgent.load(
            DQN_CHECKPOINT,
            map_location="cpu",
        )
    )

    print(
        "=== SR-04 DQN vs FARTHEST ==="
    )

    print(
        "SchemaGAN device:",
        reconstructor.device,
    )

    print(
        "DQN checkpoint:",
        DQN_CHECKPOINT,
    )

    print(
        "Volumes/grid:",
        args.volumes_per_grid,
    )

    print(
        "Max CPT:",
        args.max_cpt,
    )

    step_rows = []
    action_rows = []

    selected_manifest = {
        "study": "SR-04",
        "seed": args.seed,
        "volumes_per_grid": (
            args.volumes_per_grid
        ),
        "max_cpt": args.max_cpt,
        "groups": {},
    }

    total_volumes = 0

    for grid_label, entries in (
        manifest["groups"].items()
    ):

        selected_entries = (
            select_balanced_entries(
                entries,
                args.volumes_per_grid,
                rng,
            )
        )

        selected_manifest[
            "groups"
        ][grid_label] = (
            selected_entries
        )

        print()
        print(
            "=== GRID",
            grid_label,
            "===",
        )

        for vi, entry in enumerate(
            selected_entries,
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

                truth = d[
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

            family = meta.get(
                "family",
                "",
            )

            initial_xy = (
                choose_initial_center(
                    valid
                )
            )

            # --------------------------------------------
            # Independent simulators, exact same start.
            # --------------------------------------------

            dqn_sim = (
                SyntheticCPTSimulator(
                    reconstructor,
                    truth,
                    valid,
                    grid,
                )
            )

            farthest_sim = (
                SyntheticCPTSimulator(
                    reconstructor,
                    truth,
                    valid,
                    grid,
                )
            )

            dqn_state = dqn_sim.reset(
                [initial_xy]
            )

            farthest_state = (
                farthest_sim.reset(
                    [initial_xy]
                )
            )

            # Record CPT-1.
            record_state(
                step_rows,
                policy="dqn",
                grid_label=grid_label,
                path=path,
                family=family,
                sim=dqn_sim,
                n_cpt=1,
            )

            record_state(
                step_rows,
                policy="farthest",
                grid_label=grid_label,
                path=path,
                family=family,
                sim=farthest_sim,
                n_cpt=1,
            )

            # --------------------------------------------
            # Sequential CPT-2 ... max CPT
            # --------------------------------------------

            for n_cpt in range(
                2,
                args.max_cpt + 1,
            ):

                # ========================================
                # DQN
                # ========================================

                if len(
                    dqn_state.candidate_xy
                ) > 0:

                    scores = (
                        agent.scores(
                            dqn_state
                        )
                    )

                    dqn_action = (
                        agent.act(
                            dqn_state,
                            epsilon=0.0,
                        )
                    )

                    q_score = float(
                        scores[dqn_action]
                    )

                    x, y = map(
                        int,
                        dqn_state.candidate_xy[
                            dqn_action
                        ],
                    )

                    (
                        dqn_state,
                        reward,
                        dqn_done,
                        dqn_info,
                    ) = dqn_sim.step(
                        dqn_action
                    )

                    action_rows.append({
                        "policy": "dqn",
                        "grid": grid_label,
                        "volume": path.name,
                        "family": family,
                        "n_cpt": n_cpt,
                        "x": x,
                        "y": y,
                        "x_m": (
                            dqn_info[
                                "x_m"
                            ]
                        ),
                        "y_m": (
                            dqn_info[
                                "y_m"
                            ]
                        ),
                        "q_score": (
                            q_score
                        ),
                        "reward_rmse_improvement": (
                            reward
                        ),
                    })

                    record_state(
                        step_rows,
                        policy="dqn",
                        grid_label=(
                            grid_label
                        ),
                        path=path,
                        family=family,
                        sim=dqn_sim,
                        n_cpt=n_cpt,
                    )

                # ========================================
                # FARTHEST
                # ========================================

                if len(
                    farthest_state.candidate_xy
                ) > 0:

                    farthest_action = (
                        farthest_action_index(
                            farthest_sim
                        )
                    )

                    x, y = map(
                        int,
                        farthest_state.candidate_xy[
                            farthest_action
                        ],
                    )

                    (
                        farthest_state,
                        reward,
                        farthest_done,
                        farthest_info,
                    ) = farthest_sim.step(
                        farthest_action
                    )

                    action_rows.append({
                        "policy": (
                            "farthest"
                        ),
                        "grid": grid_label,
                        "volume": path.name,
                        "family": family,
                        "n_cpt": n_cpt,
                        "x": x,
                        "y": y,
                        "x_m": (
                            farthest_info[
                                "x_m"
                            ]
                        ),
                        "y_m": (
                            farthest_info[
                                "y_m"
                            ]
                        ),
                        "q_score": "",
                        "reward_rmse_improvement": (
                            reward
                        ),
                    })

                    record_state(
                        step_rows,
                        policy="farthest",
                        grid_label=(
                            grid_label
                        ),
                        path=path,
                        family=family,
                        sim=farthest_sim,
                        n_cpt=n_cpt,
                    )

                if (
                    len(
                        dqn_state.candidate_xy
                    ) == 0
                    and len(
                        farthest_state.candidate_xy
                    ) == 0
                ):
                    break

            total_volumes += 1

            if (
                vi % 5 == 0
                or vi
                == len(
                    selected_entries
                )
            ):
                print(
                    f"{vi}/"
                    f"{len(selected_entries)} "
                    "volumes"
                )

    # --------------------------------------------------
    # Save raw
    # --------------------------------------------------

    write_csv(
        OUT_DIR
        / "per_volume_step.csv",
        step_rows,
    )

    write_csv(
        OUT_DIR
        / "actions.csv",
        action_rows,
    )

    (
        OUT_DIR
        / "pilot_manifest.json"
    ).write_text(
        json.dumps(
            selected_manifest,
            indent=2,
        )
        + "\n"
    )

    # --------------------------------------------------
    # Aggregate by policy/grid/count
    # --------------------------------------------------

    groups = defaultdict(list)

    for row in step_rows:

        key = (
            row["policy"],
            row["grid"],
            int(
                row["n_cpt"]
            ),
        )

        groups[key].append(row)

    boot_rng = (
        np.random.default_rng(
            args.seed + 1000
        )
    )

    aggregate = []

    metric_names = [
        "mae",
        "rmse",
        "p90",
        "frac_le_010",
        "frac_le_015",
        "frac_le_020",
        "max_nearest_distance_m",
    ]

    for (
        policy,
        grid_label,
        n_cpt,
    ), rows in sorted(
        groups.items()
    ):

        out = {
            "policy": policy,
            "grid": grid_label,
            "n_cpt": n_cpt,
            "n_volumes": len(rows),
        }

        for metric in metric_names:

            values = np.array([
                float(
                    r[metric]
                )
                for r in rows
            ])

            mean, low, high = (
                bootstrap_ci(
                    values,
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

        aggregate.append(out)

    write_csv(
        OUT_DIR
        / "aggregate_by_policy_grid_count.csv",
        aggregate,
    )

    # --------------------------------------------------
    # Sufficiency frontier
    # lower CI >= 90%
    # --------------------------------------------------

    frontier_rows = []

    print()
    print(
        "=== CONSERVATIVE SUFFICIENCY FRONTIER ==="
    )

    for grid_label in sorted(
        {
            r["grid"]
            for r in aggregate
        }
    ):

        print()
        print(grid_label)

        for policy in (
            "dqn",
            "farthest",
        ):

            rows = [
                r
                for r in aggregate
                if (
                    r["grid"]
                    == grid_label
                    and r["policy"]
                    == policy
                )
            ]

            rows.sort(
                key=lambda r: (
                    int(
                        r["n_cpt"]
                    )
                )
            )

            first_pass = None

            for row in rows:

                if (
                    float(
                        row[
                            "frac_le_015_ci95_low"
                        ]
                    )
                    >= 0.90
                ):
                    first_pass = row
                    break

            if first_pass is None:

                last = rows[-1]

                print(
                    f"  {policy:8s}: "
                    "NOT reached "
                    f"(max {last['n_cpt']} CPT, "
                    f"mean="
                    f"{float(last['frac_le_015_mean'])*100:.1f}%, "
                    f"CI-low="
                    f"{float(last['frac_le_015_ci95_low'])*100:.1f}%)"
                )

                frontier_rows.append({
                    "grid": grid_label,
                    "policy": policy,
                    "first_sufficient_cpt": "",
                    "status": (
                        "not_reached"
                    ),
                })

            else:

                print(
                    f"  {policy:8s}: "
                    f"{first_pass['n_cpt']} CPT "
                    f"(mean="
                    f"{float(first_pass['frac_le_015_mean'])*100:.1f}%, "
                    f"CI-low="
                    f"{float(first_pass['frac_le_015_ci95_low'])*100:.1f}%)"
                )

                frontier_rows.append({
                    "grid": grid_label,
                    "policy": policy,
                    "first_sufficient_cpt": (
                        first_pass[
                            "n_cpt"
                        ]
                    ),
                    "status": (
                        "reached"
                    ),
                })

    write_csv(
        OUT_DIR
        / "sufficiency_frontier.csv",
        frontier_rows,
    )

    print()
    print(
        "Total evaluated volumes:",
        total_volumes,
    )

    print()
    print("Saved:")
    print(
        OUT_DIR
        / "per_volume_step.csv"
    )
    print(
        OUT_DIR
        / "actions.csv"
    )
    print(
        OUT_DIR
        / "aggregate_by_policy_grid_count.csv"
    )
    print(
        OUT_DIR
        / "sufficiency_frontier.csv"
    )
    print(
        OUT_DIR
        / "pilot_manifest.json"
    )

    print()
    print(
        "Q-score is recorded only as a "
        "diagnostic signal, NOT as a "
        "stopping criterion."
    )


if __name__ == "__main__":
    main()
