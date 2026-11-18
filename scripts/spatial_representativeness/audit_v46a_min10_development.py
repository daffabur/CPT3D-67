from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Load sklearn/scipy binary extensions BEFORE torch/DQN.
# This avoids the current NumPy/SciPy extension-loading conflict.
import scipy.spatial
import scipy.stats
import sklearn
from scipy.ndimage import distance_transform_edt

from synthetic3d.v2.core import Grid


ROOT = Path.home() / "Developer" / "cpt_super_learn"

DATA_DIR = (
    ROOT
    / "data/synthetic3d_v4_geosyn"
    / "v46a_min10_development_100"
    / "test"
)

SCHEMAGAN_CHECKPOINT = (
    ROOT
    / "model_export/SchemaGAN_V4_2"
    / "SchemaGAN_V4_2.pt"
)

NORMALIZATION = (
    ROOT
    / "model_export/SchemaGAN_V4_2"
    / "normalization.json"
)

DQN_CHECKPOINT = (
    ROOT
    / "results/dqn_v46a_experimental"
    / "training_5000/best.pt"
)

UNCERTAINTY_MODEL = (
    ROOT
    / "results/uncertainty_v3"
    / "uncertainty_v3_quantile.joblib"
)

UNCERTAINTY_META = (
    ROOT
    / "results/uncertainty_v3"
    / "model_metadata.json"
)

SUFFICIENCY_MODEL = (
    ROOT
    / "results/uncertainty_v3_1_scale_aware"
    / "v3_1_scale_aware_sufficiency.joblib"
)

SUFFICIENCY_META = (
    ROOT
    / "results/uncertainty_v3_1_scale_aware"
    / "model_metadata.json"
)

OUT_DIR = (
    ROOT
    / "results/spatial_representativeness"
    / "v46a_min10_development_audit"
)

MIN_DISTANCE_FRACTION = 0.10
MAX_CPT = 32

IC_TOLERANCE = 0.15
ACTUAL_THRESHOLD = 0.90

PROB_THRESHOLD = 0.98
PERSISTENCE = 3

MAX_FALSE_STOP = 0.05
MIN_STOPS = 10


V3_FEATURES = [
    "distance_to_cpt_m",
    "depth_m",
    "predicted_ic",
    "lateral_gradient",
    "vertical_gradient",
    "nearby_cpt_fraction_25m",
    "n_cpt",
    "dx_m",
    "dy_m",
    "dz_m",
]


V31_FEATURES = [
    "predicted_coverage",
    "max_distance_fraction",
    "n_cpt",
    "site_width_m",
    "site_height_m",
    "site_diagonal_m",
    "site_area_m2",
    "mean_cpt_spacing_m",
    "max_nearest_distance_m",
    "aspect_ratio",
]


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


def grid_label(grid: Grid) -> str:
    def fmt(v):
        v = float(v)
        return (
            str(int(v))
            if v.is_integer()
            else f"{v:g}"
        )

    return (
        f"{grid.nx}x{grid.ny}x{grid.nz}"
        f"_r{fmt(grid.dx_m)}x"
        f"{fmt(grid.dy_m)}x"
        f"{fmt(grid.dz_m)}m"
    )


def choose_initial_center(valid):
    nx, ny, _ = valid.shape

    candidates = np.argwhere(
        valid.all(axis=2)
    )

    if len(candidates) == 0:
        raise RuntimeError(
            "No full-depth valid CPT column."
        )

    center = np.array([
        (nx - 1) / 2.0,
        (ny - 1) / 2.0,
    ])

    d = np.linalg.norm(
        candidates.astype(float)
        - center[None, :],
        axis=1,
    )

    x, y = candidates[
        np.argmin(d)
    ]

    return int(x), int(y)


def geometry(grid):
    width = (
        (grid.nx - 1)
        * float(grid.dx_m)
    )

    height = (
        (grid.ny - 1)
        * float(grid.dy_m)
    )

    diagonal = float(
        np.hypot(width, height)
    )

    area = width * height

    aspect = (
        max(width, height)
        / max(min(width, height), 1e-6)
    )

    return (
        width,
        height,
        diagonal,
        area,
        aspect,
    )


def observed_xy(sim):
    return np.argwhere(
        sim.observed_mask.any(axis=2)
    )


def nearest_distances(
    sim,
    candidates,
):
    obs = observed_xy(sim)

    cx = candidates[:, 0:1].astype(float)
    cy = candidates[:, 1:2].astype(float)

    ox = obs[:, 0][None, :].astype(float)
    oy = obs[:, 1][None, :].astype(float)

    ddx = (
        cx - ox
    ) * float(sim.grid.dx_m)

    ddy = (
        cy - oy
    ) * float(sim.grid.dy_m)

    return np.sqrt(
        ddx ** 2 + ddy ** 2
    ).min(axis=1)


def max_nearest_distance(sim):
    coords = np.argwhere(
        sim.valid.any(axis=2)
    )

    if len(coords) == 0:
        return 0.0

    return float(
        nearest_distances(
            sim,
            coords,
        ).max()
    )


def evaluate_state(
    sim,
    *,
    n_cpt,
    uncertainty_model,
    correction,
    sufficiency_model,
):
    truth = np.asarray(
        sim._truth,
        dtype=np.float32,
    )

    valid = np.asarray(
        sim.valid,
        dtype=bool,
    )

    observed = np.asarray(
        sim.observed_mask,
        dtype=bool,
    )

    prediction = np.asarray(
        sim.prediction,
        dtype=np.float32,
    )

    grid = sim.grid

    dx = float(grid.dx_m)
    dy = float(grid.dy_m)
    dz = float(grid.dz_m)

    nx, ny, nz = truth.shape

    hidden = (
        valid
        & (~observed)
    )

    if not hidden.any():
        predicted_coverage = 1.0
        actual_coverage = 1.0

    else:
        # ================================================
        # Frozen Uncertainty V3 feature construction
        # ================================================

        measured_xy = observed.any(
            axis=2
        )

        distance_xy = (
            distance_transform_edt(
                ~measured_xy,
                sampling=(dx, dy),
            )
        )

        gx, gy, gz = np.gradient(
            prediction,
            dx,
            dy,
            dz,
        )

        lateral_gradient = np.sqrt(
            gx ** 2 + gy ** 2
        )

        vertical_gradient = np.abs(
            gz
        )

        chosen = np.argwhere(
            measured_xy
        )

        x_m = np.arange(nx) * dx
        y_m = np.arange(ny) * dy

        xx, yy = np.meshgrid(
            x_m,
            y_m,
            indexing="ij",
        )

        nearby_count = np.zeros(
            (nx, ny),
            dtype=np.float32,
        )

        for x, y in chosen:
            d = np.sqrt(
                (xx - x * dx) ** 2
                + (yy - y * dy) ** 2
            )

            nearby_count += (
                d <= 25.0
            ).astype(np.float32)

        nearby_fraction = (
            nearby_count
            / float(n_cpt)
        )

        coords = np.argwhere(
            hidden
        )

        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        Xv3 = pd.DataFrame({
            "distance_to_cpt_m":
                distance_xy[x, y],

            "depth_m":
                z.astype(float) * dz,

            "predicted_ic":
                prediction[x, y, z],

            "lateral_gradient":
                lateral_gradient[x, y, z],

            "vertical_gradient":
                vertical_gradient[x, y, z],

            "nearby_cpt_fraction_25m":
                nearby_fraction[x, y],

            "n_cpt":
                np.full(
                    len(coords),
                    n_cpt,
                    dtype=float,
                ),

            "dx_m":
                np.full(
                    len(coords),
                    dx,
                ),

            "dy_m":
                np.full(
                    len(coords),
                    dy,
                ),

            "dz_m":
                np.full(
                    len(coords),
                    dz,
                ),
        })[V3_FEATURES]

        raw = uncertainty_model.predict(
            Xv3
        )

        predicted_bound = (
            np.maximum(
                0.0,
                np.asarray(
                    raw,
                    dtype=float,
                ),
            )
            + correction
        )

        predicted_coverage = float(
            np.mean(
                predicted_bound
                <= IC_TOLERANCE
            )
        )

        abs_error = np.abs(
            prediction[x, y, z]
            - truth[x, y, z]
        )

        actual_coverage = float(
            np.mean(
                abs_error
                <= IC_TOLERANCE
            )
        )

    (
        width,
        height,
        diagonal,
        area,
        aspect,
    ) = geometry(grid)

    max_distance = (
        max_nearest_distance(sim)
    )

    max_fraction = (
        max_distance
        / max(diagonal, 1e-6)
    )

    mean_spacing = float(
        np.sqrt(
            area / max(n_cpt, 1)
        )
    )

    X31 = pd.DataFrame([{
        "predicted_coverage":
            predicted_coverage,

        "max_distance_fraction":
            max_fraction,

        "n_cpt":
            n_cpt,

        "site_width_m":
            width,

        "site_height_m":
            height,

        "site_diagonal_m":
            diagonal,

        "site_area_m2":
            area,

        "mean_cpt_spacing_m":
            mean_spacing,

        "max_nearest_distance_m":
            max_distance,

        "aspect_ratio":
            aspect,
    }])[V31_FEATURES]

    probability = float(
        sufficiency_model.predict_proba(
            X31
        )[0, 1]
    )

    return {
        "predicted_coverage":
            predicted_coverage,

        "actual_coverage":
            actual_coverage,

        "actual_sufficient":
            actual_coverage
            >= ACTUAL_THRESHOLD,

        "max_nearest_distance_m":
            max_distance,

        "max_distance_fraction":
            max_fraction,

        "sufficiency_probability":
            probability,

        "positive_gate":
            probability
            >= PROB_THRESHOLD,
    }


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(rows)


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ================================================
    # Load frozen models / metadata
    # ================================================

    # IMPORTANT:
    # Load sklearn/joblib models BEFORE importing
    # PyTorch/DQN binary stack.
    from sklearn.ensemble import (
        HistGradientBoostingRegressor,
        HistGradientBoostingClassifier,
    )

    uncertainty_model = joblib.load(
        UNCERTAINTY_MODEL
    )

    uncertainty_meta = json.loads(
        UNCERTAINTY_META.read_text()
    )

    correction = float(
        uncertainty_meta.get(
            "calibration_correction",
            uncertainty_meta.get(
                "correction",
                0.0,
            ),
        )
    )

    sufficiency_model = joblib.load(
        SUFFICIENCY_MODEL
    )

    print("V3/V3.1 sklearn models: LOAD PASS")

    # Only now load the PyTorch/DQN stack.
    from dqn3d.v46.adapter import FrozenV4Reconstructor
    from dqn3d.v46.agent import CandidateDQNAgent
    from dqn3d.v46.simulator import SyntheticCPTSimulator

    reconstructor = FrozenV4Reconstructor(
        checkpoint=SCHEMAGAN_CHECKPOINT,
        normalization_path=NORMALIZATION,
    )

    agent, _ = CandidateDQNAgent.load(
        DQN_CHECKPOINT,
        map_location="cpu",
    )

    print("SchemaGAN/DQN models: LOAD PASS")

    suff_meta = json.loads(
        SUFFICIENCY_META.read_text()
    )

    assert np.isclose(
        float(
            suff_meta[
                "probability_threshold"
            ]
        ),
        PROB_THRESHOLD,
        atol=1e-9,
    )

    assert (
        int(
            suff_meta["persistence"]
        )
        == PERSISTENCE
    )

    paths = sorted(
        DATA_DIR.glob("*.npz")
    )

    assert len(paths) == 100, (
        f"Expected 100 volumes, "
        f"got {len(paths)}"
    )

    print(
        "=== V4.6-A MIN-10% "
        "DEVELOPMENT AUDIT ==="
    )

    print(
        "Volumes:",
        len(paths),
    )

    print(
        "Minimum distance:",
        "10% site diagonal",
    )

    print(
        "V3 correction:",
        correction,
    )

    print(
        "V3.1 probability threshold:",
        PROB_THRESHOLD,
    )

    print(
        "Persistence:",
        PERSISTENCE,
    )

    print(
        "NO TRAINING / NO TUNING"
    )

    step_rows = []
    decision_rows = []

    for vi, path in enumerate(
        paths,
        start=1,
    ):
        with np.load(
            path,
            allow_pickle=False,
        ) as d:

            truth = np.asarray(
                d["Ic"],
                dtype=np.float32,
            )

            valid = np.asarray(
                d["valid_mask"],
                dtype=bool,
            )

            meta = json.loads(
                str(
                    d[
                        "metadata_json"
                    ]
                )
            )

        grid = build_grid(meta)
        grid_name = grid_label(grid)

        (
            _width,
            _height,
            diagonal,
            _area,
            _aspect,
        ) = geometry(grid)

        min_distance_m = (
            MIN_DISTANCE_FRACTION
            * diagonal
        )

        first_xy = (
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
            [first_xy]
        )

        volume_steps = []

        # ============================================
        # Evaluate CPT 1
        # ============================================

        metrics = evaluate_state(
            sim,
            n_cpt=1,
            uncertainty_model=(
                uncertainty_model
            ),
            correction=correction,
            sufficiency_model=(
                sufficiency_model
            ),
        )

        volume_steps.append({
            "n_cpt": 1,
            **metrics,
        })

        step_rows.append({
            "volume":
                path.name,
            "grid":
                grid_name,
            "family":
                meta.get(
                    "family",
                    "",
                ),
            "n_cpt":
                1,
            "selected_x":
                first_xy[0],
            "selected_y":
                first_xy[1],
            "nearest_existing_cpt_m":
                "",
            "q_value":
                "",
            "minimum_distance_m":
                min_distance_m,
            **metrics,
        })

        blocked = False

        # ============================================
        # CPT 2 .. 32
        # ============================================

        for n_cpt in range(
            2,
            MAX_CPT + 1,
        ):
            candidates = (
                state.candidate_xy
            )

            if len(candidates) == 0:
                blocked = True
                break

            nearest = nearest_distances(
                sim,
                candidates,
            )

            eligible = np.where(
                nearest + 1e-9
                >= min_distance_m
            )[0]

            if len(eligible) == 0:
                blocked = True
                break

            scores = agent.scores(
                state
            )

            action = int(
                eligible[
                    np.argmax(
                        scores[eligible]
                    )
                ]
            )

            q_value = float(
                scores[action]
            )

            selected_distance = float(
                nearest[action]
            )

            sx, sy = map(
                int,
                candidates[action],
            )

            (
                state,
                _reward,
                _done,
                _info,
            ) = sim.step(
                action
            )

            metrics = evaluate_state(
                sim,
                n_cpt=n_cpt,
                uncertainty_model=(
                    uncertainty_model
                ),
                correction=correction,
                sufficiency_model=(
                    sufficiency_model
                ),
            )

            volume_steps.append({
                "n_cpt":
                    n_cpt,
                **metrics,
            })

            step_rows.append({
                "volume":
                    path.name,
                "grid":
                    grid_name,
                "family":
                    meta.get(
                        "family",
                        "",
                    ),
                "n_cpt":
                    n_cpt,
                "selected_x":
                    sx,
                "selected_y":
                    sy,
                "nearest_existing_cpt_m":
                    selected_distance,
                "q_value":
                    q_value,
                "minimum_distance_m":
                    min_distance_m,
                **metrics,
            })

        # ============================================
        # Frozen sequential STOP decision
        # ============================================

        consecutive = 0
        stop_n = None
        stop_actual = None
        stop_probability = None

        first_actual = None

        for row in volume_steps:
            if (
                first_actual is None
                and row[
                    "actual_sufficient"
                ]
            ):
                first_actual = int(
                    row["n_cpt"]
                )

            if row["positive_gate"]:
                consecutive += 1
            else:
                consecutive = 0

            if (
                consecutive >= PERSISTENCE
                and stop_n is None
            ):
                stop_n = int(
                    row["n_cpt"]
                )

                stop_actual = bool(
                    row[
                        "actual_sufficient"
                    ]
                )

                stop_probability = float(
                    row[
                        "sufficiency_probability"
                    ]
                )

        decision_rows.append({
            "volume":
                path.name,
            "grid":
                grid_name,
            "family":
                meta.get(
                    "family",
                    "",
                ),
            "site_diagonal_m":
                diagonal,
            "minimum_distance_m":
                min_distance_m,
            "steps_evaluated":
                len(volume_steps),
            "blocked":
                blocked,
            "first_actual_sufficient_cpt":
                (
                    first_actual
                    if first_actual
                    is not None
                    else ""
                ),
            "stop_cpt":
                (
                    stop_n
                    if stop_n
                    is not None
                    else ""
                ),
            "stop_probability":
                (
                    stop_probability
                    if stop_probability
                    is not None
                    else ""
                ),
            "stopped":
                stop_n is not None,
            "safe_stop":
                (
                    stop_n is not None
                    and stop_actual
                ),
            "false_stop":
                (
                    stop_n is not None
                    and not stop_actual
                ),
            "ever_actual_sufficient":
                first_actual is not None,
            "stop_delay_from_first_actual":
                (
                    stop_n - first_actual
                    if (
                        stop_n is not None
                        and first_actual
                        is not None
                    )
                    else ""
                ),
        })

        if (
            vi % 10 == 0
            or vi == len(paths)
        ):
            print(
                f"{vi}/100 volumes"
            )

    # ================================================
    # Final metrics
    # ================================================

    decisions = pd.DataFrame(
        decision_rows
    )

    n = len(decisions)

    stops = int(
        decisions["stopped"].sum()
    )

    tp = int(
        decisions["safe_stop"].sum()
    )

    fp = int(
        decisions["false_stop"].sum()
    )

    actual_positive = int(
        decisions[
            "ever_actual_sufficient"
        ].sum()
    )

    fn = int(
        (
            decisions[
                "ever_actual_sufficient"
            ]
            & ~decisions[
                "safe_stop"
            ]
        ).sum()
    )

    false_stop_rate = (
        fp / stops
        if stops
        else 0.0
    )

    safe_stop_recall = (
        tp / actual_positive
        if actual_positive
        else 0.0
    )

    stop_values = (
        pd.to_numeric(
            decisions["stop_cpt"],
            errors="coerce",
        )
        .dropna()
        .to_numpy()
    )

    safe_stop_values = (
        pd.to_numeric(
            decisions.loc[
                decisions["safe_stop"],
                "stop_cpt",
            ],
            errors="coerce",
        )
        .dropna()
        .to_numpy()
    )

    delays = (
        pd.to_numeric(
            decisions.loc[
                decisions["safe_stop"],
                "stop_delay_from_first_actual",
            ],
            errors="coerce",
        )
        .dropna()
        .to_numpy()
    )

    unresolved = int(
        (~decisions["stopped"]).sum()
    )

    blocked = int(
        decisions["blocked"].sum()
    )

    pass_false_stop = (
        stops >= MIN_STOPS
        and false_stop_rate
        <= MAX_FALSE_STOP
    )

    print()
    print(
        "======================================"
    )
    print(
        "DEVELOPMENT AUDIT RESULT"
    )
    print(
        "======================================"
    )

    print(
        f"Volumes             : {n}"
    )

    print(
        f"STOP                : {stops}"
    )

    print(
        f"Safe STOP (TP)      : {tp}"
    )

    print(
        f"False STOP (FP)     : {fp}"
    )

    print(
        f"Actual-positive vols: "
        f"{actual_positive}"
    )

    print(
        f"False-stop rate     : "
        f"{false_stop_rate*100:.2f}%"
    )

    print(
        f"Safe-stop recall    : "
        f"{safe_stop_recall*100:.2f}%"
    )

    print(
        f"Unresolved by 32    : "
        f"{unresolved}"
    )

    print(
        f"Blocked by 10% rule : "
        f"{blocked}"
    )

    if len(stop_values):
        print(
            f"Median STOP CPT     : "
            f"{np.median(stop_values):.1f}"
        )

        print(
            f"Mean STOP CPT       : "
            f"{np.mean(stop_values):.2f}"
        )

    if len(safe_stop_values):
        print(
            f"Median SAFE STOP CPT: "
            f"{np.median(safe_stop_values):.1f}"
        )

    if len(delays):
        print(
            f"Median STOP delay   : "
            f"{np.median(delays):.1f} CPT"
        )

    print()
    print(
        "Acceptance false-stop <= 5%:",
        (
            "PASS"
            if pass_false_stop
            else "FAIL"
        ),
    )

    # ================================================
    # Per-grid report
    # ================================================

    print()
    print("=== PER GRID ===")

    grid_rows = []

    for grid_name, g in (
        decisions.groupby("grid")
    ):
        gs = int(
            g["stopped"].sum()
        )

        gtp = int(
            g["safe_stop"].sum()
        )

        gfp = int(
            g["false_stop"].sum()
        )

        gap = int(
            g[
                "ever_actual_sufficient"
            ].sum()
        )

        gfalse = (
            gfp / gs
            if gs
            else 0.0
        )

        grecall = (
            gtp / gap
            if gap
            else 0.0
        )

        gstop = pd.to_numeric(
            g["stop_cpt"],
            errors="coerce",
        ).dropna()

        row = {
            "grid":
                grid_name,
            "volumes":
                len(g),
            "stops":
                gs,
            "safe_stops":
                gtp,
            "false_stops":
                gfp,
            "false_stop_rate":
                gfalse,
            "actual_positive":
                gap,
            "safe_stop_recall":
                grecall,
            "median_stop_cpt":
                (
                    float(
                        gstop.median()
                    )
                    if len(gstop)
                    else ""
                ),
            "unresolved":
                int(
                    (~g["stopped"]).sum()
                ),
            "blocked":
                int(
                    g["blocked"].sum()
                ),
        }

        grid_rows.append(row)

        print(
            f"{grid_name}"
            f" | STOP {gs:2d}/20"
            f" | safe {gtp:2d}"
            f" | false {gfp:2d}"
            f" ({gfalse*100:5.1f}%)"
            f" | recall "
            f"{grecall*100:5.1f}%"
            f" | median "
            f"{row['median_stop_cpt']}"
        )

    write_csv(
        OUT_DIR / "per_step.csv",
        step_rows,
    )

    write_csv(
        OUT_DIR / "decisions.csv",
        decision_rows,
    )

    write_csv(
        OUT_DIR / "per_grid.csv",
        grid_rows,
    )

    summary = {
        "validation":
            "DEVELOPMENT_AUDIT",

        "training":
            False,

        "volumes":
            n,

        "minimum_distance_fraction":
            MIN_DISTANCE_FRACTION,

        "probability_threshold":
            PROB_THRESHOLD,

        "persistence":
            PERSISTENCE,

        "actual_threshold":
            ACTUAL_THRESHOLD,

        "stops":
            stops,

        "safe_stops_tp":
            tp,

        "false_stops_fp":
            fp,

        "actual_positive_volumes":
            actual_positive,

        "false_stop_rate":
            false_stop_rate,

        "safe_stop_recall":
            safe_stop_recall,

        "unresolved_by_32":
            unresolved,

        "blocked":
            blocked,

        "median_stop_cpt":
            (
                float(
                    np.median(
                        stop_values
                    )
                )
                if len(stop_values)
                else None
            ),

        "mean_stop_cpt":
            (
                float(
                    np.mean(
                        stop_values
                    )
                )
                if len(stop_values)
                else None
            ),

        "median_safe_stop_cpt":
            (
                float(
                    np.median(
                        safe_stop_values
                    )
                )
                if len(
                    safe_stop_values
                )
                else None
            ),

        "median_stop_delay":
            (
                float(
                    np.median(
                        delays
                    )
                )
                if len(delays)
                else None
            ),

        "acceptance_false_stop_le_5pct":
            bool(
                pass_false_stop
            ),
    }

    (
        OUT_DIR / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "Saved:",
        OUT_DIR,
    )


if __name__ == "__main__":
    main()
