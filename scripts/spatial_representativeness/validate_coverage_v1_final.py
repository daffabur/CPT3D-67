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
    / "v46a_min10_final_validation_100"
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
    / "v46a_min10_final_validation"
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




# ============================================================
# COVERAGE V1 FINAL CANDIDATE
#
# FROZEN BEFORE OPENING THIS FINAL SET.
# ============================================================

COVERAGE_MODEL_FINAL = (
    Path.home()
    / "Developer/cpt_super_learn"
    / "results/coverage_v1_stop_gate"
    / "coverage_v1_classifier.joblib"
)

COVERAGE_FINAL_DATA = (
    Path.home()
    / "Developer/cpt_super_learn"
    / "data/synthetic3d_v4_geosyn"
    / "v46a_coverage_v1_final_validation_100"
    / "test"
)

COVERAGE_FINAL_OUT = (
    Path.home()
    / "Developer/cpt_super_learn"
    / "results/spatial_representativeness"
    / "coverage_v1_final_validation"
)

COVERAGE_THRESHOLD_FINAL = 0.925
COVERAGE_PERSISTENCE_FINAL = 5


def evaluate_state(
    sim,
    *,
    n_cpt,
    coverage_model,
):
    """
    Coverage V1:
    predict P(|Ic error| <= 0.15) for every hidden valid voxel,
    then average the probabilities to estimate site coverage.

    Actual coverage is evaluated using ALL hidden valid voxels.
    """

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

        # ====================================================
        # EXACT Coverage V1 10-feature construction.
        # Same base features as Uncertainty V3.
        # ====================================================

        measured_xy = observed.any(
            axis=2
        )

        distance_xy = (
            distance_transform_edt(
                ~measured_xy,
                sampling=(
                    dx,
                    dy,
                ),
            )
        )


        gx, gy, gz = np.gradient(
            prediction,
            dx,
            dy,
            dz,
        )

        lateral_gradient = np.sqrt(
            gx ** 2
            + gy ** 2
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

        for cx, cy in chosen:

            d = np.sqrt(
                (
                    xx
                    - cx * dx
                ) ** 2
                +
                (
                    yy
                    - cy * dy
                ) ** 2
            )

            nearby_count += (
                d <= 25.0
            ).astype(
                np.float32
            )


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


        Xcov = pd.DataFrame({
            "distance_to_cpt_m":
                distance_xy[
                    x,
                    y,
                ],

            "depth_m":
                z.astype(float)
                * dz,

            "predicted_ic":
                prediction[
                    x,
                    y,
                    z,
                ],

            "lateral_gradient":
                lateral_gradient[
                    x,
                    y,
                    z,
                ],

            "vertical_gradient":
                vertical_gradient[
                    x,
                    y,
                    z,
                ],

            "nearby_cpt_fraction_25m":
                nearby_fraction[
                    x,
                    y,
                ],

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


        safe_probability = (
            coverage_model.predict_proba(
                Xcov
            )[:, 1]
        )


        predicted_coverage = float(
            np.mean(
                safe_probability
            )
        )


        # ====================================================
        # Ground truth for final evaluation only.
        # Never supplied to the gate.
        # ====================================================

        abs_error = np.abs(
            prediction[
                x,
                y,
                z,
            ]
            -
            truth[
                x,
                y,
                z,
            ]
        )

        actual_coverage = float(
            np.mean(
                abs_error
                <= IC_TOLERANCE
            )
        )


    (
        _width,
        _height,
        diagonal,
        _area,
        _aspect,
    ) = geometry(
        grid
    )


    max_distance = (
        max_nearest_distance(
            sim
        )
    )

    max_fraction = (
        max_distance
        / max(
            diagonal,
            1e-6,
        )
    )


    return {
        "predicted_coverage":
            predicted_coverage,

        "actual_coverage":
            actual_coverage,

        "actual_sufficient":
            (
                actual_coverage
                >= ACTUAL_THRESHOLD
            ),

        "max_nearest_distance_m":
            max_distance,

        "max_distance_fraction":
            max_fraction,

        "positive_gate":
            (
                predicted_coverage
                >= COVERAGE_THRESHOLD_FINAL
            ),
    }


def first_true_run(
    rows,
    *,
    key,
    run_length,
):
    consecutive = 0

    for row in rows:

        if bool(
            row[key]
        ):
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= run_length:
            return int(
                row["n_cpt"]
            )

    return None


def first_positive_gate_stop(
    rows,
):
    consecutive = 0

    for row in rows:

        if row[
            "positive_gate"
        ]:
            consecutive += 1
        else:
            consecutive = 0

        if (
            consecutive
            >= COVERAGE_PERSISTENCE_FINAL
        ):
            return row

    return None


def main():

    COVERAGE_FINAL_OUT.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Verify frozen final-set manifest BEFORE evaluation.
    # ========================================================

    manifest_path = (
        COVERAGE_FINAL_DATA.parent
        / "evaluator_manifest.json"
    )

    manifest = json.loads(
        manifest_path.read_text()
    )

    candidate = manifest[
        "candidate_under_test"
    ]

    assert np.isclose(
        float(
            candidate[
                "minimum_distance_fraction"
            ]
        ),
        0.10,
    )

    assert np.isclose(
        float(
            candidate[
                "coverage_threshold"
            ]
        ),
        COVERAGE_THRESHOLD_FINAL,
    )

    assert (
        int(
            candidate[
                "persistence"
            ]
        )
        ==
        COVERAGE_PERSISTENCE_FINAL
    )

    assert np.isclose(
        float(
            candidate[
                "actual_ic_tolerance"
            ]
        ),
        IC_TOLERANCE,
    )

    assert np.isclose(
        float(
            candidate[
                "actual_sufficiency_threshold"
            ]
        ),
        ACTUAL_THRESHOLD,
    )


    # ========================================================
    # Load sklearn Coverage V1 BEFORE PyTorch binary stack.
    # ========================================================

    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
    )

    coverage_model = joblib.load(
        COVERAGE_MODEL_FINAL
    )

    print(
        "Coverage V1 model: LOAD PASS"
    )


    # ========================================================
    # Frozen SchemaGAN + DQN.
    # ========================================================

    from dqn3d.v46.adapter import (
        FrozenV4Reconstructor,
    )

    from dqn3d.v46.agent import (
        CandidateDQNAgent,
    )

    from dqn3d.v46.simulator import (
        SyntheticCPTSimulator,
    )


    reconstructor = FrozenV4Reconstructor(
        checkpoint=SCHEMAGAN_CHECKPOINT,
        normalization_path=NORMALIZATION,
    )

    agent, _ = CandidateDQNAgent.load(
        DQN_CHECKPOINT,
        map_location="cpu",
    )

    print(
        "SchemaGAN/DQN models: LOAD PASS"
    )


    paths = sorted(
        COVERAGE_FINAL_DATA.glob(
            "*.npz"
        )
    )

    assert len(paths) == 100, (
        f"Expected 100 final volumes, "
        f"got {len(paths)}"
    )


    print()
    print(
        "======================================"
    )
    print(
        "COVERAGE V1 FINAL UNTOUCHED VALIDATION"
    )
    print(
        "======================================"
    )

    print(
        "Volumes             :",
        len(paths),
    )

    print(
        "DQN minimum distance:",
        "10% site diagonal",
    )

    print(
        "Coverage threshold  :",
        COVERAGE_THRESHOLD_FINAL,
    )

    print(
        "Persistence         :",
        COVERAGE_PERSISTENCE_FINAL,
    )

    print(
        "Actual threshold    :",
        ACTUAL_THRESHOLD,
    )

    print(
        "Ic tolerance        :",
        IC_TOLERANCE,
    )

    print(
        "Max CPT             :",
        MAX_CPT,
    )

    print(
        "NO TRAINING / NO TUNING"
    )

    print()


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


        grid = build_grid(
            meta
        )

        grid_name = grid_label(
            grid
        )

        (
            _width,
            _height,
            diagonal,
            _area,
            _aspect,
        ) = geometry(
            grid
        )


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

        family = meta.get(
            "family",
            "",
        )


        # ====================================================
        # CPT 1
        # ====================================================

        metrics = evaluate_state(
            sim,
            n_cpt=1,
            coverage_model=(
                coverage_model
            ),
        )

        volume_steps.append({
            "n_cpt":
                1,
            **metrics,
        })

        step_rows.append({
            "volume":
                path.name,

            "grid":
                grid_name,

            "family":
                family,

            "n_cpt":
                1,

            "selected_x":
                int(
                    first_xy[0]
                ),

            "selected_y":
                int(
                    first_xy[1]
                ),

            "nearest_existing_cpt_m":
                np.nan,

            "q_value":
                np.nan,

            "minimum_distance_m":
                min_distance_m,

            **metrics,
        })


        blocked = False


        # ====================================================
        # CPT 2 ... 32
        # ====================================================

        for n_cpt in range(
            2,
            MAX_CPT + 1,
        ):

            candidates = (
                state.candidate_xy
            )


            if len(
                candidates
            ) == 0:

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


            if len(
                eligible
            ) == 0:

                blocked = True
                break


            scores = agent.scores(
                state
            )


            action = int(
                eligible[
                    np.argmax(
                        scores[
                            eligible
                        ]
                    )
                ]
            )


            q_value = float(
                scores[
                    action
                ]
            )

            selected_distance = float(
                nearest[
                    action
                ]
            )


            sx, sy = map(
                int,
                candidates[
                    action
                ],
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
                coverage_model=(
                    coverage_model
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
                    family,

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


        # ====================================================
        # Frozen Coverage V1 STOP.
        # ====================================================

        first_actual = next(
            (
                int(
                    row[
                        "n_cpt"
                    ]
                )
                for row
                in volume_steps
                if row[
                    "actual_sufficient"
                ]
            ),
            None,
        )


        first_stable5 = first_true_run(
            volume_steps,
            key="actual_sufficient",
            run_length=5,
        )


        stop_row = (
            first_positive_gate_stop(
                volume_steps
            )
        )


        if stop_row is None:

            stop_n = None
            stop_actual = False
            stop_actual_coverage = None
            stop_predicted_coverage = None

        else:

            stop_n = int(
                stop_row[
                    "n_cpt"
                ]
            )

            stop_actual = bool(
                stop_row[
                    "actual_sufficient"
                ]
            )

            stop_actual_coverage = float(
                stop_row[
                    "actual_coverage"
                ]
            )

            stop_predicted_coverage = float(
                stop_row[
                    "predicted_coverage"
                ]
            )


        decision_rows.append({
            "volume":
                path.name,

            "grid":
                grid_name,

            "family":
                family,

            "site_diagonal_m":
                diagonal,

            "minimum_distance_m":
                min_distance_m,

            "steps_evaluated":
                len(
                    volume_steps
                ),

            "blocked":
                blocked,

            "first_actual_sufficient_cpt":
                first_actual,

            "first_actual_stable5_cpt":
                first_stable5,

            "stop_cpt":
                stop_n,

            "stop_predicted_coverage":
                stop_predicted_coverage,

            "stop_actual_coverage":
                stop_actual_coverage,

            "safe_stop":
                (
                    stop_n
                    is not None
                    and stop_actual
                ),

            "false_stop":
                (
                    stop_n
                    is not None
                    and not stop_actual
                ),

            "stop_delay_from_first_actual":
                (
                    stop_n
                    - first_actual
                    if (
                        stop_n
                        is not None
                        and first_actual
                        is not None
                    )
                    else None
                ),

            "stop_delay_from_stable5":
                (
                    stop_n
                    - first_stable5
                    if (
                        stop_n
                        is not None
                        and first_stable5
                        is not None
                    )
                    else None
                ),
        })


        if (
            vi % 5 == 0
            or vi == len(paths)
        ):

            print(
                f"{vi:3d}/100"
                f" | {path.name}"
                f" | {grid_name}",
                flush=True,
            )


    # ========================================================
    # Save raw final outputs.
    # ========================================================

    steps_df = pd.DataFrame(
        step_rows
    )

    decisions_df = pd.DataFrame(
        decision_rows
    )


    steps_df.to_csv(
        COVERAGE_FINAL_OUT
        / "per_step.csv",
        index=False,
    )

    decisions_df.to_csv(
        COVERAGE_FINAL_OUT
        / "decisions.csv",
        index=False,
    )


    # ========================================================
    # Overall metrics.
    # ========================================================

    stops = int(
        decisions_df[
            "stop_cpt"
        ].notna().sum()
    )

    safe_stops = int(
        decisions_df[
            "safe_stop"
        ].sum()
    )

    false_stops = int(
        decisions_df[
            "false_stop"
        ].sum()
    )

    false_stop_rate = (
        false_stops
        / stops
        if stops
        else 0.0
    )


    actual_positive = int(
        decisions_df[
            "first_actual_sufficient_cpt"
        ].notna().sum()
    )


    stable_positive = int(
        decisions_df[
            "first_actual_stable5_cpt"
        ].notna().sum()
    )


    ever_safe_recall = (
        safe_stops
        / actual_positive
        if actual_positive
        else 0.0
    )


    stable_mask = (
        decisions_df[
            "first_actual_stable5_cpt"
        ].notna()
    )

    stable_safe = int(
        decisions_df.loc[
            stable_mask,
            "safe_stop",
        ].sum()
    )

    stable_recall = (
        stable_safe
        / stable_positive
        if stable_positive
        else 0.0
    )


    unresolved = int(
        decisions_df[
            "stop_cpt"
        ].isna().sum()
    )

    blocked_count = int(
        decisions_df[
            "blocked"
        ].sum()
    )


    safe_rows = (
        decisions_df[
            decisions_df[
                "safe_stop"
            ]
        ]
    )


    median_safe_stop = (
        float(
            safe_rows[
                "stop_cpt"
            ].median()
        )
        if len(
            safe_rows
        )
        else None
    )


    mean_safe_stop = (
        float(
            safe_rows[
                "stop_cpt"
            ].mean()
        )
        if len(
            safe_rows
        )
        else None
    )


    delay_rows = (
        safe_rows[
            "stop_delay_from_first_actual"
        ]
        .dropna()
    )

    median_delay = (
        float(
            delay_rows.median()
        )
        if len(
            delay_rows
        )
        else None
    )


    # ========================================================
    # Per-grid final metrics.
    # ========================================================

    grid_rows = []


    for grid_name, g in (
        decisions_df.groupby(
            "grid"
        )
    ):

        gs = int(
            g[
                "stop_cpt"
            ].notna().sum()
        )

        gsafe = int(
            g[
                "safe_stop"
            ].sum()
        )

        gfalse = int(
            g[
                "false_stop"
            ].sum()
        )

        gfalse_rate = (
            gfalse / gs
            if gs
            else 0.0
        )


        gactual = int(
            g[
                "first_actual_sufficient_cpt"
            ].notna().sum()
        )

        gstable_mask = (
            g[
                "first_actual_stable5_cpt"
            ].notna()
        )

        gstable = int(
            gstable_mask.sum()
        )

        gstable_safe = int(
            g.loc[
                gstable_mask,
                "safe_stop",
            ].sum()
        )


        grid_rows.append({
            "grid":
                grid_name,

            "volumes":
                len(g),

            "stops":
                gs,

            "safe_stops":
                gsafe,

            "false_stops":
                gfalse,

            "false_stop_rate":
                gfalse_rate,

            "actual_positive":
                gactual,

            "stable5_positive":
                gstable,

            "stable5_safe_stops":
                gstable_safe,

            "stable5_recall":
                (
                    gstable_safe
                    / gstable
                    if gstable
                    else 0.0
                ),

            "median_safe_stop_cpt":
                (
                    float(
                        g.loc[
                            g[
                                "safe_stop"
                            ],
                            "stop_cpt",
                        ].median()
                    )
                    if gsafe
                    else None
                ),

            "unresolved":
                int(
                    g[
                        "stop_cpt"
                    ].isna().sum()
                ),

            "blocked":
                int(
                    g[
                        "blocked"
                    ].sum()
                ),
        })


    per_grid_df = pd.DataFrame(
        grid_rows
    )


    per_grid_df.to_csv(
        COVERAGE_FINAL_OUT
        / "per_grid.csv",
        index=False,
    )


    max_grid_false = float(
        per_grid_df[
            "false_stop_rate"
        ].max()
    )


    # ========================================================
    # PRE-REGISTERED SAFETY ACCEPTANCE.
    # Efficiency metrics are reported, not retuned.
    # ========================================================

    safety_pass = bool(
        stops >= 10
        and false_stop_rate <= 0.05
        and max_grid_false <= 0.05
    )


    summary = {
        "validation":
            "FINAL_UNTOUCHED",

        "training":
            False,

        "tuning":
            False,

        "volumes":
            100,

        "coverage_model":
            "Coverage V1 direct voxel-safe estimator",

        "coverage_features":
            V3_FEATURES,

        "minimum_distance_fraction":
            0.10,

        "coverage_threshold":
            COVERAGE_THRESHOLD_FINAL,

        "persistence":
            COVERAGE_PERSISTENCE_FINAL,

        "actual_ic_tolerance":
            IC_TOLERANCE,

        "actual_sufficiency_threshold":
            ACTUAL_THRESHOLD,

        "stops":
            stops,

        "safe_stops":
            safe_stops,

        "false_stops":
            false_stops,

        "false_stop_rate":
            false_stop_rate,

        "max_per_grid_false_stop_rate":
            max_grid_false,

        "actual_positive_volumes":
            actual_positive,

        "stable5_positive_volumes":
            stable_positive,

        "ever_safe_stop_recall":
            ever_safe_recall,

        "stable5_safe_stop_recall":
            stable_recall,

        "unresolved_by_32":
            unresolved,

        "blocked_by_10pct_rule":
            blocked_count,

        "median_safe_stop_cpt":
            median_safe_stop,

        "mean_safe_stop_cpt":
            mean_safe_stop,

        "median_stop_delay_from_first_actual":
            median_delay,

        "acceptance": {
            "minimum_total_stops":
                10,

            "overall_false_stop_max":
                0.05,

            "per_grid_false_stop_max":
                0.05,

            "safety_pass":
                safety_pass,
        },

        "important":
            (
                "Final untouched result. "
                "Do not retune this candidate "
                "using this validation set."
            ),
    }


    (
        COVERAGE_FINAL_OUT
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )


    # ========================================================
    # Console report.
    # ========================================================

    print()
    print(
        "======================================"
    )
    print(
        "FINAL UNTOUCHED VALIDATION RESULT"
    )
    print(
        "======================================"
    )

    print(
        "Volumes              :",
        100,
    )

    print(
        "STOP                 :",
        stops,
    )

    print(
        "Safe STOP            :",
        safe_stops,
    )

    print(
        "False STOP           :",
        false_stops,
    )

    print(
        "False-stop rate      :",
        f"{false_stop_rate*100:.2f}%",
    )

    print(
        "Max grid false-stop  :",
        f"{max_grid_false*100:.2f}%",
    )

    print(
        "Actual-positive vols :",
        actual_positive,
    )

    print(
        "Stable-5 positive    :",
        stable_positive,
    )

    print(
        "Ever safe recall     :",
        f"{ever_safe_recall*100:.2f}%",
    )

    print(
        "Stable-5 recall      :",
        f"{stable_recall*100:.2f}%",
    )

    print(
        "Unresolved by 32     :",
        unresolved,
    )

    print(
        "Blocked by 10% rule  :",
        blocked_count,
    )

    print(
        "Median SAFE STOP CPT :",
        median_safe_stop,
    )

    print(
        "Median STOP delay    :",
        median_delay,
    )

    print()
    print(
        "Safety acceptance    :",
        (
            "PASS"
            if safety_pass
            else "FAIL"
        ),
    )


    print()
    print(
        "=== PER GRID ==="
    )

    for row in grid_rows:

        print(
            f"{row['grid']}"
            f" | STOP "
            f"{row['stops']:2d}/20"
            f" | safe "
            f"{row['safe_stops']:2d}"
            f" | false "
            f"{row['false_stops']:2d}"
            f" "
            f"({row['false_stop_rate']*100:5.1f}%)"
            f" | stable "
            f"{row['stable5_positive']:2d}"
            f" | recall "
            f"{row['stable5_recall']*100:5.1f}%"
            f" | median "
            f"{row['median_safe_stop_cpt']}"
        )


    print()
    print(
        "Saved:",
        COVERAGE_FINAL_OUT,
    )


if __name__ == "__main__":
    main()
