from pathlib import Path
import argparse
import csv
import json

import joblib
import numpy as np
import pandas as pd
import torch

from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

from dqn3d.v46.agent import CandidateDQNAgent
from dqn3d.v46.adapter import FrozenV4Reconstructor
from dqn3d.v46.runner import _entries, _case, _initial
from dqn3d.v46.service import recommend


ROOT = Path.home() / "Developer/cpt_super_learn"

DATASET = (
    ROOT
    / "data/synthetic3d_v4_geosyn/full_24k"
)

DQN_CKPT = (
    ROOT
    / "results/dqn_v46a_experimental"
    / "training_5000/best.pt"
)

SGAN_CKPT = (
    ROOT
    / "model_export/SchemaGAN_V4_2"
    / "SchemaGAN_V4_2.pt"
)

NORMALIZATION = (
    ROOT
    / "model_export/SchemaGAN_V4_2"
    / "normalization.json"
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

parser = argparse.ArgumentParser(
    description="GeoSyn DQN V4.6-A Sufficiency Gate evaluation"
)
parser.add_argument("--start-index", type=int, default=500)
parser.add_argument("--stop-index", type=int, default=800)
parser.add_argument("--output-dir", type=Path, default=None)

args = parser.parse_args()

START = args.start_index
STOP = args.stop_index

if START < 0 or STOP <= START:
    parser.error("Require 0 <= start-index < stop-index")

OUT = (
    args.output_dir
    if args.output_dir is not None
    else ROOT / "results/sufficiency_gate_v3_dqn_1to16"
).resolve()

MAX_CPT = 16
SEED = 20261109

IC_TOLERANCE = 0.15
SUFFICIENCY_COVERAGE = 0.90

FEATURES = [
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


def reveal(
    truth,
    valid,
    observed_ic,
    observed_mask,
    xy,
):
    x, y = xy

    zvalid = valid[x, y]

    observed_ic[
        x, y, zvalid
    ] = truth[
        x, y, zvalid
    ]

    observed_mask[
        x, y, zvalid
    ] = True


def evaluate_state(
    truth,
    valid,
    grid,
    prediction,
    observed_mask,
    selected,
    n_cpt,
    uncertainty_model,
    correction,
):
    nx, ny, nz = truth.shape

    dx = float(grid.dx_m)
    dy = float(grid.dy_m)
    dz = float(grid.dz_m)

    measured_xy = observed_mask.any(
        axis=2
    )

    distance_xy = distance_transform_edt(
        ~measured_xy,
        sampling=(dx, dy),
    ).astype(np.float32)

    grad_x, grad_y, grad_z = np.gradient(
        prediction,
        dx,
        dy,
        dz,
    )

    lateral = np.sqrt(
        grad_x ** 2
        + grad_y ** 2
    )

    vertical = np.abs(
        grad_z
    )

    xx, yy = np.meshgrid(
        np.arange(nx) * dx,
        np.arange(ny) * dy,
        indexing="ij",
    )

    nearby = np.zeros(
        (nx, ny),
        dtype=np.float32,
    )

    for x, y in selected:

        distance_to_site = np.sqrt(
            (xx - x * dx) ** 2
            +
            (yy - y * dy) ** 2
        )

        nearby += (
            distance_to_site <= 25.0
        ).astype(np.float32)

    nearby /= float(n_cpt)

    hidden = (
        valid
        & (~observed_mask)
    )

    coords = np.where(
        hidden
    )

    x = coords[0]
    y = coords[1]
    z = coords[2]

    if len(x) == 0:
        raise RuntimeError(
            "No hidden voxels"
        )

    feature_frame = pd.DataFrame({
        "distance_to_cpt_m":
            distance_xy[x, y],

        "depth_m":
            z * dz,

        "predicted_ic":
            prediction[x, y, z],

        "lateral_gradient":
            lateral[x, y, z],

        "vertical_gradient":
            vertical[x, y, z],

        "nearby_cpt_fraction_25m":
            nearby[x, y],

        "n_cpt":
            np.full(
                len(x),
                n_cpt,
            ),

        "dx_m":
            np.full(
                len(x),
                dx,
            ),

        "dy_m":
            np.full(
                len(x),
                dy,
            ),

        "dz_m":
            np.full(
                len(x),
                dz,
            ),
    })[FEATURES]

    estimated_error = np.maximum(
        0.0,
        uncertainty_model.predict(
            feature_frame
        ),
    ) + correction

    actual_error = np.abs(
        prediction[x, y, z]
        - truth[x, y, z]
    )

    predicted_coverage = float(
        np.mean(
            estimated_error
            <= IC_TOLERANCE
        )
    )

    actual_coverage = float(
        np.mean(
            actual_error
            <= IC_TOLERANCE
        )
    )

    predicted_sufficient = (
        predicted_coverage
        >= SUFFICIENCY_COVERAGE
    )

    actual_sufficient = (
        actual_coverage
        >= SUFFICIENCY_COVERAGE
    )

    actual_rmse = float(
        np.sqrt(
            np.mean(
                actual_error ** 2
            )
        )
    )

    return {
        "actual_coverage":
            actual_coverage,

        "predicted_coverage":
            predicted_coverage,

        "actual_sufficient":
            int(actual_sufficient),

        "predicted_sufficient":
            int(predicted_sufficient),

        "actual_mae":
            float(
                actual_error.mean()
            ),

        "actual_rmse":
            actual_rmse,

        "actual_p90_error":
            float(
                np.quantile(
                    actual_error,
                    0.90,
                )
            ),

        "estimated_mean_bound":
            float(
                estimated_error.mean()
            ),

        "estimated_p90_bound":
            float(
                np.quantile(
                    estimated_error,
                    0.90,
                )
            ),

        "hidden_voxels":
            int(len(x)),
    }


device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(
    "=== FINAL SUFFICIENCY GATE TEST ==="
)
print(
    "GeoSyn test :",
    START,
    "-",
    STOP - 1,
)
print(
    "Volumes     :",
    STOP - START,
)
print(
    "CPT         : 1-16"
)
print(
    "Policy      : DQN V4.6-A"
)
print(
    "Reconstruct : SchemaGAN V4.2"
)
print(
    "Uncertainty : V3 quantile"
)
print(
    "Criterion   : predicted coverage >= 90%"
)
print(
    "Ic tolerance: ±0.15"
)
print(
    "Device      :",
    device,
)
print()


# Load Uncertainty before initializing DQN / SchemaGAN.
# Prevents NumPy/SciPy import failure after reconstructor initialization.
uncertainty_model = joblib.load(
    UNCERTAINTY_MODEL
)

agent, _ = CandidateDQNAgent.load(
    DQN_CKPT,
    map_location=device,
)

reconstructor = FrozenV4Reconstructor(
    checkpoint=SGAN_CKPT,
    normalization_path=NORMALIZATION,
)

uncertainty_meta = json.loads(
    UNCERTAINTY_META.read_text()
)

if uncertainty_meta["features"] != FEATURES:
    raise RuntimeError(
        "Uncertainty feature mismatch"
    )

correction = float(
    uncertainty_meta[
        "calibration_correction"
    ]
)

entries = _entries(
    DATASET,
    "test",
    STOP,
)[START:STOP]


rows = []
frontier_rows = []


for vi, entry in enumerate(entries):

    truth, valid, grid = _case(
        DATASET,
        entry,
    )

    initial = _initial(
        grid,
        valid,
        entry,
        1,
        SEED,
    )

    selected = [
        tuple(
            map(int, xy)
        )
        for xy in initial
    ]

    observed_ic = np.zeros_like(
        truth,
        dtype=np.float32,
    )

    observed_mask = np.zeros_like(
        valid,
        dtype=bool,
    )

    for xy in selected:
        reveal(
            truth,
            valid,
            observed_ic,
            observed_mask,
            xy,
        )

    volume_rows = []

    for n_cpt in range(
        1,
        MAX_CPT + 1,
    ):

        prediction, _ = reconstructor.predict(
            observed_ic,
            observed_mask,
            valid,
            grid,
        )

        prediction = np.asarray(
            prediction,
            dtype=np.float32,
        )

        result = evaluate_state(
            truth,
            valid,
            grid,
            prediction,
            observed_mask,
            selected,
            n_cpt,
            uncertainty_model,
            correction,
        )

        row = {
            "volume_index":
                START + vi,

            "n_cpt":
                n_cpt,

            **result,
        }

        rows.append(row)
        volume_rows.append(row)

        if n_cpt == MAX_CPT:
            break

        recommendation = recommend(
            reconstructor,
            agent,
            observed_ic,
            observed_mask,
            valid,
            grid,
            top_k=1,
        )

        if (
            recommendation.get(
                "status"
            )
            != "ok"
        ):
            raise RuntimeError(
                f"DQN failed volume "
                f"{START + vi}, "
                f"CPT={n_cpt}"
            )

        candidate = (
            recommendation[
                "ranked_candidates"
            ][0]
        )

        xy = (
            int(
                candidate[
                    "x_index"
                ]
            ),
            int(
                candidate[
                    "y_index"
                ]
            ),
        )

        selected.append(
            xy
        )

        reveal(
            truth,
            valid,
            observed_ic,
            observed_mask,
            xy,
        )


    first_actual = next(
        (
            r["n_cpt"]
            for r in volume_rows
            if r[
                "actual_sufficient"
            ] == 1
        ),
        None,
    )

    first_predicted = next(
        (
            r["n_cpt"]
            for r in volume_rows
            if r[
                "predicted_sufficient"
            ] == 1
        ),
        None,
    )


    unsafe_first_stop = 0

    if first_predicted is not None:

        pred_row = next(
            r
            for r in volume_rows
            if r["n_cpt"]
            == first_predicted
        )

        unsafe_first_stop = int(
            pred_row[
                "actual_sufficient"
            ] == 0
        )


    if (
        first_actual is not None
        and first_predicted is not None
    ):
        stop_delay = (
            first_predicted
            - first_actual
        )
    else:
        stop_delay = np.nan


    frontier_rows.append({
        "volume_index":
            START + vi,

        "first_actual_stop":
            (
                first_actual
                if first_actual
                is not None
                else ""
            ),

        "first_predicted_stop":
            (
                first_predicted
                if first_predicted
                is not None
                else ""
            ),

        "unsafe_first_stop":
            unsafe_first_stop,

        "stop_delay_cpt":
            stop_delay,
    })


    if (
        vi == 0
        or (vi + 1) % 25 == 0
    ):
        print(
            f"[{vi+1:3d}/{len(entries)}]"
        )


per_state_path = (
    OUT
    / "per_state_gate.csv"
)

with per_state_path.open(
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


frontier_path = (
    OUT
    / "per_volume_frontier.csv"
)

with frontier_path.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            frontier_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        frontier_rows
    )


actual_cov = np.asarray([
    r["actual_coverage"]
    for r in rows
])

pred_cov = np.asarray([
    r["predicted_coverage"]
    for r in rows
])

actual_stop = np.asarray([
    r["actual_sufficient"]
    for r in rows
], dtype=bool)

pred_stop = np.asarray([
    r["predicted_sufficient"]
    for r in rows
], dtype=bool)


tp = int(
    np.sum(
        actual_stop
        & pred_stop
    )
)

fp = int(
    np.sum(
        (~actual_stop)
        & pred_stop
    )
)

fn = int(
    np.sum(
        actual_stop
        & (~pred_stop)
    )
)

tn = int(
    np.sum(
        (~actual_stop)
        & (~pred_stop)
    )
)

coverage_mae = float(
    np.mean(
        np.abs(
            pred_cov
            - actual_cov
        )
    )
)

coverage_bias = float(
    np.mean(
        pred_cov
        - actual_cov
    )
)

rho, rho_p = spearmanr(
    pred_cov,
    actual_cov,
)

false_stop_rate = (
    fp / (tp + fp)
    if (tp + fp) > 0
    else np.nan
)

missed_stop_rate = (
    fn / (tp + fn)
    if (tp + fn) > 0
    else np.nan
)


print()
print("=" * 78)
print("FINAL GATE SUMMARY")
print("=" * 78)

print(
    "States evaluated       :",
    len(rows),
)

print(
    "Coverage MAE           :",
    f"{coverage_mae:.4f}",
)

print(
    "Coverage bias          :",
    f"{coverage_bias:+.4f}",
)

print(
    "Coverage Spearman      :",
    f"{rho:+.4f}",
    f"(p={rho_p:.3g})",
)

print()
print(
    "STOP TP / FP           :",
    tp,
    "/",
    fp,
)

print(
    "STOP FN / TN           :",
    fn,
    "/",
    tn,
)

print(
    "False-stop rate        :",
    (
        f"{false_stop_rate*100:.2f}%"
        if np.isfinite(
            false_stop_rate
        )
        else "N/A"
    ),
)

print(
    "Missed-stop rate       :",
    (
        f"{missed_stop_rate*100:.2f}%"
        if np.isfinite(
            missed_stop_rate
        )
        else "N/A"
    ),
)


first_actual = [
    r["first_actual_stop"]
    for r in frontier_rows
    if r["first_actual_stop"] != ""
]

first_pred = [
    r["first_predicted_stop"]
    for r in frontier_rows
    if r["first_predicted_stop"] != ""
]

unsafe = sum(
    r["unsafe_first_stop"]
    for r in frontier_rows
)

never_pred = sum(
    r["first_predicted_stop"] == ""
    for r in frontier_rows
)

never_actual = sum(
    r["first_actual_stop"] == ""
    for r in frontier_rows
)

print()
print(
    "Volumes actual STOP <=16 :",
    len(first_actual),
    "/",
    len(frontier_rows),
)

print(
    "Volumes predicted STOP <=16:",
    len(first_pred),
    "/",
    len(frontier_rows),
)

print(
    "Unsafe FIRST STOP volumes :",
    unsafe,
    "/",
    len(frontier_rows),
)

print(
    "Never predicted STOP <=16 :",
    never_pred,
)

print(
    "Never actual STOP <=16    :",
    never_actual,
)


both = [
    r
    for r in frontier_rows
    if (
        r["first_actual_stop"] != ""
        and
        r["first_predicted_stop"]
        != ""
    )
]

if both:

    delays = np.asarray([
        r["stop_delay_cpt"]
        for r in both
    ], dtype=float)

    print(
        "Mean predicted-stop delay:",
        f"{delays.mean():+.2f} CPT",
    )

    print(
        "Median stop delay        :",
        f"{np.median(delays):+.1f} CPT",
    )


print()
print("=" * 78)
print("BY CPT COUNT")
print("=" * 78)

print(
    f"{'CPT':>3}"
    f" {'ActualCov':>11}"
    f" {'PredCov':>11}"
    f" {'ActualSTOP':>12}"
    f" {'PredSTOP':>11}"
    f" {'FalseSTOP':>11}"
)

print("-" * 78)


for count in range(
    1,
    MAX_CPT + 1,
):

    subset = [
        r
        for r in rows
        if r["n_cpt"] == count
    ]

    a_cov = np.mean([
        r["actual_coverage"]
        for r in subset
    ])

    p_cov = np.mean([
        r["predicted_coverage"]
        for r in subset
    ])

    a_stop = np.mean([
        r["actual_sufficient"]
        for r in subset
    ]) * 100

    p_stop = np.mean([
        r["predicted_sufficient"]
        for r in subset
    ]) * 100

    false_stop = sum(
        (
            r["predicted_sufficient"]
            == 1
            and
            r["actual_sufficient"]
            == 0
        )
        for r in subset
    )

    print(
        f"{count:>3}"
        f" {a_cov:>10.1%}"
        f" {p_cov:>10.1%}"
        f" {a_stop:>11.1f}%"
        f" {p_stop:>10.1f}%"
        f" {false_stop:>11}"
    )


print()
print(
    "Per-state:",
    per_state_path,
)

print(
    "Frontier :",
    frontier_path,
)

print()
print("=== COMPLETE ===")
