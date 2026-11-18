from pathlib import Path
from types import SimpleNamespace
import json

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from dqn3d.v45.adapter import FrozenV4Reconstructor


ROOT = Path.home() / "Developer" / "cpt_super_learn"

DATA_DIR = (
    ROOT
    / "data"
    / "synthetic3d_v4_geosyn"
    / "combined_gate_validation_100"
    / "test"
)

ACTIONS_PATH = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "combined_gate_external_v46a"
    / "actions.csv"
)

UNCERTAINTY_ROOT = (
    ROOT
    / "results"
    / "uncertainty_v3"
)

UNCERTAINTY_MODEL = (
    UNCERTAINTY_ROOT
    / "uncertainty_v3_quantile.joblib"
)

UNCERTAINTY_METADATA = (
    UNCERTAINTY_ROOT
    / "model_metadata.json"
)

SGAN_CHECKPOINT = (
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
    / "uncertainty_v3_gate_external"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CPT_COUNTS = list(range(1, 33))

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


def parse_metadata(raw):
    if isinstance(raw, np.ndarray):
        raw = raw.item()

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    return json.loads(str(raw))


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

    distance = np.linalg.norm(
        candidates.astype(float)
        - center[None, :],
        axis=1,
    )

    x, y = candidates[
        np.argmin(distance)
    ]

    return int(x), int(y)


def grid_label(grid):
    return (
        f"{grid.nx}x{grid.ny}x{grid.nz}"
        f"_r{grid.dx_m:g}x"
        f"{grid.dy_m:g}x"
        f"{grid.dz_m:g}m"
    )


# ============================================================
# Load frozen models
# ============================================================

uncertainty_model = joblib.load(
    UNCERTAINTY_MODEL
)

uncertainty_meta = json.loads(
    UNCERTAINTY_METADATA.read_text()
)

assert (
    uncertainty_meta["features"]
    == FEATURES
)

correction = float(
    uncertainty_meta[
        "calibration_correction"
    ]
)

reconstructor = FrozenV4Reconstructor(
    checkpoint=SGAN_CHECKPOINT,
    normalization_path=NORMALIZATION,
)


# ============================================================
# Load exact V4.6-A trajectory
# ============================================================

actions = pd.read_csv(
    ACTIONS_PATH
)

actions = actions[
    actions["policy"] == "dqn"
].copy()

actions["n_cpt"] = (
    actions["n_cpt"].astype(int)
)

actions["x"] = actions["x"].astype(int)
actions["y"] = actions["y"].astype(int)


paths = sorted(
    DATA_DIR.glob("*.npz")
)

assert len(paths) == 100, (
    f"Expected 100 volumes, got {len(paths)}"
)


print(
    "=== UNCERTAINTY V2 SUFFICIENCY GATE ==="
)
print(
    "SchemaGAN device:",
    reconstructor.device,
)
print(
    "Volumes:",
    len(paths),
)
print(
    "CPT counts:",
    CPT_COUNTS,
)
print(
    "Ic tolerance:",
    IC_TOLERANCE,
)
print(
    "Actual sufficiency coverage:",
    SUFFICIENCY_COVERAGE,
)
print()


rows = []


# ============================================================
# Evaluate all volumes
# ============================================================

for vi, path in enumerate(paths, start=1):

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        truth = np.asarray(
            data["Ic"],
            dtype=np.float32,
        )

        valid = np.asarray(
            data["valid_mask"],
            dtype=bool,
        )

        meta = parse_metadata(
            data["metadata_json"]
        )

    g = meta["grid"]

    grid = SimpleNamespace(
        nx=int(g["nx"]),
        ny=int(g["ny"]),
        nz=int(g["nz"]),
        dx_m=float(g["dx_m"]),
        dy_m=float(g["dy_m"]),
        dz_m=float(g["dz_m"]),
    )

    dx = grid.dx_m
    dy = grid.dy_m
    dz = grid.dz_m

    nx, ny, nz = truth.shape

    initial_xy = choose_initial_center(
        valid
    )

    volume_actions = (
        actions[
            actions["volume"]
            == path.name
        ]
        .sort_values("n_cpt")
    )

    assert len(volume_actions) == 31, (
        f"{path.name}: expected 31 "
        f"DQN actions, got "
        f"{len(volume_actions)}"
    )

    xx, yy = np.meshgrid(
        np.arange(nx) * dx,
        np.arange(ny) * dy,
        indexing="ij",
    )

    for count in CPT_COUNTS:

        # -----------------------------------------------
        # Exact DQN trajectory:
        #
        # CPT 1 = center
        # n_cpt=2 row = second CPT
        # n_cpt=3 row = third CPT
        # etc.
        # -----------------------------------------------

        chosen = [initial_xy]

        if count > 1:

            selected_actions = (
                volume_actions[
                    volume_actions["n_cpt"]
                    <= count
                ]
            )

            for _, r in (
                selected_actions.iterrows()
            ):
                chosen.append(
                    (
                        int(r["x"]),
                        int(r["y"]),
                    )
                )

        assert len(chosen) == count

        chosen = np.asarray(
            chosen,
            dtype=np.int32,
        )

        # -----------------------------------------------
        # Build sparse CPT observation
        # -----------------------------------------------

        observed = np.zeros_like(
            valid,
            dtype=bool,
        )

        for x, y in chosen:
            observed[x, y, :] = (
                valid[x, y, :]
            )

        sparse = np.where(
            observed,
            truth,
            0.0,
        ).astype(np.float32)

        prediction, _ = (
            reconstructor.predict(
                sparse,
                observed,
                valid,
                grid,
            )
        )

        prediction = np.asarray(
            prediction,
            dtype=np.float32,
        )

        # -----------------------------------------------
        # Uncertainty V2 features
        # -----------------------------------------------

        measured_xy = observed.any(
            axis=2
        )

        distance_xy = (
            distance_transform_edt(
                ~measured_xy,
                sampling=(dx, dy),
            )
            .astype(np.float32)
        )

        grad_x, grad_y, grad_z = (
            np.gradient(
                prediction,
                dx,
                dy,
                dz,
            )
        )

        lateral = np.sqrt(
            grad_x ** 2
            + grad_y ** 2
        )

        vertical = np.abs(
            grad_z
        )

        nearby = np.zeros(
            (nx, ny),
            dtype=np.float32,
        )

        for x, y in chosen:

            distance_to_site = np.sqrt(
                (xx - x * dx) ** 2
                +
                (yy - y * dy) ** 2
            )

            nearby += (
                distance_to_site <= 25.0
            ).astype(np.float32)

        nearby /= float(count)

        hidden = (
            valid
            & (~observed)
        )

        coords = np.where(hidden)

        x = coords[0]
        y = coords[1]
        z = coords[2]

        assert len(x) > 0

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
                    count,
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

        # -----------------------------------------------
        # Ground-truth actual error
        # ONLY for evaluation, not app inference.
        # -----------------------------------------------

        actual_error = np.abs(
            prediction[x, y, z]
            -
            truth[x, y, z]
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

        rows.append({
            "volume": path.name,
            "grid": grid_label(grid),
            "family": meta.get(
                "family",
                "",
            ),
            "n_cpt": count,

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
        })

    if (
        vi % 10 == 0
        or vi == len(paths)
    ):
        print(
            f"{vi}/{len(paths)} volumes"
        )


# ============================================================
# Save per-volume results
# ============================================================

result = pd.DataFrame(rows)

out_csv = (
    OUT_DIR
    / "per_volume_uncertainty_gate.csv"
)

result.to_csv(
    out_csv,
    index=False,
)


# ============================================================
# Summary
# ============================================================

print()
print("=== OVERALL ===")

actual = (
    result["actual_sufficient"]
    .to_numpy(dtype=bool)
)

predicted = (
    result["predicted_sufficient"]
    .to_numpy(dtype=bool)
)

tp = int(
    np.sum(predicted & actual)
)

tn = int(
    np.sum(
        (~predicted)
        & (~actual)
    )
)

fp = int(
    np.sum(
        predicted
        & (~actual)
    )
)

fn = int(
    np.sum(
        (~predicted)
        & actual
    )
)

print("TP:", tp)
print("TN:", tn)
print(
    "FALSE STOP (FP):",
    fp,
)
print(
    "Missed stop (FN):",
    fn,
)

if predicted.sum() > 0:
    false_stop_fraction = (
        fp / predicted.sum()
    )
else:
    false_stop_fraction = 0.0

print(
    "False-stop fraction among STOP decisions:",
    f"{false_stop_fraction * 100:.1f}%",
)

coverage_mae = float(
    np.mean(
        np.abs(
            result[
                "predicted_coverage"
            ]
            -
            result[
                "actual_coverage"
            ]
        )
    )
)

coverage_bias = float(
    np.mean(
        result[
            "predicted_coverage"
        ]
        -
        result[
            "actual_coverage"
        ]
    )
)

pearson = float(
    result[
        [
            "predicted_coverage",
            "actual_coverage",
        ]
    ]
    .corr(
        method="pearson"
    )
    .iloc[0, 1]
)

spearman = float(
    result[
        [
            "predicted_coverage",
            "actual_coverage",
        ]
    ]
    .corr(
        method="spearman"
    )
    .iloc[0, 1]
)

print()
print(
    "Coverage MAE:",
    f"{coverage_mae:.4f}",
)

print(
    "Coverage bias (pred - actual):",
    f"{coverage_bias:+.4f}",
)

print(
    "Pearson:",
    f"{pearson:.4f}",
)

print(
    "Spearman:",
    f"{spearman:.4f}",
)


# ============================================================
# Per CPT count
# ============================================================

print()
print("=== BY CPT COUNT ===")

summary_rows = []

for count in CPT_COUNTS:

    sub = result[
        result["n_cpt"] == count
    ]

    a = (
        sub["actual_sufficient"]
        .to_numpy(dtype=bool)
    )

    p = (
        sub["predicted_sufficient"]
        .to_numpy(dtype=bool)
    )

    fp_count = int(
        np.sum(
            p & (~a)
        )
    )

    predicted_stop = int(
        np.sum(p)
    )

    false_stop_rate = (
        fp_count / predicted_stop
        if predicted_stop > 0
        else 0.0
    )

    row = {
        "n_cpt": count,

        "actual_coverage_mean":
            float(
                sub[
                    "actual_coverage"
                ].mean()
            ),

        "predicted_coverage_mean":
            float(
                sub[
                    "predicted_coverage"
                ].mean()
            ),

        "actual_sufficient_n":
            int(a.sum()),

        "predicted_stop_n":
            predicted_stop,

        "false_stop_n":
            fp_count,

        "false_stop_fraction":
            false_stop_rate,
    }

    summary_rows.append(row)

    print()
    print(
        f"CPT={count}"
    )

    print(
        "  Actual coverage mean   :",
        f"{row['actual_coverage_mean']*100:.1f}%",
    )

    print(
        "  Predicted coverage mean:",
        f"{row['predicted_coverage_mean']*100:.1f}%",
    )

    print(
        "  Actual sufficient      :",
        f"{row['actual_sufficient_n']}/"
        f"{len(sub)}",
    )

    print(
        "  Predicted STOP         :",
        f"{predicted_stop}/"
        f"{len(sub)}",
    )

    print(
        "  False STOP             :",
        fp_count,
    )

    print(
        "  False-stop fraction    :",
        f"{false_stop_rate*100:.1f}%",
    )


pd.DataFrame(
    summary_rows
).to_csv(
    OUT_DIR
    / "summary_by_cpt_count.csv",
    index=False,
)


# ============================================================
# Per grid + CPT count
# ============================================================

grid_rows = []

for (
    grid_name,
    count
), sub in result.groupby(
    ["grid", "n_cpt"]
):

    a = (
        sub["actual_sufficient"]
        .to_numpy(dtype=bool)
    )

    p = (
        sub["predicted_sufficient"]
        .to_numpy(dtype=bool)
    )

    fp_count = int(
        np.sum(
            p & (~a)
        )
    )

    predicted_stop = int(
        p.sum()
    )

    grid_rows.append({
        "grid": grid_name,
        "n_cpt": int(count),

        "actual_coverage_mean":
            float(
                sub[
                    "actual_coverage"
                ].mean()
            ),

        "predicted_coverage_mean":
            float(
                sub[
                    "predicted_coverage"
                ].mean()
            ),

        "actual_sufficient_n":
            int(a.sum()),

        "predicted_stop_n":
            predicted_stop,

        "false_stop_n":
            fp_count,
    })


pd.DataFrame(
    grid_rows
).to_csv(
    OUT_DIR
    / "summary_by_grid_count.csv",
    index=False,
)


print()
print("Saved:")
print(out_csv)
print(
    OUT_DIR
    / "summary_by_cpt_count.csv"
)
print(
    OUT_DIR
    / "summary_by_grid_count.csv"
)

print()
print(
    "NOTE: actual coverage uses ground truth "
    "ONLY for validation."
)

print(
    "The app would only have access to "
    "predicted coverage."
)
