from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path.home() / "Developer" / "cpt_super_learn"

DATA_DIR = (
    ROOT
    / "data"
    / "synthetic3d_v4_geosyn"
    / "qscore_validation_100"
    / "test"
)

UNCERTAINTY_CSV = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "uncertainty_sufficiency_gate"
    / "per_volume_uncertainty_gate.csv"
)

ACTIONS_CSV = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "qstop_validation_v46a"
    / "actions.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "combined_sufficiency_gate"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CPT_COUNTS = [1, 3, 8]

ACTUAL_SUFFICIENCY = 0.90

# Minimum number of STOP decisions accepted
# during calibration.
MIN_STOPS = 30

# Main safety target.
MAX_FALSE_STOP = 0.05


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


def calculate_spatial_support(
    valid,
    chosen,
    dx,
    dy,
):
    valid_xy = valid.any(axis=2)

    coords = np.argwhere(
        valid_xy
    )

    if len(coords) == 0:
        raise RuntimeError(
            "No valid XY locations."
        )

    chosen = np.asarray(
        chosen,
        dtype=float,
    )

    # Physical distances from every valid XY
    # to every observed CPT.
    delta_x = (
        coords[:, None, 0]
        - chosen[None, :, 0]
    ) * dx

    delta_y = (
        coords[:, None, 1]
        - chosen[None, :, 1]
    ) * dy

    distances = np.sqrt(
        delta_x ** 2
        + delta_y ** 2
    )

    nearest = distances.min(axis=1)

    max_distance = float(
        nearest.max()
    )

    p90_distance = float(
        np.quantile(
            nearest,
            0.90,
        )
    )

    nx, ny, _ = valid.shape

    width = max(
        (nx - 1) * dx,
        1e-6,
    )

    height = max(
        (ny - 1) * dy,
        1e-6,
    )

    diagonal = float(
        np.sqrt(
            width ** 2
            + height ** 2
        )
    )

    return {
        "max_distance_m":
            max_distance,

        "p90_distance_m":
            p90_distance,

        "site_diagonal_m":
            diagonal,

        "max_distance_fraction":
            max_distance / diagonal,

        "p90_distance_fraction":
            p90_distance / diagonal,
    }


# ============================================================
# Load existing results
# ============================================================

uncertainty = pd.read_csv(
    UNCERTAINTY_CSV
)

actions = pd.read_csv(
    ACTIONS_CSV
)

actions = actions[
    actions["policy"] == "dqn"
].copy()

actions["n_cpt"] = (
    actions["n_cpt"].astype(int)
)

actions["x"] = (
    actions["x"].astype(int)
)

actions["y"] = (
    actions["y"].astype(int)
)


print(
    "=== BUILD SPATIAL SUPPORT ==="
)

print(
    "Uncertainty rows:",
    len(uncertainty),
)

paths = sorted(
    DATA_DIR.glob("*.npz")
)

assert len(paths) == 100, (
    f"Expected 100 volumes, "
    f"found {len(paths)}"
)


spatial_rows = []


# ============================================================
# Reconstruct CPT trajectory only.
# NO SchemaGAN inference.
# ============================================================

for vi, path in enumerate(
    paths,
    start=1,
):

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        valid = np.asarray(
            data["valid_mask"],
            dtype=bool,
        )

        meta = parse_metadata(
            data["metadata_json"]
        )

    g = meta["grid"]

    dx = float(g["dx_m"])
    dy = float(g["dy_m"])

    initial_xy = (
        choose_initial_center(
            valid
        )
    )

    volume_actions = (
        actions[
            actions["volume"]
            == path.name
        ]
        .sort_values("n_cpt")
    )

    assert len(volume_actions) == 31, (
        f"{path.name}: expected "
        f"31 DQN actions, "
        f"got {len(volume_actions)}"
    )

    for count in CPT_COUNTS:

        chosen = [
            initial_xy
        ]

        if count > 1:

            selected = (
                volume_actions[
                    volume_actions[
                        "n_cpt"
                    ]
                    <= count
                ]
            )

            for _, row in (
                selected.iterrows()
            ):
                chosen.append(
                    (
                        int(row["x"]),
                        int(row["y"]),
                    )
                )

        assert len(chosen) == count

        spatial = (
            calculate_spatial_support(
                valid,
                chosen,
                dx,
                dy,
            )
        )

        spatial_rows.append({
            "volume":
                path.name,

            "n_cpt":
                count,

            **spatial,
        })

    if vi % 20 == 0:
        print(
            f"{vi}/{len(paths)} volumes"
        )


spatial_df = pd.DataFrame(
    spatial_rows
)

assert len(spatial_df) == 300


# ============================================================
# Merge with Uncertainty V2 evaluation
# ============================================================

df = uncertainty.merge(
    spatial_df,
    on=[
        "volume",
        "n_cpt",
    ],
    how="inner",
    validate="one_to_one",
)

assert len(df) == 300, (
    f"Expected 300 merged rows, "
    f"got {len(df)}"
)

df[
    "actual_sufficient"
] = (
    df["actual_coverage"]
    >= ACTUAL_SUFFICIENCY
).astype(int)


enriched_path = (
    OUT_DIR
    / "combined_gate_enriched.csv"
)

df.to_csv(
    enriched_path,
    index=False,
)


print()
print(
    "=== DATA SUMMARY ==="
)

print(
    "Rows:",
    len(df),
)

print(
    "Actual sufficient:",
    int(
        df[
            "actual_sufficient"
        ].sum()
    ),
)

print(
    "Max-distance fraction range:",
    f"{df['max_distance_fraction'].min():.3f}"
    " .. "
    f"{df['max_distance_fraction'].max():.3f}",
)


# ============================================================
# Baseline:
# Uncertainty only at 0.85
# ============================================================

actual = (
    df["actual_sufficient"]
    .to_numpy(dtype=bool)
)

pred_cov = (
    df["predicted_coverage"]
    .to_numpy(dtype=float)
)

max_dist_frac = (
    df["max_distance_fraction"]
    .to_numpy(dtype=float)
)


baseline_stop = (
    pred_cov >= 0.85
)

baseline_tp = int(
    np.sum(
        baseline_stop
        & actual
    )
)

baseline_fp = int(
    np.sum(
        baseline_stop
        & (~actual)
    )
)

baseline_stops = int(
    baseline_stop.sum()
)

baseline_false_stop = (
    baseline_fp
    / baseline_stops
    if baseline_stops > 0
    else 0.0
)


print()
print(
    "=== UNCERTAINTY-ONLY BASELINE ==="
)

print(
    "Predicted coverage >= 0.85"
)

print(
    "STOP:",
    baseline_stops,
)

print(
    "TP:",
    baseline_tp,
)

print(
    "FP:",
    baseline_fp,
)

print(
    "False-stop:",
    f"{baseline_false_stop*100:.1f}%",
)


# ============================================================
# Combined threshold scan
#
# STOP if:
#
# predicted coverage >= U
#
# AND
#
# max nearest distance / site diagonal <= D
# ============================================================

print()
print(
    "=== COMBINED THRESHOLD SCAN ==="
)


uncertainty_thresholds = (
    np.arange(
        0.30,
        0.951,
        0.01,
    )
)

spatial_thresholds = (
    np.arange(
        0.05,
        1.001,
        0.01,
    )
)


scan_rows = []

actual_positive = int(
    actual.sum()
)


for u_threshold in (
    uncertainty_thresholds
):

    for d_threshold in (
        spatial_thresholds
    ):

        stop = (
            (pred_cov >= u_threshold)
            &
            (
                max_dist_frac
                <= d_threshold
            )
        )

        tp = int(
            np.sum(
                stop
                & actual
            )
        )

        fp = int(
            np.sum(
                stop
                & (~actual)
            )
        )

        fn = int(
            np.sum(
                (~stop)
                & actual
            )
        )

        tn = int(
            np.sum(
                (~stop)
                & (~actual)
            )
        )

        stops = int(
            stop.sum()
        )

        false_stop = (
            fp / stops
            if stops > 0
            else 0.0
        )

        recall = (
            tp
            / actual_positive
            if actual_positive > 0
            else 0.0
        )

        precision = (
            tp / stops
            if stops > 0
            else 0.0
        )

        scan_rows.append({
            "uncertainty_threshold":
                float(u_threshold),

            "max_distance_fraction_threshold":
                float(d_threshold),

            "stops":
                stops,

            "tp":
                tp,

            "fp":
                fp,

            "fn":
                fn,

            "tn":
                tn,

            "false_stop_fraction":
                false_stop,

            "precision":
                precision,

            "recall":
                recall,
        })


scan = pd.DataFrame(
    scan_rows
)


scan_path = (
    OUT_DIR
    / "combined_gate_scan.csv"
)

scan.to_csv(
    scan_path,
    index=False,
)


# ============================================================
# Conservative candidate selection
# ============================================================

eligible = scan[
    (
        scan[
            "false_stop_fraction"
        ]
        <= MAX_FALSE_STOP
    )
    &
    (
        scan["stops"]
        >= MIN_STOPS
    )
].copy()


print()
print(
    "=== PILOT COMBINED GATE ==="
)


if len(eligible) == 0:

    print(
        "NO COMBINED THRESHOLD "
        "SATISFIES:"
    )

    print(
        f"false-stop <= "
        f"{MAX_FALSE_STOP*100:.1f}%"
    )

    print(
        f"and STOP >= "
        f"{MIN_STOPS}"
    )

else:

    # Priority:
    # 1. highest recall
    # 2. more useful STOP decisions
    # 3. lower false-stop
    best = (
        eligible
        .sort_values(
            [
                "recall",
                "stops",
                "false_stop_fraction",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .iloc[0]
    )

    u = float(
        best[
            "uncertainty_threshold"
        ]
    )

    d = float(
        best[
            "max_distance_fraction_threshold"
        ]
    )

    print(
        "STOP if:"
    )

    print(
        f"predicted_coverage >= {u:.2f}"
    )

    print(
        "AND"
    )

    print(
        "max_nearest_distance / "
        "site_diagonal "
        f"<= {d:.2f}"
    )

    print()

    print(
        "STOP cases:",
        int(best["stops"]),
    )

    print(
        "TP:",
        int(best["tp"]),
    )

    print(
        "FP:",
        int(best["fp"]),
    )

    print(
        "False-stop:",
        f"{best['false_stop_fraction']*100:.1f}%",
    )

    print(
        "Recall:",
        f"{best['recall']*100:.1f}%",
    )

    print(
        "Precision:",
        f"{best['precision']*100:.1f}%",
    )


    # --------------------------------------------
    # Evaluate candidate by CPT count
    # --------------------------------------------

    print()
    print(
        "=== CANDIDATE BY CPT COUNT ==="
    )

    candidate_stop = (
        (
            df[
                "predicted_coverage"
            ]
            >= u
        )
        &
        (
            df[
                "max_distance_fraction"
            ]
            <= d
        )
    )

    df_candidate = df.copy()

    df_candidate[
        "combined_stop"
    ] = (
        candidate_stop
        .astype(int)
    )

    for count in CPT_COUNTS:

        sub = df_candidate[
            df_candidate["n_cpt"]
            == count
        ]

        stop = (
            sub["combined_stop"]
            .to_numpy(dtype=bool)
        )

        actual_sub = (
            sub["actual_sufficient"]
            .to_numpy(dtype=bool)
        )

        tp_c = int(
            np.sum(
                stop
                & actual_sub
            )
        )

        fp_c = int(
            np.sum(
                stop
                & (~actual_sub)
            )
        )

        stops_c = int(
            stop.sum()
        )

        fs_c = (
            fp_c / stops_c
            if stops_c > 0
            else 0.0
        )

        print()
        print(
            f"CPT={count}"
        )

        print(
            "  STOP:",
            stops_c,
        )

        print(
            "  TP:",
            tp_c,
        )

        print(
            "  FP:",
            fp_c,
        )

        print(
            "  False-stop:",
            f"{fs_c*100:.1f}%",
        )


    # --------------------------------------------
    # Evaluate candidate by grid
    # --------------------------------------------

    print()
    print(
        "=== CANDIDATE BY GRID ==="
    )

    for grid_name, sub in (
        df_candidate.groupby("grid")
    ):

        stop = (
            sub["combined_stop"]
            .to_numpy(dtype=bool)
        )

        actual_sub = (
            sub["actual_sufficient"]
            .to_numpy(dtype=bool)
        )

        fp_g = int(
            np.sum(
                stop
                & (~actual_sub)
            )
        )

        tp_g = int(
            np.sum(
                stop
                & actual_sub
            )
        )

        stops_g = int(
            stop.sum()
        )

        fs_g = (
            fp_g / stops_g
            if stops_g > 0
            else 0.0
        )

        print(
            grid_name
        )

        print(
            f"  STOP={stops_g}"
            f" TP={tp_g}"
            f" FP={fp_g}"
            f" false-stop="
            f"{fs_g*100:.1f}%"
        )


    candidate_path = (
        OUT_DIR
        / "pilot_candidate_predictions.csv"
    )

    df_candidate.to_csv(
        candidate_path,
        index=False,
    )

    print()
    print(
        "Candidate saved:",
        candidate_path,
    )


# ============================================================
# Show best safe candidates
# ============================================================

if len(eligible) > 0:

    print()
    print(
        "=== TOP SAFE CANDIDATES ==="
    )

    top = (
        eligible
        .sort_values(
            [
                "recall",
                "stops",
                "false_stop_fraction",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .head(10)
    )

    for _, row in top.iterrows():

        print(
            f"U>={row['uncertainty_threshold']:.2f}"
            f" | D<={row['max_distance_fraction_threshold']:.2f}"
            f" | STOP={int(row['stops'])}"
            f" | FP={int(row['fp'])}"
            f" | false-stop="
            f"{row['false_stop_fraction']*100:.1f}%"
            f" | recall="
            f"{row['recall']*100:.1f}%"
        )


print()
print("Saved:")
print(enriched_path)
print(scan_path)

print()
print(
    "IMPORTANT:"
)

print(
    "This is CALIBRATION only."
)

print(
    "Do not use the selected threshold "
    "as a final app rule until it is "
    "validated on a separate unseen set."
)
