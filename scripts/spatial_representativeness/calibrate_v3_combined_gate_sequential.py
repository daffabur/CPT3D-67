from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path.home() / "Developer" / "cpt_super_learn"

DATA_DIR = (
    ROOT
    / "data"
    / "synthetic3d_v4_geosyn"
    / "v3_gate_calibration_100"
    / "test"
)

UNCERTAINTY_CSV = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "uncertainty_v3_gate_calibration"
    / "per_volume_uncertainty_gate.csv"
)

ACTIONS_CSV = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "v3_gate_calibration_v46a"
    / "actions.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "v3_combined_gate_calibration"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ACTUAL_THRESHOLD = 0.90

# Sama seperti acceptance calibration sebelumnya.
MAX_FALSE_STOP = 0.05
MIN_STOPS = 30


def parse_meta(raw):
    if isinstance(raw, np.ndarray):
        raw = raw.item()

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    return json.loads(str(raw))


def initial_center(valid):
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

    dist = np.linalg.norm(
        candidates.astype(float)
        - center[None, :],
        axis=1,
    )

    x, y = candidates[
        np.argmin(dist)
    ]

    return int(x), int(y)


def spatial_fraction(
    valid,
    chosen,
    dx,
    dy,
):
    coords = np.argwhere(
        valid.any(axis=2)
    )

    chosen = np.asarray(
        chosen,
        dtype=float,
    )

    ddx = (
        coords[:, None, 0]
        - chosen[None, :, 0]
    ) * dx

    ddy = (
        coords[:, None, 1]
        - chosen[None, :, 1]
    ) * dy

    nearest = np.sqrt(
        ddx ** 2
        + ddy ** 2
    ).min(axis=1)

    nx, ny, _ = valid.shape

    width = (nx - 1) * dx
    height = (ny - 1) * dy

    diagonal = np.sqrt(
        width ** 2
        + height ** 2
    )

    return float(
        nearest.max()
        / max(diagonal, 1e-6)
    )


# ============================================================
# Load existing results
# ============================================================

unc = pd.read_csv(
    UNCERTAINTY_CSV
)

actions = pd.read_csv(
    ACTIONS_CSV
)

actions = actions[
    actions["policy"] == "dqn"
].copy()

for c in [
    "n_cpt",
    "x",
    "y",
]:
    actions[c] = (
        actions[c]
        .astype(int)
    )


print(
    "=== BUILD SPATIAL SUPPORT 1-32 ==="
)

print(
    "Uncertainty rows:",
    len(unc),
)

assert len(unc) == 3200, (
    f"Expected 3200 rows, got {len(unc)}"
)


# ============================================================
# Reconstruct exact DQN trajectory
# ============================================================

spatial_rows = []

paths = sorted(
    DATA_DIR.glob("*.npz")
)

assert len(paths) == 100


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

        meta = parse_meta(
            data["metadata_json"]
        )

    grid = meta["grid"]

    dx = float(
        grid["dx_m"]
    )

    dy = float(
        grid["dy_m"]
    )

    first = initial_center(
        valid
    )

    va = (
        actions[
            actions["volume"]
            == path.name
        ]
        .sort_values("n_cpt")
    )

    assert len(va) == 31, (
        f"{path.name}: expected 31 actions, "
        f"got {len(va)}"
    )

    chosen = [first]

    # CPT 1
    spatial_rows.append({
        "volume": path.name,
        "n_cpt": 1,
        "max_distance_fraction":
            spatial_fraction(
                valid,
                chosen,
                dx,
                dy,
            ),
    })

    # CPT 2..32
    for n in range(2, 33):

        row = va[
            va["n_cpt"] == n
        ]

        assert len(row) == 1

        r = row.iloc[0]

        chosen.append(
            (
                int(r["x"]),
                int(r["y"]),
            )
        )

        assert len(chosen) == n

        spatial_rows.append({
            "volume": path.name,
            "n_cpt": n,
            "max_distance_fraction":
                spatial_fraction(
                    valid,
                    chosen,
                    dx,
                    dy,
                ),
        })

    if vi % 20 == 0:
        print(
            f"{vi}/100 volumes"
        )


spatial = pd.DataFrame(
    spatial_rows
)

assert len(spatial) == 3200


# ============================================================
# Merge
# ============================================================

df = unc.merge(
    spatial,
    on=[
        "volume",
        "n_cpt",
    ],
    validate="one_to_one",
)

assert len(df) == 3200

df = df.sort_values(
    [
        "volume",
        "n_cpt",
    ]
).reset_index(drop=True)

df["actual_sufficient"] = (
    df["actual_coverage"]
    >= ACTUAL_THRESHOLD
)


enriched_path = (
    OUT_DIR
    / "calibration_enriched_1_32.csv"
)

df.to_csv(
    enriched_path,
    index=False,
)


print()
print(
    "Spatial fraction range:",
    f"{df['max_distance_fraction'].min():.3f}"
    " .. "
    f"{df['max_distance_fraction'].max():.3f}",
)


# ============================================================
# Sequential evaluation
#
# For each volume:
# find FIRST CPT where gate is satisfied.
#
# Once STOP happens, later CPT counts do not exist
# from the application's perspective.
# ============================================================

def evaluate_gate(
    u_threshold,
    d_threshold,
):
    rows = []

    for volume, sub in df.groupby(
        "volume",
        sort=False,
    ):

        sub = sub.sort_values(
            "n_cpt"
        )

        actual_candidates = sub[
            sub["actual_sufficient"]
        ]

        if len(actual_candidates):
            first_actual = int(
                actual_candidates.iloc[0][
                    "n_cpt"
                ]
            )
        else:
            first_actual = None

        gate = (
            (
                sub["predicted_coverage"]
                >= u_threshold
            )
            &
            (
                sub[
                    "max_distance_fraction"
                ]
                <= d_threshold
            )
        )

        candidates = sub[
            gate
        ]

        if len(candidates) == 0:
            rows.append({
                "volume":
                    volume,
                "stopped":
                    False,
                "stop_n":
                    np.nan,
                "actual_at_stop":
                    False,
                "false_stop":
                    False,
                "first_actual_n":
                    first_actual,
                "delay":
                    np.nan,
            })
            continue

        first_stop = candidates.iloc[0]

        stop_n = int(
            first_stop["n_cpt"]
        )

        actual_at_stop = bool(
            first_stop[
                "actual_sufficient"
            ]
        )

        delay = np.nan

        if (
            actual_at_stop
            and first_actual is not None
        ):
            delay = (
                stop_n
                - first_actual
            )

        rows.append({
            "volume":
                volume,
            "stopped":
                True,
            "stop_n":
                stop_n,
            "actual_at_stop":
                actual_at_stop,
            "false_stop":
                not actual_at_stop,
            "first_actual_n":
                first_actual,
            "delay":
                delay,
        })

    result = pd.DataFrame(
        rows
    )

    stops = int(
        result["stopped"].sum()
    )

    fp = int(
        result["false_stop"].sum()
    )

    tp = int(
        (
            result["stopped"]
            & result["actual_at_stop"]
        ).sum()
    )

    false_stop_rate = (
        fp / stops
        if stops > 0
        else 0.0
    )

    reachable = int(
        result[
            "first_actual_n"
        ].notna().sum()
    )

    safe_stop_recall = (
        tp / reachable
        if reachable > 0
        else 0.0
    )

    median_stop_n = (
        float(
            result.loc[
                result["stopped"],
                "stop_n",
            ].median()
        )
        if stops
        else np.nan
    )

    safe_delays = (
        result.loc[
            result[
                "actual_at_stop"
            ],
            "delay",
        ]
        .dropna()
    )

    median_delay = (
        float(
            safe_delays.median()
        )
        if len(safe_delays)
        else np.nan
    )

    return {
        "u_threshold":
            float(u_threshold),
        "d_threshold":
            float(d_threshold),
        "stops":
            stops,
        "tp":
            tp,
        "fp":
            fp,
        "false_stop_fraction":
            false_stop_rate,
        "reachable_sufficient_volumes":
            reachable,
        "safe_stop_recall":
            safe_stop_recall,
        "median_stop_n":
            median_stop_n,
        "median_delay":
            median_delay,
    }, result


# ============================================================
# Scan calibration thresholds
# ============================================================

print()
print(
    "=== SEQUENTIAL COMBINED GATE SCAN ==="
)

scan_rows = []

for u in np.arange(
    0.30,
    0.951,
    0.01,
):

    for d in np.arange(
        0.20,
        0.601,
        0.01,
    ):

        metrics, _ = evaluate_gate(
            u,
            d,
        )

        scan_rows.append(
            metrics
        )


scan = pd.DataFrame(
    scan_rows
)

scan_path = (
    OUT_DIR
    / "sequential_threshold_scan.csv"
)

scan.to_csv(
    scan_path,
    index=False,
)


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
    "=== PILOT SEQUENTIAL CANDIDATE ==="
)


if len(eligible) == 0:

    print(
        "NO threshold satisfies:"
    )

    print(
        "false-stop <= 5%"
    )

    print(
        f"STOP volumes >= {MIN_STOPS}"
    )

else:

    # Safety first.
    # Then maximize useful safe stopping.
    best = (
        eligible
        .sort_values(
            [
                "safe_stop_recall",
                "stops",
                "false_stop_fraction",
                "median_delay",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        )
        .iloc[0]
    )

    u = float(
        best["u_threshold"]
    )

    d = float(
        best["d_threshold"]
    )

    print(
        "FROZEN CANDIDATE:"
    )

    print(
        f"predicted_coverage >= {u:.2f}"
    )

    print(
        "AND"
    )

    print(
        "max_distance/site_diagonal "
        f"<= {d:.2f}"
    )

    print()

    print(
        "STOP volumes:",
        int(best["stops"]),
    )

    print(
        "Safe STOP / TP:",
        int(best["tp"]),
    )

    print(
        "False STOP / FP:",
        int(best["fp"]),
    )

    print(
        "False-stop fraction:",
        f"{best['false_stop_fraction']*100:.1f}%",
    )

    print(
        "Volumes that become actually sufficient:",
        int(
            best[
                "reachable_sufficient_volumes"
            ]
        ),
    )

    print(
        "Safe-stop recall:",
        f"{best['safe_stop_recall']*100:.1f}%",
    )

    print(
        "Median stop CPT:",
        f"{best['median_stop_n']:.1f}",
    )

    print(
        "Median delay after first actual sufficiency:",
        f"{best['median_delay']:.1f} CPT",
    )


    metrics, decisions = (
        evaluate_gate(
            u,
            d,
        )
    )

    decision_path = (
        OUT_DIR
        / "pilot_sequential_decisions.csv"
    )

    decisions.to_csv(
        decision_path,
        index=False,
    )


    print()
    print(
        "=== FIRST STOP DISTRIBUTION ==="
    )

    dist = (
        decisions.loc[
            decisions["stopped"],
            "stop_n",
        ]
        .value_counts()
        .sort_index()
    )

    for n, count in dist.items():

        print(
            f"CPT={int(n):2d}: "
            f"{int(count)} volumes"
        )


    print()
    print(
        "=== TOP SAFE CANDIDATES ==="
    )

    top = (
        eligible
        .sort_values(
            [
                "safe_stop_recall",
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

    for _, r in top.iterrows():

        print(
            f"U>={r['u_threshold']:.2f}"
            f" | D<={r['d_threshold']:.2f}"
            f" | STOP={int(r['stops'])}"
            f" | FP={int(r['fp'])}"
            f" | FS="
            f"{r['false_stop_fraction']*100:.1f}%"
            f" | safe-recall="
            f"{r['safe_stop_recall']*100:.1f}%"
        )


print()
print("Saved:")
print(enriched_path)
print(scan_path)

print()
print(
    "IMPORTANT: calibration only."
)

print(
    "Do not change the external validation set."
)
