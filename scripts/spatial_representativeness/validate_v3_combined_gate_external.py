from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path.home() / "Developer" / "cpt_super_learn"

DATA_DIR = (
    ROOT
    / "data"
    / "synthetic3d_v4_geosyn"
    / "combined_gate_validation_100"
    / "test"
)

UNCERTAINTY_CSV = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "uncertainty_v3_gate_external"
    / "per_volume_uncertainty_gate.csv"
)

ACTIONS_CSV = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "combined_gate_external_v46a"
    / "actions.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "v3_combined_gate_external"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FROZEN FROM CALIBRATION.
# DO NOT CHANGE USING THIS EXTERNAL SET.
# ============================================================

U_THRESHOLD = 0.72
D_THRESHOLD = 0.52

ACTUAL_THRESHOLD = 0.90

MAX_FALSE_STOP = 0.05
MIN_STOPS = 10


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
# Load results
# ============================================================

unc = pd.read_csv(
    UNCERTAINTY_CSV
)

assert len(unc) == 3200, (
    f"Expected 3200 uncertainty rows, "
    f"got {len(unc)}"
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


# ============================================================
# Build external spatial support.
# No SchemaGAN inference.
# ============================================================

print(
    "=== BUILD EXTERNAL SPATIAL SUPPORT ==="
)

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


# ============================================================
# Sequential evaluation.
#
# For each site:
# first gate hit = app STOP point.
#
# Later CPT counts are ignored.
# ============================================================

decision_rows = []


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
        first_actual_n = int(
            actual_candidates.iloc[0][
                "n_cpt"
            ]
        )
    else:
        first_actual_n = None

    gate = (
        (
            sub["predicted_coverage"]
            >= U_THRESHOLD
        )
        &
        (
            sub["max_distance_fraction"]
            <= D_THRESHOLD
        )
    )

    candidates = sub[
        gate
    ]

    if len(candidates) == 0:

        decision_rows.append({
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
                first_actual_n,
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
        and first_actual_n is not None
    ):
        delay = (
            stop_n
            - first_actual_n
        )


    decision_rows.append({
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
            first_actual_n,
        "delay":
            delay,
    })


decisions = pd.DataFrame(
    decision_rows
)

assert len(decisions) == 100


# ============================================================
# Metrics
# ============================================================

n_stop = int(
    decisions["stopped"].sum()
)

tp = int(
    (
        decisions["stopped"]
        & decisions["actual_at_stop"]
    ).sum()
)

fp = int(
    decisions["false_stop"].sum()
)

false_stop = (
    fp / n_stop
    if n_stop > 0
    else 0.0
)

reachable = int(
    decisions[
        "first_actual_n"
    ].notna().sum()
)

safe_recall = (
    tp / reachable
    if reachable > 0
    else 0.0
)

median_stop = (
    float(
        decisions.loc[
            decisions["stopped"],
            "stop_n",
        ].median()
    )
    if n_stop
    else np.nan
)

safe_delays = (
    decisions.loc[
        decisions["actual_at_stop"],
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


print()
print(
    "=== V3 COMBINED GATE EXTERNAL VALIDATION ==="
)

print()
print(
    "FROZEN RULE:"
)

print(
    "predicted_coverage >= 0.72"
)

print(
    "AND max_distance/site_diagonal <= 0.52"
)

print()
print(
    "Volumes:",
    len(decisions),
)

print(
    "Volumes that become actually sufficient:",
    reachable,
)

print(
    "STOP volumes:",
    n_stop,
)

print(
    "Safe STOP / TP:",
    tp,
)

print(
    "False STOP / FP:",
    fp,
)

print(
    "False-stop fraction:",
    f"{false_stop*100:.1f}%",
)

print(
    "Safe-stop recall:",
    f"{safe_recall*100:.1f}%",
)

print(
    "Median stop CPT:",
    f"{median_stop:.1f}",
)

print(
    "Median delay:",
    f"{median_delay:.1f} CPT",
)


# ============================================================
# First STOP distribution
# ============================================================

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


# ============================================================
# False stops detail
# ============================================================

print()
print(
    "=== FALSE STOP DETAILS ==="
)

false_rows = decisions[
    decisions["false_stop"]
]

if len(false_rows) == 0:

    print(
        "NONE"
    )

else:

    for _, r in false_rows.iterrows():

        print(
            r["volume"],
            "| stop CPT=",
            int(r["stop_n"]),
            "| first actual sufficient=",
            r["first_actual_n"],
        )


# ============================================================
# Acceptance
# ============================================================

enough = (
    n_stop >= MIN_STOPS
)

safe = (
    false_stop
    <= MAX_FALSE_STOP
)


print()
print(
    "=== EXTERNAL PILOT ACCEPTANCE ==="
)

print(
    f"STOP volumes >= {MIN_STOPS}:",
    "PASS"
    if enough
    else "FAIL",
)

print(
    "False-stop <= 5%:",
    "PASS"
    if safe
    else "FAIL",
)

print()

if enough and safe:

    print(
        "RESULT: V3 SEQUENTIAL COMBINED GATE "
        "PASSED EXTERNAL PILOT VALIDATION"
    )

else:

    print(
        "RESULT: V3 SEQUENTIAL COMBINED GATE "
        "DID NOT PASS EXTERNAL PILOT VALIDATION"
    )


# ============================================================
# Save
# ============================================================

df.to_csv(
    OUT_DIR
    / "external_enriched_1_32.csv",
    index=False,
)

decisions.to_csv(
    OUT_DIR
    / "external_sequential_decisions.csv",
    index=False,
)

print()
print("Saved:")
print(
    OUT_DIR
    / "external_enriched_1_32.csv"
)
print(
    OUT_DIR
    / "external_sequential_decisions.csv"
)
