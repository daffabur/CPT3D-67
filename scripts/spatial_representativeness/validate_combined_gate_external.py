from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path.home() / "Developer" / "cpt_super_learn"

DATA = ROOT / "data/synthetic3d_v4_geosyn/combined_gate_validation_100/test"

UNC = (
    ROOT
    / "results/spatial_representativeness/"
    "uncertainty_sufficiency_gate_external/"
    "per_volume_uncertainty_gate.csv"
)

ACTIONS = (
    ROOT
    / "results/spatial_representativeness/"
    "combined_gate_external_v46a/actions.csv"
)

U_THRESHOLD = 0.85
D_THRESHOLD = 0.48
ACTUAL_THRESHOLD = 0.90
CPT_COUNTS = [1, 3, 8]


def meta(raw):
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode()
    return json.loads(str(raw))


def center(valid):
    nx, ny, _ = valid.shape
    c = np.argwhere(valid.all(axis=2))
    target = np.array([
        (nx - 1) / 2,
        (ny - 1) / 2,
    ])
    d = np.linalg.norm(
        c.astype(float) - target[None, :],
        axis=1,
    )
    x, y = c[np.argmin(d)]
    return int(x), int(y)


def max_dist_fraction(valid, chosen, dx, dy):
    xy = np.argwhere(valid.any(axis=2))
    chosen = np.asarray(chosen, dtype=float)

    ddx = (
        xy[:, None, 0]
        - chosen[None, :, 0]
    ) * dx

    ddy = (
        xy[:, None, 1]
        - chosen[None, :, 1]
    ) * dy

    nearest = np.sqrt(
        ddx**2 + ddy**2
    ).min(axis=1)

    nx, ny, _ = valid.shape

    diagonal = np.sqrt(
        ((nx - 1) * dx) ** 2
        + ((ny - 1) * dy) ** 2
    )

    return float(
        nearest.max()
        / max(diagonal, 1e-6)
    )


unc = pd.read_csv(UNC)

actions = pd.read_csv(ACTIONS)
actions = actions[
    actions["policy"] == "dqn"
].copy()

for c in ["n_cpt", "x", "y"]:
    actions[c] = actions[c].astype(int)


spatial = []

for path in sorted(DATA.glob("*.npz")):

    with np.load(
        path,
        allow_pickle=False,
    ) as d:
        valid = np.asarray(
            d["valid_mask"],
            dtype=bool,
        )
        m = meta(d["metadata_json"])

    g = m["grid"]
    dx = float(g["dx_m"])
    dy = float(g["dy_m"])

    first = center(valid)

    va = (
        actions[
            actions["volume"] == path.name
        ]
        .sort_values("n_cpt")
    )

    assert len(va) == 31

    for n in CPT_COUNTS:

        chosen = [first]

        if n > 1:
            for _, r in va[
                va["n_cpt"] <= n
            ].iterrows():
                chosen.append(
                    (int(r["x"]), int(r["y"]))
                )

        assert len(chosen) == n

        spatial.append({
            "volume": path.name,
            "n_cpt": n,
            "max_distance_fraction":
                max_dist_fraction(
                    valid,
                    chosen,
                    dx,
                    dy,
                ),
        })


df = unc.merge(
    pd.DataFrame(spatial),
    on=["volume", "n_cpt"],
    validate="one_to_one",
)

assert len(df) == 300

actual = (
    df["actual_coverage"]
    >= ACTUAL_THRESHOLD
)

stop = (
    (df["predicted_coverage"] >= U_THRESHOLD)
    &
    (
        df["max_distance_fraction"]
        <= D_THRESHOLD
    )
)

tp = int((stop & actual).sum())
fp = int((stop & ~actual).sum())
fn = int((~stop & actual).sum())
tn = int((~stop & ~actual).sum())

n_stop = int(stop.sum())

false_stop = (
    fp / n_stop
    if n_stop
    else 0.0
)

recall = (
    tp / int(actual.sum())
    if actual.sum()
    else 0.0
)

print("=== COMBINED GATE EXTERNAL VALIDATION ===")
print()
print("FROZEN RULE:")
print("predicted_coverage >= 0.85")
print("AND max_distance/site_diagonal <= 0.48")
print()

print("Rows:", len(df))
print("Actual sufficient:", int(actual.sum()))
print("STOP decisions:", n_stop)
print("TP:", tp)
print("FP / FALSE STOP:", fp)
print("FN:", fn)
print("TN:", tn)

print()
print(
    "False-stop fraction:",
    f"{false_stop*100:.1f}%"
)
print(
    "Recall:",
    f"{recall*100:.1f}%"
)

print()
print("=== BY CPT COUNT ===")

for n in CPT_COUNTS:
    mask = df["n_cpt"] == n

    s = stop[mask]
    a = actual[mask]

    sn = int(s.sum())
    tpn = int((s & a).sum())
    fpn = int((s & ~a).sum())

    fs = (
        fpn / sn
        if sn
        else 0.0
    )

    print(
        f"CPT={n}: "
        f"STOP={sn} "
        f"TP={tpn} "
        f"FP={fpn} "
        f"false-stop={fs*100:.1f}%"
    )

print()
print("=== PILOT ACCEPTANCE ===")

enough = n_stop >= 10
safe = false_stop <= 0.05

print(
    "STOP cases >= 10:",
    "PASS" if enough else "FAIL"
)

print(
    "False-stop <= 5%:",
    "PASS" if safe else "FAIL"
)

print()

if enough and safe:
    print(
        "RESULT: COMBINED GATE PASSED "
        "EXTERNAL PILOT VALIDATION"
    )
else:
    print(
        "RESULT: COMBINED GATE DID NOT PASS "
        "EXTERNAL PILOT VALIDATION"
    )
