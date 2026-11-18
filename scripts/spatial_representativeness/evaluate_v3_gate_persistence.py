from pathlib import Path
import pandas as pd

ROOT = Path.home() / "Developer" / "cpt_super_learn"

SETS = {
    "calibration": (
        ROOT
        / "results/spatial_representativeness/"
        "v3_combined_gate_calibration/"
        "calibration_enriched_1_32.csv"
    ),
    "development_external": (
        ROOT
        / "results/spatial_representativeness/"
        "v3_combined_gate_external/"
        "external_enriched_1_32.csv"
    ),
}

U_THRESHOLD = 0.72
D_THRESHOLD = 0.52
ACTUAL_THRESHOLD = 0.90


def evaluate(df, persistence):
    decisions = []

    for volume, sub in df.groupby("volume"):
        sub = sub.sort_values("n_cpt").copy()

        gate = (
            (sub["predicted_coverage"] >= U_THRESHOLD)
            &
            (sub["max_distance_fraction"] <= D_THRESHOLD)
        ).to_numpy()

        actual = (
            sub["actual_coverage"] >= ACTUAL_THRESHOLD
        ).to_numpy()

        counts = sub["n_cpt"].to_numpy()

        stop_idx = None

        for i in range(
            persistence - 1,
            len(sub),
        ):
            window = gate[
                i - persistence + 1:
                i + 1
            ]

            if window.all():
                stop_idx = i
                break

        if stop_idx is None:
            decisions.append({
                "volume": volume,
                "stopped": False,
                "stop_n": None,
                "false_stop": False,
            })
            continue

        decisions.append({
            "volume": volume,
            "stopped": True,
            "stop_n": int(counts[stop_idx]),
            "false_stop": not bool(actual[stop_idx]),
        })

    out = pd.DataFrame(decisions)

    stops = int(out["stopped"].sum())
    fp = int(out["false_stop"].sum())

    false_stop_rate = (
        fp / stops
        if stops else 0.0
    )

    return stops, fp, false_stop_rate


print("=== V3 GATE PERSISTENCE DEVELOPMENT TEST ===")
print()
print("Base rule:")
print("U >= 0.72")
print("D <= 0.52")

for name, path in SETS.items():

    df = pd.read_csv(path)

    print()
    print("===", name.upper(), "===")

    for persistence in [1, 2, 3]:

        stops, fp, fs = evaluate(
            df,
            persistence,
        )

        print(
            f"Persistence={persistence}"
            f" | STOP={stops}"
            f" | FP={fp}"
            f" | false-stop={fs*100:.1f}%"
        )
