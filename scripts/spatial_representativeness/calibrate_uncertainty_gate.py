from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "Developer" / "cpt_super_learn"

CSV = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "uncertainty_sufficiency_gate"
    / "per_volume_uncertainty_gate.csv"
)

df = pd.read_csv(CSV)

ACTUAL_THRESHOLD = 0.90

actual = (
    df["actual_coverage"].to_numpy()
    >= ACTUAL_THRESHOLD
)

pred_cov = (
    df["predicted_coverage"].to_numpy()
)

print("=== UNCERTAINTY GATE THRESHOLD SCAN ===")
print()
print(
    f"{'Threshold':>10} "
    f"{'Stops':>7} "
    f"{'TP':>5} "
    f"{'FP':>5} "
    f"{'FalseStop':>10} "
    f"{'Recall':>9}"
)

rows = []

for threshold in np.arange(
    0.30,
    0.951,
    0.01,
):
    predicted_stop = (
        pred_cov >= threshold
    )

    tp = int(
        np.sum(
            predicted_stop & actual
        )
    )

    fp = int(
        np.sum(
            predicted_stop & ~actual
        )
    )

    fn = int(
        np.sum(
            ~predicted_stop & actual
        )
    )

    stops = int(
        predicted_stop.sum()
    )

    false_stop = (
        fp / stops
        if stops > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    rows.append({
        "threshold": threshold,
        "stops": stops,
        "tp": tp,
        "fp": fp,
        "false_stop_fraction":
            false_stop,
        "recall":
            recall,
    })

    print(
        f"{threshold:10.2f} "
        f"{stops:7d} "
        f"{tp:5d} "
        f"{fp:5d} "
        f"{false_stop*100:9.1f}% "
        f"{recall*100:8.1f}%"
    )


result = pd.DataFrame(rows)

# Conservative pilot rule:
# false-stop <= 5%
# at least 20 STOP cases
eligible = result[
    (result["false_stop_fraction"] <= 0.05)
    & (result["stops"] >= 20)
]

print()
print("=== PILOT CANDIDATE ===")

if len(eligible) == 0:
    print(
        "No threshold satisfies "
        "false-stop <=5%."
    )
else:
    # Among safe thresholds,
    # choose highest recall.
    best = (
        eligible
        .sort_values(
            [
                "recall",
                "stops",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    print(
        f"Predicted coverage >= "
        f"{best['threshold']:.2f}"
    )

    print(
        f"STOP cases : "
        f"{int(best['stops'])}"
    )

    print(
        f"False stop : "
        f"{best['false_stop_fraction']*100:.1f}%"
    )

    print(
        f"Recall     : "
        f"{best['recall']*100:.1f}%"
    )

out = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "uncertainty_sufficiency_gate"
    / "threshold_scan.csv"
)

result.to_csv(
    out,
    index=False,
)

print()
print("Saved:", out)
