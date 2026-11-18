from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("results/coverage_calibrator_v1")
TEST = ROOT / "final_test/per_state_gate.csv"
MODEL = ROOT / "coverage_calibrator_v1.joblib"
META = ROOT / "metadata.json"

THRESHOLD = 0.975
ACTUAL_TARGET = 0.90

df = pd.read_csv(TEST)
meta = json.loads(META.read_text())

required = {
    "volume_index", "n_cpt",
    "actual_coverage", "predicted_coverage"
}

assert required.issubset(df.columns)
assert len(df) == 3200
assert df.volume_index.nunique() == 200
assert set(df.volume_index) == set(range(1100, 1300))
assert not df.duplicated(["volume_index", "n_cpt"]).any()
assert df[list(required)].notna().all().all()
assert df.n_cpt.between(1, 16).all()
assert df.groupby("volume_index").n_cpt.nunique().eq(16).all()

assert meta["train_volumes"] == [800, 999]
assert meta["validation_volumes"] == [1000, 1099]

df = df.sort_values(
    ["volume_index", "n_cpt"]
).reset_index(drop=True)

def features(data):
    p = data.predicted_coverage.to_numpy()
    n = data.n_cpt.to_numpy() / 16.0
    return np.column_stack([p, n, p*n, n*n])

# Load the previously trained frozen calibrator.
model = joblib.load(MODEL)

df["calibrated_coverage"] = np.clip(
    model.predict(features(df)), 0, 1
)

actual = df.actual_coverage.to_numpy()
actual_stop = actual >= ACTUAL_TARGET

def evaluate(label, column, threshold):
    predicted = df[column].to_numpy()
    stop = predicted >= threshold

    tp = int(np.sum(stop & actual_stop))
    fp = int(np.sum(stop & ~actual_stop))
    fn = int(np.sum(~stop & actual_stop))
    tn = int(np.sum(~stop & ~actual_stop))

    frontier = []

    for volume, group in df.groupby("volume_index"):
        first_pred = group[
            group[column] >= threshold
        ].head(1)

        first_actual = group[
            group.actual_coverage >= ACTUAL_TARGET
        ].head(1)

        pred_n = (
            int(first_pred.n_cpt.iloc[0])
            if len(first_pred) else None
        )
        actual_n = (
            int(first_actual.n_cpt.iloc[0])
            if len(first_actual) else None
        )

        unsafe = (
            pred_n is not None
            and float(first_pred.actual_coverage.iloc[0])
            < ACTUAL_TARGET
        )

        frontier.append({
            "volume_index": volume,
            "first_actual_stop": actual_n,
            "first_predicted_stop": pred_n,
            "unsafe_first_stop": int(unsafe),
            "stop_delay_cpt": (
                pred_n - actual_n
                if pred_n is not None and actual_n is not None
                else np.nan
            )
        })

    frontier = pd.DataFrame(frontier)

    stopped = frontier.first_predicted_stop.notna()
    safe = stopped & (frontier.unsafe_first_stop == 0)
    delays = frontier.stop_delay_cpt.dropna()

    summary = {
        "model": label,
        "threshold": threshold,
        "coverage_mae": float(
            np.mean(np.abs(predicted - actual))
        ),
        "coverage_bias": float(
            np.mean(predicted - actual)
        ),
        "spearman": float(
            spearmanr(predicted, actual).statistic
        ),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "false_stop_rate": (
            fp / (tp + fp) if tp + fp else None
        ),
        "missed_stop_rate": (
            fn / (tp + fn) if tp + fn else None
        ),
        "predicted_stop_volumes": int(stopped.sum()),
        "safe_first_stop_volumes": int(safe.sum()),
        "unsafe_first_stop_volumes": int(
            frontier.unsafe_first_stop.sum()
        ),
        "actual_stop_volumes": int(
            frontier.first_actual_stop.notna().sum()
        ),
        "mean_stop_delay_cpt": (
            float(delays.mean()) if len(delays) else None
        ),
        "median_stop_delay_cpt": (
            float(delays.median()) if len(delays) else None
        )
    }

    print(f"\n=== {label} ===")
    print(f"Threshold          : {threshold:.3f}")
    print(f"Coverage MAE       : {summary['coverage_mae']:.4f}")
    print(f"Coverage bias      : {summary['coverage_bias']:+.4f}")
    print(f"Spearman           : {summary['spearman']:.4f}")
    print(f"TP / FP / FN / TN  : {tp} / {fp} / {fn} / {tn}")
    print(f"False-stop rate    : {summary['false_stop_rate']:.2%}")
    print(f"Missed-stop rate   : {summary['missed_stop_rate']:.2%}")
    print(f"Volumes STOP       : {summary['predicted_stop_volumes']}/200")
    print(f"Safe first STOP    : {summary['safe_first_stop_volumes']}/200")
    print(f"Unsafe first STOP  : {summary['unsafe_first_stop_volumes']}/200")
    print(f"Actual STOP <=16   : {summary['actual_stop_volumes']}/200")
    print(f"Mean STOP delay    : {summary['mean_stop_delay_cpt']}")
    print(f"Median STOP delay  : {summary['median_stop_delay_cpt']}")

    return summary, frontier

print("=== FINAL COVERAGE CALIBRATOR V1 TEST ===")
print("GeoSyn       : 1100-1299")
print("Volumes      : 200")
print("CPT          : 1-16")
print("Threshold    : 0.975 (fixed from validation)")

raw, raw_frontier = evaluate(
    "Raw Uncertainty V3",
    "predicted_coverage", 0.90
)

cal, calibrated_frontier = evaluate(
    "Calibrated Ridge V1",
    "calibrated_coverage", THRESHOLD
)

df.to_csv(
    ROOT / "final_test/calibrated_predictions.csv",
    index=False
)

calibrated_frontier.to_csv(
    ROOT / "final_test/calibrated_frontier.csv",
    index=False
)

(ROOT / "final_test/final_comparison.json").write_text(
    json.dumps(
        {"raw": raw, "calibrated": cal},
        indent=2
    )
)

print("\n=== FINAL COMPARISON ===")
print("MAE improvement:",
      round((raw["coverage_mae"] - cal["coverage_mae"])
            / raw["coverage_mae"] * 100, 2), "%")
print("Unsafe first STOP:",
      raw["unsafe_first_stop_volumes"],
      "->", cal["unsafe_first_stop_volumes"])
print("Safe stopped volumes:",
      raw["safe_first_stop_volumes"],
      "->", cal["safe_first_stop_volumes"])

print("\n=== COMPLETE ===")
