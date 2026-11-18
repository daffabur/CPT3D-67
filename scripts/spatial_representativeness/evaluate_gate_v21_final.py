from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path("results/sufficiency_gate_v2")
TEST = ROOT / "final_test/per_state_gate.csv"

THRESHOLD = 0.995
MIN_CPT = 5
ACTUAL_TARGET = 0.90

df = pd.read_csv(TEST)

assert len(df) == 3200
assert set(df.volume_index) == set(range(1700, 1900))
assert not df.duplicated(
    ["volume_index", "n_cpt"]
).any()

assert df.groupby("volume_index").n_cpt.apply(
    lambda x: set(x) == set(range(1, 17))
).all()

df = df.sort_values(
    ["volume_index", "n_cpt"]
).reset_index(drop=True)

df["delta_predicted"] = (
    df.groupby("volume_index")
    .predicted_coverage.diff()
    .fillna(0)
)

df["delta_p90"] = (
    df.groupby("volume_index")
    .estimated_p90_bound.diff()
    .fillna(0)
)

df["target"] = (
    df.actual_coverage >= ACTUAL_TARGET
).astype(int)

metadata = json.loads(
    (ROOT / "model_metadata.json").read_text()
)

assert metadata["training"] == [1300, 1499]
assert metadata["calibration"] == [1500, 1599]
assert metadata["validation"] == [1600, 1699]
assert metadata["auto_stop_approved"] is False

features = metadata["features"]

assert df[features].notna().all().all()
assert np.isfinite(
    df[features].to_numpy(dtype=float)
).all()

model = joblib.load(
    ROOT / "gate_v2_classifier.joblib"
)

platt = joblib.load(
    ROOT / "gate_v2_platt.joblib"
)

p_raw = model.predict_proba(
    df[features]
)[:, 1]

p_clipped = np.clip(
    p_raw, 1e-6, 1 - 1e-6
)

logit = np.log(
    p_clipped / (1 - p_clipped)
)

df["p_sufficient"] = platt.predict_proba(
    logit.reshape(-1, 1)
)[:, 1]

print("=== FINAL GATE V2.1 TEST ===")
print("GeoSyn    : 1700-1899")
print("Volumes   : 200")
print("CPT       : 1-16")
print("Threshold :", THRESHOLD)
print("Minimum CPT:", MIN_CPT)

print("\n=== PROBABILITY QUALITY ===")
print(
    "AUC:",
    round(
        roc_auc_score(
            df.target,
            df.p_sufficient
        ), 4
    )
)
print(
    "Brier:",
    round(
        brier_score_loss(
            df.target,
            df.p_sufficient
        ), 4
    )
)

def evaluate(label, stop_mask):
    actual = df.target.astype(bool)

    tp = int((stop_mask & actual).sum())
    fp = int((stop_mask & ~actual).sum())
    fn = int((~stop_mask & actual).sum())
    tn = int((~stop_mask & ~actual).sum())

    frontier = []

    for vid, group in df.groupby("volume_index"):
        gstop = group.loc[
            stop_mask.loc[group.index]
        ]

        first = gstop.head(1)

        actual_first = group.loc[
            group.target == 1
        ].head(1)

        pred_n = (
            int(first.n_cpt.iloc[0])
            if len(first) else None
        )

        actual_n = (
            int(actual_first.n_cpt.iloc[0])
            if len(actual_first) else None
        )

        unsafe = (
            len(first) > 0
            and first.target.iloc[0] == 0
        )

        frontier.append({
            "volume_index": int(vid),
            "first_predicted_stop": pred_n,
            "first_actual_stop": actual_n,
            "unsafe_first_stop": int(unsafe),
            "stop_delay_cpt": (
                pred_n - actual_n
                if pred_n is not None
                and actual_n is not None
                else np.nan
            ),
        })

    frontier = pd.DataFrame(frontier)

    stopped = frontier.first_predicted_stop.notna()
    unsafe_count = int(
        frontier.unsafe_first_stop.sum()
    )

    safe_count = int(stopped.sum()) - unsafe_count
    delays = frontier.stop_delay_cpt.dropna()

    summary = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "false_stop_rate": (
            fp / (tp + fp) if tp + fp else None
        ),
        "missed_stop_rate": (
            fn / (tp + fn) if tp + fn else None
        ),
        "stopped_volumes": int(stopped.sum()),
        "safe_first_stop": safe_count,
        "unsafe_first_stop": unsafe_count,
        "actual_stop_volumes": int(
            frontier.first_actual_stop.notna().sum()
        ),
        "mean_first_stop_cpt": (
            float(
                frontier.loc[
                    stopped, "first_predicted_stop"
                ].mean()
            ) if stopped.any() else None
        ),
        "mean_stop_delay": (
            float(delays.mean())
            if len(delays) else None
        ),
    }

    print(f"\n=== {label} ===")
    print("TP / FP / FN / TN:", tp, fp, fn, tn)
    print("False-stop rate:",
          f"{summary['false_stop_rate']:.2%}"
          if summary['false_stop_rate'] is not None
          else "N/A")
    print("Missed-stop rate:",
          f"{summary['missed_stop_rate']:.2%}")
    print("Stopped volumes:", summary["stopped_volumes"])
    print("Safe first STOP:", safe_count)
    print("Unsafe first STOP:", unsafe_count)
    print("Actual STOP volumes:",
          summary["actual_stop_volumes"])
    print("Mean first STOP CPT:",
          summary["mean_first_stop_cpt"])
    print("Mean STOP delay:",
          summary["mean_stop_delay"])

    return summary, frontier

raw_stop = df.predicted_coverage >= 0.90

v21_stop = (
    (df.p_sufficient >= THRESHOLD)
    & (df.n_cpt >= MIN_CPT)
)

raw, raw_frontier = evaluate(
    "RAW UNCERTAINTY V3",
    raw_stop
)

v21, v21_frontier = evaluate(
    "PROBABILISTIC GATE V2.1",
    v21_stop
)

df.to_csv(
    ROOT / "final_test/gate_v21_predictions.csv",
    index=False
)

v21_frontier.to_csv(
    ROOT / "final_test/gate_v21_frontier.csv",
    index=False
)

summary = {
    "configuration": {
        "threshold": THRESHOLD,
        "minimum_cpt": MIN_CPT,
        "actual_target": ACTUAL_TARGET,
        "status": "experimental",
        "auto_stop_approved": False,
    },
    "raw": raw,
    "gate_v21": v21,
}

(ROOT / "final_test/gate_v21_final.json").write_text(
    json.dumps(summary, indent=2)
)

print("\n=== FINAL COMPARISON ===")
print(
    "Unsafe first STOP:",
    raw["unsafe_first_stop"],
    "->",
    v21["unsafe_first_stop"]
)
print(
    "Safe first STOP:",
    raw["safe_first_stop"],
    "->",
    v21["safe_first_stop"]
)
print(
    "Stopped volumes:",
    raw["stopped_volumes"],
    "->",
    v21["stopped_volumes"]
)

print("\n=== COMPLETE ===")
