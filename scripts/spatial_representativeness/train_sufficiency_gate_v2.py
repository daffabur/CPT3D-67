from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
)

ROOT = Path("results/sufficiency_gate_v2")
ROOT.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "training": (1300, 1500),
    "calibration": (1500, 1600),
    "validation": (1600, 1700),
}

REQUIRED = [
    "volume_index",
    "n_cpt",
    "actual_coverage",
    "predicted_coverage",
    "estimated_mean_bound",
    "estimated_p90_bound",
]

FEATURES = [
    "predicted_coverage",
    "n_cpt",
    "estimated_mean_bound",
    "estimated_p90_bound",
    "delta_predicted",
    "delta_p90",
]

data = {}

for name, (start, stop) in SPLITS.items():
    path = ROOT / name / "per_state_gate.csv"
    df = pd.read_csv(path)

    assert all(c in df.columns for c in REQUIRED)
    assert set(df.volume_index) == set(range(start, stop))
    assert len(df) == (stop - start) * 16
    assert not df.duplicated(["volume_index", "n_cpt"]).any()
    assert df[REQUIRED].notna().all().all()

    df = df.sort_values(
        ["volume_index", "n_cpt"]
    ).reset_index(drop=True)

    assert df.groupby("volume_index").n_cpt.apply(
        lambda x: set(x) == set(range(1, 17))
    ).all()

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
        df.actual_coverage >= 0.90
    ).astype(int)

    assert np.isfinite(
        df[FEATURES].to_numpy(dtype=float)
    ).all()

    data[name] = df

assert len(
    set(data["training"].volume_index)
    & set(data["calibration"].volume_index)
) == 0

assert len(
    set(data["training"].volume_index)
    & set(data["validation"].volume_index)
) == 0

assert len(
    set(data["calibration"].volume_index)
    & set(data["validation"].volume_index)
) == 0

train = data["training"]
cal = data["calibration"]
val = data["validation"]

print("=== SUFFICIENCY GATE V2 ===")
print("Training:", len(train))
print("Calibration:", len(cal))
print("Validation:", len(val))

# 1. Train probabilistic classifier.
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=67,
    )
)

model.fit(
    train[FEATURES],
    train["target"]
)

# 2. Platt calibration on separate volumes.
p_cal_raw = model.predict_proba(
    cal[FEATURES]
)[:, 1]

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

platt = LogisticRegression(
    C=1.0,
    max_iter=2000,
    random_state=67,
)

platt.fit(
    logit(p_cal_raw).reshape(-1, 1),
    cal["target"],
)

def predict_probability(df):
    raw = model.predict_proba(
        df[FEATURES]
    )[:, 1]

    calibrated = platt.predict_proba(
        logit(raw).reshape(-1, 1)
    )[:, 1]

    return raw, calibrated

print("\n=== PROBABILITY EVALUATION ===")

for name, df in data.items():
    raw, calibrated = predict_probability(df)

    df["p_raw"] = raw
    df["p_sufficient"] = calibrated

    y = df["target"]

    print(f"\n{name.upper()}")

    for label, p in [
        ("Raw", raw),
        ("Platt", calibrated),
    ]:
        print(
            f"{label:6s}"
            f" AUC={roc_auc_score(y, p):.4f}"
            f" Brier={brier_score_loss(y, p):.4f}"
            f" LogLoss={log_loss(y, p):.4f}"
        )

    df.to_csv(
        ROOT / name / "gate_v2_predictions.csv",
        index=False
    )

# 3. Threshold sweep on validation ONLY.
print("\n=== VALIDATION THRESHOLD SWEEP ===")

results = []

for threshold in np.unique(
    np.r_[np.arange(0.50, 1.00, 0.025), 0.99, 0.995]
):
    stop = val.p_sufficient >= threshold
    actual = val.target.astype(bool)

    tp = int((stop & actual).sum())
    fp = int((stop & ~actual).sum())
    fn = int((~stop & actual).sum())
    tn = int((~stop & ~actual).sum())

    stopped = 0
    unsafe = 0
    first_cpt = []

    for _, group in val.groupby("volume_index"):
        first = group[
            group.p_sufficient >= threshold
        ].head(1)

        if first.empty:
            continue

        stopped += 1
        first_cpt.append(int(first.n_cpt.iloc[0]))

        if first.actual_coverage.iloc[0] < 0.90:
            unsafe += 1

    results.append({
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "false_stop_rate": (
            fp / (tp + fp) if tp + fp else np.nan
        ),
        "missed_stop_rate": (
            fn / (tp + fn) if tp + fn else np.nan
        ),
        "stopped_volumes": stopped,
        "unsafe_first_stop": unsafe,
        "safe_first_stop": stopped - unsafe,
        "mean_first_stop_cpt": (
            float(np.mean(first_cpt))
            if first_cpt else np.nan
        ),
    })

sweep = pd.DataFrame(results)

print(
    sweep[
        [
            "threshold",
            "fp",
            "false_stop_rate",
            "missed_stop_rate",
            "stopped_volumes",
            "unsafe_first_stop",
            "safe_first_stop",
        ]
    ].round(4).to_string(index=False)
)

sweep.to_csv(
    ROOT / "validation/threshold_sweep.csv",
    index=False
)

# Do not automatically approve a threshold.
candidates = sweep[
    (sweep.unsafe_first_stop == 0)
    & (sweep.stopped_volumes >= 10)
]

print("\n=== DEVELOPMENT CANDIDATES ===")

if len(candidates):
    print(
        candidates[
            [
                "threshold",
                "stopped_volumes",
                "safe_first_stop",
                "missed_stop_rate",
            ]
        ].round(4).to_string(index=False)
    )
else:
    print("No candidate meets the exploratory criteria.")

joblib.dump(
    model,
    ROOT / "gate_v2_classifier.joblib"
)

joblib.dump(
    platt,
    ROOT / "gate_v2_platt.joblib"
)

metadata = {
    "model": "Probabilistic Sufficiency Gate V2",
    "features": FEATURES,
    "training": [1300, 1499],
    "calibration": [1500, 1599],
    "validation": [1600, 1699],
    "actual_coverage_target": 0.90,
    "max_cpt": 16,
    "status": "experimental",
    "auto_stop_approved": False,
}

(ROOT / "model_metadata.json").write_text(
    json.dumps(metadata, indent=2)
)

print("\n=== GATE V2 TRAINING COMPLETE ===")
print("Models saved:", ROOT)
print("Final test: NOT USED")
