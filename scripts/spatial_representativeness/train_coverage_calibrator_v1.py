from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr

ROOT = Path("results/coverage_calibrator_v1")
TRAIN = ROOT / "calibration/per_state_gate.csv"
VAL = ROOT / "validation/per_state_gate.csv"

train = pd.read_csv(TRAIN)
val = pd.read_csv(VAL)

required = {
    "volume_index", "n_cpt",
    "actual_coverage", "predicted_coverage"
}

for name, df, start, stop in [
    ("train", train, 800, 1000),
    ("validation", val, 1000, 1100),
]:
    assert required.issubset(df.columns)
    assert df["volume_index"].between(start, stop - 1).all()
    assert df["volume_index"].nunique() == stop - start
    assert not df[list(required)].isna().any().any()
    assert df["n_cpt"].between(1, 16).all()
    assert not df.duplicated(["volume_index", "n_cpt"]).any()

assert not (
    set(train.volume_index) & set(val.volume_index)
)

def features(df):
    p = df["predicted_coverage"].to_numpy()
    n = df["n_cpt"].to_numpy() / 16.0

    return np.column_stack([
        p,
        n,
        p * n,
        n * n,
    ])

X_train = features(train)
y_train = train["actual_coverage"].to_numpy()

model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=10.0)
)

model.fit(X_train, y_train)

def report(name, df):
    actual = df["actual_coverage"].to_numpy()
    raw = df["predicted_coverage"].to_numpy()

    calibrated = np.clip(
        model.predict(features(df)),
        0.0, 1.0
    )

    print(f"\n=== {name} ===")

    for label, predicted in [
        ("Raw V3", raw),
        ("Calibrated V1", calibrated),
    ]:
        mae = np.mean(np.abs(predicted - actual))
        bias = np.mean(predicted - actual)
        rho = spearmanr(predicted, actual).statistic

        stop = predicted >= 0.90
        sufficient = actual >= 0.90

        tp = np.sum(stop & sufficient)
        fp = np.sum(stop & ~sufficient)
        fn = np.sum(~stop & sufficient)

        false_rate = fp / (tp + fp) if tp + fp else 0
        missed_rate = fn / (tp + fn) if tp + fn else 0

        print(
            f"{label:16s}"
            f" MAE={mae:.4f}"
            f" Bias={bias:+.4f}"
            f" Spearman={rho:.4f}"
            f" FP={fp}"
            f" FalseStop={false_rate:.2%}"
            f" MissedStop={missed_rate:.2%}"
        )

    return calibrated

report("TRAINING", train)
calibrated_val = report("VALIDATION", val)

output = val.copy()
output["calibrated_coverage"] = calibrated_val

output.to_csv(
    ROOT / "validation/calibrated_predictions.csv",
    index=False
)

joblib.dump(
    model,
    ROOT / "coverage_calibrator_v1.joblib"
)

metadata = {
    "model": "Ridge Coverage Calibrator V1",
    "train_volumes": [800, 999],
    "validation_volumes": [1000, 1099],
    "features": [
        "predicted_coverage",
        "normalized_n_cpt",
        "interaction",
        "squared_n_cpt",
    ],
    "status": "experimental_not_approved_for_auto_stop",
    "stop_threshold_for_diagnostics": 0.90,
}

(ROOT / "metadata.json").write_text(
    json.dumps(metadata, indent=2)
)

print("\n=== CALIBRATOR V1 COMPLETE ===")
print("Model:", ROOT / "coverage_calibrator_v1.joblib")
print("Validation:", ROOT / "validation/calibrated_predictions.csv")
