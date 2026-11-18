from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

import joblib
import numpy as np

ROOT = Path.home() / "Developer/cpt_super_learn"
SOURCE = ROOT / "results/sufficiency_gate_v2"
EXPORT = ROOT / "model_export/SufficiencyGate_V2_1"

EXPORT.mkdir(parents=True, exist_ok=True)

FILES = {
    "gate_v2_classifier.joblib":
        SOURCE / "gate_v2_classifier.joblib",

    "gate_v2_platt.joblib":
        SOURCE / "gate_v2_platt.joblib",

    "model_metadata.json":
        SOURCE / "model_metadata.json",

    "gate_v21_final.json":
        SOURCE / "final_test/gate_v21_final.json",
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

print("=== SUFFICIENCY GATE V2.1 FREEZE ===")

# Validate files before exporting.
for name, source in FILES.items():
    if not source.is_file():
        raise FileNotFoundError(source)

metadata = json.loads(
    FILES["model_metadata.json"].read_text()
)

final = json.loads(
    FILES["gate_v21_final.json"].read_text()
)

assert metadata["features"] == [
    "predicted_coverage",
    "n_cpt",
    "estimated_mean_bound",
    "estimated_p90_bound",
    "delta_predicted",
    "delta_p90",
]

assert metadata["training"] == [1300, 1499]
assert metadata["calibration"] == [1500, 1599]
assert metadata["validation"] == [1600, 1699]

config = final["configuration"]

assert config["threshold"] == 0.995
assert config["minimum_cpt"] == 5
assert config["actual_target"] == 0.90
assert config["auto_stop_approved"] is False

assert final["gate_v21"]["unsafe_first_stop"] == 0
assert final["gate_v21"]["stopped_volumes"] == 51
assert final["gate_v21"]["safe_first_stop"] == 51

# Do not overwrite an existing frozen export.
if any(EXPORT.iterdir()):
    raise SystemExit(
        f"Export directory is not empty: {EXPORT}\n"
        "Existing frozen files were not overwritten."
    )

classifier = joblib.load(
    FILES["gate_v2_classifier.joblib"]
)

platt = joblib.load(
    FILES["gate_v2_platt.joblib"]
)

scaler = classifier.named_steps["standardscaler"]
logistic = classifier.named_steps["logisticregression"]

assert len(logistic.classes_) == 2
assert logistic.classes_.tolist() == [0, 1]
assert platt.classes_.tolist() == [0, 1]

def tolist(x):
    return np.asarray(x).tolist()

# Export numerical parameters for Swift implementation.
swift_parameters = {
    "model_name": "SufficiencyGate_V2_1",
    "model_type": "StandardScaler_LogisticRegression_Platt",
    "features": metadata["features"],

    "scaler": {
        "mean": tolist(scaler.mean_),
        "scale": tolist(scaler.scale_),
    },

    "classifier": {
        "coef": tolist(logistic.coef_[0]),
        "intercept": float(logistic.intercept_[0]),
    },

    "platt": {
        "coef": float(platt.coef_[0][0]),
        "intercept": float(platt.intercept_[0]),
        "input": "logit(raw_probability)",
    },

    "decision": {
        "probability_threshold": 0.995,
        "minimum_cpt": 5,
        "actual_coverage_target": 0.90,
        "ic_tolerance": 0.15,
        "maximum_evaluated_cpt": 16,
        "auto_stop_approved": False,
    },
}

# Copy frozen checkpoint artifacts.
for name, source in FILES.items():
    shutil.copy2(source, EXPORT / name)

(EXPORT / "gate_v21_swift_parameters.json").write_text(
    json.dumps(swift_parameters, indent=2)
)

acceptance = {
    "name": "Sufficiency Gate V2.1",
    "status": "experimental_frozen",
    "exported_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "evaluation": {
        "dataset": "GeoSyn V4",
        "volume_range": [1700, 1899],
        "volumes": 200,
        "states": 3200,
        "cpt_range": [1, 16],
        "auc": 0.9519,
        "brier_score": 0.0933,
        "unsafe_first_stop": 0,
        "safe_first_stop": 51,
        "stopped_volumes": 51,
        "missed_stop_rate": 0.6360,
    },
    "limitations": [
        "Only evaluated on synthetic GeoSyn data.",
        "Zero observed unsafe stops does not guarantee zero risk.",
        "Not validated for field geotechnical investigation.",
        "Not a substitute for SNI 8460:2017 requirements.",
        "Automatic field stopping is not approved.",
        "Independence from upstream model-development data "
        "has not been fully audited.",
    ],
}

(EXPORT / "acceptance_report.json").write_text(
    json.dumps(acceptance, indent=2)
)

manifest = {}

for path in sorted(EXPORT.iterdir()):
    if path.is_file():
        manifest[path.name] = {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }

(EXPORT / "manifest_sha256.json").write_text(
    json.dumps(manifest, indent=2)
)

print("\n=== EXPORTED FILES ===")

for path in sorted(EXPORT.iterdir()):
    print(path.name)

print("\n=== ACCEPTANCE ===")
print("Status           : experimental_frozen")
print("Threshold        : 0.995")
print("Minimum CPT      : 5")
print("Unsafe STOP      : 0 / 200")
print("Safe STOP        : 51 / 200")
print("Auto STOP field  : NOT APPROVED")

print("\nExport directory:", EXPORT)
print("\n=== FREEZE COMPLETE ===")
