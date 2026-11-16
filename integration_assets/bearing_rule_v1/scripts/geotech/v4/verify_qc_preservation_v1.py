from pathlib import Path
import ast
import json
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial import cKDTree

BASE = Path("scripts/geotech/v4")
ROOT = Path("results")

DATA = Path(
    "data/synthetic3d_v4_geosyn_qcfs/"
    "joint_v3_expanded_1200"
)
LABELS = ROOT / (
    "bearing_layer_v1/candidate_labels_v1/masks"
)
MANIFEST = ROOT / (
    "bearing_layer_v1/"
    "geometry_aware_pilot_v3/manifest.json"
)
CHECKPOINT = ROOT / (
    "qc_fs_joint_v3/spatial_cnn_v3/best.pt"
)

SEED = 68032026
CHANNELS = 24
IDW_K = 4
Q_REF = 5.0
F_REF = 0.10
MAX_LOG_CORRECTION = 1.2
DETECTION_THRESHOLDS = (10., 12., 15.)

CPTS = (1, 3, 8, 16, 32)
PATTERNS = ("random", "distributed")

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def extract(path, names):
    tree = ast.parse(path.read_text())

    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(
            node, (ast.FunctionDef, ast.ClassDef)
        )
    }

    for name in names:
        if name not in definitions:
            raise RuntimeError(f"Missing {name}: {path}")

        exec(
            compile(
                ast.Module(
                    body=[definitions[name]],
                    type_ignores=[]
                ),
                str(path),
                "exec"
            ),
            globals()
        )

extract(
    BASE / "train_spatial_cnn_v2.py",
    (
        "seed_for",
        "load_volume",
        "prepare",
        "tensor",
        "ResidualBlock",
        "SpatialCNN",
        "predict",
    )
)
extract(
    BASE / "train_spatial_cnn_v3.py",
    ("get_sites",)
)

state = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True
)

model = SpatialCNN().to(DEVICE)
model.load_state_dict(
    state["state_dict"], strict=True
)
model.eval()

files = json.loads(
    MANIFEST.read_text()
)["val_files"]

print("=== QC PRESERVATION VERIFICATION ===")
print("Checkpoint epoch:", state["epoch"])
print("Device:", DEVICE)
print("Scenario: A_thin_variable")
print("Target: Q15-T2")

with torch.inference_mode():
    for count in (3, 8, 32):
        observed_count = 0
        candidate_count = 0
        cnn_error_sum = 0.
        idw_error_sum = 0.
        cnn_recovered = 0
        idw_recovered = 0
        positive_cases = 0
        direct_hit_cases = 0

        for name in files:
            q, f, valid, spacing, scenario = (
                load_volume(DATA / name)
            )

            if scenario != "A_thin_variable":
                continue

            with np.load(
                LABELS / name, allow_pickle=False
            ) as d:
                target = np.asarray(
                    d["q15_t2m"], bool
                )

            target &= valid

            if not target.any():
                continue

            for pattern in PATTERNS:
                sites = get_sites(
                    valid, spacing,
                    name, pattern, epoch=-1
                )[:count]

                batch = prepare(
                    q, f, valid, spacing, sites
                )

                predicted_q, _, _, _ = predict(
                    model, batch
                )

                cnn = (
                    predicted_q.detach()
                    .cpu().numpy()
                    .transpose(2, 1, 0)
                )

                idw = np.asarray(
                    batch["idw_q"]
                ).transpose(2, 1, 0)

                observed = np.asarray(
                    batch["observed"], bool
                ).transpose(2, 1, 0)

                observed &= valid
                measured_candidate = (
                    observed & target
                )

                if cnn.shape != q.shape:
                    raise RuntimeError(
                        f"Axis mismatch: {name}"
                    )

                positive_cases += 1
                direct_hit_cases += int(
                    measured_candidate.any()
                )

                observed_count += int(
                    observed.sum()
                )
                candidate_count += int(
                    measured_candidate.sum()
                )

                cnn_error_sum += float(
                    np.abs(
                        cnn[observed]
                        - q[observed]
                    ).sum()
                )

                idw_error_sum += float(
                    np.abs(
                        idw[observed]
                        - q[observed]
                    ).sum()
                )

                cnn_recovered += int(
                    np.count_nonzero(
                        cnn[measured_candidate] >= 15.
                    )
                )

                idw_recovered += int(
                    np.count_nonzero(
                        idw[measured_candidate] >= 15.
                    )
                )

        print(f"\n=== CPT {count} ===")
        print("Positive cases:", positive_cases)
        print("Direct-hit cases:", direct_hit_cases)
        print("Observed voxels:", observed_count)
        print(
            "CNN observed qc MAE:",
            f"{cnn_error_sum/observed_count:.6f} MPa"
        )
        print(
            "IDW observed qc MAE:",
            f"{idw_error_sum/observed_count:.6f} MPa"
        )
        print(
            "Measured candidate voxels:",
            candidate_count
        )

        if candidate_count:
            print(
                "CNN retains qc >=15:",
                f"{cnn_recovered}/{candidate_count}",
                f"({100*cnn_recovered/candidate_count:.2f}%)"
            )
            print(
                "IDW retains qc >=15:",
                f"{idw_recovered}/{candidate_count}",
                f"({100*idw_recovered/candidate_count:.2f}%)"
            )

print("\n=== VERIFICATION COMPLETE ===")
print("Frozen checkpoints unchanged.")
