from pathlib import Path
import ast
import csv
import json
import sys

import numpy as np

BASE = Path("scripts/geotech/v4")
sys.path.insert(0, str(BASE.resolve()))

from bearing_rule_engine_v1 import detect_candidates

DATA = Path(
    "data/synthetic3d_v4_geosyn_qcfs/"
    "joint_v3_expanded_1200"
)

ROOT = Path("results/bearing_layer_v1")

LABELS = ROOT / "candidate_labels_v1/masks"

MANIFEST = (
    ROOT / "geometry_aware_pilot_v3/manifest.json"
)

GENERATOR = (
    BASE / "build_bearing_candidate_labels_v1.py"
)

OUT = ROOT / "rule_engine_v1_validation"

if OUT.exists():
    raise SystemExit(
        f"Output exists; refusing overwrite: {OUT}"
    )

# Read the actual fraction used by the frozen
# synthetic label generator. Never assume its value.
tree = ast.parse(GENERATOR.read_text())
fraction_nodes = []

for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "MIN_LATERAL_FRACTION"
            ):
                fraction_nodes.append(node.value)

if len(fraction_nodes) != 1:
    raise RuntimeError(
        "Cannot identify MIN_LATERAL_FRACTION"
    )

lateral_fraction = float(
    ast.literal_eval(fraction_nodes[0])
)

files = json.loads(
    MANIFEST.read_text()
)["val_files"]

if len(files) != 30:
    raise RuntimeError(
        f"Expected 30 volumes, got {len(files)}"
    )

thresholds = (10., 12., 15.)
thicknesses = (2., 4., 6.)

rows = []
failures = 0

print("=== BEARING RULE ENGINE V1 VALIDATION ===")
print("Validation volumes:", len(files))
print("QC thresholds:", thresholds)
print("Thicknesses:", thicknesses)
print("Lateral fraction:", lateral_fraction)
print("Expected axis order: X,Y,Z")
print("Independent test: untouched")

for i, name in enumerate(files, 1):
    with np.load(
        DATA / name,
        allow_pickle=False
    ) as d:
        if "qc_mpa" not in d:
            raise RuntimeError(
                f"Missing qc_mpa in {name}. "
                f"Available keys: {list(d.files)}"
            )

        qc = np.asarray(
            d["qc_mpa"], dtype=np.float32
        )

        valid = np.asarray(
            d["valid_mask"], dtype=bool
        )

        raw = d["metadata_json"].item()

    if isinstance(raw, bytes):
        raw = raw.decode()

    grid = json.loads(str(raw))["grid"]

    spacing = (
        float(grid["dx_m"]),
        float(grid["dy_m"]),
        float(grid["dz_m"])
    )

    with np.load(
        LABELS / name,
        allow_pickle=False
    ) as label_data:

        for threshold in thresholds:
            for thickness in thicknesses:
                key = (
                    f"q{int(threshold)}"
                    f"_t{int(thickness)}m"
                )

                if key not in label_data:
                    raise RuntimeError(
                        f"Missing label {key}: {name}"
                    )

                truth = np.asarray(
                    label_data[key],
                    dtype=bool
                )

                result = detect_candidates(
                    qc,
                    valid,
                    spacing,
                    qc_min_mpa=threshold,
                    min_thickness_m=thickness,
                    min_lateral_fraction=lateral_fraction
                )

                pred = result["mask"]

                if pred.shape != truth.shape:
                    raise RuntimeError(
                        f"Shape mismatch: {name}, {key}"
                    )

                differences = int(
                    np.count_nonzero(pred != truth)
                )

                failures += int(differences > 0)

                rows.append({
                    "file": name,
                    "target": key,
                    "equal": int(differences == 0),
                    "different_voxels": differences,
                    "truth_voxels": int(truth.sum()),
                    "pred_voxels": int(pred.sum()),
                    "components": result[
                        "component_count"
                    ],
                })

    if i % 10 == 0:
        print(
            f"Validated {i}/{len(files)} volumes",
            flush=True
        )

# Small synthetic smoke test for adjustable sliders.
qc = np.zeros((10, 10, 5), dtype=np.float32)
valid = np.ones_like(qc, dtype=bool)

qc[1:7, 1:7, 1:4] = 16.0

smoke = detect_candidates(
    qc,
    valid,
    spacing_m=(2., 2., 1.),
    qc_min_mpa=15.,
    min_thickness_m=2.,
    min_lateral_fraction=0.05,
)

assert smoke["candidate_voxels"] > 0

stricter = detect_candidates(
    qc,
    valid,
    spacing_m=(2., 2., 1.),
    qc_min_mpa=17.,
    min_thickness_m=2.,
    min_lateral_fraction=0.05,
)

assert stricter["candidate_voxels"] == 0

OUT.mkdir(parents=True)

with (OUT / "parity.csv").open(
    "w", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0])
    )
    writer.writeheader()
    writer.writerows(rows)

total = len(rows)
passed = total - failures

print("\n=== VALIDATION RESULTS ===")
print("Compared masks:", total)
print("Exact matches:", passed)
print("Mismatched masks:", failures)
print("Total differing voxels:", sum(
    r["different_voxels"] for r in rows
))
print("Interactive criteria smoke test: PASS")

if failures:
    print("\nFirst mismatches:")
    for r in rows:
        if not r["equal"]:
            print(
                r["file"],
                r["target"],
                r["different_voxels"]
            )
            failures -= 1
            if failures == 0:
                break

    raise SystemExit(
        "FAIL: Rule Engine does not match labels."
    )

print("\n=== RULE ENGINE V1 VALIDATED ===")
print("Parity:", OUT / "parity.csv")
print("No training performed.")
print("Frozen models unchanged.")
print("Independent test untouched.")
