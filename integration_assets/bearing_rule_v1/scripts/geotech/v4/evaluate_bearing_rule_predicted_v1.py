from pathlib import Path
from collections import defaultdict
import ast
import csv
import json
import sys

import numpy as np
import torch

BASE = Path("scripts/geotech/v4")
SOURCE = BASE / "verify_qc_preservation_v1.py"
OUT = Path(
    "results/bearing_layer_v1/"
    "rule_engine_predicted_v1"
)

if OUT.exists():
    raise SystemExit(
        f"Output already exists: {OUT}"
    )

if not SOURCE.is_file():
    raise SystemExit(f"Missing: {SOURCE}")

# Reuse the previously verified CNN V3 setup.
tree = ast.parse(SOURCE.read_text())
cuts = [
    i for i, node in enumerate(tree.body)
    if isinstance(node, ast.With)
]
if len(cuts) != 1:
    raise RuntimeError("Unexpected source structure")

exec(
    compile(
        ast.Module(
            body=tree.body[:cuts[0]],
            type_ignores=[]
        ),
        str(SOURCE),
        "exec"
    ),
    globals()
)

sys.path.insert(0, str(BASE.resolve()))
from bearing_rule_engine_v1 import detect_candidates

COUNTS = (3, 8, 32)
THRESHOLDS = (10., 12., 15.)
THICKNESSES = (2., 4., 6.)
LATERAL_FRACTION = 0.05

rows = []

def metrics(truth, pred, valid):
    tp = int((truth & pred & valid).sum())
    fp = int((~truth & pred & valid).sum())
    fn = int((truth & ~pred & valid).sum())
    tn = int((~truth & ~pred & valid).sum())
    return tp, fp, fn, tn

print("\n=== BEARING RULE ON PREDICTED QC ===")
print("Validation volumes:", len(files))
print("CPT counts:", COUNTS)
print("QC thresholds:", THRESHOLDS)
print("Thicknesses:", THICKNESSES)
print("Lateral fraction:", LATERAL_FRACTION)
print("Models: CNN V3, IDW")
print("Independent test: untouched", flush=True)

with torch.inference_mode():
    for count in COUNTS:
        for i, name in enumerate(files, 1):
            q, f, valid, spacing, scenario = (
                load_volume(DATA / name)
            )

            with np.load(
                LABELS / name, allow_pickle=False
            ) as label_data:
                truths = {
                    f"q{int(t)}_t{int(h)}m":
                    np.asarray(
                        label_data[f"q{int(t)}_t{int(h)}m"],
                        dtype=bool
                    ).copy()
                    for t in THRESHOLDS
                    for h in THICKNESSES
                }

            for pattern in PATTERNS:
                sites = get_sites(
                    valid, spacing, name,
                    pattern, epoch=-1
                )[:count]

                batch = prepare(
                    q, f, valid, spacing, sites
                )

                pq, _, _, _ = predict(model, batch)

                cnn = (
                    pq.detach().cpu().numpy()
                    .transpose(2, 1, 0)
                )

                idw = np.asarray(
                    batch["idw_q"],
                    dtype=np.float32
                ).transpose(2, 1, 0)

                assert (
                    cnn.shape == idw.shape
                    == valid.shape == q.shape
                )

                for method, volume in (
                    ("CNN_V3_RULE", cnn),
                    ("IDW_RULE", idw),
                ):
                    for threshold in THRESHOLDS:
                        for thickness in THICKNESSES:
                            target = (
                                f"q{int(threshold)}"
                                f"_t{int(thickness)}m"
                            )

                            result = detect_candidates(
                                volume,
                                valid,
                                spacing,
                                qc_min_mpa=threshold,
                                min_thickness_m=thickness,
                                min_lateral_fraction=LATERAL_FRACTION,
                            )

                            truth = truths[target] & valid
                            pred = result["mask"] & valid

                            tp, fp, fn, tn = metrics(
                                truth, pred, valid
                            )

                            rows.append({
                                "file": name,
                                "scenario": scenario,
                                "cpt": count,
                                "pattern": pattern,
                                "method": method,
                                "target": target,
                                "tp": tp,
                                "fp": fp,
                                "fn": fn,
                                "tn": tn,
                            })

            if i % 10 == 0:
                print(
                    f"CPT {count}: {i}/{len(files)}",
                    flush=True
                )

def percentage(num, den):
    return 100. * num / den if den else None

summary = []

for scenario in (
    "ALL",
    "A_thin_variable",
    "B_layered_moderate",
    "C_thick_continuous",
):
    for count in COUNTS:
        for threshold in THRESHOLDS:
            for thickness in THICKNESSES:
                target = (
                    f"q{int(threshold)}"
                    f"_t{int(thickness)}m"
                )

                for method in (
                    "CNN_V3_RULE",
                    "IDW_RULE",
                ):
                    selected = [
                        r for r in rows
                        if r["cpt"] == count
                        and r["target"] == target
                        and r["method"] == method
                        and (
                            scenario == "ALL"
                            or r["scenario"] == scenario
                        )
                    ]

                    tp = sum(r["tp"] for r in selected)
                    fp = sum(r["fp"] for r in selected)
                    fn = sum(r["fn"] for r in selected)
                    tn = sum(r["tn"] for r in selected)

                    summary.append({
                        "scenario": scenario,
                        "cpt": count,
                        "target": target,
                        "method": method,
                        "precision_percent": percentage(
                            tp, tp + fp
                        ),
                        "recall_percent": percentage(
                            tp, tp + fn
                        ),
                        "f1_percent": percentage(
                            2*tp, 2*tp + fp + fn
                        ),
                        "fpr_percent": percentage(
                            fp, fp + tn
                        ),
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                    })

OUT.mkdir(parents=True)

for filename, data in (
    ("per_case.csv", rows),
    ("summary.csv", summary),
):
    with (OUT / filename).open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(data[0])
        )
        writer.writeheader()
        writer.writerows(data)

def fmt(v):
    return f"{v:.2f}%" if v is not None else "N/A"

for scenario in (
    "ALL",
    "A_thin_variable",
    "B_layered_moderate",
    "C_thick_continuous",
):
    print(f"\n=== {scenario} Q15-T2 ===")

    for count in COUNTS:
        print(f"CPT {count}")

        for r in summary:
            if (
                r["scenario"] == scenario
                and r["cpt"] == count
                and r["target"] == "q15_t2m"
            ):
                print(
                    f"  {r['method']}: "
                    f"P={fmt(r['precision_percent'])} "
                    f"R={fmt(r['recall_percent'])} "
                    f"F1={fmt(r['f1_percent'])} "
                    f"FPR={fmt(r['fpr_percent'])} "
                    f"FP={r['fp']}"
                )

print("\n=== PREDICTED RULE EVALUATION COMPLETE ===")
print("Inference cases:", len(files)*len(COUNTS)*len(PATTERNS))
print("Evaluated masks:", len(rows))
print("Summary:", OUT / "summary.csv")
print("Frozen checkpoints unchanged.")
print("Independent test untouched.")
