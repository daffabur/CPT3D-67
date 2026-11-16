"""Export Bearing Rule Engine data for native applications.

No model training. Source CNN checkpoint remains frozen.
Volume memory order: XYZ, C-contiguous, Z-fastest.
"""

from pathlib import Path
import ast
import csv
import json
import sys

import numpy as np
import torch

BASE = Path("scripts/geotech/v4")
SOURCE = BASE / "verify_qc_preservation_v1.py"
RULE_V2 = BASE / "bearing_rule_engine_v2.py"

ROOT = Path("results/bearing_layer_v1")
PREVIOUS = ROOT / "rule_engine_v2_demo"
OUTPUT = ROOT / "app_contract_v3"

VOLUME = "v4_train_00890.npz"
COUNTS = (8, 32)

if OUTPUT.exists():
    raise SystemExit(
        f"Output already exists: {OUTPUT}"
    )

for path in (SOURCE, RULE_V2):
    if not path.is_file():
        raise SystemExit(f"Missing: {path}")

sys.path.insert(0, str(BASE.resolve()))

from bearing_rule_engine_v2 import (
    spatial_output,
    inspect_xy,
)

from bearing_rule_engine_v1 import detect_candidates

# Reuse the successfully verified CNN V3 setup,
# excluding its original evaluation loop.
tree = ast.parse(SOURCE.read_text())

cuts = [
    i for i, node in enumerate(tree.body)
    if isinstance(node, ast.With)
]

if len(cuts) != 1:
    raise RuntimeError(
        "Unexpected QC verification script structure"
    )

bootstrap = ast.Module(
    body=tree.body[:cuts[0]],
    type_ignores=[]
)

exec(
    compile(
        bootstrap, str(SOURCE), "exec"
    ),
    globals()
)

q, f, valid, spacing, scenario = load_volume(
    DATA / VOLUME
)

dx, dy, dz = map(float, spacing)

print("=== BEARING APP CONTRACT V3 ===")
print("Volume:", VOLUME)
print("Scenario:", scenario)
print("Grid:", q.shape)
print("Spacing:", spacing)
print("Source: Frozen QC-FS CNN V3")
print("Ic filter: disabled")
print("Independent test: untouched", flush=True)


def write_binary(path, arr, dtype):
    arr = np.ascontiguousarray(
        np.asarray(arr, dtype=dtype)
    )
    arr.tofile(path)

    reread = np.fromfile(path, dtype=dtype)
    if not np.array_equal(
        reread, arr.ravel()
    ):
        raise RuntimeError(
            f"Binary roundtrip failed: {path}"
        )


def optional_float(value):
    return (
        None if value is None
        else float(value)
    )


OUTPUT.mkdir(parents=True)

with torch.inference_mode():
    for count in COUNTS:
        sites = get_sites(
            valid,
            spacing,
            VOLUME,
            "distributed",
            epoch=-1
        )[:count]

        batch = prepare(
            q, f, valid, spacing, sites
        )

        predicted_q, predicted_f, _, _ = predict(
            model, batch
        )

        qc = (
            predicted_q.detach().cpu()
            .numpy().transpose(2, 1, 0)
            .astype(np.float32)
        )

        fs = (
            predicted_f.detach().cpu()
            .numpy().transpose(2, 1, 0)
            .astype(np.float32)
        )

        if (
            qc.shape != q.shape
            or fs.shape != q.shape
        ):
            raise RuntimeError(
                "Prediction grid mismatch"
            )

        result = spatial_output(
            qc=qc,
            valid=valid,
            spacing_m=spacing,
            cpt_sites=sites,
            qc_min_mpa=15.,
            min_thickness_m=2.,
            min_lateral_fraction=0.05,
        )

        # Compare against already-validated V2.
        old_path = (
            PREVIOUS / f"cpt_{count}"
            / "spatial_output.npz"
        )

        with np.load(
            old_path, allow_pickle=False
        ) as old:
            old_mask = np.asarray(
                old["candidate_mask_xyz"],
                dtype=bool
            )

        if not np.array_equal(
            result["mask"], old_mask
        ):
            raise RuntimeError(
                f"V2/V3 mask mismatch: CPT {count}"
            )

        folder = OUTPUT / f"cpt_{count}"
        folder.mkdir()

        binary_fields = {
            "qc_xyz.f32": (
                qc, "<f4"
            ),
            "fs_xyz.f32": (
                fs, "<f4"
            ),
            "valid_xyz.u8": (
                valid.astype(np.uint8), "u1"
            ),
            "candidate_xyz.u8": (
                result["mask"].astype(np.uint8),
                "u1"
            ),
            "component_xyz.i32": (
                result["component_ids"], "<i4"
            ),
            "nearest_cpt_xy.f32": (
                result["nearest_cpt_distance_m"],
                "<f4"
            ),
        }

        for filename, (array, dtype) in (
            binary_fields.items()
        ):
            write_binary(
                folder / filename,
                array,
                dtype
            )

        intervals_path = folder / "intervals.csv"

        fields = [
            "x_index",
            "y_index",
            "top_depth_m",
            "bottom_depth_m",
            "thickness_m",
            "nearest_cpt_m",
            "component_ids",
        ]

        with intervals_path.open(
            "w", newline=""
        ) as fh:
            writer = csv.DictWriter(
                fh, fieldnames=fields
            )
            writer.writeheader()

            for item in result["intervals"]:
                writer.writerow({
                    k: json.dumps(item[k])
                    if k == "component_ids"
                    else item[k]
                    for k in fields
                })

        summary = result["summary"]

        contract = {
            "schema": "bearing_app_contract_v3",
            "schema_version": 3,
            "source_volume": VOLUME,
            "source_scenario": scenario,
            "prediction_source": "QC-FS CNN V3",
            "model_checkpoint_epoch": int(
                state["epoch"]
            ),
            "cpt_count": count,
            "sampling_pattern": "distributed",
            "grid": {
                "shape_xyz": list(q.shape),
                "spacing_m": [
                    dx, dy, dz
                ],
                "array_order": "X,Y,Z",
                "memory_layout": (
                    "C-contiguous, Z fastest"
                ),
                "voxel_origin": (
                    "grid cell boundary"
                ),
                "depth_positive_down": True,
                "depth_reference": (
                    "top of synthetic grid"
                ),
                "flat_index_formula": (
                    "(x * ny + y) * nz + z"
                ),
            },
            "binary_files": {
                filename: {
                    "dtype": dtype,
                    "shape": (
                        list(q.shape)
                        if array.ndim == 3
                        else [q.shape[0], q.shape[1]]
                    ),
                }
                for filename, (
                    array, dtype
                ) in binary_fields.items()
            },
            "cpt_sites_xy_index": [
                [int(x), int(y)]
                for x, y in sites
            ],
            "default_criteria": {
                "qc_min_mpa": 15.0,
                "min_thickness_m": 2.0,
                "min_lateral_fraction": 0.05,
                "ic_filter_enabled": False,
            },
            "depth_summary": {
                "status": summary["status"],
                "shallowest_top_depth_m":
                    optional_float(
                        summary[
                            "shallowest_top_depth_m"
                        ]
                    ),
                "deepest_bottom_depth_m":
                    optional_float(
                        summary[
                            "deepest_bottom_depth_m"
                        ]
                    ),
                "max_contiguous_thickness_m":
                    optional_float(
                        summary[
                            "max_contiguous_thickness_m"
                        ]
                    ),
                "candidate_voxels": int(
                    summary["candidate_voxels"]
                ),
                "component_count": int(
                    summary["component_count"]
                ),
                "interval_count": int(
                    summary["column_interval_count"]
                ),
            },
            "warnings": [
                "Screening candidate only.",
                "Not a bearing-capacity estimate.",
                "No candidate detected is not proof "
                "that no suitable stratum exists.",
                "CPT distance is not calibrated "
                "uncertainty.",
                "Ic rule is not validated.",
            ],
        }

        (folder / "contract.json").write_text(
            json.dumps(
                contract,
                indent=2,
                allow_nan=False
            )
        )

        # Test changing sliders without new
        # CNN inference: reuse the saved QC volume.
        reload_qc = np.fromfile(
            folder / "qc_xyz.f32",
            dtype="<f4"
        ).reshape(q.shape)

        reload_valid = np.fromfile(
            folder / "valid_xyz.u8",
            dtype="u1"
        ).reshape(q.shape).astype(bool)

        for threshold, thickness in (
            (12., 2.),
            (15., 2.),
            (15., 4.),
            (17., 2.),
        ):
            recalculated = detect_candidates(
                qc=reload_qc,
                valid=reload_valid,
                spacing_m=spacing,
                qc_min_mpa=threshold,
                min_thickness_m=thickness,
                min_lateral_fraction=0.05,
            )

            if (
                threshold == 15.
                and thickness == 2.
                and not np.array_equal(
                    recalculated["mask"],
                    old_mask
                )
            ):
                raise RuntimeError(
                    "Slider recomputation mismatch"
                )

        print(f"\n=== CPT {count} ===")
        print("V2/V3 mask parity: PASS")
        print("Binary roundtrip: PASS")
        print("Slider recomputation: PASS")
        print(
            "Candidate voxels:",
            summary["candidate_voxels"]
        )
        print(
            "Components:",
            summary["component_count"]
        )
        print(
            "Depth intervals:",
            summary["column_interval_count"]
        )
        print("Contract:", folder / "contract.json")

print("\n=== APP CONTRACT V3 COMPLETE ===")
print("Exported:", OUTPUT)
print("No training performed.")
print("Frozen models unchanged.")
print("Independent test untouched.")
