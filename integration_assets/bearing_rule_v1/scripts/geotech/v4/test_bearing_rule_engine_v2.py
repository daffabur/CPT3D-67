from pathlib import Path
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
    "rule_engine_v2_demo"
)

if OUT.exists():
    raise SystemExit(
        f"Output already exists: {OUT}"
    )

sys.path.insert(0, str(BASE.resolve()))

from bearing_rule_engine_v2 import (
    spatial_output,
    inspect_xy,
)

# Reuse previously validated CNN V3 loading.
tree = ast.parse(SOURCE.read_text())

cuts = [
    i for i, node in enumerate(tree.body)
    if isinstance(node, ast.With)
]
if len(cuts) != 1:
    raise RuntimeError(
        "Unexpected QC verification script structure"
    )

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

# Smoke test: two distinct intervals in one column.
q_test = np.zeros(
    (12, 12, 10), dtype=np.float32
)
v_test = np.ones_like(q_test, dtype=bool)

q_test[2:10, 2:10, 1:4] = 16.0
q_test[2:10, 2:10, 6:9] = 16.0

smoke = spatial_output(
    q_test,
    v_test,
    (2., 2., 1.),
    cpt_sites=[(3, 3)],
    qc_min_mpa=15.,
    min_thickness_m=2.,
)

column = inspect_xy(smoke, 4, 4)

assert len(column["intervals"]) == 2
assert column["intervals"][0]["top_depth_m"] == 1.
assert column["intervals"][0]["bottom_depth_m"] == 4.
assert column["intervals"][1]["top_depth_m"] == 6.
assert column["intervals"][1]["bottom_depth_m"] == 9.

print("=== BEARING RULE ENGINE V2 ===")
print("Synthetic two-layer smoke test: PASS")

# Demonstrate on one fixed validation volume.
name = files[0]
q, f, valid, spacing, scenario = (
    load_volume(DATA / name)
)

print("Demo volume:", name)
print("Scenario:", scenario)
print("Device:", DEVICE)
print("Threshold: qc >=15 MPa")
print("Thickness: >=2m")
print("Continuity: 5%")
print("Ic filtering: disabled (not yet validated)")

OUT.mkdir(parents=True)

with torch.inference_mode():
    for count in (8, 32):
        sites = get_sites(
            valid,
            spacing,
            name,
            "distributed",
            epoch=-1
        )[:count]

        batch = prepare(
            q, f, valid, spacing, sites
        )

        pq, _, _, _ = predict(
            model, batch
        )

        qc_pred = (
            pq.detach().cpu().numpy()
            .transpose(2, 1, 0)
        )

        result = spatial_output(
            qc=qc_pred,
            valid=valid,
            spacing_m=spacing,
            cpt_sites=sites,
            qc_min_mpa=15.,
            min_thickness_m=2.,
            min_lateral_fraction=0.05,
        )

        # Verify V2 preserves V1 candidate mask.
        from bearing_rule_engine_v1 import (
            detect_candidates
        )

        ref = detect_candidates(
            qc_pred,
            valid,
            spacing,
            qc_min_mpa=15.,
            min_thickness_m=2.,
            min_lateral_fraction=0.05,
        )["mask"]

        assert np.array_equal(
            result["mask"], ref
        )

        folder = OUT / f"cpt_{count}"
        folder.mkdir()

        np.savez_compressed(
            folder / "spatial_output.npz",
            candidate_mask_xyz=(
                result["mask"].astype(np.uint8)
            ),
            component_ids_xyz=(
                result["component_ids"].astype(
                    np.int32
                )
            ),
            top_depth_m_xy=(
                result["top_depth_m"]
            ),
            bottom_depth_m_xy=(
                result["bottom_depth_m"]
            ),
            total_thickness_m_xy=(
                result["total_thickness_m"]
            ),
            interval_count_xy=(
                result["interval_count"]
            ),
            nearest_cpt_distance_m_xy=(
                result["nearest_cpt_distance_m"]
            ),
            cpt_sites_xy=np.asarray(
                sites, dtype=np.int32
            ),
            spacing_m=np.asarray(
                spacing, dtype=np.float32
            ),
        )

        metadata = {
            "source_volume": name,
            "source_scenario": scenario,
            "prediction_source": (
                "QC-FS CNN V3 frozen"
            ),
            "sampling": "distributed",
            "summary": result["summary"],
            "components": result["components"],
            "example_marker": inspect_xy(
                result,
                int(sites[0][0]),
                int(sites[0][1])
            ),
        }

        (folder / "summary.json").write_text(
            json.dumps(
                metadata,
                indent=2,
                allow_nan=False
            )
        )

        interval_fields = [
            "x_index",
            "y_index",
            "x_center_m",
            "y_center_m",
            "top_depth_m",
            "bottom_depth_m",
            "thickness_m",
            "component_ids",
            "nearest_cpt_m",
        ]

        with (
            folder / "depth_intervals.csv"
        ).open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=interval_fields
            )
            writer.writeheader()

            for interval in result["intervals"]:
                row = dict(interval)
                row["component_ids"] = json.dumps(
                    row["component_ids"]
                )
                writer.writerow(row)

        s = result["summary"]

        print(f"\n=== {count} CPT ===")
        print("Status:", s["status"])
        print(
            "Candidate voxels:",
            s["candidate_voxels"]
        )
        print(
            "Candidate columns:",
            s["candidate_columns"]
        )
        print(
            "Connected components:",
            s["component_count"]
        )
        print(
            "Depth intervals:",
            s["column_interval_count"]
        )
        print(
            "Shallowest top:",
            s["shallowest_top_depth_m"], "m"
        )
        print(
            "Deepest bottom:",
            s["deepest_bottom_depth_m"], "m"
        )
        print(
            "Max contiguous thickness:",
            s["max_contiguous_thickness_m"], "m"
        )
        print("Output:", folder)

print("\n=== RULE ENGINE V2 COMPLETE ===")
print("3D output generated successfully.")
print("Frozen checkpoints unchanged.")
print("Independent test untouched.")
