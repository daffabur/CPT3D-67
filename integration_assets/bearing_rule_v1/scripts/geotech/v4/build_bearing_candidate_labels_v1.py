from pathlib import Path
from collections import defaultdict
import csv
import json
import math

import numpy as np
from scipy import ndimage as ndi

ROOT = Path.cwd()

DATA = ROOT / (
    "data/synthetic3d_v4_geosyn_qcfs/"
    "joint_v3_expanded_1200"
)
OUT = ROOT / (
    "results/bearing_layer_v1/"
    "candidate_labels_v1"
)

QC_THRESHOLDS = (10., 12., 15.)
MIN_THICKNESS_M = (2., 4., 6.)
MIN_LATERAL_FRACTION = 0.05

if OUT.exists():
    raise SystemExit(
        f"Output exists; refusing overwrite: {OUT}"
    )

files = sorted(DATA.glob("*.npz"))
if len(files) != 1200:
    raise SystemExit(
        f"Expected 1200 volumes, found {len(files)}"
    )

# 6-neighbour connectivity in XYZ.
STRUCT_3D = ndi.generate_binary_structure(3, 1)

# Only consecutive Z voxels form vertical runs.
STRUCT_Z = np.zeros((3, 3, 3), dtype=np.uint8)
STRUCT_Z[1, 1, :] = 1


def build_candidate(q, valid, threshold, min_cells):
    high = valid & np.isfinite(q) & (q >= threshold)

    vertical_ids, n_vertical = ndi.label(
        high, structure=STRUCT_Z
    )

    lengths = np.bincount(
        vertical_ids.ravel(),
        minlength=n_vertical + 1
    )
    lengths[0] = 0

    thick = high & (
        lengths[vertical_ids] >= min_cells
    )

    connected_ids, n_components = ndi.label(
        thick, structure=STRUCT_3D
    )

    nx, ny, nz = q.shape
    min_columns = max(
        1, math.ceil(
            MIN_LATERAL_FRACTION * nx * ny
        )
    )

    accepted_ids = np.zeros(
        n_components + 1, dtype=bool
    )

    objects = ndi.find_objects(connected_ids)

    kept_components = 0

    for component_id, slices in enumerate(objects, 1):
        if slices is None:
            continue

        local = (
            connected_ids[slices] == component_id
        )

        lateral_columns = int(
            np.count_nonzero(
                np.any(local, axis=2)
            )
        )

        if lateral_columns >= min_columns:
            accepted_ids[component_id] = True
            kept_components += 1

    accepted = (
        thick & accepted_ids[connected_ids]
    )

    return accepted, kept_components, min_columns


OUT.mkdir(parents=True)
(OUT / "masks").mkdir()

rows = []
scenario_summary = defaultdict(
    lambda: defaultdict(list)
)

print("=== BEARING CANDIDATE LABELS V1 ===")
print("Volumes:", len(files))
print("QC thresholds:", QC_THRESHOLDS)
print("Thicknesses (m):", MIN_THICKNESS_M)
print("Lateral footprint minimum: 5%")
print("Labels are screening indicators only.")
print("No checkpoint or test-set modifications.")

for i, path in enumerate(files, 1):
    with np.load(path, allow_pickle=False) as d:
        q = np.asarray(
            d["qc_mpa"], dtype=np.float32
        )
        valid = np.asarray(
            d["valid_mask"], dtype=bool
        )
        meta_raw = d["metadata_json"].item()

    if isinstance(meta_raw, bytes):
        meta_raw = meta_raw.decode()

    meta = json.loads(str(meta_raw))
    grid = meta["grid"]
    scenario = meta["qcfs_extension"]["scenario"]

    nx = int(grid["nx"])
    ny = int(grid["ny"])
    nz = int(grid["nz"])

    dz = float(grid["dz_m"])

    if q.shape != (nx, ny, nz):
        raise RuntimeError(
            f"XYZ grid mismatch: {path.name}"
        )
    if q.shape != valid.shape:
        raise RuntimeError(
            f"Mask shape mismatch: {path.name}"
        )
    if not np.isfinite(dz) or dz <= 0:
        raise RuntimeError(
            f"Invalid dz: {path.name}"
        )
    if meta.get("split") != "train":
        raise RuntimeError(
            f"Unexpected source split: {path.name}"
        )

    valid_count = int(valid.sum())
    if valid_count == 0:
        raise RuntimeError(
            f"No valid voxels: {path.name}"
        )

    outputs = {}
    row = {
        "file": path.name,
        "scenario": scenario,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx_m": float(grid["dx_m"]),
        "dy_m": float(grid["dy_m"]),
        "dz_m": dz,
        "valid_voxels": valid_count,
    }

    for qc_threshold in QC_THRESHOLDS:
        for thickness in MIN_THICKNESS_M:
            cells = max(
                1, math.ceil(
                    thickness / dz - 1e-9
                )
            )

            key = (
                f"q{int(qc_threshold)}"
                f"_t{int(thickness)}m"
            )

            mask, components, min_columns = (
                build_candidate(
                    q, valid,
                    qc_threshold, cells
                )
            )

            outputs[key] = mask

            fraction = (
                float(mask.sum()) / valid_count
            )

            row[f"{key}_fraction"] = fraction
            row[f"{key}_components"] = components
            row[f"{key}_min_cells"] = cells

            scenario_summary[scenario][key].append(
                fraction
            )

    destination = (
        OUT / "masks" / path.name
    )

    np.savez_compressed(
        destination,
        **outputs
    )

    rows.append(row)

    if i % 200 == 0:
        print(
            f"Processed {i}/{len(files)}",
            flush=True
        )


with (OUT / "per_volume.csv").open(
    "w", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0])
    )
    writer.writeheader()
    writer.writerows(rows)


report = {
    "experiment": "bearing_candidate_labels_v1",
    "n_volumes": len(files),
    "source_data": str(DATA),
    "qc_thresholds_mpa": QC_THRESHOLDS,
    "minimum_thicknesses_m": MIN_THICKNESS_M,
    "minimum_lateral_fraction": MIN_LATERAL_FRACTION,
    "connectivity": "3D 6-neighbour",
    "array_axis_order": "X,Y,Z",
    "depth_direction": (
        "Not used: only vertical run thickness "
        "is evaluated"
    ),
    "ground_truth_type": (
        "Rule-derived synthetic candidate masks, "
        "not independent bearing-capacity labels"
    ),
    "test_data_used": False,
    "limitations": [
        "No foundation type or design loading",
        "No settlement assessment",
        "No underlying weak-layer check",
        "No groundwater calculation",
        "No field calibration",
        "Five-percent footprint is experimental",
        "Do not interpret negative mask as unsafe",
        "Do not interpret positive mask as suitable"
    ]
}

(OUT / "manifest.json").write_text(
    json.dumps(report, indent=2)
)

print("\n=== CANDIDATE LABEL SUMMARY ===")

for scenario in sorted(scenario_summary):
    print(f"\n{scenario}")

    for threshold in QC_THRESHOLDS:
        for thickness in MIN_THICKNESS_M:
            key = (
                f"q{int(threshold)}"
                f"_t{int(thickness)}m"
            )
            vals = scenario_summary[
                scenario
            ][key]

            print(
                f"  qc>={threshold:.0f} MPa, "
                f"thickness>={thickness:.0f}m: "
                f"{100*np.mean(vals):.3f}% "
                "mean candidate voxels"
            )

print("\n=== LABEL GENERATION COMPLETE ===")
print("Output:", OUT)
print("Masks:", OUT / "masks")
print("Summary:", OUT / "per_volume.csv")
print("Manifest:", OUT / "manifest.json")
print("All existing models unchanged.")
