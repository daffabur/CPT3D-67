from pathlib import Path
from collections import defaultdict
import csv
import json
import os
import shutil

import numpy as np


ROOT = Path.home() / "Developer" / "cpt_super_learn"

SOURCE = (
    ROOT
    / "data/synthetic3d_v4_geosyn/full_24k/test"
)

DEST = (
    ROOT
    / "data/synthetic3d_v4_geosyn"
    / "v46a_coverage_v1_final_validation_100"
)

SWEEP_CSV = (
    ROOT
    / "results/spatial_representativeness"
    / "v46a_distance_fraction_sweep"
    / "per_volume.csv"
)

SEED = 9251
PER_GRID = 20


# ============================================================
# EXCLUSIONS
# ============================================================

excluded = set()


# Development volumes used to select the 10% DQN rule.
if SWEEP_CSV.exists():
    with SWEEP_CSV.open() as f:
        for row in csv.DictReader(f):
            excluded.add(
                row["volume"]
            )


# Original V3 uncertainty development/calibration/evaluation.
for i in range(164, 188):
    excluded.add(
        f"v4_test_{i:05d}.npz"
    )


# Every previous development/final validation dataset that
# must never enter this fresh final set.
previous_sets = [
    "v3_gate_calibration_100",
    "combined_gate_validation_100",
    "v3_gate_final_validation_100",
    "v3_1_final_validation_100",
    "qscore_validation_100",

    # Current DQN+10% development set.
    "v46a_min10_development_100",

    # Previous final set already inspected.
    "v46a_min10_final_validation_100",
]


for name in previous_sets:
    p = (
        ROOT
        / "data/synthetic3d_v4_geosyn"
        / name
    )

    if not p.exists():
        continue

    for file in p.rglob("*.npz"):
        excluded.add(
            file.name
        )


print(
    "Excluded unique volumes:",
    len(excluded)
)


# ============================================================
# Read remaining GeoSyn candidates.
# ============================================================

groups = defaultdict(
    lambda: defaultdict(list)
)


def grid_label(meta):
    g = meta["grid"]

    def fmt(v):
        v = float(v)

        if v.is_integer():
            return str(int(v))

        return f"{v:g}"

    return (
        f"{int(g['nx'])}x"
        f"{int(g['ny'])}x"
        f"{int(g['nz'])}_r"
        f"{fmt(g['dx_m'])}x"
        f"{fmt(g['dy_m'])}x"
        f"{fmt(g['dz_m'])}m"
    )


for path in sorted(
    SOURCE.glob("*.npz")
):
    if path.name in excluded:
        continue

    with np.load(
        path,
        allow_pickle=False,
    ) as d:
        meta = json.loads(
            str(d["metadata_json"])
        )

    grid = grid_label(meta)

    family = meta.get(
        "family",
        "unknown",
    )

    groups[grid][family].append(
        path
    )


print()
print(
    "=== AVAILABLE AFTER ALL EXCLUSIONS ==="
)

for grid in sorted(groups):

    print(grid)

    for family in sorted(
        groups[grid]
    ):
        print(
            f"  {family}: "
            f"{len(groups[grid][family])}"
        )


# ============================================================
# Select 20 per grid.
#
# Balanced geological families:
# 7 / 7 / 6.
# ============================================================

rng = np.random.default_rng(
    SEED
)

selected = []


for grid in sorted(groups):

    families = sorted(
        groups[grid]
    )

    if len(families) != 3:
        raise RuntimeError(
            f"{grid}: expected exactly "
            f"3 families, got {families}"
        )

    quotas = [
        7,
        7,
        6,
    ]

    grid_selected = []


    for family, quota in zip(
        families,
        quotas,
    ):

        pool = groups[
            grid
        ][family]

        if len(pool) < quota:
            raise RuntimeError(
                f"{grid}/{family}: "
                f"need {quota}, "
                f"have {len(pool)}"
            )

        indices = rng.choice(
            len(pool),
            size=quota,
            replace=False,
        )

        for idx in indices:

            path = pool[
                int(idx)
            ]

            grid_selected.append({
                "file":
                    path.name,

                "grid":
                    grid,

                "family":
                    family,
            })


    assert len(
        grid_selected
    ) == PER_GRID

    selected.extend(
        grid_selected
    )


assert len(selected) == 100


names = [
    row["file"]
    for row in selected
]

assert len(
    set(names)
) == 100


overlap = (
    set(names)
    & excluded
)

if overlap:
    raise RuntimeError(
        "FINAL SET CONTAMINATION: "
        f"{sorted(overlap)}"
    )


# ============================================================
# Create fresh validation set using symlinks.
# ============================================================

if DEST.exists():
    shutil.rmtree(
        DEST
    )


test_dir = (
    DEST
    / "test"
)

test_dir.mkdir(
    parents=True,
    exist_ok=True,
)


for row in selected:

    source = (
        SOURCE
        / row["file"]
    )

    target = (
        test_dir
        / row["file"]
    )

    os.symlink(
        source,
        target,
    )


# ============================================================
# Verify composition.
# ============================================================

grid_counts = defaultdict(int)
family_counts = defaultdict(
    lambda: defaultdict(int)
)


for row in selected:

    grid_counts[
        row["grid"]
    ] += 1

    family_counts[
        row["grid"]
    ][
        row["family"]
    ] += 1


print()
print(
    "=== FRESH FINAL SET ==="
)

print(
    "Selected:",
    len(selected)
)

print(
    "Unique:",
    len(set(names))
)

print(
    "Overlap with excluded:",
    len(overlap)
)


for grid in sorted(
    grid_counts
):

    print()
    print(
        grid,
        ":",
        grid_counts[grid],
    )

    for family in sorted(
        family_counts[grid]
    ):
        print(
            " ",
            family,
            ":",
            family_counts[
                grid
            ][family],
        )


# ============================================================
# Manifest.
# ============================================================

manifest = {
    "name":
        "Coverage V1 fresh untouched "
        "final validation set",

    "seed":
        SEED,

    "volumes":
        100,

    "volumes_per_grid":
        PER_GRID,

    "purpose":
        "Fresh untouched validation of "
        "Coverage V1 sufficiency gate "
        "with frozen DQN V4.6-A "
        "10%-diagonal minimum-distance rule.",

    "candidate_under_test": {
        "reconstruction":
            "SchemaGAN V4.2 Flexible",

        "sampling":
            "DQN V4.6-A",

        "minimum_distance_fraction":
            0.10,

        "coverage_model":
            "Coverage V1 direct voxel-safe estimator",

        "coverage_threshold":
            0.925,

        "persistence":
            5,

        "actual_ic_tolerance":
            0.15,

        "actual_sufficiency_threshold":
            0.90,
    },

    "excluded_file_count":
        len(excluded),

    "overlap_with_excluded":
        0,

    "entries":
        selected,

    "important":
        "This set must not be used "
        "for tuning or model selection.",
}


DEST.mkdir(
    parents=True,
    exist_ok=True,
)


(
    DEST
    / "evaluator_manifest.json"
).write_text(
    json.dumps(
        manifest,
        indent=2,
    )
    + "\n"
)


print()
print(
    "Manifest:",
    DEST
    / "evaluator_manifest.json"
)

print()
print(
    "STATUS: FRESH UNTOUCHED FINAL SET CREATED"
)
