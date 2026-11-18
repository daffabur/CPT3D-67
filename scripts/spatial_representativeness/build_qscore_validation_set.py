from pathlib import Path
import csv
import json
import os

import numpy as np


ROOT = Path.home() / "Developer" / "cpt_super_learn"

FULL_TEST = (
    ROOT
    / "data"
    / "synthetic3d_v4_geosyn"
    / "full_24k"
    / "test"
)

OLD_RESULTS = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "dqn_v46a_vs_farthest"
    / "per_volume_step.csv"
)

OUT_ROOT = (
    ROOT
    / "data"
    / "synthetic3d_v4_geosyn"
    / "qscore_validation_100"
)

OUT_TEST = OUT_ROOT / "test"

SEED = 6702
N_PER_GRID = 20


# ------------------------------------------------------------
# Volumes already used by SR-04 V4.6-A
# ------------------------------------------------------------

with OLD_RESULTS.open() as f:
    rows = list(csv.DictReader(f))

used = {
    Path(r["volume"]).name
    for r in rows
}

used_stems = {
    Path(x).stem
    for x in used
}

print("Previously used volumes:", len(used_stems))


# ------------------------------------------------------------
# Grid identifier
# ------------------------------------------------------------

def grid_key(path: Path):
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["metadata_json"]))

    g = meta["grid"]

    return (
        f'{int(g["nx"])}x'
        f'{int(g["ny"])}x'
        f'{int(g["nz"])}_r'
        f'{g["dx_m"]:g}x'
        f'{g["dy_m"]:g}x'
        f'{g["dz_m"]:g}m'
    )


# ------------------------------------------------------------
# Collect unseen volumes
# ------------------------------------------------------------

groups = {}

for path in sorted(FULL_TEST.glob("*.npz")):

    if path.stem in used_stems:
        continue

    key = grid_key(path)
    groups.setdefault(key, []).append(path)


print()
print("Available unseen volumes:")

for key, paths in sorted(groups.items()):
    print(f"  {key}: {len(paths)}")


assert len(groups) == 5, (
    f"Expected 5 grid configurations, found {len(groups)}"
)


# ------------------------------------------------------------
# Deterministic sample: 20 / grid
# ------------------------------------------------------------

rng = np.random.default_rng(SEED)

selected = []

for key in sorted(groups):

    paths = groups[key]

    assert len(paths) >= N_PER_GRID

    idx = rng.choice(
        len(paths),
        size=N_PER_GRID,
        replace=False,
    )

    picked = [
        paths[int(i)]
        for i in idx
    ]

    selected.extend(
        (key, p)
        for p in picked
    )


assert len(selected) == 100


# ------------------------------------------------------------
# Create isolated test folder using symlinks
# ------------------------------------------------------------

OUT_TEST.mkdir(
    parents=True,
    exist_ok=True,
)

for old in OUT_TEST.glob("*.npz"):
    old.unlink()

for key, src in selected:

    dst = OUT_TEST / src.name

    os.symlink(
        src.resolve(),
        dst,
    )


# ------------------------------------------------------------
# Verify no overlap
# ------------------------------------------------------------

new_stems = {
    p.stem
    for _, p in selected
}

overlap = new_stems & used_stems

assert not overlap, (
    f"ERROR: overlap detected: {sorted(overlap)}"
)


# ------------------------------------------------------------
# Save manifest
# ------------------------------------------------------------

manifest = {
    "seed": SEED,
    "n_per_grid": N_PER_GRID,
    "total": len(selected),
    "overlap_with_sr04_v46a": 0,
    "volumes": [
        {
            "grid": key,
            "volume": p.name,
        }
        for key, p in selected
    ],
}

(OUT_ROOT / "manifest.json").write_text(
    json.dumps(
        manifest,
        indent=2,
    )
    + "\n"
)

print()
print("=== VALIDATION SET READY ===")
print("Total:", len(selected))
print("Overlap with calibration set:", len(overlap))

for key in sorted(groups):
    n = sum(
        1
        for k, _ in selected
        if k == key
    )
    print(f"{key}: {n}")

print()
print("Saved:")
print(OUT_ROOT / "manifest.json")
