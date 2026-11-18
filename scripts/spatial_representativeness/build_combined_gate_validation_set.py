from pathlib import Path
from collections import defaultdict
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

OLD_SR04 = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "dqn_v46a_vs_farthest"
    / "per_volume_step.csv"
)

OLD_QVALID = (
    ROOT
    / "data"
    / "synthetic3d_v4_geosyn"
    / "qscore_validation_100"
    / "manifest.json"
)

OUT_ROOT = (
    ROOT
    / "data"
    / "synthetic3d_v4_geosyn"
    / "combined_gate_validation_100"
)

OUT_TEST = OUT_ROOT / "test"

SEED = 6713
N_PER_GRID = 20


def read_meta(path):
    with np.load(path, allow_pickle=False) as d:
        raw = d["metadata_json"]

        if isinstance(raw, np.ndarray):
            raw = raw.item()

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

        return json.loads(str(raw))


def grid_key(meta):
    g = meta["grid"]

    return (
        f'{int(g["nx"])}x'
        f'{int(g["ny"])}x'
        f'{int(g["nz"])}'
        f'_r{float(g["dx_m"]):g}x'
        f'{float(g["dy_m"]):g}x'
        f'{float(g["dz_m"]):g}m'
    )


# ============================================================
# Exclude previous two 100-volume sets
# ============================================================

with OLD_SR04.open() as f:
    rows = list(csv.DictReader(f))

used_1 = {
    Path(r["volume"]).stem
    for r in rows
}

old_q = json.loads(
    OLD_QVALID.read_text()
)

used_2 = {
    Path(v["volume"]).stem
    for v in old_q["volumes"]
}

excluded = used_1 | used_2

print("Old SR-04 volumes :", len(used_1))
print("Old validation    :", len(used_2))
print("Total excluded    :", len(excluded))


# ============================================================
# Group remaining test volumes by grid AND geological family
# ============================================================

groups = defaultdict(
    lambda: defaultdict(list)
)

for path in sorted(
    FULL_TEST.glob("*.npz")
):

    if path.stem in excluded:
        continue

    meta = read_meta(path)

    grid = grid_key(meta)
    family = meta.get(
        "family",
        "unknown",
    )

    groups[grid][family].append(
        (path, meta)
    )


assert len(groups) == 5, (
    f"Expected 5 grids, got {len(groups)}"
)

rng = np.random.default_rng(SEED)

selected = []


# ============================================================
# Balanced selection exactly like evaluator concept:
# 20 volumes/grid distributed across families.
# ============================================================

for grid in sorted(groups):

    family_groups = groups[grid]
    families = sorted(family_groups)

    print()
    print("GRID:", grid)

    for family in families:
        print(
            " ",
            family,
            ":",
            len(family_groups[family]),
            "available",
        )

    base = N_PER_GRID // len(families)
    remainder = N_PER_GRID % len(families)

    selected_grid = []

    for i, family in enumerate(families):

        quota = (
            base
            + (
                1
                if i < remainder
                else 0
            )
        )

        pool = family_groups[family]

        if len(pool) < quota:
            raise RuntimeError(
                f"{grid}: not enough "
                f"{family} volumes "
                f"(need {quota}, have {len(pool)})"
            )

        idx = rng.choice(
            len(pool),
            size=quota,
            replace=False,
        )

        for j in idx:
            path, meta = pool[int(j)]

            selected_grid.append(
                (
                    grid,
                    family,
                    path,
                    meta,
                )
            )

    assert len(selected_grid) == N_PER_GRID

    selected.extend(
        selected_grid
    )


assert len(selected) == 100


# ============================================================
# Verify zero overlap
# ============================================================

new_stems = {
    p.stem
    for _, _, p, _ in selected
}

overlap = new_stems & excluded

assert len(overlap) == 0


# ============================================================
# Recreate isolated directory
# ============================================================

OUT_TEST.mkdir(
    parents=True,
    exist_ok=True,
)

for old in OUT_TEST.glob("*.npz"):
    old.unlink()

for _, _, src, _ in selected:

    os.symlink(
        src.resolve(),
        OUT_TEST / src.name,
    )


# ============================================================
# Write manifests
# ============================================================

manifest = {
    "seed": SEED,
    "n_per_grid": N_PER_GRID,
    "total": len(selected),
    "overlap_with_previous": 0,
    "volumes": [],
}

eval_groups = {}

for grid, family, path, _ in selected:

    manifest["volumes"].append({
        "grid": grid,
        "volume": path.name,
        "family": family,
    })

    eval_groups.setdefault(
        grid,
        [],
    ).append({
        "file": path.name,
        "family": family,
    })


(OUT_ROOT / "manifest.json").write_text(
    json.dumps(
        manifest,
        indent=2,
    )
    + "\n"
)

(
    OUT_ROOT
    / "evaluator_manifest.json"
).write_text(
    json.dumps(
        {
            "study":
                "combined-gate-external-validation",
            "seed": SEED,
            "groups": eval_groups,
        },
        indent=2,
    )
    + "\n"
)


# ============================================================
# Verify exact family distribution
# ============================================================

print()
print("=== THIRD VALIDATION SET ===")

for grid in sorted(eval_groups):

    entries = eval_groups[grid]

    family_counts = defaultdict(int)

    for entry in entries:
        family_counts[
            entry["family"]
        ] += 1

    print()
    print(grid)

    for family in sorted(family_counts):
        print(
            f"  {family}: "
            f"{family_counts[family]}"
        )

    print(
        "  TOTAL:",
        len(entries),
    )


print()
print("Total:", len(selected))
print("Overlap:", len(overlap))
