from pathlib import Path
from collections import defaultdict
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

EXTERNAL_MANIFEST = (
    ROOT
    / "data"
    / "synthetic3d_v4_geosyn"
    / "combined_gate_validation_100"
    / "manifest.json"
)

OUT_ROOT = (
    ROOT
    / "data"
    / "synthetic3d_v4_geosyn"
    / "v3_gate_calibration_100"
)

OUT_TEST = OUT_ROOT / "test"

SEED = 6721
N_PER_GRID = 20

# Uncertainty V3 used:
# train 164-175
# calibration 176-181
# evaluation 182-187
V3_USED = {
    f"v4_test_{i:05d}.npz"
    for i in range(164, 188)
}


def read_meta(path):
    with np.load(
        path,
        allow_pickle=False,
    ) as d:
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
# External set must stay completely untouched.
# ============================================================

external = json.loads(
    EXTERNAL_MANIFEST.read_text()
)

EXTERNAL_USED = {
    v["volume"]
    for v in external["volumes"]
}


# ============================================================
# We need family labels.
#
# Use original full dataset metadata if available.
# If family is not stored in metadata, infer it from unit metadata
# fields used by this GeoSyn dataset.
# ============================================================

def get_family(meta):
    # Direct field first.
    if "family" in meta:
        return str(meta["family"])

    # Some generators store it under geology / generator metadata.
    for key in [
        "geology_family",
        "structure_family",
        "scenario_family",
    ]:
        if key in meta:
            return str(meta[key])

    # Nested possibilities.
    for parent in [
        "geology",
        "generator",
        "scenario",
    ]:
        obj = meta.get(parent)

        if isinstance(obj, dict):
            for key in [
                "family",
                "geology_family",
                "structure_family",
                "scenario_family",
            ]:
                if key in obj:
                    return str(obj[key])

    raise RuntimeError(
        "Family label tidak ditemukan di metadata. "
        f"Available metadata keys: {list(meta.keys())}"
    )


groups = defaultdict(
    lambda: defaultdict(list)
)


for path in sorted(
    FULL_TEST.glob("*.npz")
):

    if path.name in V3_USED:
        continue

    if path.name in EXTERNAL_USED:
        continue

    meta = read_meta(path)

    grid = grid_key(meta)
    family = get_family(meta)

    groups[grid][family].append(
        (path, meta)
    )


print("=== AVAILABLE CLEAN POOL ===")

for grid in sorted(groups):

    print()
    print(grid)

    for family in sorted(
        groups[grid]
    ):
        print(
            f"  {family}: "
            f"{len(groups[grid][family])}"
        )


assert len(groups) == 5, (
    f"Expected 5 grids, got {len(groups)}"
)


rng = np.random.default_rng(
    SEED
)

selected = []


# ============================================================
# Balanced 20/grid.
#
# With three families:
# 7 + 7 + 6 = 20
# ============================================================

for grid in sorted(groups):

    families = sorted(
        groups[grid]
    )

    if len(families) != 3:
        raise RuntimeError(
            f"{grid}: expected 3 families, "
            f"found {families}"
        )

    base = (
        N_PER_GRID
        // len(families)
    )

    remainder = (
        N_PER_GRID
        % len(families)
    )

    selected_grid = []

    for i, family in enumerate(
        families
    ):

        quota = (
            base
            + (
                1
                if i < remainder
                else 0
            )
        )

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

            path, meta = pool[
                int(idx)
            ]

            selected_grid.append({
                "grid": grid,
                "family": family,
                "volume": path.name,
                "path": path,
            })

    assert (
        len(selected_grid)
        == N_PER_GRID
    )

    selected.extend(
        selected_grid
    )


assert len(selected) == 100


# ============================================================
# Leakage checks.
# ============================================================

names = {
    x["volume"]
    for x in selected
}

assert not (
    names & V3_USED
)

assert not (
    names & EXTERNAL_USED
)


# ============================================================
# Create symlink dataset.
# ============================================================

OUT_TEST.mkdir(
    parents=True,
    exist_ok=True,
)

for old in OUT_TEST.glob("*.npz"):
    old.unlink()

for item in selected:

    src = item["path"]

    os.symlink(
        src.resolve(),
        OUT_TEST / src.name,
    )


# ============================================================
# Manifest.
# ============================================================

manifest = {
    "study":
        "v3-gate-calibration",
    "seed":
        SEED,
    "total":
        100,
    "n_per_grid":
        N_PER_GRID,
    "overlap_with_v3_training":
        0,
    "overlap_with_external":
        0,
    "volumes": [],
}

eval_groups = {}

for item in selected:

    manifest[
        "volumes"
    ].append({
        "grid":
            item["grid"],
        "family":
            item["family"],
        "volume":
            item["volume"],
    })

    eval_groups.setdefault(
        item["grid"],
        [],
    ).append({
        "file":
            item["volume"],
        "family":
            item["family"],
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
                "v3-gate-calibration",
            "seed":
                SEED,
            "groups":
                eval_groups,
        },
        indent=2,
    )
    + "\n"
)


print()
print(
    "=== V3 GATE CALIBRATION SET ==="
)

for grid in sorted(
    eval_groups
):

    counts = defaultdict(int)

    for x in eval_groups[
        grid
    ]:
        counts[
            x["family"]
        ] += 1

    print()
    print(grid)

    for family in sorted(counts):
        print(
            f"  {family}: "
            f"{counts[family]}"
        )

    print(
        "  TOTAL:",
        sum(counts.values()),
    )


print()
print(
    "Total:",
    len(selected),
)

print(
    "Overlap V3:",
    len(names & V3_USED),
)

print(
    "Overlap external:",
    len(names & EXTERNAL_USED),
)
