from pathlib import Path
from collections import defaultdict
import json
import os
import numpy as np

ROOT = Path.home() / "Developer" / "cpt_super_learn"

BASE = (
    ROOT
    / "data"
    / "synthetic3d_v4_geosyn"
)

FULL_TEST = BASE / "full_24k" / "test"

CAL_MANIFEST = (
    BASE
    / "v3_gate_calibration_100"
    / "manifest.json"
)

DEV_MANIFEST = (
    BASE
    / "combined_gate_validation_100"
    / "manifest.json"
)

OUT_ROOT = (
    BASE
    / "v3_1_final_validation_100"
)

OUT_TEST = OUT_ROOT / "test"

SEED = 6731
N_PER_GRID = 20

V3_USED = {
    f"v4_test_{i:05d}.npz"
    for i in range(164, 188)
}


def load_manifest(path):
    d = json.loads(path.read_text())

    return {
        x["volume"]
        for x in d["volumes"]
    }


CAL_USED = load_manifest(
    CAL_MANIFEST
)

DEV_USED = load_manifest(
    DEV_MANIFEST
)

OLD_FINAL_USED = load_manifest(
    BASE
    / "v3_gate_final_validation_100"
    / "manifest.json"
)

QSCORE_USED = load_manifest(
    BASE
    / "qscore_validation_100"
    / "manifest.json"
)


def read_meta(path):
    with np.load(
        path,
        allow_pickle=False,
    ) as d:

        raw = d["metadata_json"]

        if isinstance(raw, np.ndarray):
            raw = raw.item()

        if isinstance(raw, bytes):
            raw = raw.decode()

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


def get_family(meta):
    if "family" in meta:
        return str(meta["family"])

    for key in [
        "geology_family",
        "structure_family",
        "scenario_family",
    ]:
        if key in meta:
            return str(meta[key])

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
        "Family label not found."
    )


groups = defaultdict(
    lambda: defaultdict(list)
)


for path in sorted(
    FULL_TEST.glob("*.npz")
):

    if path.name in V3_USED:
        continue

    if path.name in CAL_USED:
        continue

    if path.name in DEV_USED:
        continue

    if path.name in OLD_FINAL_USED:
        continue

    if path.name in QSCORE_USED:
        continue

    meta = read_meta(path)

    groups[
        grid_key(meta)
    ][
        get_family(meta)
    ].append(path)


assert len(groups) == 5

rng = np.random.default_rng(SEED)

selected = []


for grid in sorted(groups):

    families = sorted(
        groups[grid]
    )

    assert len(families) == 3

    base = N_PER_GRID // 3
    remainder = N_PER_GRID % 3

    grid_selected = []

    for i, family in enumerate(
        families
    ):

        quota = (
            base
            + (1 if i < remainder else 0)
        )

        pool = groups[grid][family]

        if len(pool) < quota:
            raise RuntimeError(
                f"{grid}/{family}: "
                f"need {quota}, have {len(pool)}"
            )

        ids = rng.choice(
            len(pool),
            size=quota,
            replace=False,
        )

        for idx in ids:

            path = pool[int(idx)]

            grid_selected.append({
                "grid": grid,
                "family": family,
                "volume": path.name,
                "path": path,
            })

    assert len(grid_selected) == 20

    selected.extend(
        grid_selected
    )


assert len(selected) == 100


names = {
    x["volume"]
    for x in selected
}

assert not (names & V3_USED)
assert not (names & CAL_USED)
assert not (names & DEV_USED)
assert not (names & OLD_FINAL_USED)
assert not (names & QSCORE_USED)


OUT_TEST.mkdir(
    parents=True,
    exist_ok=True,
)

for old in OUT_TEST.glob("*.npz"):
    old.unlink()


for item in selected:

    os.symlink(
        item["path"].resolve(),
        OUT_TEST / item["volume"],
    )


manifest = {
    "study":
        "v3-gate-final-validation",
    "seed":
        SEED,
    "total":
        100,
    "volumes": [],
}

groups_out = {}


for item in selected:

    manifest["volumes"].append({
        "grid": item["grid"],
        "family": item["family"],
        "volume": item["volume"],
    })

    groups_out.setdefault(
        item["grid"],
        [],
    ).append({
        "file": item["volume"],
        "family": item["family"],
    })


(OUT_ROOT / "manifest.json").write_text(
    json.dumps(
        manifest,
        indent=2,
    ) + "\n"
)

(
    OUT_ROOT
    / "evaluator_manifest.json"
).write_text(
    json.dumps(
        {
            "study":
                "v3-gate-final-validation",
            "seed":
                SEED,
            "groups":
                groups_out,
        },
        indent=2,
    ) + "\n"
)


print(
    "=== FINAL V3 VALIDATION SET ==="
)

for grid in sorted(
    groups_out
):

    counts = defaultdict(int)

    for item in groups_out[grid]:
        counts[
            item["family"]
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
print("Total:", len(selected))
print(
    "Overlap V3:",
    len(names & V3_USED),
)
print(
    "Overlap calibration:",
    len(names & CAL_USED),
)
print(
    "Overlap development:",
    len(names & DEV_USED),
)
print(
    "Overlap old final:",
    len(names & OLD_FINAL_USED),
)
print(
    "Overlap qscore:",
    len(names & QSCORE_USED),
)
