from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path.home() / "Developer" / "cpt_super_learn"

INPUT = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "single_cpt_by_grid"
    / "per_volume_bins.csv"
)

SEED = 67
N_BOOT = 2000

TOLERANCE_METRIC = "sgan_frac_le_015"
REQUIRED_COVERAGE = 0.90


def bootstrap_ci(values, rng):
    values = np.asarray(values, dtype=float)

    mean = float(values.mean())

    if len(values) <= 1:
        return mean, mean, mean

    samples = rng.choice(
        values,
        size=(N_BOOT, len(values)),
        replace=True,
    )

    means = samples.mean(axis=1)

    lo, hi = np.quantile(
        means,
        [0.025, 0.975],
    )

    return mean, float(lo), float(hi)


with INPUT.open() as f:
    rows = list(csv.DictReader(f))


groups = defaultdict(list)

for row in rows:

    key = (
        row["grid"],
        row["family"],
        float(row["distance_from_m"]),
        float(row["distance_to_m"]),
    )

    groups[key].append(
        float(row[TOLERANCE_METRIC])
    )


rng = np.random.default_rng(SEED)

aggregated = defaultdict(list)

for (
    grid,
    family,
    lo,
    hi,
), values in groups.items():

    mean, ci_lo, ci_hi = bootstrap_ci(
        values,
        rng,
    )

    aggregated[
        (grid, family)
    ].append({
        "lo": lo,
        "hi": hi,
        "mean": mean,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n": len(values),
    })


print(
    "=== SR-02 GEOLOGICAL FAMILY ANALYSIS ==="
)

print(
    "Criterion: |error Ic| <= 0.15 "
    "for >= 90% voxels"
)

print()


for key in sorted(aggregated):

    grid, family = key

    bins = sorted(
        aggregated[key],
        key=lambda x: x["lo"],
    )

    mean_frontier = None
    ci_frontier = None

    mean_failure = None
    ci_failure = None

    for b in bins:

        if mean_failure is None:
            if b["mean"] >= REQUIRED_COVERAGE:
                mean_frontier = b
            else:
                mean_failure = b

        if ci_failure is None:
            if b["ci_lo"] >= REQUIRED_COVERAGE:
                ci_frontier = b
            else:
                ci_failure = b

    print("=" * 65)
    print(grid)
    print("Family:", family)

    if mean_frontier is None:

        first = bins[0]

        print(
            "Mean support : unresolved / no passing bin"
        )

        print(
            f"First bin   : "
            f"{first['lo']:.0f}-"
            f"{first['hi']:.0f} m "
            f"({first['mean']*100:.1f}%)"
        )

    else:

        print(
            f"Mean support : through "
            f"{mean_frontier['lo']:.0f}-"
            f"{mean_frontier['hi']:.0f} m"
        )

        print(
            f"Coverage     : "
            f"{mean_frontier['mean']*100:.1f}%"
        )

        if mean_failure:
            print(
                f"Next fails   : "
                f"{mean_failure['lo']:.0f}-"
                f"{mean_failure['hi']:.0f} m"
            )

    if ci_frontier is None:

        first = bins[0]

        print(
            "CI95 support : unresolved / no passing bin"
        )

        print(
            f"CI95 first   : "
            f"{first['ci_lo']*100:.1f}% "
            f"lower bound"
        )

    else:

        print(
            f"CI95 support : through "
            f"{ci_frontier['lo']:.0f}-"
            f"{ci_frontier['hi']:.0f} m"
        )

        print(
            f"CI95 range   : "
            f"{ci_frontier['ci_lo']*100:.1f}"
            f"-"
            f"{ci_frontier['ci_hi']*100:.1f}%"
        )

    print()


