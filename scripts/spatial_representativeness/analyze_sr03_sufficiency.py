from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path.home() / "Developer" / "cpt_super_learn"

INPUT = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "multicpt_support"
    / "aggregate_by_count.csv"
)

REQUIRED_COVERAGE = 0.90


with INPUT.open() as f:
    rows = list(csv.DictReader(f))


groups = defaultdict(list)

for row in rows:
    groups[row["grid"]].append(row)


print("=== SR-03 INVESTIGATION SUFFICIENCY FRONTIER ===")
print("Criterion: >=90% voxels with |error Ic| <= 0.15")
print()


for grid in sorted(groups):

    grid_rows = sorted(
        groups[grid],
        key=lambda r: int(r["n_cpt"]),
    )

    mean_pass = None
    ci_pass = None

    print("=" * 72)
    print(grid)
    print()
    print(
        f"{'CPT':>4} "
        f"{'MeanCov':>10} "
        f"{'CI95 Low':>10} "
        f"{'CI95 High':>10} "
        f"{'MAE':>9} "
        f"{'MaxDist':>10}"
    )

    for r in grid_rows:

        n = int(r["n_cpt"])

        mean_cov = float(
            r["sgan_frac_le_015_mean"]
        )

        ci_low = float(
            r["sgan_frac_le_015_ci95_low"]
        )

        ci_high = float(
            r["sgan_frac_le_015_ci95_high"]
        )

        mae = float(
            r["sgan_mae_mean"]
        )

        maxdist = float(
            r["max_nearest_distance_m_mean"]
        )

        print(
            f"{n:4d} "
            f"{mean_cov*100:9.2f}% "
            f"{ci_low*100:9.2f}% "
            f"{ci_high*100:9.2f}% "
            f"{mae:9.5f} "
            f"{maxdist:9.1f}m"
        )

        if (
            mean_pass is None
            and mean_cov >= REQUIRED_COVERAGE
        ):
            mean_pass = n

        if (
            ci_pass is None
            and ci_low >= REQUIRED_COVERAGE
        ):
            ci_pass = n

    print()

    if mean_pass is not None:
        print(
            "Mean sufficiency first reached :",
            f"{mean_pass} CPT"
        )
    else:
        print(
            "Mean sufficiency first reached :",
            "NOT reached"
        )

    if ci_pass is not None:
        print(
            "Conservative CI95 sufficiency  :",
            f"{ci_pass} CPT"
        )
    else:
        print(
            "Conservative CI95 sufficiency  :",
            "NOT reached"
        )

    last = grid_rows[-1]

    print(
        "Coverage at max tested CPT       :",
        f"{float(last['sgan_frac_le_015_mean'])*100:.2f}%"
    )

    print(
        "Max tested CPT                   :",
        last["n_cpt"]
    )

    print()


print("=" * 72)
print(
    "NOTE: This is empirical SchemaGAN/GeoSyn sufficiency, "
    "not a geotechnical design standard."
)
