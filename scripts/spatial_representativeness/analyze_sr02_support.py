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


def load_rows():
    with INPUT.open() as f:
        return list(csv.DictReader(f))


def bootstrap_mean_ci(values, rng):
    values = np.asarray(values, dtype=float)

    mean = float(values.mean())

    if len(values) <= 1:
        return mean, mean, mean

    samples = rng.choice(
        values,
        size=(N_BOOT, len(values)),
        replace=True,
    )

    boot = samples.mean(axis=1)

    lo, hi = np.quantile(
        boot,
        [0.025, 0.975],
    )

    return mean, float(lo), float(hi)


def aggregate(rows, metric):
    groups = defaultdict(list)

    for r in rows:
        key = (
            r["grid"],
            float(r["distance_from_m"]),
            float(r["distance_to_m"]),
        )

        groups[key].append(
            float(r[metric])
        )

    rng = np.random.default_rng(SEED)

    result = defaultdict(list)

    for (grid, lo, hi), values in groups.items():

        mean, ci_lo, ci_hi = bootstrap_mean_ci(
            values,
            rng,
        )

        result[grid].append({
            "lo": lo,
            "hi": hi,
            "mean": mean,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n": len(values),
        })

    for grid in result:
        result[grid].sort(
            key=lambda x: x["lo"]
        )

    return result


def contiguous_frontier(
    bins,
    required_fraction,
    conservative=False,
):

    frontier = None
    first_evaluable = None
    first_failure = None

    for b in bins:

        if first_evaluable is None:
            first_evaluable = b

        value = (
            b["ci_lo"]
            if conservative
            else b["mean"]
        )

        if value >= required_fraction:
            frontier = b
        else:
            first_failure = b
            break

    return (
        frontier,
        first_evaluable,
        first_failure,
    )


def print_result(
    label,
    bins,
    required_fraction,
):

    print()
    print(label)

    for conservative in (False, True):

        frontier, first_bin, failure = (
            contiguous_frontier(
                bins,
                required_fraction,
                conservative,
            )
        )

        mode = (
            "Mean criterion"
            if not conservative
            else "CI95-low criterion"
        )

        print(f"  {mode}:")

        if frontier is None:

            if first_bin is None:
                print("    no evaluable bins")
                continue

            print(
                "    no contiguous bin passes"
            )

            print(
                f"    first evaluable bin = "
                f"{first_bin['lo']:.0f}-"
                f"{first_bin['hi']:.0f} m"
            )

            print(
                f"    fraction = "
                f"{first_bin['mean']*100:.1f}% "
                f"[CI95 "
                f"{first_bin['ci_lo']*100:.1f}, "
                f"{first_bin['ci_hi']*100:.1f}]%"
            )

            if first_bin["lo"] > 0:
                print(
                    f"    support boundary may lie "
                    f"below {first_bin['lo']:.0f} m "
                    f"and is unresolved by this grid"
                )

        else:

            print(
                f"    contiguous support through "
                f"{frontier['lo']:.0f}-"
                f"{frontier['hi']:.0f} m"
            )

            print(
                f"    fraction = "
                f"{frontier['mean']*100:.1f}% "
                f"[CI95 "
                f"{frontier['ci_lo']*100:.1f}, "
                f"{frontier['ci_hi']*100:.1f}]%"
            )

            if failure is not None:
                print(
                    f"    first failing bin = "
                    f"{failure['lo']:.0f}-"
                    f"{failure['hi']:.0f} m"
                )


def main():

    rows = load_rows()

    settings = [
        (
            "sgan_frac_le_010",
            0.80,
            "|error| <= 0.10, coverage >= 80%",
        ),
        (
            "sgan_frac_le_010",
            0.90,
            "|error| <= 0.10, coverage >= 90%",
        ),
        (
            "sgan_frac_le_015",
            0.80,
            "|error| <= 0.15, coverage >= 80%",
        ),
        (
            "sgan_frac_le_015",
            0.90,
            "|error| <= 0.15, coverage >= 90%",
        ),
        (
            "sgan_frac_le_015",
            0.95,
            "|error| <= 0.15, coverage >= 95%",
        ),
        (
            "sgan_frac_le_020",
            0.90,
            "|error| <= 0.20, coverage >= 90%",
        ),
    ]

    print("=== SR-02 SUPPORT SENSITIVITY ===")
    print("Input:", INPUT)

    for metric, required, title in settings:

        print()
        print("=" * 72)
        print(title)
        print("=" * 72)

        grouped = aggregate(
            rows,
            metric,
        )

        for grid in sorted(grouped):

            print_result(
                grid,
                grouped[grid],
                required,
            )


if __name__ == "__main__":
    main()
