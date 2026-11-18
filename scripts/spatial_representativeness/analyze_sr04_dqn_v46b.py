from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path.home() / "Developer" / "cpt_super_learn"

ACTIONS = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "dqn_v46b_vs_farthest"
    / "actions.csv"
)

STEPS = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "dqn_v46b_vs_farthest"
    / "per_volume_step.csv"
)


with ACTIONS.open() as f:
    actions = list(csv.DictReader(f))

with STEPS.open() as f:
    steps = list(csv.DictReader(f))


# ------------------------------------------------------------
# 1. Q-score vs actual reward
# ------------------------------------------------------------

dqn_actions = [
    r for r in actions
    if r["policy"] == "dqn"
    and r["q_score"] != ""
]


q = np.array([
    float(r["q_score"])
    for r in dqn_actions
])

reward = np.array([
    float(r["reward_rmse_improvement"])
    for r in dqn_actions
])


pearson = np.corrcoef(
    q,
    reward,
)[0, 1]


rank_q = np.argsort(
    np.argsort(q)
)

rank_reward = np.argsort(
    np.argsort(reward)
)

spearman = np.corrcoef(
    rank_q,
    rank_reward,
)[0, 1]


print("=== Q-SCORE vs ACTUAL RMSE IMPROVEMENT ===")
print("Samples :", len(q))
print(f"Pearson : {pearson:.4f}")
print(f"Spearman: {spearman:.4f}")


# ------------------------------------------------------------
# 2. Compare DQN vs Farthest per CPT count
# ------------------------------------------------------------

groups = defaultdict(list)

for r in steps:

    key = (
        r["grid"],
        int(r["n_cpt"]),
        r["policy"],
    )

    groups[key].append(r)


print()
print("=== DQN vs FARTHEST ===")


grids = sorted({
    r["grid"]
    for r in steps
})


for grid in grids:

    print()
    print("=" * 65)
    print(grid)

    counts = sorted({
        int(r["n_cpt"])
        for r in steps
        if r["grid"] == grid
    })

    print(
        f"{'CPT':>4} "
        f"{'DQN MAE':>10} "
        f"{'FAR MAE':>10} "
        f"{'DQN Cov':>10} "
        f"{'FAR Cov':>10} "
        f"{'DQN Dist':>10} "
        f"{'FAR Dist':>10}"
    )

    for n in counts:

        dqn_rows = groups.get(
            (grid, n, "dqn"),
            [],
        )

        far_rows = groups.get(
            (grid, n, "farthest"),
            [],
        )

        if not dqn_rows or not far_rows:
            continue

        dqn_mae = np.mean([
            float(r["mae"])
            for r in dqn_rows
        ])

        far_mae = np.mean([
            float(r["mae"])
            for r in far_rows
        ])

        dqn_cov = np.mean([
            float(r["frac_le_015"])
            for r in dqn_rows
        ])

        far_cov = np.mean([
            float(r["frac_le_015"])
            for r in far_rows
        ])

        dqn_dist = np.mean([
            float(
                r[
                    "max_nearest_distance_m"
                ]
            )
            for r in dqn_rows
        ])

        far_dist = np.mean([
            float(
                r[
                    "max_nearest_distance_m"
                ]
            )
            for r in far_rows
        ])

        print(
            f"{n:4d} "
            f"{dqn_mae:10.5f} "
            f"{far_mae:10.5f} "
            f"{dqn_cov*100:9.1f}% "
            f"{far_cov*100:9.1f}% "
            f"{dqn_dist:9.1f}m "
            f"{far_dist:9.1f}m"
        )


# ------------------------------------------------------------
# 3. Average reward by policy
# ------------------------------------------------------------

print()
print("=== ACTUAL RMSE IMPROVEMENT PER NEW CPT ===")


for policy in ("dqn", "farthest"):

    rows = [
        r for r in actions
        if r["policy"] == policy
    ]

    values = np.array([
        float(
            r["reward_rmse_improvement"]
        )
        for r in rows
    ])

    print(
        f"{policy:8s}: "
        f"mean={values.mean():.6f}, "
        f"median={np.median(values):.6f}, "
        f"positive={np.mean(values > 0)*100:.1f}%"
    )
