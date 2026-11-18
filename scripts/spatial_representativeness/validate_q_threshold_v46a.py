from pathlib import Path
import csv
import numpy as np


ROOT = Path.home() / "Developer" / "cpt_super_learn"

ACTIONS = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "qstop_validation_v46a"
    / "actions.csv"
)

THRESHOLD = -0.010135
NEGLIGIBLE_RMSE = 0.001


with ACTIONS.open() as f:
    rows = list(csv.DictReader(f))

rows = [
    r
    for r in rows
    if r["policy"] == "dqn"
    and r.get("q_score", "") != ""
    and r.get("reward_rmse_improvement", "") != ""
]

q = np.array(
    [float(r["q_score"]) for r in rows],
    dtype=float,
)

improvement = np.array(
    [
        float(r["reward_rmse_improvement"])
        for r in rows
    ],
    dtype=float,
)

stop = q <= THRESHOLD
rr = improvement[stop]

print(
    "=== Q-SCORE THRESHOLD EXTERNAL VALIDATION ==="
)
print()

print(f"Threshold: top_q <= {THRESHOLD:.6f}")
print()

print(f"Total DQN actions : {len(q)}")
print(f"Stop cases        : {stop.sum()}")
print(
    f"Stop frequency    : "
    f"{stop.mean() * 100:.2f}%"
)

if len(rr) == 0:
    print()
    print(
        "No Q-score values reached threshold."
    )
    raise SystemExit(0)

mean_reward = float(rr.mean())
median_reward = float(np.median(rr))
p90_reward = float(
    np.percentile(rr, 90)
)

negligible = float(
    np.mean(
        rr <= NEGLIGIBLE_RMSE
    )
)

nonpositive = float(
    np.mean(rr <= 0.0)
)

print()
print(
    "Actual benefit when STOP would trigger:"
)
print(
    f"Mean ΔRMSE   : {mean_reward:.6f}"
)
print(
    f"Median ΔRMSE : {median_reward:.6f}"
)
print(
    f"P90 ΔRMSE    : {p90_reward:.6f}"
)
print(
    f"<= 0         : "
    f"{nonpositive * 100:.1f}%"
)
print(
    f"<= 0.001     : "
    f"{negligible * 100:.1f}%"
)

enough_samples = len(rr) >= 50
pass_fraction = negligible >= 0.90
pass_p90 = (
    p90_reward <= NEGLIGIBLE_RMSE
)

print()
print("=== ACCEPTANCE ===")

print(
    "N stop >= 50          :",
    "PASS"
    if enough_samples
    else "FAIL",
)

print(
    ">=90% ΔRMSE <=0.001   :",
    "PASS"
    if pass_fraction
    else "FAIL",
)

print(
    "P90 ΔRMSE <=0.001     :",
    "PASS"
    if pass_p90
    else "FAIL",
)

print()

if (
    enough_samples
    and pass_fraction
    and pass_p90
):
    print(
        "RESULT: THRESHOLD VALIDATED "
        "ON UNSEEN VOLUMES"
    )
else:
    print(
        "RESULT: THRESHOLD "
        "NOT YET VALIDATED"
    )
