from pathlib import Path
import csv
import numpy as np


ROOT = Path.home() / "Developer" / "cpt_super_learn"

ACTIONS = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "dqn_v46a_vs_farthest"
    / "actions.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "spatial_representativeness"
    / "qscore_calibration_v46a"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


with ACTIONS.open() as f:
    rows = list(csv.DictReader(f))

print("Columns:")
print(rows[0].keys())
print()


# ============================================================
# DQN rows only
# ============================================================

dqn = [
    r for r in rows
    if r.get("policy") == "dqn"
    and r.get("q_score", "") != ""
    and r.get("reward_rmse_improvement", "") != ""
]

q = np.array(
    [float(r["q_score"]) for r in dqn],
    dtype=float,
)

reward = np.array(
    [float(r["reward_rmse_improvement"]) for r in dqn],
    dtype=float,
)

print("=== DATA ===")
print("Samples:", len(q))
print(
    f"Q range : {q.min():.6f} .. {q.max():.6f}"
)
print(
    f"Q median: {np.median(q):.6f}"
)
print(
    f"Reward mean   : {reward.mean():.6f}"
)
print(
    f"Reward median : {np.median(reward):.6f}"
)
print()


# ============================================================
# 1. Q-score deciles
# ============================================================

print("=== Q-SCORE DECILES ===")
print(
    f"{'Bin':>5} "
    f"{'N':>5} "
    f"{'Q min':>10} "
    f"{'Q max':>10} "
    f"{'Mean ΔRMSE':>12} "
    f"{'Median':>10} "
    f"{'Positive':>10} "
    f"{'<=0.0005':>11} "
    f"{'<=0.001':>10}"
)

edges = np.quantile(
    q,
    np.linspace(0, 1, 11),
)

decile_rows = []

for i in range(10):
    lo = edges[i]
    hi = edges[i + 1]

    if i == 9:
        mask = (q >= lo) & (q <= hi)
    else:
        mask = (q >= lo) & (q < hi)

    rr = reward[mask]
    qq = q[mask]

    if len(rr) == 0:
        continue

    row = {
        "bin": i + 1,
        "n": len(rr),
        "q_min": float(qq.min()),
        "q_max": float(qq.max()),
        "mean_reward": float(rr.mean()),
        "median_reward": float(np.median(rr)),
        "positive_fraction": float(np.mean(rr > 0)),
        "le_0005_fraction": float(np.mean(rr <= 0.0005)),
        "le_001_fraction": float(np.mean(rr <= 0.001)),
    }

    decile_rows.append(row)

    print(
        f"{i+1:5d} "
        f"{len(rr):5d} "
        f"{qq.min():10.5f} "
        f"{qq.max():10.5f} "
        f"{rr.mean():12.6f} "
        f"{np.median(rr):10.6f} "
        f"{np.mean(rr > 0)*100:9.1f}% "
        f"{np.mean(rr <= 0.0005)*100:10.1f}% "
        f"{np.mean(rr <= 0.001)*100:9.1f}%"
    )


# ============================================================
# 2. Scan candidate stop thresholds
#
# Stop rule:
#     top_q <= threshold
#
# We then ask:
# When we would stop, how often was the TRUE RMSE benefit
# actually negligible?
# ============================================================

print()
print("=== CANDIDATE STOP THRESHOLDS ===")

print(
    f"{'Q percentile':>12} "
    f"{'Q threshold':>12} "
    f"{'N stop':>8} "
    f"{'% samples':>10} "
    f"{'Mean Δ':>10} "
    f"{'P90 Δ':>10} "
    f"{'<=0':>8} "
    f"{'<=.0005':>10} "
    f"{'<=.001':>9}"
)

threshold_rows = []

# We focus on lower half of Q distribution.
for pct in range(5, 55, 5):

    threshold = float(
        np.percentile(q, pct)
    )

    mask = q <= threshold
    rr = reward[mask]

    row = {
        "percentile": pct,
        "q_threshold": threshold,
        "n_stop": int(mask.sum()),
        "sample_fraction": float(mask.mean()),
        "mean_reward": float(rr.mean()),
        "median_reward": float(np.median(rr)),
        "p90_reward": float(np.percentile(rr, 90)),
        "nonpositive_fraction": float(np.mean(rr <= 0)),
        "le_0005_fraction": float(
            np.mean(rr <= 0.0005)
        ),
        "le_001_fraction": float(
            np.mean(rr <= 0.001)
        ),
    }

    threshold_rows.append(row)

    print(
        f"{pct:11d}% "
        f"{threshold:12.6f} "
        f"{len(rr):8d} "
        f"{mask.mean()*100:9.1f}% "
        f"{rr.mean():10.6f} "
        f"{np.percentile(rr,90):10.6f} "
        f"{np.mean(rr <= 0)*100:7.1f}% "
        f"{np.mean(rr <= 0.0005)*100:9.1f}% "
        f"{np.mean(rr <= 0.001)*100:8.1f}%"
    )


# ============================================================
# 3. Conservative threshold suggestion
#
# Pilot definition:
# "negligible next-CPT benefit" =
# actual RMSE improvement <= 0.001
#
# Candidate threshold is accepted only if:
# - at least 100 observations
# - >=90% of stopped actions have improvement <=0.001
# - P90 improvement <=0.001
#
# This is calibration logic, NOT a geotechnical standard.
# ============================================================

valid = [
    r
    for r in threshold_rows
    if r["n_stop"] >= 100
    and r["le_001_fraction"] >= 0.90
    and r["p90_reward"] <= 0.001
]

print()
print("=== PILOT RECOMMENDATION ===")

if valid:

    # Largest threshold satisfying conservative rule.
    best = max(
        valid,
        key=lambda r: r["q_threshold"],
    )

    print(
        "Candidate Q threshold found:"
    )
    print(
        f"top_q <= {best['q_threshold']:.6f}"
    )
    print(
        f"Calibrated on {best['n_stop']} actions "
        f"({best['sample_fraction']*100:.1f}% of DQN actions)"
    )
    print(
        "Fraction with actual improvement <= 0.001: "
        f"{best['le_001_fraction']*100:.1f}%"
    )
    print(
        f"P90 actual improvement: "
        f"{best['p90_reward']:.6f}"
    )

else:
    print(
        "NO SAFE UNIVERSAL Q-SCORE THRESHOLD FOUND."
    )
    print(
        "Do not use Q-score alone as the stopping gate."
    )


# ============================================================
# Save CSV
# ============================================================

with (
    OUT_DIR / "qscore_deciles.csv"
).open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=decile_rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(decile_rows)


with (
    OUT_DIR / "qscore_threshold_scan.csv"
).open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=threshold_rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(threshold_rows)


print()
print("Saved:")
print(
    OUT_DIR / "qscore_deciles.csv"
)
print(
    OUT_DIR / "qscore_threshold_scan.csv"
)
