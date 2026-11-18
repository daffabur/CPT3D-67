import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path.home() / "Developer/cpt_super_learn"
DATA = ROOT / "data/synthetic3d_v4_geosyn/full_24k/test"
OUT = ROOT / "results/dqn_spatial_research/coverage_baseline_v1"
COUNTS = (1, 2, 3, 5, 8, 12, 16, 24, 32)
STRATEGIES = ("random", "coverage", "coverage_interior")


def get_spacing(data):
    spacing = (10.0, 10.0)
    if "metadata_json" in data:
        try:
            raw = data["metadata_json"].item()
            meta = json.loads(str(raw))
            value = meta.get("spacing_m", meta.get("resolution_m"))
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                spacing = (float(value[0]), float(value[1]))
        except (ValueError, TypeError, KeyError, AttributeError):
            pass
    return spacing


def choose_sites(valid_xy, dx, dy, strategy, seed):
    nx, ny = valid_xy.shape
    xy = np.argwhere(valid_xy)
    if len(xy) < max(COUNTS):
        raise ValueError("Fewer than 32 valid XY cells")

    physical = xy * np.array([dx, dy])
    center = physical.mean(axis=0)
    first = int(np.argmin(np.linalg.norm(physical - center, axis=1)))

    chosen = [first]
    available = np.ones(len(xy), dtype=bool)
    available[first] = False
    rng = np.random.default_rng(seed)

    edge_distance = np.minimum.reduce([
        xy[:, 0], xy[:, 1],
        nx - 1 - xy[:, 0],
        ny - 1 - xy[:, 1],
    ]).astype(float)

    edge_distance = edge_distance * min(dx, dy)
    edge_scale = max(float(np.max(edge_distance)), 1.0)

    while len(chosen) < max(COUNTS):
        if strategy == "random":
            idx = int(rng.choice(np.flatnonzero(available)))
        else:
            diff = physical[:, None, :] - physical[chosen][None, :, :]
            nearest = np.sqrt(np.sum(diff ** 2, axis=2)).min(axis=1)

            if strategy == "coverage":
                score = nearest
            else:
                # Experimental interior preference; not a geotechnical rule.
                score = nearest * (
                    0.75 + 0.25 * edge_distance / edge_scale
                )

            score[~available] = -np.inf
            idx = int(np.argmax(score))

        chosen.append(idx)
        available[idx] = False

    return xy[np.array(chosen)]


def evaluate(ic, valid, sites, dx, dy):
    nx, ny, nz = ic.shape
    xs, ys = np.indices((nx, ny))
    coords = np.stack([xs * dx, ys * dy], axis=-1)
    site_xy = sites * np.array([dx, dy])

    distance = np.sqrt(
        np.sum(
            (coords[None, ...] - site_xy[:, None, None, :]) ** 2,
            axis=-1
        )
    )
    weights = 1 / np.maximum(distance, 1e-6) ** 2

    profiles = ic[sites[:, 0], sites[:, 1], :]
    profile_valid = valid[sites[:, 0], sites[:, 1], :]

    profiles = np.where(profile_valid, profiles, 0.0)
    weights_flat = weights.reshape(len(sites), -1).T

    numerator = weights_flat @ profiles
    denominator = weights_flat @ profile_valid.astype(float)

    observed_values = ic[sites[:, 0], sites[:, 1], :]
    observed_mask = profile_valid
    fallback = np.sum(np.where(observed_mask, observed_values, 0.0))
    fallback /= max(int(observed_mask.sum()), 1)

    predicted = np.divide(
        numerator, denominator,
        out=np.full_like(numerator, fallback),
        where=denominator > 0,
    ).reshape(nx, ny, nz)

    hidden = valid.copy()
    hidden[sites[:, 0], sites[:, 1], :] = False

    if not np.any(hidden):
        mae = float("nan")
    else:
        mae = float(np.mean(np.abs(predicted[hidden] - ic[hidden])))

    valid_xy = valid.any(axis=2)
    all_xy = np.argwhere(valid_xy) * np.array([dx, dy])

    coverage_distance = np.sqrt(
        np.sum(
            (all_xy[:, None, :] - site_xy[None, :, :]) ** 2,
            axis=2
        )
    ).min(axis=1)

    border = (
        (sites[:, 0] <= 1)
        | (sites[:, 1] <= 1)
        | (sites[:, 0] >= nx - 2)
        | (sites[:, 1] >= ny - 2)
    )

    return (
        mae,
        float(np.percentile(coverage_distance, 90)),
        float(np.mean(border) * 100),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes", type=int, default=8)
    args = parser.parse_args()

    if OUT.exists():
        raise SystemExit(
            f"STOP: Output already exists: {OUT}\n"
            "Existing research will not be overwritten."
        )

    files = sorted(DATA.glob("*.npz"))[:args.volumes]
    if not files:
        raise SystemExit(f"STOP: No NPZ files in {DATA}")

    OUT.mkdir(parents=True)
    rows = []

    for vi, path in enumerate(files):
        with np.load(path, allow_pickle=False) as data:
            if "Ic" not in data or "valid_mask" not in data:
                raise ValueError(f"Missing required fields: {path.name}")

            ic = np.asarray(data["Ic"], dtype=float)
            valid = np.asarray(data["valid_mask"], dtype=bool)
            dx, dy = get_spacing(data)

        if ic.ndim != 3 or valid.shape != ic.shape:
            raise ValueError(f"Unexpected volume shape: {path.name}")

        valid &= np.isfinite(ic)
        ic = np.where(valid, ic, 0.0)
        if valid.any(axis=2).sum() < 32:
            print(f"SKIP {path.name}: insufficient valid columns")
            continue

        for si, strategy in enumerate(STRATEGIES):
            sites = choose_sites(
                valid.any(axis=2), dx, dy, strategy,
                seed=67 + vi * 10 + si
            )

            for count in COUNTS:
                mae, p90, edge = evaluate(
                    ic, valid, sites[:count], dx, dy
                )

                rows.append({
                    "volume": path.name,
                    "strategy": strategy,
                    "cpt_count": count,
                    "hidden_ic_mae": mae,
                    "coverage_p90_m": p90,
                    "border_fraction_pct": edge,
                })

        print(f"Processed {vi + 1}/{len(files)}: {path.name}", flush=True)

    with (OUT / "per_volume.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "volume", "strategy", "cpt_count",
            "hidden_ic_mae", "coverage_p90_m",
            "border_fraction_pct"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== SPATIAL COVERAGE BENCHMARK ===")
    print("Strategy | CPT | Ic MAE | Coverage P90 m | Border %")

    summaries = []
    for strategy in STRATEGIES:
        for count in COUNTS:
            group = [
                r for r in rows
                if r["strategy"] == strategy
                and r["cpt_count"] == count
            ]
            if not group:
                continue

            result = {
                "strategy": strategy,
                "cpt_count": count,
                "mean_hidden_ic_mae": float(np.nanmean(
                    [r["hidden_ic_mae"] for r in group]
                )),
                "mean_coverage_p90_m": float(np.mean(
                    [r["coverage_p90_m"] for r in group]
                )),
                "mean_border_fraction_pct": float(np.mean(
                    [r["border_fraction_pct"] for r in group]
                )),
            }
            summaries.append(result)

            print(
                f'{strategy:18s} | {count:2d} | '
                f'{result["mean_hidden_ic_mae"]:.5f} | '
                f'{result["mean_coverage_p90_m"]:.1f} | '
                f'{result["mean_border_fraction_pct"]:.1f}%'
            )

    (OUT / "summary.json").write_text(
        json.dumps(summaries, indent=2)
    )
    print("\nOutput:", OUT)
    print("Research baseline only; DQN checkpoint not evaluated.")
    print("No automatic CPT sufficiency decision.")


if __name__ == "__main__":
    main()
