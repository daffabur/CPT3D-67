from pathlib import Path
from collections import defaultdict
import ast
import copy
import csv
import hashlib
import json
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/synthetic3d_v4_geosyn_qcfs/joint_v3_expanded_1200"
SPLITS = ROOT / "results/qc_fs_joint_v3/spatial_cnn_v3/splits.json"
SOURCE = ROOT / "scripts/geotech/v4/train_spatial_cnn_v2.py"
OUT = ROOT / "results/qc_fs_joint_v3/spatial_cnn_v3"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 68032026
CHANNELS = 24
IDW_K = 4
Q_REF = 5.0
F_REF = 0.10
MAX_LOG_CORRECTION = 1.2
DETECTION_THRESHOLDS = (10., 12., 15.)
POS_WEIGHTS = (2., 3., 4.)

CPTS = (1, 2, 3, 4, 5, 8, 12, 16, 24, 32)
PATTERNS = ("random", "distributed")

EPOCHS = 10
PATIENCE = 4
LR = 0.0005

# Fixed model-selection subset:
# 15 validation volumes per scenario.
VAL_FAST_PER_SCENARIO = 15
VAL_COUNTS = (1, 8, 32)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CHECKPOINT = OUT / "best.pt"
if CHECKPOINT.exists():
    raise RuntimeError(
        "CNN V3 checkpoint already exists; refusing overwrite."
    )

for path in (DATA, SPLITS, SOURCE):
    if not path.exists():
        raise RuntimeError(f"Missing prerequisite: {path}")

split = json.loads(SPLITS.read_text())
train_files = [DATA / n for n in split["train"]]
val_files = [DATA / n for n in split["val"]]

if len(train_files) != 960 or len(val_files) != 240:
    raise RuntimeError("Incorrect train/validation split")

all_files = train_files + val_files
if len(set(p.name for p in all_files)) != 1200:
    raise RuntimeError("Duplicated source volumes")

if any(not p.exists() for p in all_files):
    raise RuntimeError("Missing dataset file")

# Reuse the successful V2 preprocessing and
# architecture definitions, without running V2 training.
required = (
    "seed_for", "load_volume", "prepare",
    "tensor", "ResidualBlock", "SpatialCNN", "predict"
)

tree = ast.parse(SOURCE.read_text())
definitions = {}

for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        if node.name in required:
            definitions[node.name] = node

if set(definitions) != set(required):
    raise RuntimeError(
        f"Missing V2 definitions: {set(required)-set(definitions)}"
    )

for name in required:
    module = ast.Module(
        body=[definitions[name]], type_ignores=[]
    )
    exec(
        compile(module, str(SOURCE), "exec"),
        globals()
    )


def get_sites(valid, spacing, filename, pattern, epoch):
    candidates = np.argwhere(valid.all(axis=2))

    if len(candidates) < max(CPTS):
        raise RuntimeError(
            f"Insufficient complete-depth sites: {filename}"
        )

    digest = hashlib.blake2b(
        f"{SEED}/{filename}/{epoch}".encode(),
        digest_size=8
    ).digest()
    rng = np.random.default_rng(
        int.from_bytes(digest, "little")
    )

    order = rng.permutation(len(candidates))

    if pattern == "random":
        return candidates[order[:max(CPTS)]]

    dx, dy, _ = spacing
    physical = candidates * np.array([dx, dy])
    nearest_sq = np.full(len(candidates), np.inf)
    selected = []
    used = np.zeros(len(candidates), dtype=bool)

    index = int(order[0])

    for _ in range(max(CPTS)):
        selected.append(index)
        used[index] = True

        delta = physical - physical[index]
        d2 = np.sum(delta*delta, axis=1)
        nearest_sq = np.minimum(nearest_sq, d2)
        nearest_sq[used] = -1.

        index = int(np.argmax(nearest_sq))

    return candidates[selected]


def training_loss(net, batch):
    pq, pf, logits, probabilities = predict(net, batch)

    q = tensor(batch["q"])
    f = tensor(batch["f"])
    hidden = tensor(batch["hidden"].astype(np.bool_))

    qt = q[hidden]
    ft = f[hidden]
    qp = pq[hidden]
    fp = pf[hidden]

    qlog = F.smooth_l1_loss(
        torch.log(qp.clamp(min=1e-6)),
        torch.log(qt.clamp(min=1e-6)),
        beta=.20
    )
    qlinear = torch.mean(
        torch.abs(qp-qt)/10.
    )
    flog = F.smooth_l1_loss(
        torch.log(fp.clamp(min=1e-8)),
        torch.log(ft.clamp(min=1e-8)),
        beta=.25
    )

    detection = torch.zeros((), device=DEVICE)

    for j, threshold in enumerate(DETECTION_THRESHOLDS):
        labels = (qt >= threshold).float()
        detection = detection + F.binary_cross_entropy_with_logits(
            logits[j][hidden],
            labels,
            pos_weight=torch.tensor(
                POS_WEIGHTS[j], device=DEVICE
            )
        )

    detection = detection/len(DETECTION_THRESHOLDS)

    # Penalize large false high-qc regression outputs.
    low = qt < 15.
    false_high = (
        (F.relu(qp[low]-15.)/10.).square().mean()
        if low.any()
        else qp.sum()*0.
    )

    return (
        qlog
        + .20*qlinear
        + .40*flog
        + .60*detection
        + .10*false_high
    )


# Balanced, deterministic validation subset.
groups = defaultdict(list)

for path in val_files:
    with np.load(path, allow_pickle=False) as d:
        raw = d["metadata_json"].item()
    if isinstance(raw, bytes):
        raw = raw.decode()

    scenario = json.loads(str(raw))[
        "qcfs_extension"
    ]["scenario"]
    groups[scenario].append(path)

validation_subset = []
rng = np.random.default_rng(SEED)

for scenario, paths in sorted(groups.items()):
    if len(paths) != 80:
        raise RuntimeError(
            f"Expected 80 validation volumes: {scenario}"
        )
    order = rng.permutation(len(paths))
    validation_subset.extend([
        paths[int(i)]
        for i in order[:VAL_FAST_PER_SCENARIO]
    ])

(OUT / "validation_subset.json").write_text(
    json.dumps([p.name for p in validation_subset], indent=2)
)


def evaluate(net, paths, counts):
    net.eval()
    records = defaultdict(lambda: {
        "n": 0,
        "cnn_q_sse": 0.,
        "idw_q_sse": 0.,
        "cnn_f_sse": 0.,
        "idw_f_sse": 0.,
        "labels": [],
        "scores": [],
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    })

    with torch.inference_mode():
        for i, path in enumerate(paths):
            q, f, valid, spacing, scenario = load_volume(path)

            # Validation sites fixed independently of epoch.
            for j, count in enumerate(counts):
                pattern = PATTERNS[(i+j) % len(PATTERNS)]
                sites = get_sites(
                    valid, spacing, path.name,
                    pattern, epoch=-1
                )[:count]

                batch = prepare(q, f, valid, spacing, sites)
                pq, pf, logits, probabilities = (
                    predict(net, batch)
                )

                hidden = batch["hidden"]

                qt = batch["q"][hidden].astype(np.float64)
                ft = batch["f"][hidden].astype(np.float64)
                iq = batch["idw_q"][hidden].astype(np.float64)
                iff = batch["idw_f"][hidden].astype(np.float64)

                qp = pq.detach().cpu().numpy()[
                    hidden
                ].astype(np.float64)

                fp = pf.detach().cpu().numpy()[
                    hidden
                ].astype(np.float64)

                scores = probabilities.detach().cpu().numpy()[
                    :, hidden
                ].astype(np.float32)

                rec = records[(scenario, count)]

                rec["n"] += len(qt)
                rec["cnn_q_sse"] += float(
                    np.sum((qp-qt)**2)
                )
                rec["idw_q_sse"] += float(
                    np.sum((iq-qt)**2)
                )
                rec["cnn_f_sse"] += float(
                    np.sum((fp-ft)**2)
                )
                rec["idw_f_sse"] += float(
                    np.sum((iff-ft)**2)
                )

                # QC>=15 detection: pooled AP and confusion.
                y = qt >= 15.
                s = scores[2]
                predicted = s >= .5

                rec["labels"].append(y)
                rec["scores"].append(s)

                rec["tp"] += int(np.sum(y & predicted))
                rec["fp"] += int(np.sum(~y & predicted))
                rec["fn"] += int(np.sum(y & ~predicted))
                rec["tn"] += int(np.sum(~y & ~predicted))

    summary = {}
    total_cnn_q = 0.
    total_idw_q = 0.
    total_cnn_f = 0.
    total_idw_f = 0.
    macro_ap = []

    for (scenario, count), rec in sorted(records.items()):
        n = max(rec["n"], 1)
        y = np.concatenate(rec["labels"])
        s = np.concatenate(rec["scores"])

        ap = (
            float(average_precision_score(y, s))
            if y.any() and (~y).any()
            else None
        )

        if ap is not None:
            macro_ap.append(ap)

        tp, fp, fn, tn = (
            rec["tp"], rec["fp"],
            rec["fn"], rec["tn"]
        )

        summary.setdefault(scenario, {})[str(count)] = {
            "qc_rmse_cnn": float(
                np.sqrt(rec["cnn_q_sse"]/n)
            ),
            "qc_rmse_idw": float(
                np.sqrt(rec["idw_q_sse"]/n)
            ),
            "fs_rmse_cnn_kpa": float(
                1000*np.sqrt(rec["cnn_f_sse"]/n)
            ),
            "fs_rmse_idw_kpa": float(
                1000*np.sqrt(rec["idw_f_sse"]/n)
            ),
            "ap15": ap,
            "recall15": (
                100*tp/(tp+fn) if tp+fn else None
            ),
            "precision15": (
                100*tp/(tp+fp) if tp+fp else None
            ),
            "fpr15": (
                100*fp/(fp+tn) if fp+tn else None
            ),
            "positive_voxels": int(y.sum()),
        }

        total_cnn_q += rec["cnn_q_sse"]
        total_idw_q += rec["idw_q_sse"]
        total_cnn_f += rec["cnn_f_sse"]
        total_idw_f += rec["idw_f_sse"]

    qc_ratio = total_cnn_q/max(total_idw_q, 1e-9)
    fs_ratio = total_cnn_f/max(total_idw_f, 1e-9)
    ap_mean = float(np.mean(macro_ap))

    # Select using both continuous QC quality and
    # high-QC discrimination, not recall alone.
    selection = (
        .65*qc_ratio
        + .20*fs_ratio
        + .15*(1.-ap_mean)
    )

    return summary, {
        "score": float(selection),
        "qc_sse_ratio": float(qc_ratio),
        "fs_sse_ratio": float(fs_ratio),
        "macro_ap15": ap_mean,
    }


net = SpatialCNN().to(DEVICE)
optimizer = torch.optim.AdamW(
    net.parameters(),
    lr=LR,
    weight_decay=1e-4
)

print("=== SPATIAL CNN V3 FLEXIBLE TRAINING ===", flush=True)
print("Device:", DEVICE)
print("Train volumes:", len(train_files))
print("Validation volumes:", len(val_files))
print("Selection subset:", len(validation_subset))
print("Training CPT:", CPTS)
print("Sampling patterns:", PATTERNS)
print("Validation counts:", VAL_COUNTS)
print("No test evaluation.", flush=True)

best_score = float("inf")
best_epoch = None
bad_epochs = 0
history = []

for epoch in range(EPOCHS):
    net.train()

    order = np.random.default_rng(
        SEED+epoch
    ).permutation(len(train_files))

    losses = []
    patterns_used = defaultdict(int)
    counts_used = defaultdict(int)

    for position, index in enumerate(order):
        path = train_files[int(index)]

        q, f, valid, spacing, scenario = load_volume(path)

        # Balanced rotation through CPT counts/patterns.
        count = CPTS[(epoch+position) % len(CPTS)]
        pattern = PATTERNS[
            ((epoch + position//len(CPTS)) % len(PATTERNS))
        ]

        sites = get_sites(
            valid, spacing, path.name,
            pattern, epoch
        )[:count]

        batch = prepare(
            q, f, valid, spacing, sites
        )

        optimizer.zero_grad(set_to_none=True)

        loss = training_loss(net, batch)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Nonfinite loss: {path.name}"
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            net.parameters(), 1.0
        )
        optimizer.step()

        losses.append(float(loss.item()))
        counts_used[count] += 1
        patterns_used[pattern] += 1

        if (position+1) % 240 == 0:
            print(
                f"  Epoch {epoch+1}: "
                f"{position+1}/960 volumes",
                flush=True
            )

    report, metrics = evaluate(
        net, validation_subset, VAL_COUNTS
    )

    score = metrics["score"]
    improved = score < best_score-1e-4

    history.append({
        "epoch": epoch+1,
        "train_loss": float(np.mean(losses)),
        **metrics
    })

    print(
        f"Epoch {epoch+1:02d}: "
        f"loss={np.mean(losses):.4f} "
        f"selection={score:.4f} "
        f"QC_ratio={metrics['qc_sse_ratio']:.4f} "
        f"FS_ratio={metrics['fs_sse_ratio']:.4f} "
        f"AP15={metrics['macro_ap15']:.4f} "
        f"{'BEST' if improved else ''}",
        flush=True
    )

    if improved:
        best_score = score
        best_epoch = epoch+1
        bad_epochs = 0

        torch.save({
            "state_dict": copy.deepcopy(
                net.state_dict()
            ),
            "architecture": "spatial_cnn_v3_flexible",
            "base_architecture": "spatial_cnn_v2_multitask",
            "epoch": best_epoch,
            "channels": CHANNELS,
            "cpt_counts": CPTS,
            "patterns": PATTERNS,
            "thresholds": DETECTION_THRESHOLDS,
            "seed": SEED,
            "checkpoint_selection": metrics,
        }, CHECKPOINT)

    else:
        bad_epochs += 1

    if bad_epochs >= PATIENCE:
        print("Early stopping.", flush=True)
        break

# Evaluate only the best checkpoint.
state = torch.load(
    CHECKPOINT, map_location="cpu",
    weights_only=True
)
net.load_state_dict(state["state_dict"])

final, scores = evaluate(
    net, val_files, (1, 8, 16, 32)
)

report = {
    "experiment": "spatial_cnn_v3_flexible",
    "dataset": str(DATA),
    "split_manifest": str(SPLITS),
    "training_volumes": len(train_files),
    "validation_volumes": len(val_files),
    "best_epoch": best_epoch,
    "training_history": history,
    "full_validation_metrics": scores,
    "full_validation": final,
    "limitations": [
        "Synthetic dataset only",
        "No external test evaluation",
        "Fixed detection cutoff 0.5",
        "Validation alternates sampling patterns",
        "CPT counts 2/4/5/12/24 trained but not in full validation",
        "Probabilities are not field calibrated",
        "Performance does not establish bearing-stratum suitability",
    ],
}

(OUT/"metrics.json").write_text(
    json.dumps(report, indent=2)
)

with (OUT/"history.csv").open(
    "w", newline=""
) as handle:
    writer = csv.DictWriter(
        handle, fieldnames=list(history[0])
    )
    writer.writeheader()
    writer.writerows(history)


def fmt(v):
    return "N/A" if v is None else f"{v:.2f}"


print("\n=== V3 FULL VALIDATION QC>=15 ===")

for scenario, counts in sorted(final.items()):
    print(f"\n{scenario}")

    for count, m in sorted(
        counts.items(),
        key=lambda x: int(x[0])
    ):
        print(
            f"  CPT {count}: "
            f"QC_RMSE CNN/IDW="
            f"{fmt(m['qc_rmse_cnn'])}/"
            f"{fmt(m['qc_rmse_idw'])} MPa "
            f"Recall15={fmt(m['recall15'])}% "
            f"Precision15={fmt(m['precision15'])}% "
            f"FPR15={fmt(m['fpr15'])}% "
            f"AP15={fmt(m['ap15'])}"
        )

print("\n=== V3 TRAINING COMPLETE ===")
print("Best epoch:", best_epoch)
print("QC SSE ratio:", scores["qc_sse_ratio"])
print("FS SSE ratio:", scores["fs_sse_ratio"])
print("Macro AP15:", scores["macro_ap15"])
print("Checkpoint:", CHECKPOINT)
print("Metrics:", OUT/"metrics.json")
print("History:", OUT/"history.csv")
print("Independent test not accessed.")
print("Existing checkpoints unchanged.")
