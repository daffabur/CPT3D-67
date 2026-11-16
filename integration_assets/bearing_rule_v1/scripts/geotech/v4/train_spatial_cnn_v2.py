from pathlib import Path
import hashlib
import json
import random
import copy
import csv

import numpy as np
from scipy.spatial import cKDTree

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]

DATA = ROOT / (
    "data/synthetic3d_v4_geosyn_qcfs/"
    "joint_v3_pilot_120"
)

SPLITS = ROOT / (
    "results/qc_fs_joint_v3/"
    "pilot_training_v1/splits.json"
)

OUT = ROOT / (
    "results/qc_fs_joint_v3/spatial_cnn_v2"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 6811
CPTS = (1, 3, 8)
THRESHOLDS = (10., 12., 15.)

EPOCHS = 20
PATIENCE = 6
LR = 0.0007
CHANNELS = 24
IDW_K = 4

# Fixed transformations. Shared by training/inference.
Q_REF = 5.0
F_REF = 0.10
MAX_LOG_CORRECTION = 1.2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE, flush=True)

if not SPLITS.exists():
    raise RuntimeError(
        "HGB split not found: " + str(SPLITS)
    )

split_data = json.loads(SPLITS.read_text())

train_files = [DATA / n for n in split_data["train"]]
val_files = [DATA / n for n in split_data["val"]]
test_files = [DATA / n for n in split_data["test"]]

assert len(train_files) == 72
assert len(val_files) == 24
assert len(test_files) == 24

all_names = [
    p.name
    for group in (train_files, val_files, test_files)
    for p in group
]
assert len(set(all_names)) == 120

for p in train_files + val_files + test_files:
    if not p.exists():
        raise RuntimeError(f"Dataset missing: {p}")

CHECKPOINT = OUT / "best.pt"
if CHECKPOINT.exists():
    raise RuntimeError(
        "Checkpoint already exists. "
        "Existing training results will not be overwritten."
    )


def seed_for(name):
    digest = hashlib.blake2b(
        f"{SEED}/{name}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little")


def load_volume(path):
    with np.load(path, allow_pickle=False) as d:
        q = np.asarray(d["qc_mpa"], np.float32)
        f = np.asarray(d["fs_mpa"], np.float32)
        valid = np.asarray(d["valid_mask"], bool)
        raw = d["metadata_json"].item()

    if isinstance(raw, bytes):
        raw = raw.decode()

    meta = json.loads(str(raw))
    grid = meta["grid"]

    dx = float(grid["dx_m"])
    dy = float(grid["dy_m"])
    dz = float(grid["dz_m"])

    if min(dx, dy, dz) <= 0:
        raise RuntimeError(f"Invalid spacing: {path.name}")

    if q.ndim != 3 or q.shape != f.shape or q.shape != valid.shape:
        raise RuntimeError(f"Invalid volume: {path.name}")

    valid &= (
        np.isfinite(q)
        & np.isfinite(f)
        & (q > 0)
        & (f > 0)
    )

    if not valid.any():
        raise RuntimeError(f"No valid cells: {path.name}")

    return q, f, valid, (dx, dy, dz), (
        meta["qcfs_extension"]["scenario"]
    )


def chosen_sites(valid, filename, epoch=None):
    sites = np.argwhere(valid.any(axis=2))

    if len(sites) < 8:
        raise RuntimeError("Insufficient CPT sites")

    # Validation uses exactly the same sampling
    # seed and nested CPT positions as HGB V1.
    if epoch is None:
        rng = np.random.default_rng(seed_for(filename))
    else:
        rng = np.random.default_rng(
            seed_for(filename) + 100003 * (epoch + 1)
        )

    return sites[rng.permutation(len(sites))[:8]]


def prepare(q, f, valid, spacing, sites):
    """
    Input only: measured CPT qc/fs and coordinates.
    Hidden ground truth is never used as a feature.
    Source arrays are XYZ; network arrays are ZYX.
    """
    nx, ny, nz = q.shape
    dx, dy, dz = spacing

    observed = np.zeros_like(valid)
    idw_q = np.zeros_like(q)
    idw_f = np.zeros_like(f)
    distance = np.ones_like(q)

    diag = max(float(np.hypot(nx*dx, ny*dy)), 1e-8)

    xv, yv = np.meshgrid(
        np.arange(nx), np.arange(ny),
        indexing="ij"
    )
    all_xy = np.stack(
        [xv.ravel(), yv.ravel()], axis=1
    )
    physical_grid = all_xy * np.array([dx, dy])

    for z in range(nz):
        measured = np.asarray([
            (int(x), int(y))
            for x, y in sites
            if valid[int(x), int(y), z]
        ], dtype=int).reshape(-1, 2)

        if len(measured) == 0:
            raise RuntimeError(
                "No CPT measurement at a valid depth; "
                "this pilot requires full-depth CPT."
            )

        observed[
            measured[:, 0], measured[:, 1], z
        ] = True

        mq = q[measured[:, 0], measured[:, 1], z]
        mf = f[measured[:, 0], measured[:, 1], z]

        tree = cKDTree(
            measured * np.array([dx, dy])
        )

        k = min(IDW_K, len(measured))
        d, indices = tree.query(physical_grid, k=k)

        d = np.asarray(d)
        indices = np.asarray(indices)

        if k == 1:
            d = d[:, None]
            indices = indices[:, None]

        weights = 1.0 / np.maximum(d, 1e-8)**2
        weights /= weights.sum(axis=1, keepdims=True)

        idw_q[:, :, z] = (
            (weights * mq[indices]).sum(axis=1)
            .reshape(nx, ny)
        )
        idw_f[:, :, z] = (
            (weights * mf[indices]).sum(axis=1)
            .reshape(nx, ny)
        )
        distance[:, :, z] = (
            d[:, 0].reshape(nx, ny) / diag
        )

    sq = np.zeros_like(q)
    sf = np.zeros_like(f)

    sq[observed] = np.log(
        np.maximum(q[observed], 1e-6) / Q_REF
    )
    sf[observed] = np.log(
        np.maximum(f[observed], 1e-8) / F_REF
    )

    xx = np.broadcast_to(
        ((np.arange(nx) + .5) / nx)[:, None, None],
        q.shape
    )
    yy = np.broadcast_to(
        ((np.arange(ny) + .5) / ny)[None, :, None],
        q.shape
    )
    zz = np.broadcast_to(
        ((np.arange(nz) + .5) / nz)[None, None, :],
        q.shape
    )

    channels = [
        np.log(np.maximum(idw_q, 1e-6) / Q_REF),
        np.log(np.maximum(idw_f, 1e-8) / F_REF),
        sq,
        sf,
        observed.astype(np.float32),
        np.clip(distance, 0, 1),
        xx.astype(np.float32),
        yy.astype(np.float32),
        zz.astype(np.float32),
    ]

    # C, Z, Y, X for PyTorch Conv3d.
    x = np.stack(
        [a.transpose(2, 1, 0) for a in channels],
        axis=0
    ).astype(np.float32)

    hidden = valid & ~observed

    return {
        "x": x,
        "q": q.transpose(2, 1, 0),
        "f": f.transpose(2, 1, 0),
        "idw_q": idw_q.transpose(2, 1, 0),
        "idw_f": idw_f.transpose(2, 1, 0),
        "hidden": hidden.transpose(2, 1, 0),
        "observed": observed.transpose(2, 1, 0),
    }


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(
                channels, channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation
            ),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv3d(
                channels, channels,
                kernel_size=3,
                padding=1
            ),
        )

    def forward(self, x):
        return F.silu(x + .35*self.net(x))


class SpatialCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(9, CHANNELS, 3, padding=1),
            nn.SiLU(),
        )

        self.blocks = nn.Sequential(
            ResidualBlock(CHANNELS, 1),
            ResidualBlock(CHANNELS, 2),
            ResidualBlock(CHANNELS, 4),
            ResidualBlock(CHANNELS, 1),
        )

        self.head = nn.Conv3d(CHANNELS, 2, 1)

        # Start training exactly from IDW.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        y = self.head(self.blocks(self.stem(x)))

        return MAX_LOG_CORRECTION * torch.tanh(
            y / MAX_LOG_CORRECTION
        )


def tensor(a):
    return torch.from_numpy(
        np.ascontiguousarray(a)
    ).to(DEVICE)


def forward_predictions(net, batch):
    x = tensor(batch["x"])[None]

    residual = net(x)[0]

    idw_q = tensor(batch["idw_q"])
    idw_f = tensor(batch["idw_f"])

    log_q = torch.log(idw_q.clamp(min=1e-6))
    log_f = torch.log(idw_f.clamp(min=1e-8))

    pq = torch.exp(log_q + residual[0])
    pf = torch.exp(log_f + residual[1])

    # Explicit CPT preservation.
    observed = tensor(
        batch["observed"].astype(np.bool_)
    )

    pq = torch.where(
        observed, tensor(batch["q"]), pq
    )
    pf = torch.where(
        observed, tensor(batch["f"]), pf
    )

    return pq, pf


def loss_function(pq, pf, batch):
    q = tensor(batch["q"])
    f = tensor(batch["f"])
    mask = tensor(
        batch["hidden"].astype(np.bool_)
    )

    qt = q[mask]
    ft = f[mask]
    qp = pq[mask]
    fp = pf[mask]

    high_weight = 1.0 + 1.0 * (
        qt >= 15
    ).float()

    qc_log_error = F.smooth_l1_loss(
        torch.log(qp.clamp(min=1e-6)),
        torch.log(qt.clamp(min=1e-6)),
        reduction="none",
        beta=.20
    )

    qc_loss = (
        high_weight * qc_log_error
    ).mean()

    qc_linear = (
        (qp - qt).abs() / 10.
    ).mean()

    fs_loss = F.smooth_l1_loss(
        torch.log(fp.clamp(min=1e-8)),
        torch.log(ft.clamp(min=1e-8)),
        beta=.25
    )

    # Penalize confidently high predictions where
    # actual hidden qc is below the threshold.
    low = qt < 15.
    if low.any():
        false_high = (
            F.relu(qp[low] - 15.) / 10.
        ).square().mean()
    else:
        false_high = qp.sum() * 0.

    return (
        qc_loss
        + .25*qc_linear
        + .45*fs_loss
        + .15*false_high
    )


def new_acc():
    return {
        "n": 0,
        "sq_q": 0.,
        "sq_f": 0.,
        "high": {
            str(t): {
                "tp": 0, "fp": 0,
                "fn": 0, "tn": 0,
                "bias_sum": 0.,
                "high_n": 0
            }
            for t in THRESHOLDS
        }
    }


def add_metrics(acc, q, f, pq, pf):
    dq = pq-q
    df = pf-f

    acc["n"] += len(q)
    acc["sq_q"] += float(np.dot(dq, dq))
    acc["sq_f"] += float(np.dot(df, df))

    for threshold in THRESHOLDS:
        key = str(threshold)
        h = acc["high"][key]

        truth = q >= threshold
        predicted = pq >= threshold

        h["tp"] += int(np.sum(truth & predicted))
        h["fp"] += int(np.sum(~truth & predicted))
        h["fn"] += int(np.sum(truth & ~predicted))
        h["tn"] += int(np.sum(~truth & ~predicted))

        h["bias_sum"] += float(dq[truth].sum())
        h["high_n"] += int(truth.sum())


def finish(acc):
    n = max(acc["n"], 1)

    result = {
        "evaluated_voxels": acc["n"],
        "qc_rmse_mpa": float(
            np.sqrt(acc["sq_q"] / n)
        ),
        "fs_rmse_kpa": float(
            1000 * np.sqrt(acc["sq_f"] / n)
        ),
        "thresholds": {}
    }

    for threshold, h in acc["high"].items():
        tp, fp, fn, tn = (
            h["tp"], h["fp"], h["fn"], h["tn"]
        )

        result["thresholds"][threshold] = {
            "recall_percent": (
                100*tp/(tp+fn) if tp+fn else None
            ),
            "precision_percent": (
                100*tp/(tp+fp) if tp+fp else None
            ),
            "fpr_percent": (
                100*fp/(fp+tn) if fp+tn else None
            ),
            "high_bias_mpa": (
                h["bias_sum"]/h["high_n"]
                if h["high_n"] else None
            ),
            "true_high_voxels": h["high_n"],
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        }

    return result


def evaluate(net):
    net.eval()
    accumulated = {}

    with torch.no_grad():
        for index, path in enumerate(val_files, 1):
            q, f, valid, spacing, scenario = (
                load_volume(path)
            )
            sites = chosen_sites(valid, path.name)

            for count in CPTS:
                batch = prepare(
                    q, f, valid, spacing,
                    sites[:count]
                )

                pq, pf = forward_predictions(net, batch)

                mask = batch["hidden"]

                qt = batch["q"][mask].astype(np.float64)
                ft = batch["f"][mask].astype(np.float64)
                q_idw = batch["idw_q"][mask].astype(np.float64)
                f_idw = batch["idw_f"][mask].astype(np.float64)

                predicted_q = (
                    pq.detach().cpu().numpy()[mask]
                    .astype(np.float64)
                )
                predicted_f = (
                    pf.detach().cpu().numpy()[mask]
                    .astype(np.float64)
                )

                for method, qm, fm in (
                    ("idw", q_idw, f_idw),
                    ("cnn", predicted_q, predicted_f),
                ):
                    key = (scenario, count, method)
                    if key not in accumulated:
                        accumulated[key] = new_acc()
                    add_metrics(
                        accumulated[key],
                        qt, ft, qm, fm
                    )

    report = {}

    for (scenario, count, method), accumulator in (
        accumulated.items()
    ):
        report.setdefault(scenario, {})
        report[scenario].setdefault(str(count), {})
        report[scenario][str(count)][method] = (
            finish(accumulator)
        )

    # Score used only for choosing the checkpoint
    # on validation, never on the test split.
    all_cnn = sum(
        (
            acc["sq_q"]
            for (sc, count, method), acc
            in accumulated.items()
            if method == "cnn"
        ),
        0.
    )
    all_idw = sum(
        (
            acc["sq_q"]
            for (sc, count, method), acc
            in accumulated.items()
            if method == "idw"
        ),
        0.
    )

    return report, float(all_cnn/max(all_idw, 1e-9))




# ==================================================
# V2: Multi-task CNN
# Two continuous regression outputs:
# qc and fs.
#
# Three independent detection outputs:
# P(qc>=10), P(qc>=12), P(qc>=15).
# ==================================================

POS_WEIGHTS = (1.5, 2.5, 4.0)
DETECTION_THRESHOLDS = (10., 12., 15.)
CLASSIFICATION_CUTOFF = 0.5


class SpatialCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(9, CHANNELS, 3, padding=1),
            nn.SiLU()
        )
        self.blocks = nn.Sequential(
            ResidualBlock(CHANNELS, 1),
            ResidualBlock(CHANNELS, 2),
            ResidualBlock(CHANNELS, 4),
            ResidualBlock(CHANNELS, 1)
        )

        self.regression = nn.Conv3d(CHANNELS, 2, 1)
        self.detection = nn.Conv3d(CHANNELS, 3, 1)

        nn.init.zeros_(self.regression.weight)
        nn.init.zeros_(self.regression.bias)

        nn.init.zeros_(self.detection.weight)

        # Initialize detection probabilities below 0.5.
        nn.init.constant_(self.detection.bias, -1.0)

    def forward(self, x):
        latent = self.blocks(self.stem(x))

        raw_reg = self.regression(latent)
        reg = MAX_LOG_CORRECTION * torch.tanh(
            raw_reg / MAX_LOG_CORRECTION
        )

        logits = self.detection(latent)

        return reg, logits


def predict(net, batch):
    x = tensor(batch["x"])[None]

    residual, logits = net(x)
    residual = residual[0]
    logits = logits[0]

    idw_q = tensor(batch["idw_q"])
    idw_f = tensor(batch["idw_f"])

    q = torch.exp(
        torch.log(idw_q.clamp(min=1e-6))
        + residual[0]
    )

    f = torch.exp(
        torch.log(idw_f.clamp(min=1e-8))
        + residual[1]
    )

    observed = tensor(
        batch["observed"].astype(np.bool_)
    )

    # Exact preservation of measured CPT.
    q = torch.where(
        observed, tensor(batch["q"]), q
    )
    f = torch.where(
        observed, tensor(batch["f"]), f
    )

    # Monotonic probabilities:
    # P(qc>=10) >= P(qc>=12) >= P(qc>=15)
    p = torch.sigmoid(logits)
    p10 = p[0]
    p12 = torch.minimum(p10, p[1])
    p15 = torch.minimum(p12, p[2])

    probabilities = torch.stack(
        (p10, p12, p15)
    )

    # On measured CPTs, thresholds are known.
    actual_q = tensor(batch["q"])

    for j, threshold in enumerate(DETECTION_THRESHOLDS):
        probabilities[j] = torch.where(
            observed,
            (actual_q >= threshold).float(),
            probabilities[j]
        )

    return q, f, logits, probabilities


def multitask_loss(net, batch):
    q, f, logits, probs = predict(net, batch)

    true_q = tensor(batch["q"])
    true_f = tensor(batch["f"])
    hidden = tensor(
        batch["hidden"].astype(np.bool_)
    )

    qt = true_q[hidden]
    ft = true_f[hidden]
    qp = q[hidden]
    fp = f[hidden]

    qloss = F.smooth_l1_loss(
        torch.log(qp.clamp(min=1e-6)),
        torch.log(qt.clamp(min=1e-6)),
        beta=.20
    )

    qlinear = torch.mean(
        torch.abs(qp-qt) / 10.
    )

    floss = F.smooth_l1_loss(
        torch.log(fp.clamp(min=1e-8)),
        torch.log(ft.clamp(min=1e-8)),
        beta=.25
    )

    detection_loss = 0.

    for j, threshold in enumerate(DETECTION_THRESHOLDS):
        labels = (qt >= threshold).float()
        raw = logits[j][hidden]

        weight = torch.tensor(
            POS_WEIGHTS[j],
            device=DEVICE
        )

        detection_loss = detection_loss + (
            F.binary_cross_entropy_with_logits(
                raw, labels,
                pos_weight=weight
            )
        )

    detection_loss = detection_loss / 3.

    # Train regression and detection simultaneously.
    return (
        qloss
        + .20*qlinear
        + .40*floss
        + .55*detection_loss
    )


def empty_acc():
    return {
        "n": 0,
        "qc_sse": 0.,
        "fs_sse": 0.,
        "bce_sum": 0.,
        "bce_n": 0,
        "thresholds": {
            str(t): {
                "tp": 0, "fp": 0,
                "fn": 0, "tn": 0,
                "high_n": 0,
                "high_bias_sum": 0.
            }
            for t in DETECTION_THRESHOLDS
        }
    }


def add_result(acc, qt, ft, qp, fp, prob=None):
    dq = qp-qt
    df = fp-ft

    acc["n"] += len(qt)
    acc["qc_sse"] += float(np.dot(dq, dq))
    acc["fs_sse"] += float(np.dot(df, df))

    for j, threshold in enumerate(DETECTION_THRESHOLDS):
        truth = qt >= threshold

        if prob is None:
            predicted = qp >= threshold
        else:
            predicted = prob[j] >= CLASSIFICATION_CUTOFF

            clipped = np.clip(
                prob[j], 1e-6, 1.-1e-6
            )

            acc["bce_sum"] += float(
                -np.sum(
                    truth*np.log(clipped)
                    + (~truth)*np.log(1.-clipped)
                )
            )
            acc["bce_n"] += len(qt)

        h = acc["thresholds"][str(threshold)]

        h["tp"] += int(np.sum(truth & predicted))
        h["fp"] += int(np.sum(~truth & predicted))
        h["fn"] += int(np.sum(truth & ~predicted))
        h["tn"] += int(np.sum(~truth & ~predicted))

        h["high_n"] += int(truth.sum())
        h["high_bias_sum"] += float(dq[truth].sum())


def finish_acc(acc):
    n = max(acc["n"], 1)

    result = {
        "voxels": acc["n"],
        "qc_rmse_mpa": float(
            np.sqrt(acc["qc_sse"]/n)
        ),
        "fs_rmse_kpa": float(
            1000*np.sqrt(acc["fs_sse"]/n)
        ),
        "bce": (
            acc["bce_sum"]/acc["bce_n"]
            if acc["bce_n"] else None
        ),
        "thresholds": {}
    }

    for key, h in acc["thresholds"].items():
        tp, fp, fn, tn = (
            h["tp"], h["fp"], h["fn"], h["tn"]
        )

        result["thresholds"][key] = {
            "recall_percent": (
                100*tp/(tp+fn) if tp+fn else None
            ),
            "precision_percent": (
                100*tp/(tp+fp) if tp+fp else None
            ),
            "fpr_percent": (
                100*fp/(fp+tn) if fp+tn else None
            ),
            "high_bias_mpa": (
                h["high_bias_sum"]/h["high_n"]
                if h["high_n"] else None
            ),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }

    return result


def validate(net):
    net.eval()
    accumulators = {}

    with torch.no_grad():
        for path in val_files:
            q, f, valid, spacing, scenario = (
                load_volume(path)
            )

            sites = chosen_sites(valid, path.name)

            for count in CPTS:
                batch = prepare(
                    q, f, valid, spacing,
                    sites[:count]
                )

                qp, fp, logits, probs = (
                    predict(net, batch)
                )

                mask = batch["hidden"]
                qt = batch["q"][mask].astype(np.float64)
                ft = batch["f"][mask].astype(np.float64)

                qc_cnn = (
                    qp.cpu().numpy()[mask]
                    .astype(np.float64)
                )
                fs_cnn = (
                    fp.cpu().numpy()[mask]
                    .astype(np.float64)
                )

                probabilities = (
                    probs.cpu().numpy()[:, mask]
                    .astype(np.float64)
                )

                cases = (
                    (
                        "idw",
                        batch["idw_q"][mask].astype(np.float64),
                        batch["idw_f"][mask].astype(np.float64),
                        None
                    ),
                    (
                        "cnn_regression",
                        qc_cnn, fs_cnn, None
                    ),
                    (
                        "cnn_detection",
                        qc_cnn, fs_cnn, probabilities
                    ),
                )

                for method, qm, fm, pm in cases:
                    key = (scenario, count, method)

                    if key not in accumulators:
                        accumulators[key] = empty_acc()

                    add_result(
                        accumulators[key],
                        qt, ft, qm, fm, pm
                    )

    report = {}
    qc_cnn_sse = 0.
    qc_idw_sse = 0.
    fs_cnn_sse = 0.
    fs_idw_sse = 0.
    bce_total = 0.
    bce_n = 0

    for (scenario, count, method), acc in accumulators.items():
        report.setdefault(scenario, {})
        report[scenario].setdefault(str(count), {})
        report[scenario][str(count)][method] = (
            finish_acc(acc)
        )

        if method == "idw":
            qc_idw_sse += acc["qc_sse"]
            fs_idw_sse += acc["fs_sse"]
        elif method == "cnn_regression":
            qc_cnn_sse += acc["qc_sse"]
            fs_cnn_sse += acc["fs_sse"]
        elif method == "cnn_detection":
            bce_total += acc["bce_sum"]
            bce_n += acc["bce_n"]

    qc_ratio = qc_cnn_sse/max(qc_idw_sse, 1e-9)
    fs_ratio = fs_cnn_sse/max(fs_idw_sse, 1e-9)
    bce = bce_total/max(bce_n, 1)

    # Validation-only model-selection objective.
    # Detection score normalized to the BCE of
    # constant probability 0.5 (ln 2).
    selection_score = (
        .50*qc_ratio
        + .15*fs_ratio
        + .35*(bce/np.log(2.))
    )

    return report, {
        "selection_score": float(selection_score),
        "qc_sse_ratio": float(qc_ratio),
        "fs_sse_ratio": float(fs_ratio),
        "detection_bce": float(bce)
    }


net = SpatialCNN().to(DEVICE)
optimizer = torch.optim.AdamW(
    net.parameters(),
    lr=LR,
    weight_decay=1e-4
)

print("\n=== SPATIAL CNN V2 MULTI-TASK ===")
print("Train:", len(train_files))
print("Validation:", len(val_files))
print("Reserved test:", len(test_files))
print("Device:", DEVICE)
print("CPT:", CPTS)
print("Outputs: QC, FS, P10, P12, P15")

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

    for position, index in enumerate(order):
        path = train_files[int(index)]

        q, f, valid, spacing, scenario = (
            load_volume(path)
        )

        sites = chosen_sites(
            valid, path.name, epoch=epoch
        )

        count = CPTS[
            (epoch+position) % len(CPTS)
        ]

        batch = prepare(
            q, f, valid, spacing,
            sites[:count]
        )

        optimizer.zero_grad(set_to_none=True)

        loss = multitask_loss(net, batch)

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

    validation, scores = validate(net)
    current = scores["selection_score"]

    improved = current < best_score-1e-4

    history.append({
        "epoch": epoch+1,
        "loss": float(np.mean(losses)),
        **scores
    })

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} "
        f"loss={np.mean(losses):.4f} "
        f"score={current:.4f} "
        f"qc_ratio={scores['qc_sse_ratio']:.4f} "
        f"fs_ratio={scores['fs_sse_ratio']:.4f} "
        f"bce={scores['detection_bce']:.4f} "
        f"{'BEST' if improved else ''}",
        flush=True
    )

    if improved:
        best_score = current
        best_epoch = epoch+1
        bad_epochs = 0

        torch.save({
            "state_dict": copy.deepcopy(
                net.state_dict()
            ),
            "architecture": "spatial_cnn_v2_multitask",
            "epoch": best_epoch,
            "channels": CHANNELS,
            "regression_outputs": 2,
            "detection_thresholds": DETECTION_THRESHOLDS,
            "positive_weights": POS_WEIGHTS,
            "classification_cutoff": CLASSIFICATION_CUTOFF,
            "seed": SEED,
        }, CHECKPOINT)
    else:
        bad_epochs += 1

    if bad_epochs >= PATIENCE:
        print("Early stopping.", flush=True)
        break


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)
net.load_state_dict(checkpoint["state_dict"])

final, scores = validate(net)

report = {
    "experiment": "spatial_cnn_v2_multitask",
    "train_volumes": 72,
    "validation_volumes": 24,
    "reserved_test_volumes": 24,
    "best_epoch": best_epoch,
    "validation_selection": scores,
    "detection_cutoff": CLASSIFICATION_CUTOFF,
    "thresholds_mpa": DETECTION_THRESHOLDS,
    "history": history,
    "validation": final,
    "limitations": [
        "Synthetic-only",
        "No test evaluation",
        "Detection probabilities not calibrated",
        "Validation cutoff 0.5 is exploratory",
        "One fixed validation CPT layout per volume"
    ]
}

(OUT/"metrics.json").write_text(
    json.dumps(report, indent=2)
)

with (OUT/"history.csv").open(
    "w", newline=""
) as file:
    writer = csv.DictWriter(
        file, fieldnames=list(history[0])
    )
    writer.writeheader()
    writer.writerows(history)


def fmt(value):
    return (
        "N/A" if value is None
        else f"{value:.2f}"
    )


print("\n=== FINAL V2 VALIDATION SUMMARY ===")

for scenario, counts in sorted(final.items()):
    print(f"\n{scenario}")

    for count in CPTS:
        print(f"  CPT {count}")

        for method in (
            "idw",
            "cnn_regression",
            "cnn_detection"
        ):
            m = counts[str(count)][method]

            print(
                f"    {method}: "
                f"QC_RMSE={fmt(m['qc_rmse_mpa'])} MPa, "
                f"FS_RMSE={fmt(m['fs_rmse_kpa'])} kPa"
            )

            for threshold in DETECTION_THRESHOLDS:
                h = m["thresholds"][str(threshold)]
                print(
                    f"      QC>={threshold:g}: "
                    f"Recall={fmt(h['recall_percent'])}% "
                    f"Precision={fmt(h['precision_percent'])}% "
                    f"FPR={fmt(h['fpr_percent'])}%"
                )

print("\n=== TRAINING COMPLETE ===")
print("Best epoch:", best_epoch)
print("Selection score:", scores["selection_score"])
print("QC SSE ratio:", scores["qc_sse_ratio"])
print("Detection BCE:", scores["detection_bce"])
print("Checkpoint:", CHECKPOINT)
print("Report:", OUT/"metrics.json")
print("Independent test untouched.")
print("Frozen models unchanged.")
