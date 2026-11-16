from pathlib import Path
import ast
import hashlib
import json
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial import cKDTree
import coremltools as ct

BASE = Path("scripts/geotech/v4")
SRC = BASE / "train_spatial_cnn_v2.py"
SRC_V3 = BASE / "train_spatial_cnn_v3.py"
CKPT = Path(
    "results/qc_fs_joint_v3/spatial_cnn_v3/best.pt"
)
DATA = Path(
    "data/synthetic3d_v4_geosyn_qcfs/"
    "joint_v3_expanded_1200"
)
MANIFEST = Path(
    "results/bearing_layer_v1/"
    "geometry_aware_pilot_v3/manifest.json"
)
OUT = Path(
    "results/qc_fs_joint_v3/coreml_export_v1"
)
MODEL_PATH = OUT / "QCFS_CNN_V3.mlpackage"

SEED = 68032026
CHANNELS = 24
IDW_K = 4
Q_REF = 5.0
F_REF = 0.10
MAX_LOG_CORRECTION = 1.2
DETECTION_THRESHOLDS = (10., 12., 15.)
CPTS = (1, 3, 8, 16, 32)
PATTERNS = ("random", "distributed")
DEVICE = torch.device("cpu")


def extract(path, names):
    tree = ast.parse(path.read_text())
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(
            node, (ast.ClassDef, ast.FunctionDef)
        )
    }

    for name in names:
        if name not in definitions:
            raise RuntimeError(
                f"Missing definition: {name}"
            )

        exec(
            compile(
                ast.Module(
                    body=[definitions[name]],
                    type_ignores=[]
                ),
                str(path),
                "exec",
            ),
            globals(),
        )


extract(
    SRC,
    (
        "seed_for",
        "load_volume",
        "prepare",
        "tensor",
        "ResidualBlock",
        "SpatialCNN",
        "predict",
    ),
)
extract(SRC_V3, ("get_sites",))



class CoreMLGroupNorm(nn.Module):
    """GroupNorm expressed using rank-5 tensor operations."""

    def __init__(self, original):
        super().__init__()

        self.groups = original.num_groups
        self.channels = original.num_channels
        self.eps = original.eps

        if self.channels % self.groups:
            raise ValueError("Invalid channel/group count")

        self.per_group = self.channels // self.groups

        self.weight = nn.Parameter(
            original.weight.detach().clone(),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            original.bias.detach().clone(),
            requires_grad=False,
        )

    def forward(self, x):
        normalized = []

        for group in range(self.groups):
            start = group * self.per_group
            end = start + self.per_group

            part = x[:, start:end, :, :, :]

            mean = part.mean(
                dim=(1, 2, 3, 4),
                keepdim=True,
            )
            variance = (
                (part - mean) ** 2
            ).mean(
                dim=(1, 2, 3, 4),
                keepdim=True,
            )

            result = (
                (part - mean)
                * torch.rsqrt(variance + self.eps)
            )

            weight = self.weight[start:end].reshape(
                1, self.per_group, 1, 1, 1
            )
            bias = self.bias[start:end].reshape(
                1, self.per_group, 1, 1, 1
            )

            normalized.append(
                result * weight + bias
            )

        return torch.cat(normalized, dim=1)


def replace_group_norm(module):
    replaced = 0

    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm):
            setattr(
                module,
                name,
                CoreMLGroupNorm(child),
            )
            replaced += 1
        else:
            replaced += replace_group_norm(child)

    return replaced


class CoreMLWrapper(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, input_volume):
        residual, logits = self.network(
            input_volume
        )
        return residual, logits


def physical_outputs(residual, batch):
    """
    Reconstruct physical qc/fs in ZYX order,
    matching the original frozen predict() function.
    """
    reg = np.asarray(residual, dtype=np.float64)

    if reg.ndim != 5 or reg.shape[0] != 1:
        raise ValueError(
            f"Unexpected residual shape: {reg.shape}"
        )

    rq = reg[0, 0]
    rf = reg[0, 1]

    idw_q = np.asarray(
        batch["idw_q"], dtype=np.float64
    )
    idw_f = np.asarray(
        batch["idw_f"], dtype=np.float64
    )

    if rq.shape != idw_q.shape:
        raise ValueError(
            f"qc grid mismatch: {rq.shape} vs {idw_q.shape}"
        )

    if rf.shape != idw_f.shape:
        raise ValueError(
            f"fs grid mismatch: {rf.shape} vs {idw_f.shape}"
        )

    qc = np.exp(
        np.log(np.maximum(idw_q, 1e-6)) + rq
    )
    fs = np.exp(
        np.log(np.maximum(idw_f, 1e-8)) + rf
    )

    observed = np.asarray(
        batch["observed"], dtype=bool
    )

    qc = np.where(
        observed,
        np.asarray(batch["q"]),
        qc,
    )
    fs = np.where(
        observed,
        np.asarray(batch["f"]),
        fs,
    )

    return qc, fs


def errors(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    delta = np.abs(a - b)
    return float(delta.mean()), float(delta.max())


print("=== QC-FS CNN V3 CORE ML EXPORT ===")
print("PyTorch:", torch.__version__)
print("Core ML tools:", ct.__version__)

original_hash = hashlib.sha256(
    CKPT.read_bytes()
).hexdigest()

expected_hash = (
    "8a67c06704d1cb923a5e9249da506055dbe"
    "f351bb0083b60eeee573b92c1d615"
)

if original_hash != expected_hash:
    raise RuntimeError(
        "Frozen checkpoint SHA256 mismatch"
    )

state = torch.load(
    CKPT,
    map_location="cpu",
    weights_only=True,
)

network = SpatialCNN()
network.load_state_dict(
    state["state_dict"], strict=True
)
network.eval()

import copy

export_network = copy.deepcopy(network)
replaced = replace_group_norm(export_network)

if replaced != 4:
    raise RuntimeError(
        f"Expected 4 GroupNorm layers, found {replaced}"
    )

export_network.eval()
wrapper = CoreMLWrapper(export_network).eval()

print("GroupNorm layers replaced:", replaced)
print("Original frozen network preserved.")

names = json.loads(
    MANIFEST.read_text()
)["val_files"]

name = names[0]
q, f, valid, spacing, scenario = load_volume(
    DATA / name
)

nx, ny, nz = q.shape

print("Checkpoint verified:", original_hash)
print("Epoch:", state["epoch"])
print("Volume:", name)
print("Scenario:", scenario)
print("Grid XYZ:", q.shape)
print("Spacing:", spacing)

sites = get_sites(
    valid, spacing, name,
    "distributed", epoch=-1
)

if len(sites) < 32:
    raise RuntimeError(
        "Selected validation case has fewer than 32 sites"
    )

batch = prepare(
    q, f, valid, spacing, sites[:8]
)

input_sample = torch.from_numpy(
    np.ascontiguousarray(
        batch["x"][None], dtype=np.float32
    )
)

assert input_sample.shape == (
    1, 9, nz, ny, nx
)

# Validate mathematical equivalence before Core ML export.
print("\n=== GROUPNORM NUMERICAL PARITY ===")

for count in (1, 3, 8, 32):
    check_batch = prepare(
        q, f, valid, spacing, sites[:count]
    )
    check_input = torch.from_numpy(
        np.ascontiguousarray(
            check_batch["x"][None],
            dtype=np.float32,
        )
    )

    with torch.inference_mode():
        orig_reg, orig_logits = network(check_input)
        new_reg, new_logits = wrapper(check_input)

    reg_error = float(
        (orig_reg - new_reg).abs().max()
    )
    logits_error = float(
        (orig_logits - new_logits).abs().max()
    )

    print(
        f"{count} CPT: "
        f"residual_max={reg_error:.9f}, "
        f"logits_max={logits_error:.9f}"
    )

    if (
        not np.isfinite(reg_error)
        or not np.isfinite(logits_error)
        or reg_error > 1e-4
        or logits_error > 1e-4
    ):
        raise RuntimeError(
            "GroupNorm parity failed; export cancelled."
        )

print("GROUPNORM PARITY: PASS", flush=True)

OUT.mkdir(parents=True, exist_ok=True)

trace = torch.jit.trace(
    wrapper,
    input_sample,
    strict=True,
    check_trace=True,
)
trace.eval()

print("\nConverting CNN to Core ML...", flush=True)

coreml = ct.convert(
    trace,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS17,
    compute_precision=ct.precision.FLOAT32,
    inputs=[
        ct.TensorType(
            name="input_volume",
            shape=(1, 9, nz, ny, nx),
            dtype=np.float32,
        )
    ],
    outputs=[
        ct.TensorType(name="residual"),
        ct.TensorType(name="logits"),
    ],
)

if MODEL_PATH.exists():
    print("Using existing Core ML package.")
else:
    coreml.save(str(MODEL_PATH))

print("Core ML model saved:", MODEL_PATH)
print("\nLoading Core ML model for parity...", flush=True)

runtime = ct.models.MLModel(
    str(MODEL_PATH),
    compute_units=ct.ComputeUnit.CPU_ONLY,
)

overall_pass = True
report = {
    "checkpoint_sha256": original_hash,
    "checkpoint_epoch": state["epoch"],
    "source_volume": name,
    "scenario": scenario,
    "grid_xyz": [nx, ny, nz],
    "spacing_m": list(map(float, spacing)),
    "precision": "float32",
    "deployment_target": "iOS17",
    "cases": [],
}

for count in (1, 3, 8, 32):
    batch = prepare(
        q, f, valid, spacing, sites[:count]
    )

    x = np.ascontiguousarray(
        batch["x"][None], dtype=np.float32
    )

    with torch.inference_mode():
        ref_reg, ref_logits = wrapper(
            torch.from_numpy(x)
        )

    ref_reg = ref_reg.numpy()
    ref_logits = ref_logits.numpy()

    prediction = runtime.predict({
        "input_volume": x
    })

    ml_reg = np.asarray(
        prediction["residual"]
    )
    ml_logits = np.asarray(
        prediction["logits"]
    )

    if (
        ml_reg.shape != ref_reg.shape
        or ml_logits.shape != ref_logits.shape
    ):
        raise RuntimeError(
            f"Core ML output shape mismatch at {count} CPT"
        )

    reg_mae, reg_max = errors(
        ref_reg, ml_reg
    )
    logits_mae, logits_max = errors(
        ref_logits, ml_logits
    )

    torch_q, torch_f = physical_outputs(
        ref_reg, batch
    )
    core_q, core_f = physical_outputs(
        ml_reg, batch
    )

    q_mae, q_max = errors(
        torch_q, core_q
    )
    f_mae, f_max = errors(
        torch_f, core_f
    )

    observed = np.asarray(
        batch["observed"], dtype=bool
    )

    measured_q_error = float(
        np.max(np.abs(
            core_q[observed]
            - np.asarray(batch["q"])[observed]
        ))
    )
    measured_f_error = float(
        np.max(np.abs(
            core_f[observed]
            - np.asarray(batch["f"])[observed]
        ))
    )

    passed = (
        reg_max <= 0.002
        and logits_max <= 0.005
        and q_max <= 0.05
        and f_max <= 0.005
        and measured_q_error <= 1e-6
        and measured_f_error <= 1e-6
    )

    overall_pass &= passed

    row = {
        "cpt_count": count,
        "status": "PASS" if passed else "FAIL",
        "regression_mae": reg_mae,
        "regression_max_error": reg_max,
        "logits_mae": logits_mae,
        "logits_max_error": logits_max,
        "qc_mae_mpa": q_mae,
        "qc_max_error_mpa": q_max,
        "fs_mae": f_mae,
        "fs_max_error": f_max,
        "measured_qc_max_error": measured_q_error,
        "measured_fs_max_error": measured_f_error,
    }

    report["cases"].append(row)

    print(f"\n=== {count} CPT ===")
    print("Status:", row["status"])
    print(
        f"Residual max error: {reg_max:.8f}"
    )
    print(
        f"Logits max error: {logits_max:.8f}"
    )
    print(
        f"qc MAE/max: {q_mae:.8f} / {q_max:.8f} MPa"
    )
    print(
        f"fs MAE/max: {f_mae:.8f} / {f_max:.8f}"
    )
    print(
        "Measured qc/fs max error:",
        measured_q_error,
        measured_f_error,
    )

report["overall_status"] = (
    "PASS" if overall_pass else "FAIL"
)

(OUT / "parity_report.json").write_text(
    json.dumps(report, indent=2)
)

print("\n=== CORE ML PARITY RESULT ===")
print("Overall:", report["overall_status"])
print("Report:", OUT / "parity_report.json")
print("Frozen checkpoint unchanged.")

if not overall_pass:
    raise SystemExit(
        "PARITY FAILED: do not integrate this model."
    )

print("=== CORE ML EXPORT COMPLETE ===")
print("Ready for additional validation.")
