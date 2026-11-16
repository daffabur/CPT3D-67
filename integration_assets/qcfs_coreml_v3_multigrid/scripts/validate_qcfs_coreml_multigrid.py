from pathlib import Path
import ast
import json
import numpy as np
import torch
import coremltools as ct

BASE = Path("scripts/geotech/v4")
VERIFY = BASE / "verify_qc_preservation_v1.py"

OUT = Path(
    "results/qc_fs_joint_v3/"
    "coreml_export_multigrid_v1"
)

PACKAGE = OUT / "QCFS_CNN_V3.mlpackage"
REPORT = OUT / "multigrid_parity.json"

if not PACKAGE.is_dir():
    raise SystemExit("Multi-grid .mlpackage not found")

# Load the original validated PyTorch inference setup.
tree = ast.parse(VERIFY.read_text())

cuts = [
    i for i, node in enumerate(tree.body)
    if isinstance(node, ast.With)
]

if len(cuts) != 1:
    raise RuntimeError("Unexpected verification structure")

exec(
    compile(
        ast.Module(
            body=tree.body[:cuts[0]],
            type_ignores=[]
        ),
        str(VERIFY),
        "exec",
    ),
    globals(),
)

SHAPES = (
    (24, 24, 12),
    (32, 32, 16),
    (40, 40, 24),
    (48, 32, 20),
)

selected = {}

for name in files:
    q, f, valid, spacing, scenario = load_volume(
        DATA / name
    )

    shape = tuple(q.shape)

    if shape in SHAPES and shape not in selected:
        selected[shape] = name

if set(selected) != set(SHAPES):
    raise RuntimeError(
        f"Missing grids: {set(SHAPES) - set(selected)}"
    )

runtime = ct.models.MLModel(
    str(PACKAGE),
    compute_units=ct.ComputeUnit.CPU_ONLY,
)

print("=== CORE ML MULTIGRID PARITY ===")
print("Selected grids:", len(selected))

rows = []
all_pass = True

with torch.inference_mode():
    for shape in SHAPES:
        name = selected[shape]
        q, f, valid, spacing, scenario = load_volume(
            DATA / name
        )

        nx, ny, nz = q.shape

        sites = get_sites(
            valid,
            spacing,
            name,
            "distributed",
            epoch=-1
        )

        if len(sites) < 32:
            raise RuntimeError(
                f"Insufficient CPT sites: {name}"
            )

        print(
            f"\n=== GRID XYZ {shape} | {name} ===",
            flush=True
        )

        for count in (1, 8, 32):
            batch = prepare(
                q, f, valid, spacing,
                sites[:count]
            )

            x = np.ascontiguousarray(
                batch["x"][None],
                dtype=np.float32
            )

            if x.shape != (1, 9, nz, ny, nx):
                raise RuntimeError("Input shape mismatch")

            reference_q, reference_f, reference_logits, _ = (
                predict(model, batch)
            )

            reference_q = (
                reference_q.detach().cpu().numpy()
            )
            reference_f = (
                reference_f.detach().cpu().numpy()
            )
            reference_logits = (
                reference_logits.detach().cpu().numpy()
            )

            prediction = runtime.predict({
                "input_volume": x
            })

            residual = np.asarray(
                prediction["residual"],
                dtype=np.float64
            )

            logits = np.asarray(
                prediction["logits"],
                dtype=np.float64
            )

            expected_reg = (1, 2, nz, ny, nx)
            expected_logits = (1, 3, nz, ny, nx)

            if residual.shape != expected_reg:
                raise RuntimeError(
                    f"Residual shape: {residual.shape}"
                )
            if logits.shape != expected_logits:
                raise RuntimeError(
                    f"Logits shape: {logits.shape}"
                )

            observed = np.asarray(
                batch["observed"], dtype=bool
            )

            ml_q = np.exp(
                np.log(
                    np.maximum(
                        batch["idw_q"], 1e-6
                    )
                ) + residual[0, 0]
            )

            ml_f = np.exp(
                np.log(
                    np.maximum(
                        batch["idw_f"], 1e-8
                    )
                ) + residual[0, 1]
            )

            ml_q = np.where(
                observed, batch["q"], ml_q
            )
            ml_f = np.where(
                observed, batch["f"], ml_f
            )

            qc_error = float(
                np.max(np.abs(ml_q-reference_q))
            )
            fs_error = float(
                np.max(np.abs(ml_f-reference_f))
            )
            logits_error = float(
                np.max(np.abs(
                    logits[0]-reference_logits
                ))
            )

            observed_q_error = float(
                np.max(np.abs(
                    ml_q[observed]
                    - batch["q"][observed]
                ))
            )
            observed_f_error = float(
                np.max(np.abs(
                    ml_f[observed]
                    - batch["f"][observed]
                ))
            )

            passed = (
                np.isfinite(qc_error)
                and np.isfinite(fs_error)
                and np.isfinite(logits_error)
                and qc_error <= 0.05
                and fs_error <= 0.005
                and logits_error <= 0.005
                and observed_q_error <= 1e-6
                and observed_f_error <= 1e-6
            )

            all_pass &= passed

            rows.append({
                "grid_xyz": list(shape),
                "volume": name,
                "scenario": scenario,
                "cpt_count": count,
                "qc_max_error_mpa": qc_error,
                "fs_max_error": fs_error,
                "logits_max_error": logits_error,
                "observed_qc_max_error": observed_q_error,
                "observed_fs_max_error": observed_f_error,
                "status": "PASS" if passed else "FAIL",
            })

            print(
                f"{count} CPT: "
                f"{'PASS' if passed else 'FAIL'} | "
                f"qc_max={qc_error:.8f} MPa | "
                f"fs_max={fs_error:.8f} | "
                f"logits_max={logits_error:.8f}",
                flush=True
            )

REPORT.write_text(
    json.dumps({
        "model": "QCFS_CNN_V3",
        "input_order": "N,C,Z,Y,X",
        "grid_shapes_xyz": [
            list(s) for s in SHAPES
        ],
        "cases": rows,
        "overall_status": (
            "PASS" if all_pass else "FAIL"
        ),
    }, indent=2)
)

print("\n=== MULTIGRID RESULT ===")
print("Total cases:", len(rows))
print("Overall:", "PASS" if all_pass else "FAIL")
print("Report:", REPORT)
print("Frozen checkpoint unchanged.")

if not all_pass:
    raise SystemExit(
        "MULTIGRID PARITY FAILED — do not integrate."
    )

print("=== MULTIGRID VALIDATION COMPLETE ===")
