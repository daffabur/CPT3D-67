# QC-FS CNN V3 Core ML — Multi-Grid

Frozen QC-FS CNN V3 exported to Core ML.

## Supported grids (XYZ)
- 24 x 24 x 12
- 32 x 32 x 16
- 40 x 40 x 24
- 48 x 32 x 20

Core ML tensor convention: [1, 9, Z, Y, X].

## Inference
Inputs are generated using IDW interpolation,
measured CPT qc/fs, an observed mask, normalized
distance, and normalized XYZ coordinate channels.

Outputs:
- residual: 2 channels for qc/fs log corrections
- logits: 3 threshold-detection channels

The Swift application must implement preprocessing
and physical-unit postprocessing consistently with
the frozen Python reference. Preserve measured CPT
values after inference.

## Validation
12 of 12 multi-grid numerical parity cases passed
on macOS Core ML CPU_ONLY.

## Restrictions
This is a four-shape enumerated Core ML export,
not an arbitrary-grid or field-validated model.
Further testing on real iOS hardware is required.

See asset_manifest.json and validation reports.
