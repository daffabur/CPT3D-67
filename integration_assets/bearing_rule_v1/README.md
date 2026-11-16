# Bearing Rule Integration Assets

This package contains reference model checkpoints, source
implementations, and example app-contract exports copied from
cpt_super_learn.

## Models
- QC-FS CNN V3: frozen qc/fs reconstruction.
- SchemaGAN Ic V4.2: frozen Ic reconstruction.

## Bearing
- bearing_rule_engine_v1.py: candidate mask computation.
- bearing_rule_engine_v2.py: spatial and depth outputs.
- export_bearing_app_contract_v3.py: binary + JSON example.

## Grid
- Logical array axes: X,Y,Z.
- Binary volumes: C-contiguous, Z-fastest.
- Flat index: (x * ny + y) * nz + z.
- Depth positive down from the top of the grid.

## Verification
- V1 label parity: 270/270 masks matched.
- Example contracts tested for 8 and 32 CPT.
- These example volumes are synthetic, not field data.

## Important
These are reference assets, not a standalone iOS runtime.
The training scripts have dependencies on the research
repository and datasets. Native inference, numerical parity,
coordinate alignment, and application criteria still need
to be verified during integration.

The Ic filter is not yet validated as a bearing criterion.
The output indicates potential candidate strata only.

See asset_manifest.json for SHA-256 checksums.
