# BearingLayer Final V1

## Components
- QC-FS V3-D checkpoint (frozen)
- qc/fs normalization
- Flexible Bearing V1 strict threshold algorithm
- Final synthetic evaluation metrics
- Reliability and display policy

## Default experimental criteria
- qc >= 10 MPa
- Ic <= 2.2
- Minimum continuous thickness >= 1 m

All thresholds are user-adjustable.

## Algorithm input
- qc_mpa: 3D volume (Z,Y,X)
- ic: 3D volume (Z,Y,X)
- valid_mask: boolean 3D volume (Z,Y,X)
- dz_m: vertical grid spacing in metres

## Algorithm output
- Potential candidate layer starting depth (Y,X)
- NaN means no qualifying interval was detected.
- NaN is NOT proof that no bearing layer exists.

## Integration
SchemaGAN V4.2 and DQN remain separate frozen models.
QC-FS weights are included, but a standalone QC-FS inference
adapter and portable dependency test are still required before
the package can be considered fully portable.

The package is NOT a Core ML export.

## Limitations
Research prototype evaluated on synthetic data.
Not validated for engineering decisions in the field.
No calibrated per-location confidence.
Do not use for automatic investigation stopping.
Potential bearing layer does not equal confirmed
foundation bearing stratum.
