# Bearing Reliability V1

Experimental synthetic-calibration indicators for
Potential Bearing Layer Depth Map.

## Inputs
- Candidate depth map (Y,X)
- CPT grid coordinates (X,Y)
- Grid spacing dx_m, dy_m
- Optional valid XY mask

## Outputs
- Conditional depth error P90 (metres)
- Estimated detection risk map
- High-risk area percentage
- Supported and unsupported area percentages

## Calibration restrictions
- qc >= 10 MPa
- Ic <= 2.2
- Minimum continuous thickness >= 1 m
- CPT counts: 1, 3, 5, 8, 16, 32

P90 describes errors conditional on true-positive
detections, not all possible model outcomes.

High-risk classification uses an experimental
25% estimated detection-risk threshold.
Holdout evaluation found risk underestimation.

No field validation. No automatic engineering
decisions or site-investigation stopping.
