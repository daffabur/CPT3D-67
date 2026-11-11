# Bearing UI Contract V1

JSON output contract for displaying potential bearing layers
and experimental reliability estimates in iOS SwiftUI.

## Inputs
- Candidate depth map: Y,X, metres
- CPT locations: X,Y grid indices
- Horizontal resolution: dx_m, dy_m
- Optional valid XY mask
- Flexible bearing criteria

## Outputs
- Shallowest candidate depth
- Deepest candidate depth
- Depth variation
- Candidate depth map
- Per-location candidate status
- Conditional depth uncertainty P90
- Estimated detection risk map
- High risk area percentage

## Data conventions
- Grid coordinate order: X,Y
- Depth map order: Y,X
- Missing depth: JSON null
- Depth values: metres
- Reliability: experimental synthetic calibration

## Supported reliability calibration
- CPT counts: 1, 3, 5, 8, 16, 32
- qc >= 10 MPa
- Ic <= 2.2
- Minimum thickness >= 1 m

Other criteria or CPT counts must report unavailable
reliability rather than extrapolating calibration.

## Limitations
No field validation.
No automatic stopping decisions.
No bearing capacity calculation.
Potential bearing layer is not a confirmed bearing stratum.
