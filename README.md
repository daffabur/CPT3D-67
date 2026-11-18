# CPT3D-67

Research repository for adaptive CPT site investigation, 3D Ic reconstruction, uncertainty estimation, and CPT sampling recommendations.

## Project Overview

CPT3D-67 investigates how machine learning can assist geotechnical engineers in reconstructing subsurface conditions, estimating reconstruction uncertainty, recommending additional CPT locations, and assessing reconstruction sufficiency.

The application can begin with **1 CPT** and progressively incorporate additional observations.

## Model Inventory

| Model | Purpose | Location |
|---|---|---|
| SchemaGAN V4.2 Flexible | 3D Ic reconstruction | `SchemaGAN_V4_2/` |
| DQN V4.5 Extended | Previous recommendation model | `DQN_V4_5/` |
| DQN V4.6-A | Updated recommendation model | `model_export/DQN_V4_6_A_iOS/` |
| Uncertainty V2 | Previous reconstruction error model | `Uncertainty_V2/` |
| Uncertainty V3.1 | Updated iOS uncertainty export | `model_export/Uncertainty_V3_1_iOS/` |
| Sufficiency Gate V2.1 | Probabilistic reconstruction sufficiency | `model_export/SufficiencyGate_V2_1/` |
| QC-FS CNN V3 | 3D qc/fs reconstruction, frozen multigrid Core ML | `integration_assets/qcfs_coreml_v3_multigrid/` |
| Bearing Rule V1/V2 | Deterministic candidate bearing-stratum detection | `integration_assets/bearing_rule_v1/` |
| Bearing Reliability V1 | Earlier experimental reliability lookup | `model_export/BearingReliability_V1/` |

## Research Pipeline

1. **CPT Input:** Measured CPT profiles and XY coordinates.
2. **SchemaGAN V4.2:** Reconstruct 3D Ic.
3. **Uncertainty:** Estimate reconstruction error and predicted coverage.
4. **Sufficiency Gate V2.1:** Estimate whether reconstruction sufficiency criteria are met.
5. **DQN V4.6-A:** Recommend additional CPT locations when appropriate.
6. Repeat after new CPT measurements become available.

The Uncertainty V3.1 iOS export is not yet verified as equivalent to the Uncertainty V3 quantile features used in Gate V2.1. Verify compatibility before connecting them.

## Sufficiency Gate V2.1

Experimental configuration:

- Probability threshold: **0.995**
- Minimum CPT for gate eligibility: **5**
- Actual coverage target: **90%**
- Ic absolute error tolerance: **0.15**
- Evaluated CPT counts: **1-16**

The application can receive 1 CPT. The 5-CPT minimum only applies to the experimental sufficiency gate decision.

### Final GeoSyn Evaluation

Evaluated on 200 synthetic volumes and 3,200 states.

| Metric | Result |
|---|---|
| AUC | 0.9519 |
| Brier score | 0.0933 |
| Unsafe first STOP | 0/200 |
| Safe first STOP | 51/200 |
| Missed-stop rate | 63.60% |
| Mean CPT at first STOP | 6.37 |

**Status: experimental_frozen**

Zero observed unsafe STOP does not guarantee zero risk on future data.

## Repository Structure

- `SchemaGAN_V4_2/`: Frozen reconstruction assets and normalization.
- `DQN_V4_5/`: Previous recommendation model.
- `Uncertainty_V2/`: Previous uncertainty model.
- `model_export/`: Newer iOS assets and Sufficiency Gate V2.1.
- `Source_Code/`: Selected research source code.
- `Research/`: Research evaluations.
- `docs/`: Integration documentation.

## Preprocessing for iOS Integration

The frozen Core ML models require preprocessing before inference. The Python files below are reference implementations for reproducing the model inputs in Swift.

### SchemaGAN V4.2 Flexible

| Component | File |
|---|---|
| Core ML model | `SchemaGAN_V4_2/SchemaGAN_V4_2_Flexible.mlpackage` |
| Normalization | `SchemaGAN_V4_2/normalization.json` |
| Python preprocessing | `Source_Code/schemagan_v42/flexible_cpt_preprocessing.py` |
| Reference evaluation | `Source_Code/schemagan_v42/test_flexible_cpt_v42_v4b.py` |

Integration flow:

1. Receive measured CPT profiles, XY coordinates, depths, and Ic values.
2. Prepare the project grid and physical cell resolution.
3. Build sparse observations, observation masks, and spatial input features using the Python reference.
4. Apply the exported normalization and preserve the required tensor axis order.
5. Run SchemaGAN Core ML inference.
6. Convert the prediction back to physical Ic values and preserve measured CPT observations.

The reconstruction grid and number of CPT observations may vary. Do not assume a fixed number of initial CPTs.

### DQN V4.6-A

| Component | File |
|---|---|
| Core ML model | `model_export/DQN_V4_6_A_iOS/DQN_V4_6_A.mlpackage` |
| Model contract | `model_export/DQN_V4_6_A_iOS/DQN_V4_6_A_contract.json` |
| Python feature builder | `Source_Code/dqn3d/v46/features.py` |
| Parity reference | `model_export/DQN_V4_6_A_iOS/DQN_V4_6_A_parity.json` |

Integration flow:

1. Receive the reconstructed Ic volume, measured CPT mask, valid-volume mask, grid resolution, and candidate locations.
2. Select legal XY candidates according to the exported contract.
3. Calculate 18 candidate features and 11 global context features.
4. Concatenate features in the exact 29-feature order defined in the JSON contract.
5. Run DQN Core ML inference with input shape `[N,29]`.
6. Select the legal candidate with the highest Q-value and convert it into an XY recommendation.

The DQN Q-value is a candidate ranking score, not a geological confidence score or automatic stopping criterion.

### Parallel Development

| SchemaGAN Developer | DQN Developer |
|---|---|
| Implement CPT preprocessing | Implement the 29-feature builder |
| Integrate SchemaGAN Core ML | Integrate DQN Core ML |
| Produce physical Ic reconstruction | Consume reconstructed Ic and CPT state |
| Verify observation preservation | Verify legal candidates and Q-value parity |

The DQN developer can use mock reconstructed Ic volumes while SchemaGAN integration is in progress. Both developers must agree on the grid geometry, coordinate order, valid mask, measured CPT mask, Ic units, and volume axis order.

### Validation Requirements

- Compare Swift preprocessing outputs against Python reference outputs using identical inputs.
- Verify normalization and tensor dimensions.
- Verify XY/Z axis orientation and physical grid resolution.
- Verify that measured CPT values are preserved.
- Verify all 29 DQN features and recommendation ranking against the exported parity reference.
- Do not substitute older V4.5 feature calculations for V4.6-A without checking equivalence.
- The supplied Python references may require additional modules from the original research project; copying a single file does not guarantee standalone execution.

See [iOS Integration Guide](docs/IOS_INTEGRATION.md) for the remaining integration details.

## iOS Integration

See [iOS Integration Guide](docs/IOS_INTEGRATION.md) for model contracts, integration order, probability calculation, and parity testing.

## Bearing Layer Integration

### Architecture

The Bearing Layer research pipeline consists of:

1. Import measured CPT profiles and spatial coordinates.
2. Construct a shared physical grid.
3. Reconstruct 3D qc/fs using frozen QC-FS CNN V3.
4. Preserve measured CPT observations.
5. Apply deterministic Bearing Rule Engine V1/V2.
6. Export candidate masks, components, and depth intervals.
7. Visualize candidate bearing strata in the iOS application.

SchemaGAN V4.2 reconstructs Ic independently. Ic is not
currently a validated Bearing Layer classification criterion.

### Frozen QC-FS CNN V3

- Checkpoint: `integration_assets/bearing_rule_v1/results/qc_fs_joint_v3/spatial_cnn_v3/best.pt`
- Core ML: `integration_assets/qcfs_coreml_v3_multigrid/QCFS_CNN_V3.mlpackage`
- Input: `[N,C,Z,Y,X]`, nine channels.
- Outputs: residual corrections for qc/fs and auxiliary logits.
- The preprocessing and postprocessing contracts must be
  reproduced exactly in Swift.
- Measured CPT observations must be preserved.

Validated Core ML grid shapes (XYZ):

- `24 x 24 x 12`
- `32 x 32 x 16`
- `40 x 40 x 24`
- `48 x 32 x 20`

Other grid shapes are not covered by this Core ML export's
enumerated-shape validation.

Core ML multigrid parity: 12/12 cases passed on macOS.
This does not establish complete iOS device validation.

### Bearing Rule Engine

The deterministic rule engine identifies candidate regions
using qc thresholds, minimum vertical thickness, lateral
footprint filtering, and spatial connectivity.

Reference criteria include:

- Default qc threshold: 15 MPa.
- Minimum vertical thickness: 2 m.
- Minimum lateral footprint fraction: 0.05.
- Connected candidate regions in 3D.

Criteria are configurable. Changing threshold settings
does not require retraining QC-FS CNN.

Reference assets:
`integration_assets/bearing_rule_v1/`

Rule-label parity: 270/270 tests passed.

### Bearing Layer Uncertainty

Bearing Layer uncertainty and Ic reconstruction uncertainty
are different prediction targets.

- Uncertainty V3.1 concerns Ic reconstruction and sufficiency.
- A validated estimator of Bearing Layer candidate/depth
  uncertainty is not yet deployed.
- Bearing-specific uncertainty experiments are ongoing.
- Nearest-CPT distance is not a calibrated probability of error.
- No-candidate predictions do not establish candidate absence.

Do not use Ic uncertainty as a substitute for Bearing Layer
depth uncertainty without separate validation.

### Bearing Reliability and Legacy Assets

The repository also contains earlier experimental packages:

- `model_export/BearingLayer_Final_V1/`
- `model_export/BearingReliability_V1/`
- `model_export/BearingUIContract_V1/`
- `model_export/QCFS_CoreML_V1/`

These are versioned research or integration artifacts and
must not be assumed interchangeable with the current frozen
QC-FS CNN V3 multigrid and deterministic Bearing Rule pipeline.

### Integration Status

The Terravise iOS application integrates ML inference with
SwiftUI, Core ML, and its existing visualization components.

This research repository provides reference implementations,
exported models, manifests, and validation assets. Actual
Terravise UI behavior, on-device performance, and deployment
status must be verified in the Terravise repository.

Candidate bearing strata are preliminary indications, not
validated bearing capacity or foundation design approval.

## Limitations

- Results are primarily based on synthetic GeoSyn data.
- Model performance on actual construction sites has not been established.
- Reconstruction sufficiency does not establish bearing capacity, suitable foundation strata, or foundation safety.
- Model output does not replace geotechnical engineering judgment or SNI 8460:2017 requirements.
- Automatic field investigation stopping is not approved.
- Swift runtime parity and upstream training-data provenance still require verification.

## Research Progress — November 2026

### CPT Sufficiency & Optimization
- Completed initial evaluation of DQN V4.6-A against farthest-point sampling.
- Evaluated CPT sufficiency across different grid sizes and resolutions.
- Identified cases where adding more CPT measurements provided limited reconstruction improvement.
- Current focus: reducing unnecessary CPT recommendations, evaluating marginal information gain, and improving stopping criteria.
- Status: **Ongoing research — automatic stopping is not yet validated.**

### Bearing Layer Improvement
- Integrated QCFS CNN V3 for 3D qc/fs reconstruction.
- Completed initial QCFS V4 training and baseline comparison.
- Integrated experimental Bearing Uncertainty V3 into Terravise.
- Current focus: improving hidden high-qc detection at 10, 12, and 15 MPa, reducing missed candidate layers, and evaluating QCFS V4 against V3.
- Status: **Ongoing research — further validation required.**

### Latest Research Checkpoints
- DQN V4.6-A: `results/dqn_v46a_experimental/training_5000/best.pt`
- QCFS V4: `results/qc_fs_v4/combined_existing_v1/best.pt`
- Bearing Uncertainty V3: `results/bearing_layer_v1/bearing_uncertainty_v3_step9b_model/candidate_estimator.joblib`

All existing frozen baselines remain unchanged. New research improvements will be evaluated before integration into Terravise.