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

## iOS Integration

See [iOS Integration Guide](docs/IOS_INTEGRATION.md) for model contracts, integration order, probability calculation, and parity testing.

## Limitations

- Results are primarily based on synthetic GeoSyn data.
- Model performance on actual construction sites has not been established.
- Reconstruction sufficiency does not establish bearing capacity, suitable foundation strata, or foundation safety.
- Model output does not replace geotechnical engineering judgment or SNI 8460:2017 requirements.
- Automatic field investigation stopping is not approved.
- Swift runtime parity and upstream training-data provenance still require verification.
