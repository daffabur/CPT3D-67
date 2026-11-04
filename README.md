# CPT3D-67

Geotechnical machine learning models for 3D CPT reconstruction, sampling recommendations, and reconstruction risk estimation.

## Models

- SchemaGAN V4.2 Flexible: 3D Ic reconstruction from CPT measurements.
- DQN V4.5 Extended: CPT sampling location recommendations.
- Uncertainty V2: estimated reconstruction error risk.

## Structure

- SchemaGAN_V4_2/: SchemaGAN checkpoint, normalization, and Core ML model.
- DQN_V4_5/: DQN checkpoint, Core ML model, and feature manifest.
- Uncertainty_V2/: regression model, exported JSON trees, and metadata.
- Source_Code/: selected Python source code.

## Limitations

The models are research prototypes. Synthetic evaluation does not establish accuracy on real-world construction sites. Estimated Reconstruction Risk is not calibrated geological confidence.
