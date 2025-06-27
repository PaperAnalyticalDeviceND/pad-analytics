# PAD ML Workflow

A Python package for analyzing Paper Analytical Device (PAD) images using machine learning to detect and quantify pharmaceutical compounds.

## Overview

Paper Analytical Devices (PADs) are low-cost diagnostic tools that use colorimetric reactions to detect the presence and concentration of substances. This package provides a complete workflow for:

- Retrieving PAD image data from the PAD API v2
- Applying pre-trained machine learning models (Neural Networks and PLS)
- Analyzing colorimetric patterns to identify drugs and estimate concentrations
- Evaluating model performance with statistical metrics

## Key Features

- **🔬 Drug Identification**: Classify pharmaceutical compounds from PAD images
- **📊 Concentration Estimation**: Quantify drug concentrations using colorimetric analysis
- **🤖 Dual Model Support**: Neural Network (TensorFlow Lite) and PLS regression models
- **🖼️ Automated Image Processing**: Extract color features from specific PAD regions
- **📈 Performance Metrics**: Built-in RMSE calculation and accuracy assessment
- **🔌 API Integration**: Seamless connection to PAD data repository

## Installation

```bash
pip install pad-ml-workflow
```

For development:
```bash
git clone https://github.com/PaperAnalyticalDeviceND/pad-ml-workflow-v2.git
cd pad-ml-workflow-v2
pip install -e .
```

## Quick Start

```python
from pad_ml_workflow import padanalytics as pad

# List available projects
projects = pad.get_projects()

# Analyze a specific PAD card
card = pad.get_card(card_id=19208)

# Make a prediction using a pre-trained model
actual_label, prediction = pad.predict(card_id=19208, model_id=18)

# For PLS models: prediction is the concentration (float)
# For NN models: prediction is (drug_name, confidence, energy)
```

## Core Workflows

### 1. Drug Identification
Identify which pharmaceutical compound is present on a PAD:

```python
# Using a neural network classifier
actual, (predicted_drug, confidence, energy) = pad.predict(
    card_id=12345, 
    model_id=15  # NN classifier model
)
print(f"Predicted: {predicted_drug} (confidence: {confidence:.2f})")
```

### 2. Concentration Quantification
Estimate drug concentration using colorimetric analysis:

```python
# Using a PLS regression model
actual_conc, predicted_conc = pad.predict(
    card_id=12345,
    model_id=18  # PLS quantification model
)
print(f"Predicted concentration: {predicted_conc:.2f} µg/mL")
```

### 3. Batch Analysis
Process multiple samples and evaluate model performance:

```python
# Get a dataset
dataset = pad.get_dataset("FHI2022")

# Apply model to all samples
results = pad.apply_predictions_to_dataframe(dataset, model_id=18)

# Calculate performance metrics
rmse_by_drug = pad.calculate_rmse_by_api(results)
```

### 4. Visualization (Jupyter Notebooks)
Display PAD images with analysis results:

```python
# Show a single card with prediction
pad.show_prediction(card_id=12345, model_id=18)

# Display multiple cards in a grid
pad.show_cards([12345, 12346, 12347])
```

## How It Works

1. **Image Acquisition**: PAD images are captured after applying test samples
2. **Region Extraction**: The image is divided into lanes (A-L) with multiple test regions
3. **Color Analysis**: RGB/LAB values are extracted from each region
4. **Feature Vector**: Color values form a feature vector for ML models
5. **Prediction**: Models classify the drug type or estimate concentration
6. **Validation**: Results are compared against known values for accuracy

## Model Types

### Neural Networks (TensorFlow Lite)
- Used for drug classification and concentration prediction
- Returns: (prediction, probability, energy score)
- Suitable for complex non-linear patterns

### PLS (Partial Least Squares)
- Statistical regression for concentration estimation
- Returns: predicted concentration (float)
- Effective for linear relationships

## API Reference

### Data Access
- `get_projects()` - List all available projects
- `get_card(card_id)` - Retrieve specific card data
- `get_models()` - List available ML models
- `get_dataset(name)` - Load a named dataset

### Prediction
- `predict(card_id, model_id)` - Single prediction
- `apply_predictions_to_dataframe(df, model_id)` - Batch predictions

### Visualization
- `show_card(card_id)` - Display card with metadata
- `show_prediction(card_id, model_id)` - Show prediction results

### Metrics
- `calculate_rmse_by_api(results)` - RMSE by drug type

## Use Cases

- **Quality Control**: Verify pharmaceutical authenticity in field settings
- **Research**: Develop new PAD assays and validate ML models
- **Education**: Teach colorimetric analysis and machine learning
- **Deployment**: Enable drug quality testing in resource-limited settings

## Requirements

- Python >= 3.8
- TensorFlow >= 2.13.0
- OpenCV
- NumPy, Pandas, scikit-learn

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{pad_ml_workflow,
  title = {PAD ML Workflow: Machine Learning for Paper Analytical Devices},
  author = {Paper Analytical Device Team, University of Notre Dame},
  year = {2024},
  url = {https://github.com/PaperAnalyticalDeviceND/pad-ml-workflow-v2}
}
```

## License

MIT License - see [LICENSE](LICENSE) file

## Acknowledgments

This project is supported by the University of Notre Dame and the Paper Analytical Device research group.

## Links

- [PAD API Documentation](https://pad.crc.nd.edu/docs)
- [Project Homepage](https://github.com/PaperAnalyticalDeviceND/pad-ml-workflow-v2)
- [Issue Tracker](https://github.com/PaperAnalyticalDeviceND/pad-ml-workflow-v2/issues)