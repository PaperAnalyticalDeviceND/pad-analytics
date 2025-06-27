# PAD ML Workflow

[![Python Version](https://img.shields.io/pypi/pyversions/pad-ml-workflow)](https://pypi.org/project/pad-ml-workflow/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/PAD%20API%20v2-Documentation-blue?logo=swagger)](https://pad.crc.nd.edu/docs)

A comprehensive Python package for machine learning workflows using Paper Analytical Device (PAD) data from the PAD API v2. This package provides tools for data exploration, model training, and evaluation specifically designed for colorimetric analysis of paper-based diagnostic devices.

## Features

- 🔍 **Data Exploration**: Easy access to PAD API v2 for retrieving projects, cards, and samples
- 🤖 **Model Support**: Built-in support for both Neural Network (TensorFlow Lite) and PLS models
- 📊 **Analysis Tools**: Comprehensive functions for image processing and colorimetric analysis
- 🖼️ **Visualization**: Interactive widgets for displaying PAD images and results (Jupyter compatible)
- 📈 **Evaluation Metrics**: Built-in RMSE calculation and prediction accuracy assessment
- 🔧 **Preprocessing**: Image processing utilities optimized for PAD analysis

## Installation

### From PyPI (Coming Soon)
```bash
pip install pad-ml-workflow
```

### From GitHub
```bash
pip install git+https://github.com/PaperAnalyticalDeviceND/pad-ml-workflow-v2.git
```

### Development Installation
```bash
git clone https://github.com/PaperAnalyticalDeviceND/pad-ml-workflow-v2.git
cd pad-ml-workflow-v2
pip install -e .
```

## Quick Start

```python
from pad_ml_workflow import padanalytics as pad

# Get all projects
projects = pad.get_projects()

# Get a specific card
card = pad.get_card(card_id=12345)

# Make a prediction
actual_label, prediction = pad.predict(card_id=12345, model_id=18)

# For neural network models, prediction is a tuple: (label, probability, energy)
# For PLS models, prediction is a float concentration value
```

## Core Functionality

### Data Access

```python
# Get projects
projects = pad.get_projects()

# Get cards for a project
cards = pad.get_project_cards(project_id=123)
# or by project name
cards = pad.get_project_cards(project_name="MyProject")

# Get card by sample ID
cards = pad.get_card_by_sample_id(sample_id=456)
```

### Model Prediction

```python
# Get available models
models = pad.get_models()

# Make predictions
actual, predicted = pad.predict(card_id=19208, model_id=18)

# Apply predictions to entire dataset
results = pad.apply_predictions_to_dataframe(dataset_df, model_id=18)
```

### Visualization (Jupyter Notebooks)

```python
# Show a single card with details
pad.show_card(card_id=12345)

# Show multiple cards
pad.show_cards([12345, 12346, 12347])

# Show cards grouped by a column
pad.show_grouped_cards(cards_df, group_column='sample_name', images_per_row=5)

# Show prediction results
pad.show_prediction(card_id=12345, model_id=18)
```

### Image Processing

```python
from pad_ml_workflow import regionRoutine, pixelProcessing

# Process PAD image regions
results = regionRoutine.fullRoutine(image, intensity_function, {}, True, 10)

# Average pixel values (with overflow protection)
avg_r, avg_g, avg_b = pixelProcessing.avgPixels(pixel_list, image)
```

## Advanced Usage

### Working with Datasets

```python
# Get dataset list
datasets = pad.get_dataset_list()

# Get dataset by name
dataset = pad.get_dataset("FHI2022")

# Get dataset associated with a model
dataset = pad.get_dataset_from_model_id(model_id=18)
```

### Custom Analysis

```python
# Calculate RMSE by API
rmse_results = pad.calculate_rmse_by_api(predictions_df)

# Custom PLS analysis
pls_model = pad.pls('path/to/coefficients.csv')
concentration = pls_model.quantity('image.png', 'drug_name')
```

## API Reference

### Main Functions

- `get_projects()`: Retrieve all projects from PAD API
- `get_card(card_id)`: Get specific card data
- `get_models()`: List available ML models
- `predict(card_id, model_id)`: Make prediction for a card
- `show_card(card_id)`: Display card with image and metadata

### Image Processing

- `avgPixels(pixels, img)`: Calculate average RGB values with overflow protection
- `avgPixelsHSV(pixels, img)`: Calculate average HSV values
- `avgPixelsLAB(pixels, img)`: Calculate average LAB values

## Requirements

- Python >= 3.8
- TensorFlow >= 2.13.0
- OpenCV Python >= 4.5.0
- Pandas >= 1.3.0
- NumPy >= 1.21.0
- And more (see requirements.txt)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{pad_ml_workflow,
  title = {PAD ML Workflow: Machine Learning Tools for Paper Analytical Devices},
  author = {PAD ML Team},
  year = {2024},
  url = {https://github.com/PaperAnalyticalDeviceND/pad-ml-workflow-v2}
}
```

## Acknowledgments

This work is supported by the Paper Analytical Device project at the University of Notre Dame.

## Links

- [PAD API v2 Documentation](https://pad.crc.nd.edu/docs)
- [GitHub Repository](https://github.com/PaperAnalyticalDeviceND/pad-ml-workflow-v2)
- [Issue Tracker](https://github.com/PaperAnalyticalDeviceND/pad-ml-workflow-v2/issues)