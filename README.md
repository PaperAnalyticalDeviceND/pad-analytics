# PAD ML Workflow

A Python package for researchers to explore and analyze Paper Analytical Device (PAD) data, build machine learning models, and develop new analytical methods for pharmaceutical quality testing.


## About PADs

[Paper Analytical Devices (PADs)](https://padproject.nd.edu) are low-cost diagnostic tools designed to verify pharmaceutical authenticity in low-resource settings. When a dissolved drug sample is applied to a PAD card, it produces colorimetric patterns that can be analyzed to determine drug quality and composition.

This package provides programmatic access to PAD image data collected through the [PADReader mobile app](https://padproject.nd.edu) and enables researchers to:
- Explore historical PAD test data
- Apply and evaluate machine learning models
- Develop new analytical methods
- Build custom ML pipelines for PAD analysis

## Installation

```bash
pip install pad-analytics
```

Or install directly from GitHub:
```bash
pip install git+https://github.com/PaperAnalyticalDeviceND/pad-analytics.git
```

For development:
```bash
git clone https://github.com/PaperAnalyticalDeviceND/pad-analytics.git
cd pad-analytics
pip install -e .
```

### Debug Mode
By default, the package suppresses technical warnings for a cleaner user experience. To enable debug output:

```bash
PAD_DEBUG=1 python your_script.py
# or
export PAD_DEBUG=1
python -c "import pad_analytics as pad; pad.predict(19208, 18)"
```

**Note:** You may see `libpng error: Read Error` messages during prediction - these are harmless warnings from corrupted image data on the server side and do not affect the prediction results.

## Quick Start

```python
import pad_analytics as pad

# Explore available projects
projects = pad.get_projects()
print(f"Found {len(projects)} projects")

# Get PAD test cards from a specific project
cards = pad.get_project_cards(project_name="ChemoPADNNtraining2024")

# Analyze a specific PAD card
card_data = pad.get_card(card_id=19208)
print(f"Drug tested: {card_data['sample_name'].values[0]}")
print(f"Concentration: {card_data['quantity'].values[0]} %")

# Apply a pre-trained model
actual, prediction = pad.predict(card_id=19208, model_id=18)
```

## Key Features

### 1. Data Exploration
Access the complete PAD database through the [OAS-compliant API](https://pad.crc.nd.edu/openapi.json):

```python
# List all projects
projects = pad.get_projects()

# Get cards by various criteria
cards = pad.get_project_cards(project_ids=12)
cards = pad.get_card_by_sample_id(65490)

# View available ML models
models = pad.get_models()
```

### 2. Model Application
Apply pre-trained models to PAD images:

```python
# Neural Network models (for classification)
actual, (drug_name, confidence, energy) = pad.predict(card_id=19208, model_id=16)

# PLS models (for concentration quantification)
actual_conc, predicted_conc = pad.predict(card_id=19208, model_id=18)

```

### 3. Visualization
Interactive widgets for Jupyter notebooks:

```python
# Display PAD card with metadata
pad.show_card(card_id=19208)

# Show prediction results
pad.show_prediction(card_id=19208, model_id=18)

# Display multiple cards grouped by drug type
cards_df = pad.get_project_cards(project_name="ChemoPADNNtraining2024")
pad.show_grouped_cards(cards_df, group_column='sample_name')
```


## Example Notebook

See [`notebooks/using_padml_package.ipynb`](notebooks/using_padml_package.ipynb) for a comprehensive example of:
- Exploring PAD projects and data
- Applying different model types
- Visualizing results
- Evaluating model performance
- Building custom analysis pipelines

## Research Applications

This package supports various research activities:

### For Chemistry Researchers
- Analyze PAD performance across different drug formulations
- Evaluate colorimetric response patterns
- Optimize PAD card designs
- Validate new analytical methods

### For Computer Science Researchers  
- Develop new ML models for PAD analysis
- Compare algorithm performance (NN vs PLS vs custom)
- Implement novel image processing techniques
- Create ensemble methods for improved accuracy

## The PAD Workflow

1. **Sample Preparation**: Dissolve pharmaceutical sample
2. **Application**: Apply sample to PAD card
3. **Reaction**: Chemical indicators produce color patterns
4. **Imaging**: Capture with PADReader mobile app
5. **Analysis**: ML algorithms interpret patterns
6. **Results**: Determine drug identity and quality

This package focuses on steps 5-6, providing tools to analyze the collected images and develop better analytical methods.

## API Documentation

### Core Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `get_projects()` | List all PAD projects | DataFrame of projects |
| `get_card(card_id)` | Get specific card data | Card metadata + image URL |
| `predict(card_id, model_id)` | Apply model to card | (actual, prediction) |
| `get_models()` | List available models | DataFrame of models |
| `show_card(card_id)` | Display card in notebook | Interactive widget |

### Model Types

**Neural Networks (TensorFlow Lite)**
- Purpose: Drug identification and multi-class classification
- Output: `(predicted_class, probability, energy_score)`

**PLS (Partial Least Squares)**
- Purpose: Concentration quantification
- Output: `predicted_concentration` (float)

## Requirements

- Python >= 3.8
- TensorFlow >= 2.13.0
- OpenCV-Python >= 4.5.0
- NumPy, Pandas, scikit-learn
- ipywidgets (for notebook visualizations)

## Contributing

We welcome contributions from both chemistry and computer science researchers! Please see our [Contributing Guide](CONTRIBUTING.md).

## Citation

If you use this package in your research, please cite:

```bibtex
@software{pad_analytics,
  title = {PAD Analytics: Python Tools for Paper Analytical Device Research},
  author = {Paper Analytical Device Project Team},
  institution = {University of Notre Dame},
  year = {2024},
  url = {https://github.com/PaperAnalyticalDeviceND/pad-analytics}
}
```

## License

MIT License - see [LICENSE](LICENSE)

## Links

- [PAD Project Homepage](https://padproject.nd.edu)
- [PADReader Mobile App](https://padproject.nd.edu)
- [API Documentation](https://pad.crc.nd.edu/docs)
- [OpenAPI Specification](https://pad.crc.nd.edu/openapi.json)
- [GitHub Repository](https://github.com/PaperAnalyticalDeviceND/pad-analytics)

## Support

For questions about:
- PAD technology and chemistry: Visit [padproject.nd.edu](https://padproject.nd.edu)
- Package usage and ML models: Open an [issue on GitHub](https://github.com/PaperAnalyticalDeviceND/pad-analytics/issues)
- API access: Check the [API documentation](https://pad.crc.nd.edu/docs)

## Security Notice

**Note about Keras dependency**: This package uses Keras 2.14.0 which has a known security vulnerability (CVE-2024-55459) related to the `keras.utils.get_file()` function. However, **pad-analytics is not affected** because we do not use this function in our codebase.

If you are extending this package and need to download files programmatically:
- **DO NOT** use `keras.utils.get_file()` with untrusted URLs
- If you must use it, add security measures to validate any downloaded files
- Only download files from trusted sources

We plan to upgrade to Keras 3.8.0+ in a future release to eliminate this dependency concern. See [Issue #2](https://github.com/PaperAnalyticalDeviceND/pad-analytics/issues/2) for more details.
