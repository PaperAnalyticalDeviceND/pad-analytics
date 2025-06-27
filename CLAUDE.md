# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pad-analytics` is a Python package for analyzing Paper Analytical Device (PAD) images using machine learning. PADs are low-cost diagnostic tools for pharmaceutical quality testing that produce colorimetric patterns when samples are applied. This package provides ML-powered analysis of those patterns to determine drug identity and concentration.

## Development Environment Setup

```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e .[dev]

# Install with notebook dependencies
pip install -e .[notebooks]
```

## Build and Test Commands

### Testing
```bash
# Run all tests using pytest
python -m pytest tests/ -v

# Run tests with the custom test runner
python run_tests.py

# Run a single test file
python -m pytest tests/test_basic.py -v

# Test package installation
python test_simple.py
```

### Building and Publishing
```bash
# Build the package
python -m build

# Upload to PyPI
python -m twine upload dist/*

# Test installation from GitHub
pip install git+https://github.com/PaperAnalyticalDeviceND/pad-analytics.git@refactor-installable-package
```

### Development Tools
```bash
# Console script (after installation)
pad-analytics --help

# Run examples
python examples/basic_usage.py
python examples/batch_predictions.py
python examples/custom_analysis.py
```

## Architecture Overview

### Core Module Structure
The package is organized around a **layered architecture** with distinct responsibilities:

- **API Layer** (`padanalytics.py`): Primary interface for researchers, handles API communication with PAD database at https://pad.crc.nd.edu/api/v2/
- **Image Processing Layer** (`regionRoutine.py`, `pixelProcessing.py`): Extracts color features from PAD images by analyzing specific regions and converting between color spaces (RGB, HSV, LAB)
- **ML Model Layer**: Integrates two model types:
  - **Neural Networks** (TensorFlow Lite): For drug classification, returns `(drug_name, confidence, energy)`
  - **PLS Models**: For concentration quantification, returns predicted concentration as float
- **Data Management Layer** (`fileManagement.py`): Handles CSV data processing and PAD lane indexing
- **Visualization Layer**: Jupyter widget integration for interactive data exploration

### Key Architectural Patterns

**Dual Model Integration**: The `predict()` function automatically detects model type and returns appropriate format:
```python
# PLS model (model_id=18) returns (actual_conc, predicted_conc)
actual, prediction = pad.predict(card_id=19208, model_id=18)

# NN model returns (actual, (drug_name, confidence, energy))
actual, (drug, conf, energy) = pad.predict(card_id, nn_model_id)
```

**API-First Design**: All data access goes through the PAD API v2, with functions like `get_projects()`, `get_card()`, `get_models()` providing programmatic access to the research database.

**Region-Based Image Analysis**: PAD cards are divided into lanes (A-L) with multiple regions per lane. The image processing pipeline extracts average color values from each region to create feature vectors for ML models.

### Import Patterns
```python
# Primary interface for researchers
from pad_analytics import padanalytics as pad

# Direct access to image processing
from pad_analytics import regionRoutine, pixelProcessing

# Package-level import
import pad_analytics
```

## SDK Evolution Context

This package is evolving from v0.1.0 (basic package) toward v1.0.0 (full SDK). See `SDK_ROADMAP.md` for the complete evolution plan:

- **v0.2.0**: Add comprehensive error handling (`PADError`, `PADAPIError`, etc.)
- **v0.3.0**: Add configuration management and client classes
- **v0.4.0**: Enhanced CLI tools and debugging utilities
- **v1.0.0**: Full SDK with stable API and manager-based architecture

### Future SDK Architecture (v1.0.0 target)
```python
# Target SDK usage pattern
with pad.SDK() as sdk:
    result = sdk.models.predict(card_id, model_id)
    sdk.data.export_results(result, format='csv')
```

## Model and Data Context

### Model Types and IDs
- **Model 16**: Neural Network classifier (24fhiNN1classifyAPI)
- **Model 17**: Neural Network concentration (24fhiNN1concAPI) 
- **Model 18**: PLS concentration model (24fhiPLS1conc)
- **Model 19**: Neural Network concentration v2

### API Endpoints
- Base URL: `https://pad.crc.nd.edu/api/v2`
- Projects endpoint: `/projects`
- Cards endpoint: `/cards/{card_id}`
- Models endpoint: `/neural-networks/{model_id}`

### Package Structure Specifics
- Package name: `pad-analytics` (PyPI), imports as `pad_analytics`
- Source code in `src/` directory with `package_dir={"pad_analytics": "src"}`
- Console script: `pad-analytics` maps to `pad_analytics.padanalytics:main`
- All modules use relative imports (e.g., `from . import regionRoutine`)

## Important Development Notes

### Critical Dependencies
- **NumPy version constraint**: `>=1.21.0,<2.0.0` (strict upper bound to prevent compatibility issues)
- **TensorFlow**: Required for neural network model inference
- **OpenCV**: Essential for image processing operations
- **ipywidgets**: Required for Jupyter notebook visualizations

### Testing Strategy
- Unit tests in `tests/` directory using pytest framework
- Integration tests mock API responses to avoid external dependencies
- Example scripts in `examples/` serve as integration tests
- Virtual environment testing ensures clean installation

### Known Technical Debt
- `pls_model.py` is currently empty (commented out of imports)
- API calls use `verify=False` due to SSL certificate issues (temporary workaround)
- Some functions lack comprehensive error handling (addressed in v0.2.0)

### Target Users and Use Cases
- **Chemistry researchers**: PAD performance analysis, colorimetric response evaluation
- **Computer science researchers**: ML model development, algorithm comparison
- **Pharmaceutical quality engineers**: Drug quality testing in resource-limited settings