# PAD Analytics Examples

This directory contains example scripts demonstrating how to use the `pad-analytics` package.

## Available Examples

### 1. `basic_usage.py`
Demonstrates the fundamental operations:
- Connecting to the PAD API
- Fetching project and card data
- Making predictions with pre-trained models
- Listing available models

Run with:
```bash
python examples/basic_usage.py
```

### 2. `batch_predictions.py`
Shows how to work with multiple samples:
- Loading datasets
- Applying models to multiple PAD cards
- Calculating performance metrics (RMSE, MAE)
- Exporting results to CSV

Run with:
```bash
python examples/batch_predictions.py
```

### 3. `custom_analysis.py`
Demonstrates advanced usage:
- Direct image processing functions
- Feature extraction from PAD images
- Building custom analysis pipelines
- Working with color spaces (RGB, HSV, LAB)

Run with:
```bash
python examples/custom_analysis.py
```

## Prerequisites

Make sure you have `pad-analytics` installed:
```bash
pip install pad-analytics
```

Or install from source:
```bash
pip install -e .
```

## Notes

- These examples use some synthetic data for demonstration when API access is not available
- In production use, ensure you have proper API credentials if required
- The custom analysis example creates a test image file for demonstration purposes

## Additional Resources

- [Full Documentation](https://pad.crc.nd.edu/docs)
- [API Reference](https://pad.crc.nd.edu/openapi.json)
- [Jupyter Notebooks](../notebooks/)