# Development Guide

## Current Status
- **Version**: 0.1.0 (Ready for PyPI publication)
- **Package Name**: `pad-analytics`
- **Target**: Evolve from basic package to comprehensive SDK

## Quick Context for AI Assistants

### Project Overview
`pad-analytics` is a Python package for analyzing Paper Analytical Device (PAD) images using machine learning. PADs are low-cost diagnostic tools for pharmaceutical quality testing in resource-limited settings.

### Current Architecture
```
pad_analytics/
├── padanalytics.py      # Core API functions (predict, get_card, etc.)
├── regionRoutine.py     # Image processing routines
├── pixelProcessing.py   # Pixel-level analysis
├── fileManagement.py    # Data management utilities
├── intensityFind.py     # Intensity analysis
└── pad_analysis.py      # Analysis workflows
```

### Key Functions
- `predict(card_id, model_id)` - Apply ML models to PAD images
- `get_projects()` - List available PAD projects
- `get_card(card_id)` - Get PAD card data and metadata
- `apply_predictions_to_dataframe()` - Batch predictions

### Technology Stack
- **ML Models**: TensorFlow Lite (Neural Networks), PLS regression
- **Image Processing**: OpenCV, NumPy
- **Data**: Pandas, API integration with https://pad.crc.nd.edu
- **Visualization**: Matplotlib, ipywidgets (Jupyter notebooks)

### SDK Evolution Roadmap
See `SDK_ROADMAP.md` for detailed version-by-version evolution plan:
- v0.1.0: Basic package (current)
- v0.2.0: Error handling framework
- v0.3.0: Configuration management  
- v0.4.0: Developer tools
- v1.0.0: Full SDK with stable API

## For AI Assistants Helping with Future Development

### Important Context
1. **Target Users**: Chemistry researchers, computer science researchers, pharmaceutical quality engineers
2. **Use Cases**: Drug quality testing, PAD performance analysis, ML model development
3. **Integration**: Works with PADReader mobile app ecosystem
4. **API**: Uses PAD API v2 at https://pad.crc.nd.edu

### Code Patterns to Follow
```python
# Error handling (implement in v0.2.0)
try:
    result = pad.predict(card_id, model_id)
except pad.PADAPIError as e:
    logger.error(f"API error: {e}")

# Configuration (implement in v0.3.0)  
client = pad.Client(config=pad.Config.from_file("config.yml"))

# SDK style (target for v1.0.0)
with pad.SDK() as sdk:
    result = sdk.models.predict(card_id, model_id)
```

### Testing Approach
- Unit tests in `tests/` directory using pytest
- Integration tests with mock API responses
- Example scripts in `examples/` directory
- Installation testing with virtual environments

### Documentation Standards
- Docstrings for all public functions
- Type hints where possible
- Example usage in docstrings
- Keep README.md updated with major changes

## Development Environment Setup

```bash
# Create virtual environment
python -m venv pad_dev
source pad_dev/bin/activate  # On Windows: pad_dev\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
python -m pytest tests/

# Run examples
python examples/basic_usage.py
```

## Common Development Tasks

### Adding New Features
1. Update relevant module in `src/`
2. Add tests in `tests/`
3. Update examples if needed
4. Update CHANGELOG.md
5. Update version in setup.py/pyproject.toml

### Publishing New Version
1. Update version numbers
2. Update CHANGELOG.md
3. Run tests: `python -m pytest`
4. Build: `python -m build`
5. Upload: `python -m twine upload dist/*`

### Working with API
- Base URL: https://pad.crc.nd.edu/api/v2/
- Authentication: API key (when required)
- Rate limiting: Be respectful with requests
- Error handling: Always check status codes

## Troubleshooting

### Common Issues
1. **Import errors**: Check virtual environment activation
2. **Missing dependencies**: Run `pip install -e .`
3. **API errors**: Check network connection and API status
4. **Test failures**: Ensure all dependencies installed

### Getting Help
- Check GitHub Issues: https://github.com/PaperAnalyticalDeviceND/pad-analytics/issues
- Review examples in `examples/` directory
- Check PAD API documentation: https://pad.crc.nd.edu/docs

*Last Updated: 2024-06-27*