# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of pad-analytics package
- Comprehensive README with installation and usage instructions
- Unit test structure with pytest
- Package metadata for PyPI publication
- Support for both Neural Network and PLS models
- Integration with PAD API v2
- Interactive visualization widgets for Jupyter notebooks
- Example notebooks demonstrating usage
- Console script `pad-analytics` for command-line usage

### Changed
- Package name from `pad-ml-workflow` to `pad-analytics`
- Restructured code as installable Python package
- Updated all imports to use relative imports for package compatibility
- Fixed overflow issues in pixel averaging functions
- Updated numpy version constraint for compatibility

### Fixed
- Import errors when installing from GitHub
- Numerical overflow in avgPixels, avgPixelsHSV, and avgPixelsLAB functions
- Module import issues with empty pls_model.py file

## [0.1.0] - TBD

### Added
- Core functionality for PAD image analysis
- Machine learning model integration (NN and PLS)
- API client for PAD database access
- Image processing and feature extraction
- Model evaluation metrics (RMSE, accuracy)
- Data visualization tools

[Unreleased]: https://github.com/PaperAnalyticalDeviceND/pad-analytics/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/PaperAnalyticalDeviceND/pad-analytics/releases/tag/v0.1.0