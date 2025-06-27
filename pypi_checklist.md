# PyPI Publication Checklist

## 1. Package Name
- [ ] Check if "pad-ml-workflow" is available on PyPI
- [ ] Consider alternatives if taken (e.g., "padml", "pad-workflow", "paperpad-ml")

## 2. Package Metadata Updates Needed

### setup.py and pyproject.toml:
- [ ] Add author email
- [ ] Update GitHub URL to correct repository
- [ ] Add long description from README
- [ ] Add keywords for better discoverability
- [ ] Update classifiers for production readiness

## 3. Code Quality
- [ ] Add docstrings to all public functions
- [ ] Remove or fix empty modules (pls_model.py)
- [ ] Fix remaining import issues
- [ ] Add type hints (optional but recommended)

## 4. Testing
- [ ] Create tests/ directory
- [ ] Add unit tests for core functions
- [ ] Add integration tests
- [ ] Set up pytest configuration

## 5. Documentation
- [ ] Create docs/ directory
- [ ] Add API documentation
- [ ] Add installation guide
- [ ] Add tutorials/examples

## 6. Examples
- [ ] Create examples/ directory
- [ ] Add basic usage examples
- [ ] Add Jupyter notebook examples
- [ ] Add advanced usage examples

## 7. CI/CD
- [ ] Set up GitHub Actions for testing
- [ ] Add automated PyPI deployment
- [ ] Add code coverage reporting
- [ ] Add linting (flake8/black)

## 8. Version Management
- [ ] Create CHANGELOG.md
- [ ] Decide on versioning strategy (semantic versioning)
- [ ] Tag releases in git

## 9. PyPI Specific
- [ ] Create PyPI account
- [ ] Generate API token
- [ ] Test with TestPyPI first
- [ ] Build distributions (wheel and sdist)

## 10. Final Steps
- [ ] Update all relative imports
- [ ] Ensure all dependencies are listed
- [ ] Test installation from TestPyPI
- [ ] Publish to PyPI

## Commands for Publishing

```bash
# Build the package
python -m build

# Upload to TestPyPI (for testing)
python -m twine upload --repository testpypi dist/*

# Upload to PyPI (final)
python -m twine upload dist/*
```