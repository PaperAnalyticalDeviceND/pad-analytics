"""Pytest configuration and fixtures."""

import pytest
import os
import sys

# Add src to path for all tests
@pytest.fixture(scope="session", autouse=True)
def setup_path():
    """Add src directory to Python path for testing."""
    src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


@pytest.fixture
def sample_card_data():
    """Provide sample card data for testing."""
    return {
        "id": 19208,
        "sample_name": "Amoxicillin",
        "quantity": 50.0,
        "image_url": "http://example.com/card.jpg",
        "project_id": 123
    }


@pytest.fixture 
def sample_project_data():
    """Provide sample project data for testing."""
    return [
        {"id": 1, "name": "FHI2020_Stratified_Sampling", "description": "Test project 1"},
        {"id": 2, "name": "FHI2022", "description": "Test project 2"}
    ]


@pytest.fixture
def sample_prediction_data():
    """Provide sample prediction results for testing."""
    import pandas as pd
    return pd.DataFrame({
        'card_id': [19208, 19209, 19210],
        'api': ['Amoxicillin', 'Amoxicillin', 'Ciprofloxacin'],
        'actual': [50.0, 60.0, 30.0],
        'prediction': [52.5, 58.2, 31.8]
    })