"""Test padanalytics module functionality."""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
import pandas as pd

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pad_analytics


class TestPadAnalytics:
    """Test core padanalytics functions."""
    
    @patch('pad_analytics.padanalytics.requests.get')
    def test_get_projects_success(self, mock_get):
        """Test get_projects function with successful API response."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": 1, "name": "Test Project 1"},
            {"id": 2, "name": "Test Project 2"}
        ]
        mock_get.return_value = mock_response
        
        result = pad_analytics.get_projects()
        
        # Verify the result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "id" in result.columns
        assert "name" in result.columns
    
    @patch('pad_analytics.padanalytics.requests.get')
    def test_get_projects_api_error(self, mock_get):
        """Test get_projects function with API error."""
        # Mock API error
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = pad_analytics.get_projects()
        
        # Should return empty DataFrame on error
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    @patch('pad_analytics.padanalytics.requests.get')
    def test_get_card_success(self, mock_get):
        """Test get_card function with successful response."""
        card_id = 12345
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": card_id,
            "sample_name": "Test Sample",
            "quantity": 50.0,
            "image_url": "http://example.com/image.jpg"
        }
        mock_get.return_value = mock_response
        
        result = pad_analytics.get_card(card_id)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["id"] == card_id
        assert result.iloc[0]["sample_name"] == "Test Sample"


class TestPixelProcessing:
    """Test pixel processing functions."""
    
    def test_avgPixels_no_overflow(self):
        """Test that avgPixels doesn't overflow with large values."""
        import pixelProcessing
        import numpy as np
        
        # Create test image with large values
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        pixels = [(i, j) for i in range(50) for j in range(50)]  # 2500 pixels
        
        # This should not raise an overflow warning
        result = pixelProcessing.avgPixels(pixels, img)
        
        # Result should be close to 255 for all channels
        assert len(result) == 3
        assert all(250 <= val <= 255 for val in result)


class TestModelPrediction:
    """Test model prediction functionality."""
    
    @patch('padanalytics.get_card')
    @patch('padanalytics.requests.post')
    def test_predict_pls_model(self, mock_post, mock_get_card):
        """Test predict function with PLS model."""
        # Mock card data
        mock_get_card.return_value = pd.DataFrame([{
            "id": 19208,
            "sample_name": "Test Sample",
            "quantity": 50.0
        }])
        
        # Mock PLS prediction response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prediction": 57.573}
        mock_post.return_value = mock_response
        
        actual, prediction = padanalytics.predict(19208, 18)  # Model 18 is PLS
        
        assert actual == 50.0
        assert isinstance(prediction, float)
        assert prediction == 57.573
    
    def test_predict_invalid_card_id(self):
        """Test predict function with invalid card ID."""
        with pytest.raises(Exception):
            padanalytics.predict(-1, 18)


class TestDataProcessing:
    """Test data processing utilities."""
    
    def test_apply_predictions_to_dataframe_empty(self):
        """Test apply_predictions_to_dataframe with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = pad_analytics.apply_predictions_to_dataframe(empty_df, 18)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_calculate_rmse_by_api(self):
        """Test RMSE calculation function."""
        # Create test data
        test_data = pd.DataFrame({
            'api': ['Drug A', 'Drug A', 'Drug B', 'Drug B'],
            'actual': [50.0, 60.0, 30.0, 40.0],
            'prediction': [52.0, 58.0, 32.0, 38.0]
        })
        
        result = pad_analytics.calculate_rmse_by_api(test_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'api' in result.columns
        assert 'rmse' in result.columns
        assert len(result) == 2  # Two unique drugs