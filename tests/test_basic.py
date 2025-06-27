"""Basic functionality tests that work without full dependencies."""

import pytest
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestBasicFunctionality:
    """Test basic functionality without external dependencies."""
    
    def test_pixelProcessing_import(self):
        """Test that pixelProcessing module can be imported."""
        import pixelProcessing
        assert hasattr(pixelProcessing, 'avgPixels')
        assert hasattr(pixelProcessing, 'avgPixelsHSV')
        assert hasattr(pixelProcessing, 'avgPixelsLAB')
    
    def test_fileManagement_import(self):
        """Test that fileManagement module can be imported."""
        import fileManagement
        assert hasattr(fileManagement, 'genIndex')
        assert hasattr(fileManagement, 'convertToDF')
    
    def test_regionRoutine_import(self):
        """Test that regionRoutine module can be imported."""
        import regionRoutine
        assert hasattr(regionRoutine, 'fullRoutine')
    
    def test_avgPixels_functionality(self):
        """Test avgPixels function with sample data."""
        import pixelProcessing
        import numpy as np
        
        # Create simple test image
        img = np.ones((10, 10, 3), dtype=np.uint8) * 100
        pixels = [(1, 1), (2, 2), (3, 3)]
        
        result = pixelProcessing.avgPixels(pixels, img)
        
        assert len(result) == 3  # RGB values
        assert all(isinstance(val, (int, float)) for val in result)
        assert all(val == 100.0 for val in result)  # Should be 100 for all channels
    
    def test_genIndex_functionality(self):
        """Test genIndex function."""
        import fileManagement
        
        result = fileManagement.genIndex(regions=3)
        
        assert isinstance(result, list)
        assert 'Image' in result
        assert 'Contains' in result
        assert 'Drug %' in result
        assert 'PAD S#' in result
        # Should have entries for lanes A-L with 3 regions each and RGB values
        assert len(result) > 10  # Basic sanity check


class TestPackageInfo:
    """Test package information and structure."""
    
    def test_setup_py_exists(self):
        """Test that setup.py exists and is readable."""
        setup_path = os.path.join(os.path.dirname(__file__), '..', 'setup.py')
        assert os.path.exists(setup_path)
        
        with open(setup_path, 'r') as f:
            content = f.read()
            assert 'pad-ml-workflow' in content
            assert 'setup(' in content
    
    def test_readme_exists(self):
        """Test that README.md exists and has content."""
        readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
        assert os.path.exists(readme_path)
        
        with open(readme_path, 'r') as f:
            content = f.read()
            assert 'PAD ML Workflow' in content
            assert 'Installation' in content