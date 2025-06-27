"""Test basic imports and package structure."""

import pytest
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestImports:
    """Test that all modules can be imported."""
    
    def test_padanalytics_import(self):
        """Test that padanalytics module can be imported."""
        import pad_analytics
        assert hasattr(pad_analytics, 'get_projects')
        assert hasattr(pad_analytics, 'get_card')
        assert hasattr(pad_analytics, 'predict')
    
    def test_pad_helper_import(self):
        """Test that pad_helper module can be imported."""
        import pad_analytics.pad_helper
        # Basic import test
        assert pad_analytics.pad_helper is not None
    
    def test_regionRoutine_import(self):
        """Test that regionRoutine module can be imported."""
        import regionRoutine
        assert hasattr(regionRoutine, 'fullRoutine')
    
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
    
    def test_intensityFind_import(self):
        """Test that intensityFind module can be imported."""
        import intensityFind
        # Basic import test
        assert intensityFind is not None


class TestPackageStructure:
    """Test package-level imports."""
    
    def test_package_init_import(self):
        """Test that package can be imported."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        import pad_analytics
        # Check that main modules are available
        assert hasattr(pad_analytics, 'padanalytics')
        assert hasattr(pad_analytics, 'pad_analysis')