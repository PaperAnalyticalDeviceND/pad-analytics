"""
Preprocessing pipeline components for PAD Analytics.

This module provides preprocessing capabilities for different model types:
- Neural Networks (NN): RGB/HSV color space preprocessing
- PLS Models: Statistical feature extraction and normalization

Usage:
    from pad_analytics.preprocessors import NeuralNetworkPreprocessor, PLSPreprocessor
    from pad_analytics import PreprocessingPipeline
    
    # Create model-specific preprocessor
    pipeline = PreprocessingPipeline(model_id=16)  # Auto-detects NN preprocessing
    processed_data = pipeline.preprocess(dataset)
"""

from .base_preprocessor import BasePreprocessor
from .nn_preprocessor import NeuralNetworkPreprocessor
from .pls_preprocessor import PLSPreprocessor

__all__ = [
    "BasePreprocessor",
    "NeuralNetworkPreprocessor", 
    "PLSPreprocessor"
]