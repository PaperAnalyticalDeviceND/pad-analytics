"""
Model adapter components for PAD Analytics.

This module provides model adapter capabilities for different model types:
- Neural Networks (NN): TensorFlow Lite model adaptation
- PLS Models: Scikit-learn compatible model adaptation

Usage:
    from pad_analytics.adapters import NeuralNetworkAdapter, PLSAdapter
    from pad_analytics import ModelAdapter
    
    # Create model-specific adapter
    adapter = ModelAdapter(model_id=16)  # Auto-detects model type
    result = adapter.predict(card_data)
"""

from .base_adapter import BaseAdapter
from .nn_adapter import NeuralNetworkAdapter
from .pls_adapter import PLSAdapter

__all__ = [
    "BaseAdapter",
    "NeuralNetworkAdapter", 
    "PLSAdapter"
]