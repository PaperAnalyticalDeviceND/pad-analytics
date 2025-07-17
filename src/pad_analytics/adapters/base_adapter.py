"""
Base model adapter abstract class for PAD Analytics.

This module defines the interface that all model adapters must implement.
Different model types (NN, PLS) will have their own specific model loading and prediction logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Tuple
import os
import tempfile
from pathlib import Path


class BaseAdapter(ABC):
    """
    Abstract base class for all PAD model adapters.
    
    This class defines the interface that all model adapters must implement.
    Each model type (Neural Network, PLS) will have its own specific model loading and prediction logic.
    """
    
    def __init__(self, model_id: int, cache_manager: Optional[Any] = None):
        """
        Initialize the model adapter.
        
        Args:
            model_id: The model ID this adapter is configured for
            cache_manager: Optional cache manager for caching models and predictions
        """
        self.model_id = model_id
        self.cache_manager = cache_manager
        self.model = None
        self.model_metadata = None
        self.is_loaded = False
        
        # Get model information
        self.model_info = self._get_model_info()
        
    @abstractmethod
    def _get_model_info(self) -> Dict[str, Any]:
        """
        Get model information from the PAD API or local registry.
        
        Returns:
            Dictionary containing model metadata (name, URL, type, etc.)
        """
        pass
    
    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the model from URL or cache.
        
        Returns:
            True if model was loaded successfully
        """
        pass
    
    @abstractmethod
    def predict_single(self, preprocessed_data: Dict[str, Any]) -> Union[Tuple, float]:
        """
        Make a prediction for a single preprocessed data point.
        
        Args:
            preprocessed_data: Output from preprocessing pipeline
            
        Returns:
            Prediction result in model-specific format:
            - Neural Network: (drug_name, confidence, energy)
            - PLS: concentration as float
        """
        pass
    
    @abstractmethod
    def predict_batch(self, preprocessed_batch: List[Dict[str, Any]]) -> List[Union[Tuple, float]]:
        """
        Make predictions for a batch of preprocessed data.
        
        Args:
            preprocessed_batch: List of preprocessed data dictionaries
            
        Returns:
            List of prediction results
        """
        pass
    
    @abstractmethod
    def get_expected_input_format(self) -> str:
        """
        Get the expected input format for this adapter.
        
        Returns:
            String describing expected input format
        """
        pass
    
    def predict(self, card_data: Dict[str, Any], 
                preprocessor: Optional[Any] = None) -> Union[Tuple, float]:
        """
        High-level prediction method that handles preprocessing and prediction.
        
        Args:
            card_data: Raw card data dictionary
            preprocessor: Optional preprocessor instance (will create if not provided)
            
        Returns:
            Prediction result in model-specific format
        """
        # Ensure model is loaded
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError(f"Failed to load model {self.model_id}")
        
        # Create preprocessor if not provided
        if preprocessor is None:
            from ..preprocessing_pipeline import PreprocessingPipeline
            preprocessor = PreprocessingPipeline(self.model_id, self.cache_manager)
        
        # Preprocess data
        preprocessed_data = preprocessor.preprocess_single_card(card_data)
        
        # Make prediction
        return self.predict_single(preprocessed_data)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_id': self.model_id,
            'model_type': self.get_model_type(),
            'adapter_class': self.__class__.__name__,
            'is_loaded': self.is_loaded,
            'model_info': self.model_info,
            'cache_enabled': self.cache_manager is not None
        }
    
    def get_model_type(self) -> str:
        """
        Get the model type this adapter is designed for.
        
        Returns:
            String indicating model type (e.g., 'neural_network', 'pls')
        """
        return self.__class__.__name__.lower().replace('adapter', '')
    
    def is_model_loaded(self) -> bool:
        """
        Check if the model is currently loaded.
        
        Returns:
            True if model is loaded and ready for predictions
        """
        return self.is_loaded and self.model is not None
    
    def unload_model(self) -> None:
        """
        Unload the model to free memory.
        """
        self.model = None
        self.is_loaded = False
    
    def download_model(self, model_url: str, force_download: bool = False) -> Optional[str]:
        """
        Download model file from URL.
        
        Args:
            model_url: URL to download model from
            force_download: If True, download even if file exists
            
        Returns:
            Path to downloaded model file or None if failed
        """
        try:
            import requests
            from urllib.parse import urlparse
            
            # Generate filename from URL
            parsed_url = urlparse(model_url)
            filename = os.path.basename(parsed_url.path)
            
            # Use temporary directory for model storage
            temp_dir = tempfile.gettempdir()
            model_path = os.path.join(temp_dir, filename)
            
            # Check if file already exists
            if os.path.exists(model_path) and not force_download:
                return model_path
            
            # Download model
            response = requests.get(model_url, stream=True, verify=False)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return model_path
            
        except Exception as e:
            print(f"Failed to download model from {model_url}: {e}")
            return None
    
    def validate_preprocessed_data(self, preprocessed_data: Dict[str, Any]) -> bool:
        """
        Validate that preprocessed data meets adapter requirements.
        
        Args:
            preprocessed_data: Preprocessed data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        # Basic validation - subclasses can override for specific checks
        required_fields = ['card_id', 'model_id', 'preprocessor_type']
        
        for field in required_fields:
            if field not in preprocessed_data:
                return False
        
        # Check model ID matches
        if preprocessed_data.get('model_id') != self.model_id:
            return False
        
        return True
    
    def _get_cache_key(self, data_hash: str) -> str:
        """
        Generate a cache key for model predictions.
        
        Args:
            data_hash: Hash of the input data
            
        Returns:
            String cache key
        """
        return f"prediction_m{self.model_id}_{data_hash}"
    
    def _cache_prediction(self, cache_key: str, prediction: Union[Tuple, float]) -> None:
        """
        Cache prediction result if cache manager is available.
        
        Args:
            cache_key: Key to store the prediction under
            prediction: Prediction result to cache
        """
        if self.cache_manager:
            # This will be implemented when we integrate with CacheManager
            pass
    
    def _load_cached_prediction(self, cache_key: str) -> Optional[Union[Tuple, float]]:
        """
        Load cached prediction result if available.
        
        Args:
            cache_key: Key to load prediction from
            
        Returns:
            Cached prediction or None if not available
        """
        if self.cache_manager:
            # This will be implemented when we integrate with CacheManager
            pass
        return None
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        return (f"{self.__class__.__name__}(model_id={self.model_id}, "
                f"loaded={self.is_loaded})")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        model_name = self.model_info.get('name', 'Unknown') if self.model_info else 'Unknown'
        return f"{self.__class__.__name__} for {model_name} (ID: {self.model_id})"