"""
Base preprocessor abstract class for PAD Analytics.

This module defines the interface that all preprocessors must implement.
Different model types (NN, PLS) will have their own specific preprocessing logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path


class BasePreprocessor(ABC):
    """
    Abstract base class for all PAD data preprocessors.
    
    This class defines the interface that all preprocessors must implement.
    Each model type (Neural Network, PLS) will have its own specific preprocessing logic.
    """
    
    def __init__(self, model_id: int, cache_manager: Optional[Any] = None):
        """
        Initialize the preprocessor.
        
        Args:
            model_id: The model ID this preprocessor is configured for
            cache_manager: Optional cache manager for caching processed data
        """
        self.model_id = model_id
        self.cache_manager = cache_manager
        self.config = self._get_default_config()
        
    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration for this preprocessor.
        
        Returns:
            Dictionary containing default configuration parameters
        """
        pass
    
    @abstractmethod
    def preprocess_single_card(self, card_data: Dict[str, Any], 
                              image_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Preprocess a single card's data.
        
        Args:
            card_data: Dictionary containing card metadata and features
            image_path: Optional path to the card's image file
            
        Returns:
            Dictionary containing preprocessed features and metadata
        """
        pass
    
    @abstractmethod
    def preprocess_batch(self, cards_data: List[Dict[str, Any]], 
                        image_paths: Optional[List[Path]] = None) -> pd.DataFrame:
        """
        Preprocess a batch of cards efficiently.
        
        Args:
            cards_data: List of card data dictionaries
            image_paths: Optional list of image paths corresponding to cards
            
        Returns:
            DataFrame with preprocessed features and metadata
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features produced by this preprocessor.
        
        Returns:
            List of feature names in the order they appear in preprocessed data
        """
        pass
    
    @abstractmethod
    def get_expected_input_shape(self) -> Tuple[int, ...]:
        """
        Get the expected input shape for this preprocessor.
        
        Returns:
            Tuple representing the expected input shape
        """
        pass
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Update the preprocessor configuration.
        
        Args:
            config: Dictionary containing configuration parameters to update
        """
        self.config.update(config)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current preprocessor configuration.
        
        Returns:
            Dictionary containing current configuration parameters
        """
        return self.config.copy()
    
    def _get_cache_key(self, card_id: int, config_hash: str) -> str:
        """
        Generate a cache key for preprocessed data.
        
        Args:
            card_id: ID of the card being processed
            config_hash: Hash of the preprocessing configuration
            
        Returns:
            String cache key
        """
        return f"preprocessed_m{self.model_id}_c{card_id}_{config_hash}"
    
    def _cache_preprocessed_data(self, cache_key: str, data: Dict[str, Any]) -> None:
        """
        Cache preprocessed data if cache manager is available.
        
        Args:
            cache_key: Key to store the data under
            data: Preprocessed data to cache
        """
        if self.cache_manager:
            # This will be implemented when we integrate with CacheManager
            pass
    
    def _load_cached_preprocessed_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Load cached preprocessed data if available.
        
        Args:
            cache_key: Key to load data from
            
        Returns:
            Cached preprocessed data or None if not available
        """
        if self.cache_manager:
            # This will be implemented when we integrate with CacheManager
            pass
        return None
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """
        Validate that input data meets preprocessor requirements.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        # Basic validation - subclasses can override for specific checks
        return isinstance(data, dict) and 'id' in data
    
    def get_model_type(self) -> str:
        """
        Get the model type this preprocessor is designed for.
        
        Returns:
            String indicating model type (e.g., 'neural_network', 'pls')
        """
        return self.__class__.__name__.lower().replace('preprocessor', '')