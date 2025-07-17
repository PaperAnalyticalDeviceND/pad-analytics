"""
Main preprocessing pipeline for PAD Analytics.

This module provides the central PreprocessingPipeline class that automatically
selects and configures the appropriate preprocessor based on model type.
"""

from typing import Dict, Any, Optional, List, Union
import pandas as pd
from pathlib import Path

from .preprocessors import BasePreprocessor, NeuralNetworkPreprocessor, PLSPreprocessor
from .cache_manager import CacheManager


class PreprocessingPipeline:
    """
    Main preprocessing pipeline that automatically selects the appropriate
    preprocessor based on model type and provides a unified interface.
    """
    
    def __init__(self, model_id: int, 
                 cache_manager: Optional[CacheManager] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            model_id: The model ID to create preprocessor for
            cache_manager: Optional cache manager for caching processed data
            config: Optional configuration dictionary to override defaults
        """
        self.model_id = model_id
        self.cache_manager = cache_manager
        self.config = config or {}
        
        # Auto-detect and create appropriate preprocessor
        self.preprocessor = self._create_preprocessor()
        
        # Apply custom configuration if provided
        if self.config:
            self.preprocessor.set_config(self.config)
    
    def _create_preprocessor(self) -> BasePreprocessor:
        """
        Create the appropriate preprocessor based on model type.
        
        Returns:
            Configured preprocessor instance
        """
        model_type = self._detect_model_type()
        
        if model_type == 'neural_network':
            return NeuralNetworkPreprocessor(self.model_id, self.cache_manager)
        elif model_type == 'pls':
            return PLSPreprocessor(self.model_id, self.cache_manager)
        else:
            raise ValueError(f"Unknown model type for model_id {self.model_id}: {model_type}")
    
    def _detect_model_type(self) -> str:
        """
        Detect the model type based on model ID.
        
        Returns:
            String indicating model type ('neural_network' or 'pls')
        """
        # Model type detection based on known model IDs
        # This follows the mapping from the existing codebase
        neural_network_models = {16, 17, 19, 20}  # NN classification and concentration models
        pls_models = {18}  # PLS concentration models
        
        if self.model_id in neural_network_models:
            return 'neural_network'
        elif self.model_id in pls_models:
            return 'pls'
        else:
            # Default assumption for unknown models
            # Could be enhanced with API lookup in the future
            print(f"Warning: Unknown model_id {self.model_id}, assuming neural_network")
            return 'neural_network'
    
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
        return self.preprocessor.preprocess_single_card(card_data, image_path)
    
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
        return self.preprocessor.preprocess_batch(cards_data, image_paths)
    
    def preprocess_dataset(self, dataset: Union[pd.DataFrame, 'CachedDataset'],
                          max_images: Optional[int] = None,
                          use_cached_images: bool = True) -> pd.DataFrame:
        """
        Preprocess an entire dataset.
        
        Args:
            dataset: Dataset to preprocess (DataFrame or CachedDataset)
            max_images: Maximum number of images to process (None for all)
            use_cached_images: Whether to use cached images if available
            
        Returns:
            DataFrame with preprocessed features and metadata
        """
        # Handle different dataset types
        if hasattr(dataset, 'load_dataset_metadata'):
            # CachedDataset
            metadata_df = dataset.load_dataset_metadata()
            cards_data = metadata_df.to_dict('records')
        elif isinstance(dataset, pd.DataFrame):
            # Regular DataFrame
            cards_data = dataset.to_dict('records')
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")
        
        # Limit number of cards if specified
        if max_images is not None:
            cards_data = cards_data[:max_images]
        
        # Get image paths if using cached images
        image_paths = None
        if use_cached_images and hasattr(dataset, 'get_cached_image_paths'):
            image_paths = dataset.get_cached_image_paths([card['id'] for card in cards_data])
        
        # Preprocess batch
        return self.preprocess_batch(cards_data, image_paths)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features produced by this pipeline.
        
        Returns:
            List of feature names in the order they appear in preprocessed data
        """
        return self.preprocessor.get_feature_names()
    
    def get_expected_input_shape(self) -> tuple:
        """
        Get the expected input shape for this pipeline.
        
        Returns:
            Tuple representing the expected input shape
        """
        return self.preprocessor.get_expected_input_shape()
    
    def get_model_type(self) -> str:
        """
        Get the model type this pipeline is configured for.
        
        Returns:
            String indicating model type
        """
        return self.preprocessor.get_model_type()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current pipeline configuration.
        
        Returns:
            Dictionary containing current configuration parameters
        """
        return self.preprocessor.get_config()
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Update the pipeline configuration.
        
        Args:
            config: Dictionary containing configuration parameters to update
        """
        self.preprocessor.set_config(config)
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """
        Validate that input data meets pipeline requirements.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        return self.preprocessor.validate_input(data)
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about the preprocessing pipeline.
        
        Returns:
            Dictionary containing pipeline information
        """
        return {
            'model_id': self.model_id,
            'model_type': self.get_model_type(),
            'preprocessor_class': self.preprocessor.__class__.__name__,
            'expected_input_shape': self.get_expected_input_shape(),
            'num_features': len(self.get_feature_names()),
            'config': self.get_config(),
            'cache_enabled': self.cache_manager is not None
        }
    
    def __repr__(self) -> str:
        """String representation of the pipeline."""
        return (f"PreprocessingPipeline(model_id={self.model_id}, "
                f"type={self.get_model_type()}, "
                f"preprocessor={self.preprocessor.__class__.__name__})")


# Convenience functions for direct use
def create_neural_network_pipeline(model_id: int, 
                                  cache_manager: Optional[CacheManager] = None,
                                  config: Optional[Dict[str, Any]] = None) -> PreprocessingPipeline:
    """
    Create a preprocessing pipeline specifically for neural network models.
    
    Args:
        model_id: Neural network model ID
        cache_manager: Optional cache manager
        config: Optional configuration dictionary
        
    Returns:
        Configured preprocessing pipeline
    """
    pipeline = PreprocessingPipeline(model_id, cache_manager, config)
    if pipeline.get_model_type() != 'neural_network':
        raise ValueError(f"Model {model_id} is not a neural network model")
    return pipeline


def create_pls_pipeline(model_id: int, 
                       cache_manager: Optional[CacheManager] = None,
                       config: Optional[Dict[str, Any]] = None) -> PreprocessingPipeline:
    """
    Create a preprocessing pipeline specifically for PLS models.
    
    Args:
        model_id: PLS model ID
        cache_manager: Optional cache manager
        config: Optional configuration dictionary
        
    Returns:
        Configured preprocessing pipeline
    """
    pipeline = PreprocessingPipeline(model_id, cache_manager, config)
    if pipeline.get_model_type() != 'pls':
        raise ValueError(f"Model {model_id} is not a PLS model")
    return pipeline