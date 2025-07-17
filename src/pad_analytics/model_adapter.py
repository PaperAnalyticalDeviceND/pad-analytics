"""
Unified model adapter interface for PAD Analytics.

This module provides the main ModelAdapter class that automatically
selects and configures the appropriate model adapter based on model type.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd
from pathlib import Path

from .adapters import BaseAdapter, NeuralNetworkAdapter, PLSAdapter
from .cache_manager import CacheManager


class ModelAdapter:
    """
    Main model adapter that automatically selects the appropriate
    adapter based on model type and provides a unified interface.
    """
    
    def __init__(self, model_id: int, 
                 cache_manager: Optional[CacheManager] = None,
                 auto_load: bool = True):
        """
        Initialize the model adapter.
        
        Args:
            model_id: The model ID to create adapter for
            cache_manager: Optional cache manager for caching models and predictions
            auto_load: Whether to automatically load the model on initialization
        """
        self.model_id = model_id
        self.cache_manager = cache_manager
        
        # Auto-detect and create appropriate adapter
        self.adapter = self._create_adapter()
        
        # Automatically load model if requested
        if auto_load:
            self.load_model()
    
    def _create_adapter(self) -> BaseAdapter:
        """
        Create the appropriate adapter based on model type.
        
        Returns:
            Configured adapter instance
        """
        model_type = self._detect_model_type()
        
        if model_type == 'neural_network':
            return NeuralNetworkAdapter(self.model_id, self.cache_manager)
        elif model_type == 'pls':
            return PLSAdapter(self.model_id, self.cache_manager)
        else:
            raise ValueError(f"Unknown model type for model_id {self.model_id}: {model_type}")
    
    def _detect_model_type(self) -> str:
        """
        Detect the model type based on model ID.
        
        Returns:
            String indicating model type ('neural_network' or 'pls')
        """
        # Model type detection based on known model IDs
        neural_network_models = {16, 17, 19, 20}  # NN classification and concentration models
        pls_models = {18}  # PLS concentration models
        
        if self.model_id in neural_network_models:
            return 'neural_network'
        elif self.model_id in pls_models:
            return 'pls'
        else:
            # Default assumption for unknown models
            print(f"Warning: Unknown model_id {self.model_id}, assuming neural_network")
            return 'neural_network'
    
    def load_model(self) -> bool:
        """
        Load the model.
        
        Returns:
            True if model was loaded successfully
        """
        return self.adapter.load_model()
    
    def predict(self, card_data: Dict[str, Any]) -> Union[Tuple[str, float, float], float]:
        """
        Make a prediction for a single card.
        
        Args:
            card_data: Raw card data dictionary
            
        Returns:
            Prediction result:
            - Neural Network: (drug_name, confidence, energy)
            - PLS: concentration as float
        """
        return self.adapter.predict(card_data)
    
    def predict_batch(self, cards_data: List[Dict[str, Any]]) -> List[Union[Tuple[str, float, float], float]]:
        """
        Make predictions for a batch of cards.
        
        Args:
            cards_data: List of card data dictionaries
            
        Returns:
            List of prediction results
        """
        # Create preprocessing pipeline
        from .preprocessing_pipeline import PreprocessingPipeline
        preprocessor = PreprocessingPipeline(self.model_id, self.cache_manager)
        
        # Preprocess batch
        preprocessed_batch = preprocessor.preprocess_batch(cards_data)
        
        # Convert to list of dictionaries
        preprocessed_list = preprocessed_batch.to_dict('records')
        
        # Make predictions
        return self.adapter.predict_batch(preprocessed_list)
    
    def predict_dataset(self, dataset: Union[pd.DataFrame, 'CachedDataset'],
                       max_cards: Optional[int] = None) -> pd.DataFrame:
        """
        Make predictions for an entire dataset.
        
        Args:
            dataset: Dataset to make predictions for
            max_cards: Maximum number of cards to process (None for all)
            
        Returns:
            DataFrame with predictions added
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
        if max_cards is not None:
            cards_data = cards_data[:max_cards]
        
        # Make predictions
        predictions = self.predict_batch(cards_data)
        
        # Create results DataFrame
        results_df = pd.DataFrame(cards_data)
        
        # Add predictions
        if self.get_model_type() == 'neural_network':
            # Neural network predictions are tuples
            results_df['predicted_drug'] = [pred[0] for pred in predictions]
            results_df['confidence'] = [pred[1] for pred in predictions]
            results_df['energy'] = [pred[2] for pred in predictions]
        else:
            # PLS predictions are floats
            results_df['predicted_concentration'] = predictions
        
        return results_df
    
    def get_model_type(self) -> str:
        """
        Get the model type this adapter is configured for.
        
        Returns:
            String indicating model type
        """
        return self.adapter.get_model_type()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.
        
        Returns:
            Dictionary containing model information
        """
        return self.adapter.get_model_info()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded model.
        
        Returns:
            Dictionary containing model summary information
        """
        return self.adapter.get_model_summary()
    
    def is_loaded(self) -> bool:
        """
        Check if the model is currently loaded.
        
        Returns:
            True if model is loaded and ready for predictions
        """
        return self.adapter.is_model_loaded()
    
    def unload_model(self) -> None:
        """
        Unload the model to free memory.
        """
        self.adapter.unload_model()
    
    def get_expected_input_format(self) -> str:
        """
        Get the expected input format for this adapter.
        
        Returns:
            String describing expected input format
        """
        return self.adapter.get_expected_input_format()
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        return (f"ModelAdapter(model_id={self.model_id}, "
                f"type={self.get_model_type()}, "
                f"loaded={self.is_loaded()})")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        model_info = self.get_model_info()
        model_name = model_info.get('model_info', {}).get('name', 'Unknown')
        return f"ModelAdapter for {model_name} (ID: {self.model_id}, Type: {self.get_model_type()})"


# Convenience functions for direct use
def create_neural_network_adapter(model_id: int, 
                                 cache_manager: Optional[CacheManager] = None,
                                 auto_load: bool = True) -> ModelAdapter:
    """
    Create a model adapter specifically for neural network models.
    
    Args:
        model_id: Neural network model ID
        cache_manager: Optional cache manager
        auto_load: Whether to automatically load the model
        
    Returns:
        Configured model adapter
    """
    adapter = ModelAdapter(model_id, cache_manager, auto_load=False)
    if adapter.get_model_type() != 'neural_network':
        raise ValueError(f"Model {model_id} is not a neural network model")
    
    if auto_load:
        adapter.load_model()
    
    return adapter


def create_pls_adapter(model_id: int, 
                      cache_manager: Optional[CacheManager] = None,
                      auto_load: bool = True) -> ModelAdapter:
    """
    Create a model adapter specifically for PLS models.
    
    Args:
        model_id: PLS model ID
        cache_manager: Optional cache manager
        auto_load: Whether to automatically load the model
        
    Returns:
        Configured model adapter
    """
    adapter = ModelAdapter(model_id, cache_manager, auto_load=False)
    if adapter.get_model_type() != 'pls':
        raise ValueError(f"Model {model_id} is not a PLS model")
    
    if auto_load:
        adapter.load_model()
    
    return adapter


def get_available_models() -> Dict[int, Dict[str, Any]]:
    """
    Get information about all available models.
    
    Returns:
        Dictionary mapping model IDs to model information
    """
    models = {}
    
    # Neural network models
    for model_id in [16, 17, 19, 20]:
        try:
            adapter = ModelAdapter(model_id, auto_load=False)
            models[model_id] = adapter.get_model_info()
        except Exception as e:
            print(f"Warning: Could not get info for NN model {model_id}: {e}")
    
    # PLS models
    for model_id in [18]:
        try:
            adapter = ModelAdapter(model_id, auto_load=False)
            models[model_id] = adapter.get_model_info()
        except Exception as e:
            print(f"Warning: Could not get info for PLS model {model_id}: {e}")
    
    return models