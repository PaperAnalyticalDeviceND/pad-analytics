"""
Unified model adapter interface for PAD Analytics.

This module provides the main ModelAdapter class that automatically
selects and configures the appropriate model adapter based on model type.
"""

from typing import Dict, Any, Optional, List, Union, Tuple, Awaitable
import pandas as pd
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from .adapters import BaseAdapter, NeuralNetworkAdapter, PLSAdapter
from .cache_manager import CacheManager
from .performance_monitor import performance_monitor, get_global_monitor


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
        
        # Performance monitoring
        self.performance_monitor = get_global_monitor()
        
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
    
    @performance_monitor("model_adapter_predict")
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
    
    def predict_batch(self, cards_data: List[Dict[str, Any]], 
                     parallel: bool = True, 
                     max_workers: Optional[int] = None) -> List[Union[Tuple[str, float, float], float]]:
        """
        Make predictions for a batch of cards with optional parallel processing.
        
        Args:
            cards_data: List of card data dictionaries
            parallel: Whether to use parallel processing for large batches
            max_workers: Maximum number of parallel workers (None for auto)
            
        Returns:
            List of prediction results
        """
        # For small batches, use regular processing
        if len(cards_data) <= 10 or not parallel:
            return self._predict_batch_sequential(cards_data)
        
        # For larger batches, use parallel processing
        return self._predict_batch_parallel(cards_data, max_workers)
    
    def _predict_batch_sequential(self, cards_data: List[Dict[str, Any]]) -> List[Union[Tuple[str, float, float], float]]:
        """
        Sequential batch prediction (original method).
        
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
    
    def _predict_batch_parallel(self, cards_data: List[Dict[str, Any]], 
                               max_workers: Optional[int] = None) -> List[Union[Tuple[str, float, float], float]]:
        """
        Parallel batch prediction for large datasets.
        
        Args:
            cards_data: List of card data dictionaries
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of prediction results
        """
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 8)  # Don't overwhelm the system
        
        # Split into chunks for parallel processing
        chunk_size = max(1, len(cards_data) // max_workers)
        chunks = [cards_data[i:i + chunk_size] for i in range(0, len(cards_data), chunk_size)]
        
        # Process chunks in parallel using threads (not processes due to model state)
        all_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._predict_batch_sequential, chunk): chunk 
                for chunk in chunks
            }
            
            # Collect results maintaining order
            chunk_results = {}
            from concurrent.futures import as_completed
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                chunk_index = chunks.index(chunk)
                try:
                    chunk_results[chunk_index] = future.result()
                except Exception as e:
                    print(f"Chunk processing failed: {e}")
                    # Create placeholder results for failed chunk
                    placeholder = self._get_placeholder_result()
                    chunk_results[chunk_index] = [placeholder] * len(chunk)
        
        # Combine results in order
        for i in sorted(chunk_results.keys()):
            all_results.extend(chunk_results[i])
        
        return all_results
    
    def _get_placeholder_result(self) -> Union[Tuple[str, float, float], float]:
        """
        Get a placeholder result for failed predictions.
        
        Returns:
            Placeholder result based on model type
        """
        if self.get_model_type() == 'neural_network':
            return ("unknown", 0.0, 0.0)
        else:
            return 0.0
    
    async def predict_async(self, card_data: Dict[str, Any]) -> Union[Tuple[str, float, float], float]:
        """
        Make an asynchronous prediction for a single card.
        
        Args:
            card_data: Raw card data dictionary
            
        Returns:
            Prediction result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, card_data)
    
    async def predict_batch_async(self, cards_data: List[Dict[str, Any]], 
                                 parallel: bool = True,
                                 max_workers: Optional[int] = None) -> List[Union[Tuple[str, float, float], float]]:
        """
        Make asynchronous predictions for a batch of cards.
        
        Args:
            cards_data: List of card data dictionaries
            parallel: Whether to use parallel processing for large batches
            max_workers: Maximum number of parallel workers (None for auto)
            
        Returns:
            List of prediction results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict_batch, cards_data, parallel, max_workers)
    
    async def predict_dataset_async(self, dataset: Union[pd.DataFrame, 'CachedDataset'],
                                   max_cards: Optional[int] = None,
                                   batch_size: int = 100,
                                   progress_callback: Optional[callable] = None) -> pd.DataFrame:
        """
        Make asynchronous predictions for an entire dataset with progress tracking.
        
        Args:
            dataset: Dataset to make predictions for
            max_cards: Maximum number of cards to process (None for all)
            batch_size: Size of batches for processing
            progress_callback: Optional callback function for progress updates
            
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
        
        # Process in batches
        all_predictions = []
        total_batches = (len(cards_data) + batch_size - 1) // batch_size
        
        for i in range(0, len(cards_data), batch_size):
            batch = cards_data[i:i + batch_size]
            batch_predictions = await self.predict_batch_async(batch, parallel=True)
            all_predictions.extend(batch_predictions)
            
            # Call progress callback if provided
            if progress_callback:
                batch_num = (i // batch_size) + 1
                progress_callback(batch_num, total_batches, len(batch))
        
        # Create results DataFrame
        results_df = pd.DataFrame(cards_data)
        
        # Add predictions
        if self.get_model_type() == 'neural_network':
            # Neural network predictions are tuples
            results_df['predicted_drug'] = [pred[0] for pred in all_predictions]
            results_df['confidence'] = [pred[1] for pred in all_predictions]
            results_df['energy'] = [pred[2] for pred in all_predictions]
        else:
            # PLS predictions are floats
            results_df['predicted_concentration'] = all_predictions
        
        return results_df
    
    def predict_dataset(self, dataset: Union[pd.DataFrame, 'CachedDataset'],
                       max_cards: Optional[int] = None,
                       parallel: bool = True) -> pd.DataFrame:
        """
        Make predictions for an entire dataset with improved parallel processing.
        
        Args:
            dataset: Dataset to make predictions for
            max_cards: Maximum number of cards to process (None for all)
            parallel: Whether to use parallel processing for large datasets
            
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
        
        # Make predictions with improved parallel processing
        predictions = self.predict_batch(cards_data, parallel=parallel)
        
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