"""
Neural Network preprocessor for PAD Analytics.

This module handles preprocessing for neural network models, including:
- Image loading and resizing
- Cropping to active PAD area
- Normalization for model input
- Feature extraction from image data
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO

from .base_preprocessor import BasePreprocessor


class NeuralNetworkPreprocessor(BasePreprocessor):
    """
    Preprocessor for Neural Network models.
    
    Handles image preprocessing including cropping, resizing, and normalization
    according to the requirements of PAD neural network models.
    """
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration for neural network preprocessing.
        
        Returns:
            Dictionary containing default configuration parameters
        """
        return {
            # Image preprocessing parameters
            'crop_box': (71, 359, 71 + 636, 359 + 490),  # (left, top, right, bottom)
            'target_size': (454, 454),  # (width, height)
            'resize_method': Image.BICUBIC,
            'input_shape': (454, 454, 3),  # (height, width, channels)
            'dtype': np.float32,
            
            # Normalization parameters (disabled to match original nn_predict behavior)
            'normalize': False,
            'mean': [0.485, 0.456, 0.406],  # ImageNet defaults (not used when normalize=False)
            'std': [0.229, 0.224, 0.225],   # ImageNet defaults (not used when normalize=False)
            
            # Performance parameters
            'batch_size': 32,
            'num_workers': 4,
        }
    
    def preprocess_single_card(self, card_data: Dict[str, Any], 
                              image_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Preprocess a single card's data for neural network inference.
        
        Args:
            card_data: Dictionary containing card metadata and features
            image_path: Optional path to the card's image file
            
        Returns:
            Dictionary containing preprocessed features and metadata
        """
        if not self.validate_input(card_data):
            raise ValueError(f"Invalid card data: {card_data}")
        
        # Get image URL or path (check both 'image_url' and 'url' fields)
        image_url = card_data.get('image_url') or card_data.get('url')
        if not image_url and not image_path:
            raise ValueError("Either image_url/url in card_data or image_path must be provided")
        
        # Load and preprocess image
        if image_path:
            img = Image.open(image_path)
        else:
            img = self._load_image_from_url(image_url)
        
        # Preprocess image
        preprocessed_array = self._preprocess_image(img)
        
        # Prepare result
        result = {
            'card_id': card_data['id'],
            'preprocessed_image': preprocessed_array,
            'original_shape': img.size,
            'processed_shape': preprocessed_array.shape,
            'model_id': self.model_id,
            'preprocessor_type': 'neural_network'
        }
        
        # Add metadata
        for key in ['sample_name', 'sample_id', 'lane', 'concentration']:
            if key in card_data:
                result[key] = card_data[key]
        
        return result
    
    def preprocess_batch(self, cards_data: List[Dict[str, Any]], 
                        image_paths: Optional[List[Path]] = None) -> pd.DataFrame:
        """
        Preprocess a batch of cards efficiently with parallel image loading.
        
        Args:
            cards_data: List of card data dictionaries
            image_paths: Optional list of image paths corresponding to cards
            
        Returns:
            DataFrame with preprocessed features and metadata
        """
        if not cards_data:
            return pd.DataFrame()
        
        # Try optimized batch processing first
        try:
            return self._preprocess_batch_optimized(cards_data, image_paths)
        except Exception as e:
            print(f"Optimized batch preprocessing failed, falling back to sequential: {e}")
            # Fallback to sequential processing
            return self._preprocess_batch_sequential(cards_data, image_paths)
    
    def _preprocess_batch_optimized(self, cards_data: List[Dict[str, Any]], 
                                   image_paths: Optional[List[Path]] = None) -> pd.DataFrame:
        """
        Optimized batch preprocessing with parallel image loading.
        
        Args:
            cards_data: List of card data dictionaries
            image_paths: Optional list of image paths corresponding to cards
            
        Returns:
            DataFrame with preprocessed features and metadata
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        # Prepare work items
        work_items = []
        for i, card_data in enumerate(cards_data):
            if not self.validate_input(card_data):
                print(f"Warning: Invalid card data at index {i}, skipping")
                continue
            
            image_path = image_paths[i] if image_paths else None
            work_items.append((i, card_data, image_path))
        
        if not work_items:
            return pd.DataFrame()
        
        results = {}
        max_workers = min(self.config.get('num_workers', 4), len(work_items))
        
        # Process in parallel with thread safety
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all preprocessing tasks
            future_to_item = {
                executor.submit(self._preprocess_single_item_safe, item): item 
                for item in work_items
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_item):
                original_index, result = future.result()
                if result is not None:
                    results[original_index] = result
        
        if not results:
            return pd.DataFrame()
        
        # Sort results by original index to maintain order
        sorted_results = [results[i] for i in sorted(results.keys())]
        
        # Convert to DataFrame
        df = pd.DataFrame(sorted_results)
        
        # Create optimized batch structure
        if len(df) > 1:
            try:
                # Stack all preprocessed images for efficient batch operations
                image_arrays = np.stack([arr.squeeze(0) for arr in df['preprocessed_image'].values])
                df['image_batch'] = [image_arrays] * len(df)  # Reference to batch
                df['batch_index'] = range(len(df))
            except Exception as e:
                print(f"Warning: Could not create image batch: {e}")
                # Fall back to individual arrays
                df['image_batch'] = df['preprocessed_image']
                df['batch_index'] = range(len(df))
        else:
            df['image_batch'] = df['preprocessed_image']
            df['batch_index'] = range(len(df))
        
        return df
    
    def _preprocess_batch_sequential(self, cards_data: List[Dict[str, Any]], 
                                    image_paths: Optional[List[Path]] = None) -> pd.DataFrame:
        """
        Sequential batch preprocessing (fallback method).
        
        Args:
            cards_data: List of card data dictionaries
            image_paths: Optional list of image paths corresponding to cards
            
        Returns:
            DataFrame with preprocessed features and metadata
        """
        results = []
        
        for i, card_data in enumerate(cards_data):
            try:
                image_path = image_paths[i] if image_paths else None
                result = self.preprocess_single_card(card_data, image_path)
                results.append(result)
            except Exception as e:
                # Log error but continue processing other cards
                print(f"Warning: Failed to preprocess card {card_data.get('id', 'unknown')}: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Separate image arrays from metadata
        try:
            image_arrays = np.stack(df['preprocessed_image'].values)
            df['image_batch'] = [image_arrays] * len(df)  # Reference to batch
            df['batch_index'] = range(len(df))
        except Exception as e:
            print(f"Warning: Could not create image batch: {e}")
            df['image_batch'] = df['preprocessed_image']
            df['batch_index'] = range(len(df))
        
        return df
    
    def _preprocess_single_item_safe(self, item: tuple) -> tuple:
        """
        Thread-safe wrapper for preprocessing a single item.
        
        Args:
            item: Tuple of (index, card_data, image_path)
            
        Returns:
            Tuple of (original_index, result_dict or None)
        """
        original_index, card_data, image_path = item
        
        try:
            result = self.preprocess_single_card(card_data, image_path)
            return (original_index, result)
        except Exception as e:
            print(f"Warning: Failed to preprocess card {card_data.get('id', 'unknown')} at index {original_index}: {e}")
            return (original_index, None)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features produced by this preprocessor.
        
        Returns:
            List of feature names in the order they appear in preprocessed data
        """
        height, width, channels = self.config['input_shape']
        
        # For neural networks, features are pixel values
        feature_names = []
        for h in range(height):
            for w in range(width):
                for c in range(channels):
                    feature_names.append(f'pixel_{h}_{w}_{c}')
        
        return feature_names
    
    def get_expected_input_shape(self) -> Tuple[int, ...]:
        """
        Get the expected input shape for this preprocessor.
        
        Returns:
            Tuple representing the expected input shape (height, width, channels)
        """
        return self.config['input_shape']
    
    def _load_image_from_url(self, image_url: str) -> Image.Image:
        """
        Load an image from a URL.
        
        Args:
            image_url: URL to the image
            
        Returns:
            PIL Image object
        """
        try:
            response = requests.get(image_url, verify=False)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            raise ValueError(f"Failed to load image from URL {image_url}: {e}")
    
    def _preprocess_image(self, img: Image.Image) -> np.ndarray:
        """
        Preprocess a PIL image for neural network input.
        
        Args:
            img: PIL Image object
            
        Returns:
            Preprocessed numpy array ready for model input
        """
        # Crop to active PAD area
        img = img.crop(self.config['crop_box'])
        
        # Resize to target size
        img = img.resize(self.config['target_size'], self.config['resize_method'])
        
        # Convert to numpy array
        img_array = np.asarray(img)
        
        # Ensure 3 channels (RGB)
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] > 3:
            img_array = img_array[:, :, :3]
        
        # Normalize if enabled
        if self.config['normalize']:
            img_array = img_array.astype(self.config['dtype']) / 255.0
            
            # Apply mean and std normalization
            mean = np.array(self.config['mean'], dtype=self.config['dtype'])
            std = np.array(self.config['std'], dtype=self.config['dtype'])
            img_array = (img_array - mean) / std
        else:
            img_array = img_array.astype(self.config['dtype'])
        
        # Ensure correct dtype before reshaping
        img_array = img_array.astype(self.config['dtype'])
        
        # Reshape for model input (add batch dimension)
        img_array = img_array.reshape(1, *self.config['input_shape'])
        
        return img_array
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """
        Validate that input data meets neural network preprocessor requirements.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        if not super().validate_input(data):
            return False
        
        # Check for required fields
        required_fields = ['id']
        for field in required_fields:
            if field not in data:
                return False
        
        # Check for image source (either 'image_url' or 'url')
        has_image_url = ('image_url' in data and data['image_url']) or ('url' in data and data['url'])
        # image_path will be checked separately in preprocess_single_card
        
        return has_image_url  # At least one image source must be available
    
    def get_model_compatible_array(self, preprocessed_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract model-compatible array from preprocessed data.
        
        Args:
            preprocessed_data: Output from preprocess_single_card
            
        Returns:
            Numpy array ready for model inference
        """
        return preprocessed_data['preprocessed_image']