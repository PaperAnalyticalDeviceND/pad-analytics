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
        Preprocess a batch of cards efficiently.
        
        Args:
            cards_data: List of card data dictionaries
            image_paths: Optional list of image paths corresponding to cards
            
        Returns:
            DataFrame with preprocessed features and metadata
        """
        if not cards_data:
            return pd.DataFrame()
        
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
        image_arrays = np.stack(df['preprocessed_image'].values)
        df['image_batch'] = [image_arrays] * len(df)  # Reference to batch
        df['batch_index'] = range(len(df))
        
        return df
    
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