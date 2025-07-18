"""
PLS (Partial Least Squares) preprocessor for PAD Analytics.

This module handles preprocessing for PLS models, including:
- Region-based feature extraction from PAD images
- RGB color space analysis across 12 lanes (A-L)
- Statistical preprocessing for concentration prediction
- Integration with regionRoutine for standardized feature extraction
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import cv2 as cv
import requests
from io import BytesIO
from PIL import Image

from .base_preprocessor import BasePreprocessor
from .. import regionRoutine


class PLSPreprocessor(BasePreprocessor):
    """
    Preprocessor for PLS (Partial Least Squares) models.
    
    Handles region-based feature extraction from PAD images using the existing
    regionRoutine module to extract RGB color features from 12 lanes (A-L) 
    with 10 regions each, resulting in 360 features (12 lanes × 10 regions × 3 colors).
    """
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration for PLS preprocessing.
        
        Returns:
            Dictionary containing default configuration parameters
        """
        return {
            # Region extraction parameters
            'num_lanes': 12,  # A through L
            'num_regions_per_lane': 10,
            'lane_labels': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'],
            'color_channels': ['R', 'G', 'B'],
            
            # Feature extraction parameters
            'region_function': 'findMaxIntensitiesFiltered',  # regionRoutine function
            'normalize_features': True,
            'feature_scaling': 'standard',  # 'standard', 'minmax', or 'none'
            
            # Image processing parameters
            'image_format': 'BGR',  # OpenCV default
            'required_image_size': None,  # Will be validated by regionRoutine
            
            # Performance parameters
            'batch_size': 16,
            'cache_features': True,
        }
    
    def preprocess_single_card(self, card_data: Dict[str, Any], 
                              image_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Preprocess a single card's data for PLS model inference.
        
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
        
        # Load image
        if image_path:
            img = cv.imread(str(image_path))
        else:
            img = self._load_image_from_url(image_url)
        
        if img is None:
            raise ValueError(f"Failed to load image from {image_path or image_url}")
        
        # Extract features using regionRoutine
        features = self._extract_pls_features(img)
        
        # Prepare result
        result = {
            'card_id': card_data['id'],
            'features': features,
            'feature_names': self.get_feature_names(),
            'num_features': len(features),
            'model_id': self.model_id,
            'preprocessor_type': 'pls'
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
        
        # Expand features into separate columns
        if 'features' in df.columns:
            feature_names = self.get_feature_names()
            features_df = pd.DataFrame(df['features'].tolist(), columns=feature_names)
            df = pd.concat([df.drop('features', axis=1), features_df], axis=1)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features produced by this preprocessor.
        
        Returns:
            List of feature names in the order they appear in preprocessed data
        """
        feature_names = []
        
        # Generate feature names following the format: A1-R, A1-G, A1-B, A2-R, etc.
        for lane in self.config['lane_labels']:
            for region in range(1, self.config['num_regions_per_lane'] + 1):
                for color in self.config['color_channels']:
                    feature_names.append(f"{lane}{region}-{color}")
        
        return feature_names
    
    def get_expected_input_shape(self) -> Tuple[int, ...]:
        """
        Get the expected input shape for this preprocessor.
        
        Returns:
            Tuple representing the expected input shape (num_features,)
        """
        num_features = (self.config['num_lanes'] * 
                       self.config['num_regions_per_lane'] * 
                       len(self.config['color_channels']))
        return (num_features,)
    
    def _load_image_from_url(self, image_url: str) -> np.ndarray:
        """
        Load an image from a URL and convert to OpenCV format.
        
        Args:
            image_url: URL to the image
            
        Returns:
            OpenCV image array (BGR format)
        """
        try:
            response = requests.get(image_url, verify=False)
            response.raise_for_status()
            
            # Convert to PIL Image first
            pil_image = Image.open(BytesIO(response.content))
            
            # Convert to numpy array
            img_array = np.array(pil_image)
            
            # Convert RGB to BGR for OpenCV
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv.cvtColor(img_array, cv.COLOR_RGB2BGR)
            
            return img_array
        except Exception as e:
            raise ValueError(f"Failed to load image from URL {image_url}: {e}")
    
    def _extract_pls_features(self, img: np.ndarray) -> List[float]:
        """
        Extract PLS features from a PAD image using regionRoutine.
        
        Args:
            img: OpenCV image array
            
        Returns:
            List of feature values in the order specified by get_feature_names()
        """
        try:
            # Use regionRoutine to extract features
            # This follows the same pattern as the pls.quantity() method
            features_dict = {}
            
            # Apply the region routine with the specified function
            if self.config['region_function'] == 'findMaxIntensitiesFiltered':
                features_dict = regionRoutine.fullRoutine(
                    img, 
                    regionRoutine.intFind.findMaxIntensitiesFiltered, 
                    features_dict, 
                    True, 
                    10
                )
            else:
                # Default fallback - could be extended for other functions
                features_dict = regionRoutine.fullRoutine(
                    img, 
                    regionRoutine.intFind.findMaxIntensitiesFiltered, 
                    features_dict, 
                    True, 
                    10
                )
            
            # Convert dictionary to ordered list following the feature names
            features = []
            for lane in self.config['lane_labels']:
                for region in range(1, self.config['num_regions_per_lane'] + 1):
                    for color in self.config['color_channels']:
                        feature_key = f"{lane}{region}-{color}"
                        feature_value = features_dict.get(feature_key, 0.0)
                        features.append(float(feature_value))
            
            # Apply normalization if enabled
            if self.config['normalize_features']:
                features = self._normalize_features(features)
            
            return features
            
        except Exception as e:
            raise ValueError(f"Failed to extract PLS features from image: {e}")
    
    def _normalize_features(self, features: List[float]) -> List[float]:
        """
        Normalize features according to the configuration.
        
        Args:
            features: List of raw feature values
            
        Returns:
            List of normalized feature values
        """
        if self.config['feature_scaling'] == 'none':
            return features
        
        features_array = np.array(features)
        
        if self.config['feature_scaling'] == 'standard':
            # Standard scaling: (x - mean) / std
            mean = np.mean(features_array)
            std = np.std(features_array)
            if std > 0:
                features_array = (features_array - mean) / std
        
        elif self.config['feature_scaling'] == 'minmax':
            # Min-max scaling: (x - min) / (max - min)
            min_val = np.min(features_array)
            max_val = np.max(features_array)
            if max_val > min_val:
                features_array = (features_array - min_val) / (max_val - min_val)
        
        return features_array.tolist()
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """
        Validate that input data meets PLS preprocessor requirements.
        
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
    
    def get_features_for_prediction(self, preprocessed_data: Dict[str, Any]) -> List[float]:
        """
        Extract features in the format expected by PLS models.
        
        Args:
            preprocessed_data: Output from preprocess_single_card
            
        Returns:
            List of feature values ready for PLS model prediction
        """
        return preprocessed_data['features']