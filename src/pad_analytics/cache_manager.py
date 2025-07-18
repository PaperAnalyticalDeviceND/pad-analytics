"""
PAD Analytics Data Caching System - Phase 1: Basic Image Caching

This module implements the foundational caching infrastructure for PAD analytics,
focusing on eliminating redundant image downloads and enabling offline research workflows.
"""

import os
import hashlib
import json
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse
import requests
from PIL import Image
import pandas as pd
import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super().default(obj)


class CacheManager:
    """
    Professional caching system for PAD analytics data.
    
    Implements hierarchical caching with:
    - Raw image storage
    - Metadata persistence  
    - Cache integrity verification
    - Automatic cleanup mechanisms
    - Preprocessed data caching (Phase 2)
    """
    
    def __init__(self, cache_dir: str = "~/.pad_cache", max_cache_size_gb: float = 5.0):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Base directory for cache storage
            max_cache_size_gb: Maximum cache size in gigabytes
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.max_cache_size = max_cache_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        
        # Cache subdirectories
        self.raw_images_dir = self.cache_dir / "raw_images"
        self.metadata_dir = self.cache_dir / "metadata"
        self.datasets_dir = self.cache_dir / "datasets"
        self.preprocessed_dir = self.cache_dir / "preprocessed"  # Phase 2: Preprocessed data
        self.models_dir = self.cache_dir / "models"  # Phase 3: Model files
        
        # Create cache structure
        self._initialize_cache_structure()
        
    def _initialize_cache_structure(self):
        """Create cache directory structure."""
        for directory in [self.raw_images_dir, self.metadata_dir, self.datasets_dir, self.preprocessed_dir, self.models_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Create cache info file
        cache_info_file = self.cache_dir / "cache_info.json"
        if not cache_info_file.exists():
            cache_info = {
                "version": "1.0",
                "created": time.time(),
                "last_cleanup": time.time(),
                "max_size_gb": self.max_cache_size / (1024**3)
            }
            with open(cache_info_file, 'w') as f:
                json.dump(cache_info, f, indent=2, cls=NumpyJSONEncoder)
    
    def get_image_cache_key(self, image_url: str) -> str:
        """
        Generate a unique cache key for an image URL.
        
        Args:
            image_url: URL of the PAD image
            
        Returns:
            Unique cache key (hash) for the image
        """
        # Use URL hash as cache key for consistency
        return hashlib.md5(image_url.encode()).hexdigest()
    
    def is_image_cached(self, image_url: str) -> bool:
        """
        Check if an image is already cached.
        
        Args:
            image_url: URL of the PAD image
            
        Returns:
            True if image is cached, False otherwise
        """
        cache_key = self.get_image_cache_key(image_url)
        image_path = self.raw_images_dir / f"{cache_key}.png"
        metadata_path = self.raw_images_dir / f"{cache_key}.json"
        
        return image_path.exists() and metadata_path.exists()
    
    def cache_image(self, image_url: str, card_metadata: Optional[Dict] = None) -> str:
        """
        Download and cache an image with metadata.
        
        Args:
            image_url: URL of the PAD image to cache
            card_metadata: Optional metadata about the card/image
            
        Returns:
            Path to cached image file
            
        Raises:
            Exception: If download or caching fails
        """
        cache_key = self.get_image_cache_key(image_url)
        image_path = self.raw_images_dir / f"{cache_key}.png"
        metadata_path = self.raw_images_dir / f"{cache_key}.json"
        
        # If already cached, return existing path
        if self.is_image_cached(image_url):
            return str(image_path)
        
        try:
            # Download image
            print(f"Downloading and caching image: {image_url}")
            response = requests.get(image_url, stream=True, verify=False, timeout=30)
            response.raise_for_status()
            
            # Save image
            with open(image_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            
            # Verify image can be opened
            with Image.open(image_path) as img:
                img_size = img.size
            
            # Save metadata
            metadata = {
                "url": image_url,
                "cache_key": cache_key,
                "cached_at": time.time(),
                "file_size": os.path.getsize(image_path),
                "image_size": img_size,
                "card_metadata": card_metadata or {}
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, cls=NumpyJSONEncoder)
            
            print(f"✅ Image cached successfully: {cache_key}")
            return str(image_path)
            
        except Exception as e:
            # Clean up partial downloads
            for path in [image_path, metadata_path]:
                if path.exists():
                    path.unlink()
            raise Exception(f"Failed to cache image {image_url}: {e}")
    
    def get_cached_image_path(self, image_url: str) -> Optional[str]:
        """
        Get path to cached image if it exists.
        
        Args:
            image_url: URL of the PAD image
            
        Returns:
            Path to cached image or None if not cached
        """
        if self.is_image_cached(image_url):
            cache_key = self.get_image_cache_key(image_url)
            return str(self.raw_images_dir / f"{cache_key}.png")
        return None
    
    def get_image_metadata(self, image_url: str) -> Optional[Dict]:
        """
        Get cached metadata for an image.
        
        Args:
            image_url: URL of the PAD image
            
        Returns:
            Metadata dictionary or None if not cached
        """
        if self.is_image_cached(image_url):
            cache_key = self.get_image_cache_key(image_url)
            metadata_path = self.raw_images_dir / f"{cache_key}.json"
            
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def cache_dataset_metadata(self, dataset_name: str, dataset_df: pd.DataFrame) -> str:
        """
        Cache dataset metadata for faster subsequent access.
        
        Args:
            dataset_name: Name of the dataset
            dataset_df: DataFrame containing dataset information
            
        Returns:
            Path to cached dataset file
        """
        dataset_path = self.datasets_dir / f"{dataset_name}.csv"
        metadata_path = self.datasets_dir / f"{dataset_name}_info.json"
        
        # Save dataset as CSV for compatibility (future: upgrade to parquet when pyarrow available)
        dataset_df.to_csv(dataset_path, index=False)
        
        # Save metadata
        metadata = {
            "dataset_name": dataset_name,
            "cached_at": time.time(),
            "num_records": len(dataset_df),
            "columns": list(dataset_df.columns),
            "file_size": os.path.getsize(dataset_path)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Dataset cached: {dataset_name} ({len(dataset_df)} records)")
        return str(dataset_path)
    
    def get_cached_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Load cached dataset if available.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame or None if not cached
        """
        dataset_path = self.datasets_dir / f"{dataset_name}.csv"
        
        if dataset_path.exists():
            try:
                return pd.read_csv(dataset_path)
            except Exception as e:
                print(f"Warning: Failed to load cached dataset {dataset_name}: {e}")
                return None
        return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "cache_dir": str(self.cache_dir),
            "total_size_mb": 0,
            "num_images": 0,
            "num_datasets": 0,
            "num_preprocessed": 0,  # Phase 2: Preprocessed data count
            "num_models": 0,  # Phase 3: Model files count
            "oldest_entry": None,
            "newest_entry": None
        }
        
        # Count images and calculate size
        image_times = []
        for img_file in self.raw_images_dir.glob("*.png"):
            stats["total_size_mb"] += os.path.getsize(img_file) / (1024 * 1024)
            stats["num_images"] += 1
            
            # Get cached time from metadata
            metadata_file = img_file.with_suffix('.json')
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        image_times.append(metadata.get('cached_at', 0))
                except:
                    pass
        
        # Count datasets
        for dataset_file in self.datasets_dir.glob("*.csv"):
            stats["total_size_mb"] += os.path.getsize(dataset_file) / (1024 * 1024)
            stats["num_datasets"] += 1
        
        # Count preprocessed data (Phase 2)
        for preprocessed_file in self.preprocessed_dir.glob("*.json"):
            stats["total_size_mb"] += os.path.getsize(preprocessed_file) / (1024 * 1024)
            stats["num_preprocessed"] += 1
        
        # Count models (Phase 3)
        for model_file in self.models_dir.glob("*"):
            if model_file.is_file() and not model_file.name.endswith('_metadata.json'):
                stats["total_size_mb"] += os.path.getsize(model_file) / (1024 * 1024)
                stats["num_models"] += 1
        
        # Calculate time ranges
        if image_times:
            stats["oldest_entry"] = min(image_times)
            stats["newest_entry"] = max(image_times)
        
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        return stats
    
    def _get_cache_size_bytes(self) -> int:
        """Get total cache size in bytes."""
        total_size = 0
        for directory in [self.raw_images_dir, self.metadata_dir, self.datasets_dir, self.preprocessed_dir, self.models_dir]:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        return total_size
    
    def cleanup_old_cache(self, max_age_days: int = 30) -> Dict[str, int]:
        """
        Clean up old cache entries to free space.
        
        Args:
            max_age_days: Maximum age of cache entries in days
            
        Returns:
            Dictionary with cleanup statistics
        """
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        
        cleaned_images = 0
        cleaned_size_mb = 0
        
        # Clean old images
        for metadata_file in self.raw_images_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                if metadata.get('cached_at', 0) < cutoff_time:
                    # Remove image and metadata
                    image_file = metadata_file.with_suffix('.png')
                    
                    if image_file.exists():
                        cleaned_size_mb += os.path.getsize(image_file) / (1024 * 1024)
                        image_file.unlink()
                    
                    cleaned_size_mb += os.path.getsize(metadata_file) / (1024 * 1024)
                    metadata_file.unlink()
                    cleaned_images += 1
                    
            except Exception as e:
                print(f"Warning: Failed to process {metadata_file}: {e}")
        
        # Update cache info
        cache_info_file = self.cache_dir / "cache_info.json"
        if cache_info_file.exists():
            try:
                with open(cache_info_file, 'r') as f:
                    cache_info = json.load(f)
                cache_info["last_cleanup"] = time.time()
                
                with open(cache_info_file, 'w') as f:
                    json.dump(cache_info, f, indent=2, cls=NumpyJSONEncoder)
            except:
                pass
        
        return {
            "cleaned_images": cleaned_images,
            "cleaned_size_mb": round(cleaned_size_mb, 2)
        }
    
    def clear_cache(self, confirm: bool = False) -> bool:
        """
        Clear entire cache (use with caution).
        
        Args:
            confirm: Must be True to actually clear cache
            
        Returns:
            True if cache was cleared
        """
        if not confirm:
            print("Cache not cleared. Use confirm=True to actually clear cache.")
            return False
        
        try:
            shutil.rmtree(self.cache_dir)
            self._initialize_cache_structure()
            print("✅ Cache cleared successfully")
            return True
        except Exception as e:
            print(f"Failed to clear cache: {e}")
            return False
    
    # Phase 2: Preprocessed data caching methods
    def cache_preprocessed_data(self, card_id: int, model_id: int, preprocessed_data: Dict[str, Any], 
                               config_hash: str = None) -> bool:
        """
        Cache preprocessed data for a specific card and model.
        
        Args:
            card_id: ID of the card
            model_id: ID of the model
            preprocessed_data: Preprocessed data dictionary
            config_hash: Optional configuration hash for cache invalidation
            
        Returns:
            True if data was cached successfully
        """
        try:
            # Generate cache key
            cache_key = f"card_{card_id}_model_{model_id}"
            if config_hash:
                cache_key += f"_config_{config_hash}"
            
            # Create cache file path
            cache_file = self.preprocessed_dir / f"{cache_key}.json"
            
            # Prepare cache data
            cache_data = {
                'card_id': card_id,
                'model_id': model_id,
                'config_hash': config_hash,
                'cached_at': time.time(),
                'preprocessed_data': preprocessed_data
            }
            
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, cls=NumpyJSONEncoder, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Failed to cache preprocessed data for card {card_id}, model {model_id}: {e}")
            return False
    
    def load_preprocessed_data(self, card_id: int, model_id: int, 
                              config_hash: str = None) -> Optional[Dict[str, Any]]:
        """
        Load cached preprocessed data for a specific card and model.
        
        Args:
            card_id: ID of the card
            model_id: ID of the model
            config_hash: Optional configuration hash for cache validation
            
        Returns:
            Cached preprocessed data or None if not found
        """
        try:
            # Generate cache key
            cache_key = f"card_{card_id}_model_{model_id}"
            if config_hash:
                cache_key += f"_config_{config_hash}"
            
            # Check cache file
            cache_file = self.preprocessed_dir / f"{cache_key}.json"
            if not cache_file.exists():
                return None
            
            # Load cached data
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Validate cache data
            if cache_data.get('card_id') != card_id or cache_data.get('model_id') != model_id:
                return None
            
            # Check config hash if provided
            if config_hash and cache_data.get('config_hash') != config_hash:
                return None
            
            return cache_data.get('preprocessed_data')
            
        except Exception as e:
            print(f"Failed to load preprocessed data for card {card_id}, model {model_id}: {e}")
            return None
    
    def is_preprocessed_data_cached(self, card_id: int, model_id: int, 
                                   config_hash: str = None) -> bool:
        """
        Check if preprocessed data is cached for a specific card and model.
        
        Args:
            card_id: ID of the card
            model_id: ID of the model
            config_hash: Optional configuration hash for cache validation
            
        Returns:
            True if preprocessed data is cached
        """
        return self.load_preprocessed_data(card_id, model_id, config_hash) is not None
    
    def clear_preprocessed_cache(self, model_id: int = None) -> bool:
        """
        Clear cached preprocessed data, optionally for a specific model.
        
        Args:
            model_id: Optional model ID to clear cache for specific model only
            
        Returns:
            True if cache was cleared successfully
        """
        try:
            if model_id is None:
                # Clear all preprocessed data
                for file_path in self.preprocessed_dir.glob("*.json"):
                    file_path.unlink()
                print("✅ All preprocessed cache cleared")
            else:
                # Clear for specific model
                pattern = f"*_model_{model_id}_*.json"
                cleared_count = 0
                for file_path in self.preprocessed_dir.glob(pattern):
                    file_path.unlink()
                    cleared_count += 1
                print(f"✅ Cleared {cleared_count} preprocessed cache entries for model {model_id}")
            
            return True
            
        except Exception as e:
            print(f"Failed to clear preprocessed cache: {e}")
            return False
    
    # Phase 3: Model caching methods
    def cache_model(self, model_id: int, model_data: bytes, model_info: Dict[str, Any]) -> bool:
        """
        Cache a model file and its metadata.
        
        Args:
            model_id: ID of the model
            model_data: Raw model file data
            model_info: Model information dictionary
            
        Returns:
            True if model was cached successfully
        """
        try:
            # Create models directory if it doesn't exist
            models_dir = self.cache_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Generate filename based on model info
            model_name = model_info.get('name', f'model_{model_id}')
            file_extension = model_info.get('file_extension', '.bin')
            model_filename = f"{model_name}{file_extension}"
            
            # Save model file
            model_path = models_dir / model_filename
            with open(model_path, 'wb') as f:
                f.write(model_data)
            
            # Save model metadata
            metadata = {
                'model_id': model_id,
                'model_info': model_info,
                'cached_at': time.time(),
                'file_path': str(model_path),
                'file_size': len(model_data)
            }
            
            metadata_path = models_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, cls=NumpyJSONEncoder, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Failed to cache model {model_id}: {e}")
            return False
    
    def load_cached_model(self, model_id: int) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """
        Load cached model file and metadata.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Tuple of (model_data, model_info) or None if not cached
        """
        try:
            models_dir = self.cache_dir / "models"
            if not models_dir.exists():
                return None
            
            # Find model metadata file
            for metadata_file in models_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    if metadata.get('model_id') == model_id:
                        # Load model file
                        model_path = Path(metadata['file_path'])
                        if model_path.exists():
                            with open(model_path, 'rb') as f:
                                model_data = f.read()
                            
                            return model_data, metadata['model_info']
                        
                except Exception as e:
                    print(f"Error reading model metadata {metadata_file}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            print(f"Failed to load cached model {model_id}: {e}")
            return None
    
    def is_model_cached(self, model_id: int) -> bool:
        """
        Check if a model is cached.
        
        Args:
            model_id: ID of the model
            
        Returns:
            True if model is cached
        """
        return self.load_cached_model(model_id) is not None
    
    def clear_model_cache(self, model_id: int = None) -> bool:
        """
        Clear cached models, optionally for a specific model.
        
        Args:
            model_id: Optional model ID to clear cache for specific model only
            
        Returns:
            True if cache was cleared successfully
        """
        try:
            models_dir = self.cache_dir / "models"
            if not models_dir.exists():
                return True
            
            if model_id is None:
                # Clear all models
                import shutil
                shutil.rmtree(models_dir)
                models_dir.mkdir(exist_ok=True)
                print("✅ All model cache cleared")
            else:
                # Clear specific model
                cleared_count = 0
                for metadata_file in models_dir.glob("*_metadata.json"):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        if metadata.get('model_id') == model_id:
                            # Remove model file
                            model_path = Path(metadata['file_path'])
                            if model_path.exists():
                                model_path.unlink()
                                cleared_count += 1
                            
                            # Remove metadata file
                            metadata_file.unlink()
                            
                    except Exception as e:
                        print(f"Error clearing model metadata {metadata_file}: {e}")
                
                print(f"✅ Cleared {cleared_count} model cache entries for model {model_id}")
            
            return True
            
        except Exception as e:
            print(f"Failed to clear model cache: {e}")
            return False