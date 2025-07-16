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
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
import requests
from PIL import Image
import pandas as pd


class CacheManager:
    """
    Professional caching system for PAD analytics data.
    
    Implements hierarchical caching with:
    - Raw image storage
    - Metadata persistence  
    - Cache integrity verification
    - Automatic cleanup mechanisms
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
        
        # Create cache structure
        self._initialize_cache_structure()
        
    def _initialize_cache_structure(self):
        """Create cache directory structure."""
        for directory in [self.raw_images_dir, self.metadata_dir, self.datasets_dir]:
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
                json.dump(cache_info, f, indent=2)
    
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
                json.dump(metadata, f, indent=2)
            
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
        
        # Calculate time ranges
        if image_times:
            stats["oldest_entry"] = min(image_times)
            stats["newest_entry"] = max(image_times)
        
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        return stats
    
    def _get_cache_size_bytes(self) -> int:
        """Get total cache size in bytes."""
        total_size = 0
        for directory in [self.raw_images_dir, self.metadata_dir, self.datasets_dir]:
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
                    json.dump(cache_info, f, indent=2)
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