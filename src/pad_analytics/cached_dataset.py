"""
PAD Analytics Cached Dataset - Phase 1 Implementation

This module provides a professional interface for working with cached PAD datasets,
eliminating redundant downloads and enabling offline research workflows.
"""

import time
from typing import Optional, List, Dict, Any
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from .cache_manager import CacheManager

# Conditional imports to avoid ipywidgets dependency
try:
    from .padanalytics import get_dataset_cards, get_card
    _PADANALYTICS_AVAILABLE = True
except ImportError as e:
    _PADANALYTICS_AVAILABLE = False
    import warnings
    warnings.warn(f"padanalytics not available for caching: {e}")

try:
    from .dataset_manager import DatasetManager
    _DATASET_MANAGER_AVAILABLE = True
except ImportError:
    _DATASET_MANAGER_AVAILABLE = False


class CachedDataset:
    """
    Professional cached dataset for PAD analytics research.
    
    Features:
    - Automatic image caching and reuse
    - Offline capability after initial download
    - Progress tracking for large datasets
    - Metadata persistence and integrity
    - Efficient batch operations
    """
    
    def __init__(self, dataset_name: str, cache_dir: str = "~/.pad_cache"):
        """
        Initialize a cached dataset.
        
        Args:
            dataset_name: Name of the PAD dataset (e.g., "FHI2020_Stratified_Sampling")
            cache_dir: Base directory for cache storage
        """
        self.dataset_name = dataset_name
        self.cache_manager = CacheManager(cache_dir)
        self._dataset_df: Optional[pd.DataFrame] = None
        
    @property
    def dataset_df(self) -> Optional[pd.DataFrame]:
        """Get the dataset DataFrame (loads from cache or API if needed)."""
        if self._dataset_df is None:
            self.load_dataset_metadata()
        return self._dataset_df
    
    def load_dataset_metadata(self) -> pd.DataFrame:
        """
        Load dataset metadata from cache or API.
        
        Returns:
            Dataset DataFrame
        """
        print(f"Loading dataset: {self.dataset_name}")
        
        # Try to load from cache first
        cached_df = self.cache_manager.get_cached_dataset(self.dataset_name)
        if cached_df is not None:
            print(f"✅ Loaded dataset from cache ({len(cached_df)} records)")
            self._dataset_df = cached_df
            return cached_df
        
        # Load from API and cache
        if not _PADANALYTICS_AVAILABLE:
            raise Exception("padanalytics module not available - cannot fetch from API")
            
        print("📡 Fetching dataset from PAD API...")
        try:
            dataset_df = get_dataset_cards(self.dataset_name)
            if dataset_df is not None and not dataset_df.empty:
                # Cache the dataset metadata
                self.cache_manager.cache_dataset_metadata(self.dataset_name, dataset_df)
                self._dataset_df = dataset_df
                return dataset_df
            else:
                raise Exception(f"Dataset {self.dataset_name} not found or empty")
        except Exception as e:
            raise Exception(f"Failed to load dataset {self.dataset_name}: {e}")
    
    def download_and_cache_images(self, 
                                 max_workers: int = 8, 
                                 max_images: Optional[int] = None,
                                 force_refresh: bool = False) -> Dict[str, Any]:
        """
        Download and cache all images in the dataset.
        
        Args:
            max_workers: Number of parallel download threads
            max_images: Maximum number of images to download (for testing)
            force_refresh: Re-download images even if cached
            
        Returns:
            Dictionary with download statistics
        """
        if self.dataset_df is None:
            self.load_dataset_metadata()
        
        dataset_subset = self.dataset_df.head(max_images) if max_images else self.dataset_df
        total_images = len(dataset_subset)
        
        print(f"🚀 Starting image caching for {total_images} images from {self.dataset_name}")
        print(f"Using {max_workers} parallel workers")
        
        stats = {
            "total_images": total_images,
            "cached_new": 0,
            "already_cached": 0,
            "failed": 0,
            "start_time": time.time()
        }
        
        def cache_single_image(row_data):
            """Cache a single image with error handling."""
            card_id, image_info = row_data
            try:
                if not _PADANALYTICS_AVAILABLE:
                    return "failed", "padanalytics module not available"
                    
                # Get full card information for metadata
                card_df = get_card(card_id=card_id)
                if card_df is None or card_df.empty:
                    return "failed", f"Could not get card data for {card_id}"
                
                # Construct image URL
                image_url = "https://pad.crc.nd.edu/" + card_df.processed_file_location.values[0]
                
                # Check if already cached
                if not force_refresh and self.cache_manager.is_image_cached(image_url):
                    return "already_cached", card_id
                
                # Cache the image with metadata
                card_metadata = {
                    "card_id": card_id,
                    "sample_name": card_df.sample_name.values[0],
                    "sample_id": card_df.sample_id.values[0],
                    "quantity": card_df.quantity.values[0] if 'quantity' in card_df.columns else None,
                    "dataset": self.dataset_name
                }
                
                cached_path = self.cache_manager.cache_image(image_url, card_metadata)
                return "cached_new", card_id
                
            except Exception as e:
                return "failed", f"Card {card_id}: {e}"
        
        # Prepare data for parallel processing
        image_data = [(int(row["id"]), row) for _, row in dataset_subset.iterrows()]
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_data = {executor.submit(cache_single_image, data): data for data in image_data}
            
            # Collect results with progress tracking
            completed = 0
            for future in as_completed(future_to_data):
                completed += 1
                
                result_type, result_data = future.result()
                stats[result_type] += 1
                
                if result_type == "failed":
                    print(f"❌ Failed: {result_data}")
                
                # Progress indicator
                if completed % 50 == 0 or completed == total_images:
                    elapsed = time.time() - stats["start_time"]
                    print(f"Progress: {completed}/{total_images} ({completed/total_images*100:.1f}%) "
                          f"- Elapsed: {elapsed:.1f}s")
        
        # Final statistics
        stats["end_time"] = time.time()
        stats["total_time"] = stats["end_time"] - stats["start_time"]
        
        print(f"\n✅ Image caching completed!")
        print(f"📊 Results:")
        print(f"   • Total images: {stats['total_images']}")
        print(f"   • Newly cached: {stats['cached_new']}")
        print(f"   • Already cached: {stats['already_cached']}")
        print(f"   • Failed: {stats['failed']}")
        print(f"   • Total time: {stats['total_time']:.1f}s")
        
        if stats['cached_new'] > 0:
            print(f"   • Avg time per new image: {stats['total_time']/stats['cached_new']:.2f}s")
        
        return stats
    
    def get_cached_image_path(self, card_id: int) -> Optional[str]:
        """
        Get path to cached image for a specific card.
        
        Args:
            card_id: The PAD card ID
            
        Returns:
            Path to cached image or None if not cached
        """
        if self.dataset_df is None:
            self.load_dataset_metadata()
        
        # Find the card in the dataset
        card_row = self.dataset_df[self.dataset_df['id'] == card_id]
        if card_row.empty:
            return None
        
        # Get the image URL (would need card details)
        if not _PADANALYTICS_AVAILABLE:
            return None
            
        try:
            card_df = get_card(card_id=card_id)
            if card_df is None or card_df.empty:
                return None
            
            image_url = "https://pad.crc.nd.edu/" + card_df.processed_file_location.values[0]
            return self.cache_manager.get_cached_image_path(image_url)
            
        except Exception:
            return None
    
    def get_cache_coverage(self) -> Dict[str, Any]:
        """
        Calculate cache coverage statistics for this dataset.
        
        Returns:
            Dictionary with coverage statistics
        """
        if self.dataset_df is None:
            self.load_dataset_metadata()
        
        total_cards = len(self.dataset_df)
        cached_count = 0
        
        # Sample a subset for performance (checking all could be slow)
        sample_size = min(100, total_cards)
        sample_cards = self.dataset_df.sample(n=sample_size, random_state=42)
        
        for _, row in sample_cards.iterrows():
            if self.get_cached_image_path(int(row['id'])) is not None:
                cached_count += 1
        
        # Estimate coverage based on sample
        estimated_coverage = (cached_count / sample_size) * 100
        
        return {
            "dataset_name": self.dataset_name,
            "total_cards": total_cards,
            "sample_size": sample_size,
            "sample_cached": cached_count,
            "estimated_coverage_percent": round(estimated_coverage, 1)
        }
    
    def is_offline_ready(self, sample_size: int = 50) -> bool:
        """
        Check if dataset is ready for offline use.
        
        Args:
            sample_size: Number of cards to sample for checking
            
        Returns:
            True if sufficient images are cached for offline use
        """
        coverage = self.get_cache_coverage()
        return coverage["estimated_coverage_percent"] > 80.0  # 80% threshold
    
    def __len__(self) -> int:
        """Return number of cards in the dataset."""
        if self.dataset_df is None:
            self.load_dataset_metadata()
        return len(self.dataset_df)
    
    def __repr__(self) -> str:
        """String representation of the cached dataset."""
        if self.dataset_df is None:
            return f"CachedDataset('{self.dataset_name}', not loaded)"
        
        coverage = self.get_cache_coverage()
        return (f"CachedDataset('{self.dataset_name}', "
                f"{len(self.dataset_df)} cards, "
                f"~{coverage['estimated_coverage_percent']}% cached)")


def create_cached_dataset(dataset_name: str, 
                         cache_dir: str = "~/.pad_cache",
                         download_images: bool = False,
                         max_images: Optional[int] = None,
                         max_workers: int = 8) -> CachedDataset:
    """
    Convenience function to create and optionally populate a cached dataset.
    
    Args:
        dataset_name: Name of the PAD dataset
        cache_dir: Cache directory path
        download_images: Whether to download images immediately
        max_images: Maximum number of images to download
        max_workers: Number of parallel download workers
        
    Returns:
        CachedDataset instance
    """
    dataset = CachedDataset(dataset_name, cache_dir)
    
    if download_images:
        dataset.download_and_cache_images(
            max_workers=max_workers,
            max_images=max_images
        )
    
    return dataset