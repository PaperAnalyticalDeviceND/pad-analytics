"""
Dataset Manager for PAD Analytics

Provides a hybrid approach to dataset management:
- Fetches dynamic dataset catalog from padproject.info
- Preserves static model-dataset mappings from CSV
- Merges information from both sources
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import pandas as pd
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages dataset information from both dynamic catalog and static mappings."""
    
    def __init__(self, cache_duration_hours: int = 1, cache_dir: Optional[str] = None):
        """
        Initialize the DatasetManager.
        
        Args:
            cache_duration_hours: How long to cache the catalog (default: 1 hour)
            cache_dir: Directory for cache files (default: package data dir)
        """
        self.catalog_url = "https://padproject.info/pad_dataset_registry/api/catalog.json"
        self.cache_duration = timedelta(hours=cache_duration_hours)
        
        # Set up cache directory
        if cache_dir is None:
            # Use the package data directory
            module_dir = os.path.dirname(os.path.realpath(__file__))
            self.cache_dir = os.path.join(module_dir, "data", ".cache")
        else:
            self.cache_dir = cache_dir
            
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "dataset_catalog.json")
        
        # Load static model mappings
        self._model_mapping = None
        self._catalog_cache = None
        
    def _load_model_mapping(self) -> pd.DataFrame:
        """Load the static model-dataset mapping CSV."""
        if self._model_mapping is None:
            mapping_file = self._get_mapping_file_path()
            self._model_mapping = pd.read_csv(mapping_file)
            # Clean up column names (remove leading/trailing spaces)
            self._model_mapping.columns = self._model_mapping.columns.str.strip()
        return self._model_mapping
    
    def _get_mapping_file_path(self):
        """Get the correct path to the model dataset mapping file."""
        # Try to get the file from package resources (when installed)
        try:
            # For resource file access
            try:
                from importlib import resources
            except ImportError:
                # Python < 3.9 fallback
                import importlib_resources as resources
                
            package_path = resources.files("pad_analytics")
            mapping_file = package_path / "data" / "model_dataset_mapping.csv"
            if mapping_file.exists():
                return str(mapping_file)
        except (ImportError, AttributeError, FileNotFoundError):
            pass
        
        # Fallback: try path relative to this module (development mode)
        module_dir = os.path.dirname(os.path.realpath(__file__))
        package_data_path = os.path.join(module_dir, "data", "model_dataset_mapping.csv")
        if os.path.exists(package_data_path):
            return package_data_path
        
        # Fallback: try relative path from current working directory
        relative_path = "./data/model_dataset_mapping.csv"
        if os.path.exists(relative_path):
            return relative_path
        
        # Final fallback: try path relative to project root
        package_root = os.path.dirname(os.path.dirname(module_dir))
        fallback_path = os.path.join(package_root, "data", "model_dataset_mapping.csv")
        if os.path.exists(fallback_path):
            return fallback_path
        
        # If none found, return the package path (will cause error with helpful message)
        return package_data_path
    
    def _should_refresh_cache(self) -> bool:
        """Check if the cache needs to be refreshed."""
        if not os.path.exists(self.cache_file):
            return True
            
        # Check cache age
        cache_time = datetime.fromtimestamp(os.path.getmtime(self.cache_file))
        return datetime.now() - cache_time > self.cache_duration
    
    def _fetch_catalog(self) -> Dict:
        """Fetch the dataset catalog from the API."""
        try:
            logger.info(f"Fetching dataset catalog from {self.catalog_url}")
            response = requests.get(self.catalog_url, timeout=30, verify=False)
            response.raise_for_status()
            catalog_data = response.json()
            
            # Save to cache
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'data': catalog_data
                }, f, indent=2)
                
            return catalog_data
            
        except Exception as e:
            logger.error(f"Failed to fetch dataset catalog: {e}")
            # Try to load from cache even if expired
            if os.path.exists(self.cache_file):
                logger.warning("Using expired cache due to fetch failure")
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    return cache_data.get('data', {})
            raise
    
    def get_dataset_catalog(self, force_refresh: bool = False) -> Dict:
        """
        Get the dataset catalog with caching.
        
        Args:
            force_refresh: Force refresh the cache even if not expired
            
        Returns:
            Dictionary containing the full dataset catalog
        """
        if self._catalog_cache is None or self._should_refresh_cache() or force_refresh:
            self._catalog_cache = self._fetch_catalog()
        elif self._catalog_cache is None and os.path.exists(self.cache_file):
            # Load from cache
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
                self._catalog_cache = cache_data.get('data', {})
                
        return self._catalog_cache
    
    def get_dataset_list(self) -> List[str]:
        """
        Get list of all available datasets.
        
        Combines datasets from:
        1. Dynamic catalog
        2. Static CSV mapping
        
        Returns:
            List of unique dataset names
        """
        datasets = set()
        
        # Get datasets from catalog
        try:
            catalog = self.get_dataset_catalog()
            # Handle both list and dict formats
            if isinstance(catalog, list):
                for dataset in catalog:
                    if 'name' in dataset:
                        datasets.add(dataset['name'])
            elif isinstance(catalog, dict) and 'dataset' in catalog:
                for dataset in catalog['dataset']:
                    if 'name' in dataset:
                        datasets.add(dataset['name'])
        except Exception as e:
            logger.warning(f"Could not fetch dynamic catalog: {e}")
        
        # Add datasets from static mapping
        mapping_df = self._load_model_mapping()
        datasets.update(mapping_df['Dataset Name'].dropna().unique())
        
        return sorted(list(datasets))
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get comprehensive dataset information.
        
        Merges information from:
        1. Dynamic catalog (metadata, schema, etc.)
        2. Static mapping (model associations)
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with all available dataset information
        """
        info = {
            'name': dataset_name,
            'source': 'hybrid'  # Indicates data source
        }
        
        # Get info from catalog
        catalog_info = {}
        try:
            catalog = self.get_dataset_catalog()
            # Handle both list and dict formats
            datasets_to_search = []
            if isinstance(catalog, list):
                datasets_to_search = catalog
            elif isinstance(catalog, dict) and 'dataset' in catalog:
                datasets_to_search = catalog['dataset']
                
            for dataset in datasets_to_search:
                if dataset.get('name') == dataset_name:
                    catalog_info = dataset
                    break
                        
            if catalog_info:
                info.update({
                    'description': catalog_info.get('description'),
                    'record_count': catalog_info.get('recordCount'),
                    'file_count': catalog_info.get('fileCount'),
                    'version': catalog_info.get('version'),
                    'date_published': catalog_info.get('datePublished'),
                    'url': catalog_info.get('url'),
                    'api_url': catalog_info.get('API_URL'),
                    'distribution': catalog_info.get('distribution', []),
                    'schema': catalog_info.get('datasetSchema'),
                    'splits': catalog_info.get('dataSplits'),
                    'readme_url': catalog_info.get('readme_url'),
                    'source': 'catalog'
                })
        except Exception as e:
            logger.warning(f"Could not fetch catalog info for {dataset_name}: {e}")
        
        # Get model associations from static mapping
        mapping_df = self._load_model_mapping()
        dataset_rows = mapping_df[mapping_df['Dataset Name'] == dataset_name]
        
        if not dataset_rows.empty:
            models = []
            for _, row in dataset_rows.iterrows():
                if pd.notna(row['Model ID']):
                    models.append({
                        'model_id': int(row['Model ID']),
                        'model_name': row['Model Name'],
                        'endpoint_url': row['Endpoint URL']
                    })
            
            info['models'] = models
            info['training_dataset_url'] = dataset_rows.iloc[0]['Training Dataset']
            info['test_dataset_url'] = dataset_rows.iloc[0]['Test Dataset']
            
            # If no catalog info, mark as static only
            if 'description' not in info:
                info['source'] = 'static'
        
        return info
    
    def get_dataset_urls(self, dataset_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get training and test dataset URLs.
        
        First tries the static mapping, then falls back to catalog.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (training_url, test_url)
        """
        # First try static mapping
        mapping_df = self._load_model_mapping()
        dataset_rows = mapping_df[mapping_df['Dataset Name'] == dataset_name]
        
        if not dataset_rows.empty:
            train_url = dataset_rows.iloc[0]['Training Dataset']
            test_url = dataset_rows.iloc[0]['Test Dataset']
            if pd.notna(train_url) or pd.notna(test_url):
                return (
                    train_url if pd.notna(train_url) else None,
                    test_url if pd.notna(test_url) else None
                )
        
        # Try to get from catalog
        info = self.get_dataset_info(dataset_name)
        
        # Look for training/test files in distribution
        train_url = None
        test_url = None
        
        if 'distribution' in info:
            for dist in info['distribution']:
                name = dist.get('name', '').lower()
                if 'train' in name or 'dev' in name:
                    train_url = dist.get('contentUrl')
                elif 'test' in name:
                    test_url = dist.get('contentUrl')
        
        # If still not found, try data splits
        if 'splits' in info:
            splits = info['splits']
            # This would need more logic to construct URLs from splits
            # For now, we'll just return what we have
            
        return (train_url, test_url)
    
    def get_models_for_dataset(self, dataset_name: str) -> List[Dict]:
        """
        Get all models that use a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of model information dictionaries
        """
        info = self.get_dataset_info(dataset_name)
        return info.get('models', [])
    
    def get_dataset_from_model_id(self, model_id: int) -> Optional[str]:
        """
        Get dataset name used by a specific model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Dataset name or None if not found
        """
        mapping_df = self._load_model_mapping()
        model_rows = mapping_df[mapping_df['Model ID'] == model_id]
        
        if not model_rows.empty:
            return model_rows.iloc[0]['Dataset Name']
        return None