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
    
    def get_datasets(self) -> pd.DataFrame:
        """Get clean overview of all available datasets.
        
        Combines information from dynamic catalog and static mappings to provide
        a comprehensive dataset overview with documentation links.
        
        Returns:
            pd.DataFrame: Dataset overview with the following columns:
                - Dataset Name (str): Name of the dataset
                - Total Records (int): Combined training + testing records
                - Description (str): Dataset description from catalog
                - Documentation (str): Link to dataset readme
                - Source (str): Data source - "catalog", "static", or "hybrid"
                
        Example:
            >>> dm = DatasetManager()
            >>> datasets_df = dm.get_datasets()
            >>> print(datasets_df[['Dataset Name', 'Total Records']].head())
            Dataset Name                         Total Records
            FHI2020_Stratified_Sampling         8001
            FHI2021                             1500
        """
        datasets = []
        
        # Get all unique dataset names
        dataset_names = self.get_dataset_list()
        
        for name in dataset_names:
            info = self.get_dataset_info(name)
            
            # Calculate total records
            total_records = info.get('record_count', 'N/A')
            
            # Get description
            description = info.get('description', '')
            if not description and 'models' in info and info['models']:
                # For static-only datasets, create description from models
                model_count = len(info['models'])
                description = f"Dataset used by {model_count} model(s)"
            
            # Generate documentation link
            doc_link = f"https://padproject.info/pad_dataset_registry/datasets/{name}/readme/"
            
            # Determine source
            source = info.get('source', 'unknown')
            
            datasets.append({
                'Dataset Name': name,
                'Total Records': total_records,
                'Description': description,
                'Documentation': doc_link,
                'Source': source
            })
        
        return pd.DataFrame(datasets)
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get comprehensive dataset information.
        
        Merges information from dynamic catalog (metadata, schema) and static 
        mapping (model associations) to provide complete dataset details.
        
        Args:
            dataset_name (str): Name of the dataset to retrieve information for.
            
        Returns:
            dict: Dataset information including:
                - name (str): Dataset name
                - source (str): Data source - "catalog", "static", or "hybrid"
                - description (str): Dataset description (if available)
                - record_count (int): Total number of records
                - models (list): List of models using this dataset
                - training_dataset_url (str): URL to training data
                - test_dataset_url (str): URL to test data
                - Additional metadata from catalog (if available)
                
        Example:
            >>> info = dm.get_dataset_info("FHI2020_Stratified_Sampling")
            >>> print(f"Dataset has {info['record_count']} records")
            Dataset has 8001 records
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
    
    def get_dataset_name_from_model_id(self, model_id: int) -> Optional[str]:
        """Get dataset name used by a specific model.
        
        Looks up which dataset was used to train and test the specified model.
        
        Args:
            model_id (int): The model ID to look up.
            
        Returns:
            str or None: Dataset name (e.g., "FHI2020_Stratified_Sampling") or
                None if model ID not found.
                
        Example:
            >>> dataset_name = dm.get_dataset_name_from_model_id(16)
            >>> print(dataset_name)
            FHI2020_Stratified_Sampling
        """
        mapping_df = self._load_model_mapping()
        model_rows = mapping_df[mapping_df['Model ID'] == model_id]
        
        if not model_rows.empty:
            return model_rows.iloc[0]['Dataset Name']
        return None
    
    def get_dataset_from_model_id(self, model_id: int) -> Optional[str]:
        """
        DEPRECATED: Use get_dataset_name_from_model_id() instead.
        
        Get dataset name used by a specific model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Dataset name or None if not found
        """
        import warnings
        warnings.warn(
            "get_dataset_from_model_id() is deprecated and will be removed in a future version. "
            "Use get_dataset_name_from_model_id() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_dataset_name_from_model_id(model_id)
    
    def get_dataset_cards(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Get all cards (samples) from a specific dataset.
        
        Loads and combines training and test data for the specified dataset.
        Returns a clean view without implementation details like 'is_train' column.
        
        Args:
            dataset_name (str): Name of the dataset to load cards from.
            
        Returns:
            pd.DataFrame or None: Combined dataset with all cards/samples, or None if
                dataset not found. Columns typically include:
                - id: Card ID
                - sample_id: Sample identifier  
                - sample_name: Drug/sample name
                - quantity: Concentration/amount
                - url: Image URL
                - Additional metadata columns
                
        Note:
            The 'is_train' column is NOT included in the output for a clean view.
            Use get_model_data() if you need train/test distinction.
                
        Example:
            >>> cards = dm.get_dataset_cards("FHI2020_Stratified_Sampling")
            >>> print(f"Loaded {len(cards)} cards")
            Loaded 8001 cards
        """
        # Get dataset URLs
        train_url, test_url = self.get_dataset_urls(dataset_name)
        
        if train_url is None and test_url is None:
            logger.warning(f"No dataset URLs found for: {dataset_name}")
            return None
        
        train_df = None
        test_df = None
        
        # Load training data
        if train_url:
            try:
                train_df = pd.read_csv(train_url)
            except Exception as e:
                logger.error(f"Error loading training data for {dataset_name}: {e}")
        
        # Load test data
        if test_url:
            try:
                test_df = pd.read_csv(test_url)
            except Exception as e:
                logger.error(f"Error loading test data for {dataset_name}: {e}")
        
        # Combine datasets without is_train column
        if train_df is not None and test_df is not None:
            data_df = pd.concat([train_df, test_df], ignore_index=True)
        elif train_df is not None:
            data_df = train_df
        elif test_df is not None:
            data_df = test_df
        else:
            logger.error(f"Failed to load any data for dataset: {dataset_name}")
            return None
        
        return data_df
    
    def get_model_data(self, model_id: int, data_type: str = "all") -> Optional[pd.DataFrame]:
        """Get training, testing, or all data for a specific model.
        
        Retrieves dataset used by a specific model, with options to get
        training data only, test data only, or combined dataset.
        
        Args:
            model_id (int): The model ID to retrieve data for.
            data_type (str): Type of data to return. Options:
                - "train": Training data only (no is_train column)
                - "test": Test data only (no is_train column)
                - "all": Combined data (includes is_train column)
                
        Returns:
            pd.DataFrame or None: Requested dataset or None if model not found.
                When data_type="all", includes 'is_train' column (1=train, 0=test).
                
        Raises:
            ValueError: If data_type is not one of "train", "test", or "all".
            
        Example:
            >>> # Get all data with train/test indicator
            >>> all_data = dm.get_model_data(16, "all")
            >>> print(f"Training: {len(all_data[all_data['is_train']==1])}")
            Training: 5923
            
            >>> # Get only training data (no is_train column)
            >>> train_data = dm.get_model_data(16, "train")
            >>> print(f"Training samples: {len(train_data)}")
            Training samples: 5923
        """
        if data_type not in ["train", "test", "all"]:
            raise ValueError("data_type must be 'train', 'test', or 'all'")
        
        # Get dataset name for this model
        dataset_name = self.get_dataset_name_from_model_id(model_id)
        if dataset_name is None:
            logger.warning(f"No dataset found for model ID: {model_id}")
            return None
        
        # Get dataset URLs
        train_url, test_url = self.get_dataset_urls(dataset_name)
        
        train_df = None
        test_df = None
        
        # Load training data if needed
        if data_type in ["train", "all"] and train_url:
            try:
                train_df = pd.read_csv(train_url)
                if data_type == "all":
                    train_df['is_train'] = 1
            except Exception as e:
                logger.error(f"Error loading training data: {e}")
        
        # Load test data if needed
        if data_type in ["test", "all"] and test_url:
            try:
                test_df = pd.read_csv(test_url)
                if data_type == "all":
                    test_df['is_train'] = 0
            except Exception as e:
                logger.error(f"Error loading test data: {e}")
        
        # Return requested data
        if data_type == "train":
            return train_df
        elif data_type == "test":
            return test_df
        else:  # data_type == "all"
            if train_df is not None and test_df is not None:
                return pd.concat([train_df, test_df], ignore_index=True)
            elif train_df is not None:
                return train_df
            elif test_df is not None:
                return test_df
            else:
                logger.error(f"Failed to load any data for model {model_id}")
                return None