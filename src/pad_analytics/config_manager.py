"""
Configuration management system for PAD Analytics.

This module provides a centralized configuration management system that handles
settings for preprocessing, model adapters, caching, performance monitoring,
and other components of the PAD Analytics pipeline.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, asdict, field
import copy


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipelines."""
    # Neural Network preprocessing
    nn_crop_box: tuple = (71, 359, 71 + 636, 359 + 490)
    nn_target_size: tuple = (454, 454)
    nn_normalize: bool = False
    nn_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    nn_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # PLS preprocessing
    pls_num_lanes: int = 12
    pls_num_regions: int = 10
    pls_num_colors: int = 3
    
    # General preprocessing
    batch_size: int = 32
    num_workers: int = 4
    parallel_processing: bool = True


@dataclass
class ModelConfig:
    """Configuration for model adapters."""
    # Model loading
    auto_load: bool = True
    download_timeout: int = 60
    retry_attempts: int = 3
    
    # Prediction settings
    enable_batch_optimization: bool = True
    batch_chunk_size: int = 32
    parallel_batch_threshold: int = 10
    max_parallel_workers: Optional[int] = None
    
    # Model specific settings
    neural_network_models: List[int] = field(default_factory=lambda: [16, 17, 19, 20])
    pls_models: List[int] = field(default_factory=lambda: [18])


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    enabled: bool = True
    cache_dir: str = "~/.pad_cache"
    
    # Cache policies
    max_cache_size_mb: int = 1000
    auto_cleanup: bool = True
    cleanup_threshold_mb: int = 800
    
    # Cache types
    enable_image_cache: bool = True
    enable_metadata_cache: bool = True
    enable_dataset_cache: bool = True
    enable_preprocessing_cache: bool = True
    enable_model_cache: bool = True
    enable_prediction_cache: bool = False  # Usually disabled for freshness
    
    # Cache expiration (in hours)
    image_cache_ttl: int = 24
    metadata_cache_ttl: int = 6
    dataset_cache_ttl: int = 12
    preprocessing_cache_ttl: int = 4
    model_cache_ttl: int = 168  # 1 week
    prediction_cache_ttl: int = 1


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""
    enabled: bool = True
    detailed_monitoring: bool = True
    
    # Metrics collection
    collect_timing: bool = True
    collect_memory: bool = True
    collect_cpu: bool = True
    collect_throughput: bool = True
    
    # Storage
    max_metrics_history: int = 1000
    auto_export: bool = False
    export_interval_hours: int = 24
    export_path: Optional[str] = None


@dataclass
class APIConfig:
    """Configuration for PAD API interactions."""
    base_url: str = "https://pad.crc.nd.edu/api/v2"
    timeout: int = 30
    verify_ssl: bool = False  # Due to SSL certificate issues
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Rate limiting
    enable_rate_limiting: bool = True
    requests_per_minute: int = 60
    
    # Caching API responses
    cache_api_responses: bool = True
    api_cache_ttl: int = 300  # 5 minutes


@dataclass
class ValidationConfig:
    """Configuration for data validation and quality checks."""
    enabled: bool = True
    strict_mode: bool = False
    
    # Input validation
    validate_card_data: bool = True
    validate_image_urls: bool = True
    validate_preprocessing_output: bool = True
    
    # Quality checks
    check_image_quality: bool = True
    min_image_size: tuple = (100, 100)
    max_image_size: tuple = (4000, 4000)
    
    # Error handling
    skip_invalid_data: bool = True
    log_validation_errors: bool = True


@dataclass
class PADAnalyticsConfig:
    """Main configuration container for PAD Analytics."""
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    api: APIConfig = field(default_factory=APIConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    # Global settings
    debug_mode: bool = False
    log_level: str = "INFO"
    random_seed: Optional[int] = None


class ConfigManager:
    """
    Configuration manager for PAD Analytics.
    
    Handles loading, saving, and managing configuration settings from
    various sources including files, environment variables, and defaults.
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config_file = Path(config_file) if config_file else None
        self._config = PADAnalyticsConfig()
        
        # Load configuration from various sources
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load configuration from file and environment variables."""
        # 1. Start with defaults (already loaded)
        
        # 2. Load from config file if provided
        if self.config_file and self.config_file.exists():
            self._load_from_file(self.config_file)
        
        # 3. Load from standard config locations
        self._load_from_standard_locations()
        
        # 4. Override with environment variables
        self._load_from_environment()
        
        # 5. Validate and post-process configuration
        self._validate_configuration()
    
    def _load_from_file(self, file_path: Path) -> None:
        """Load configuration from a file."""
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yml', '.yaml']:
                    data = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path.suffix}")
            
            self._merge_config_data(data)
            
        except Exception as e:
            print(f"Warning: Failed to load config from {file_path}: {e}")
    
    def _load_from_standard_locations(self) -> None:
        """Load configuration from standard locations."""
        # Check for config files in standard locations
        config_locations = [
            Path.cwd() / "pad_analytics.yml",
            Path.cwd() / "pad_analytics.yaml",
            Path.cwd() / "pad_analytics.json",
            Path.home() / ".pad_analytics" / "config.yml",
            Path.home() / ".pad_analytics" / "config.yaml",
            Path.home() / ".pad_analytics" / "config.json",
        ]
        
        for config_path in config_locations:
            if config_path.exists():
                self._load_from_file(config_path)
                break
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            'PAD_DEBUG_MODE': ('debug_mode', bool),
            'PAD_LOG_LEVEL': ('log_level', str),
            'PAD_CACHE_ENABLED': ('cache.enabled', bool),
            'PAD_CACHE_DIR': ('cache.cache_dir', str),
            'PAD_CACHE_SIZE_MB': ('cache.max_cache_size_mb', int),
            'PAD_API_BASE_URL': ('api.base_url', str),
            'PAD_API_TIMEOUT': ('api.timeout', int),
            'PAD_PERFORMANCE_ENABLED': ('performance.enabled', bool),
            'PAD_BATCH_SIZE': ('preprocessing.batch_size', int),
            'PAD_NUM_WORKERS': ('preprocessing.num_workers', int),
            'PAD_PARALLEL_PROCESSING': ('preprocessing.parallel_processing', bool),
        }
        
        for env_var, (config_path, value_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Convert value to appropriate type
                    if value_type == bool:
                        converted_value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif value_type == int:
                        converted_value = int(env_value)
                    elif value_type == float:
                        converted_value = float(env_value)
                    else:
                        converted_value = env_value
                    
                    # Set the configuration value
                    self._set_nested_config(config_path, converted_value)
                    
                except Exception as e:
                    print(f"Warning: Failed to parse environment variable {env_var}={env_value}: {e}")
    
    def _merge_config_data(self, data: Dict[str, Any]) -> None:
        """Merge configuration data into current config."""
        def merge_dicts(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    merge_dicts(target[key], value)
                else:
                    target[key] = value
        
        # Convert config to dict, merge, and convert back
        config_dict = asdict(self._config)
        merge_dicts(config_dict, data)
        
        # Reconstruct config object
        self._config = self._dict_to_config(config_dict)
    
    def _dict_to_config(self, data: Dict[str, Any]) -> PADAnalyticsConfig:
        """Convert dictionary to configuration object."""
        return PADAnalyticsConfig(
            preprocessing=PreprocessingConfig(**data.get('preprocessing', {})),
            model=ModelConfig(**data.get('model', {})),
            cache=CacheConfig(**data.get('cache', {})),
            performance=PerformanceConfig(**data.get('performance', {})),
            api=APIConfig(**data.get('api', {})),
            validation=ValidationConfig(**data.get('validation', {})),
            debug_mode=data.get('debug_mode', False),
            log_level=data.get('log_level', 'INFO'),
            random_seed=data.get('random_seed', None)
        )
    
    def _set_nested_config(self, path: str, value: Any) -> None:
        """Set a nested configuration value using dot notation."""
        parts = path.split('.')
        current = self._config
        
        for part in parts[:-1]:
            current = getattr(current, part)
        
        setattr(current, parts[-1], value)
    
    def _validate_configuration(self) -> None:
        """Validate and post-process configuration."""
        # Expand user paths
        self._config.cache.cache_dir = os.path.expanduser(self._config.cache.cache_dir)
        
        # Set reasonable defaults for None values
        if self._config.model.max_parallel_workers is None:
            import multiprocessing as mp
            self._config.model.max_parallel_workers = min(mp.cpu_count(), 8)
        
        # Validate numeric ranges
        self._config.preprocessing.batch_size = max(1, self._config.preprocessing.batch_size)
        self._config.preprocessing.num_workers = max(1, self._config.preprocessing.num_workers)
        self._config.cache.max_cache_size_mb = max(10, self._config.cache.max_cache_size_mb)
        self._config.api.timeout = max(5, self._config.api.timeout)
    
    def get_config(self) -> PADAnalyticsConfig:
        """Get the current configuration."""
        return copy.deepcopy(self._config)
    
    def get_preprocessing_config(self) -> PreprocessingConfig:
        """Get preprocessing configuration."""
        return copy.deepcopy(self._config.preprocessing)
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return copy.deepcopy(self._config.model)
    
    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration."""
        return copy.deepcopy(self._config.cache)
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration."""
        return copy.deepcopy(self._config.performance)
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration."""
        return copy.deepcopy(self._config.api)
    
    def get_validation_config(self) -> ValidationConfig:
        """Get validation configuration."""
        return copy.deepcopy(self._config.validation)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        self._merge_config_data(updates)
        self._validate_configuration()
    
    def save_config(self, file_path: Optional[Union[str, Path]] = None,
                   format: str = 'yaml') -> None:
        """
        Save current configuration to file.
        
        Args:
            file_path: Path to save config file (uses loaded file if None)
            format: Output format ('yaml' or 'json')
        """
        if file_path is None:
            file_path = self.config_file
        
        if file_path is None:
            raise ValueError("No config file path specified")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self._config)
        
        with open(file_path, 'w') as f:
            if format.lower() in ['yml', 'yaml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self._config = PADAnalyticsConfig()
        self._validate_configuration()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        return {
            'preprocessing': {
                'batch_size': self._config.preprocessing.batch_size,
                'num_workers': self._config.preprocessing.num_workers,
                'parallel_processing': self._config.preprocessing.parallel_processing,
            },
            'model': {
                'auto_load': self._config.model.auto_load,
                'enable_batch_optimization': self._config.model.enable_batch_optimization,
                'max_parallel_workers': self._config.model.max_parallel_workers,
            },
            'cache': {
                'enabled': self._config.cache.enabled,
                'cache_dir': self._config.cache.cache_dir,
                'max_cache_size_mb': self._config.cache.max_cache_size_mb,
            },
            'performance': {
                'enabled': self._config.performance.enabled,
                'detailed_monitoring': self._config.performance.detailed_monitoring,
            },
            'api': {
                'base_url': self._config.api.base_url,
                'timeout': self._config.api.timeout,
                'verify_ssl': self._config.api.verify_ssl,
            },
            'validation': {
                'enabled': self._config.validation.enabled,
                'strict_mode': self._config.validation.strict_mode,
            },
            'global': {
                'debug_mode': self._config.debug_mode,
                'log_level': self._config.log_level,
            }
        }


# Global configuration manager instance
_global_config_manager = None


def get_global_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager


def get_config() -> PADAnalyticsConfig:
    """Get the global configuration."""
    return get_global_config_manager().get_config()


def update_global_config(updates: Dict[str, Any]) -> None:
    """Update the global configuration."""
    get_global_config_manager().update_config(updates)


def reset_global_config() -> None:
    """Reset the global configuration to defaults."""
    get_global_config_manager().reset_to_defaults()