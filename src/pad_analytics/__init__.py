"""PAD Analytics Package

A complete workflow for machine learning models using data from the PAD API v2.
"""

__version__ = "0.2.2"

# Import all main functions from padanalytics module to package level
try:
    from .padanalytics import (
        get_data_api,
        get_card_issues,
        get_projects,
        get_project_cards,
        get_card,
        get_project_by_id,
        get_project_by_name,
        get_project,
        load_image_from_url,
        show_card,
        show_grouped_cards,
        show_cards_from_df,
        show_cards,
        get_models,
        get_model,
        predict,
        show_prediction,
        apply_predictions_to_dataframe,
        get_model_dataset_mapping,
        get_dataset_list,
        get_datasets,
        get_dataset_name_from_model_id,
        get_dataset_cards,
        get_model_data,
        get_dataset_info,
        get_dataset_manager,
        calculate_rmse,
        calculate_rmse_by_api,
        download_file,
        standardize_names,
    )
    _PADANALYTICS_IMPORTED = True
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import main functions from padanalytics: {e}")
    _PADANALYTICS_IMPORTED = False

# Import other modules for advanced users (with error handling)
try:
    from . import pad_analysis
    from . import pad_helper
    from . import fileManagement
    from . import intensityFind
    from . import pixelProcessing
    from . import regionRoutine
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import some submodules: {e}")

# Build __all__ list based on successful imports
__all__ = ["__version__"]

if _PADANALYTICS_IMPORTED:
    __all__.extend([
        # Main functions from padanalytics
        "get_data_api",
        "get_card_issues", 
        "get_projects",
        "get_project_cards",
        "get_card",
        "get_project_by_id",
        "get_project_by_name", 
        "get_project",
        "load_image_from_url",
        "show_card",
        "show_grouped_cards",
        "show_cards_from_df",
        "show_cards",
        "get_models",
        "get_model", 
        "predict",
        "show_prediction",
        "apply_predictions_to_dataframe",
        "get_model_dataset_mapping",
        "get_dataset_list",
        "get_datasets", 
        "get_dataset_name_from_model_id",
        "get_dataset_cards",
        "get_model_data",
        "get_dataset_info",
        "get_dataset_manager",
        "calculate_rmse",
        "calculate_rmse_by_api",
        "download_file",
        "standardize_names",
    ])

# Data Caching System (NEW in v0.2.3)
try:
    from .cache_manager import CacheManager
    from .cached_dataset import CachedDataset
    from .cached_predictions import (
        predict_with_cache,
        apply_predictions_to_dataframe_with_cache
    )
    
    __all__.extend([
        "CacheManager",
        "CachedDataset", 
        "predict_with_cache",
        "apply_predictions_to_dataframe_with_cache"
    ])
    _CACHING_IMPORTED = True
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import caching system: {e}")
    _CACHING_IMPORTED = False

# Preprocessing Pipeline System (NEW in v0.2.3 - Phase 2)
try:
    from .preprocessing_pipeline import PreprocessingPipeline, create_neural_network_pipeline, create_pls_pipeline
    from .preprocessors import BasePreprocessor, NeuralNetworkPreprocessor, PLSPreprocessor
    
    __all__.extend([
        "PreprocessingPipeline",
        "create_neural_network_pipeline",
        "create_pls_pipeline",
        "BasePreprocessor",
        "NeuralNetworkPreprocessor",
        "PLSPreprocessor"
    ])
    _PREPROCESSING_IMPORTED = True
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import preprocessing pipeline: {e}")
    _PREPROCESSING_IMPORTED = False

# Model Adapter System (NEW in v0.2.3 - Phase 3)
try:
    from .model_adapter import ModelAdapter, create_neural_network_adapter, create_pls_adapter, get_available_models
    from .adapters import BaseAdapter, NeuralNetworkAdapter, PLSAdapter
    
    __all__.extend([
        "ModelAdapter",
        "create_neural_network_adapter",
        "create_pls_adapter",
        "get_available_models",
        "BaseAdapter",
        "NeuralNetworkAdapter",
        "PLSAdapter"
    ])
    _MODEL_ADAPTER_IMPORTED = True
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import model adapter: {e}")
    _MODEL_ADAPTER_IMPORTED = False

# Add available submodules
for module_name in ["pad_analysis", "pad_helper", "fileManagement", "intensityFind", "pixelProcessing", "regionRoutine"]:
    if module_name in globals():
        __all__.append(module_name)
