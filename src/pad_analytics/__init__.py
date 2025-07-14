"""PAD Analytics Package

A complete workflow for machine learning models using data from the PAD API v2.
"""

__version__ = "0.2.1"

# Import all main functions from padanalytics module to package level
try:
    from .padanalytics import (
        get_data_api,
        get_card_issues,
        get_projects,
        get_project_cards,
        get_card_by_id,
        get_card_by_sample_id,
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
        get_dataset_from_model_id,
        get_dataset_name_from_model_id,
        get_dataset,
        get_dataset_cards,
        get_model_data,
        get_dataset_info,
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
        "get_card_by_id",
        "get_card_by_sample_id",
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
        "get_dataset_from_model_id",
        "get_dataset_name_from_model_id",
        "get_dataset",
        "get_dataset_cards",
        "get_model_data",
        "get_dataset_info",
        "calculate_rmse",
        "calculate_rmse_by_api",
        "download_file",
        "standardize_names",
    ])

# Add available submodules
for module_name in ["pad_analysis", "pad_helper", "fileManagement", "intensityFind", "pixelProcessing", "regionRoutine"]:
    if module_name in globals():
        __all__.append(module_name)
