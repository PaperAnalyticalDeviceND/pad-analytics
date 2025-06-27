"""PAD Analytics Package

A complete workflow for machine learning models using data from the PAD API v2.
"""

__version__ = "0.1.0"

# Import all main functions from padanalytics module to package level
from .padanalytics import (
    get_data_api,
    get_card_issues,
    get_projects,
    get_project_cards,
    get_card_by_id,
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
    get_dataset_from_model_id,
    get_dataset,
    calculate_rmse,
    calculate_rmse_by_api,
)

# Import other modules for advanced users
from . import pad_analysis
from . import pad_helper
from . import fileManagement
from . import intensityFind
from . import pixelProcessing
from . import regionRoutine

__all__ = [
    # Main functions from padanalytics
    "get_data_api",
    "get_card_issues", 
    "get_projects",
    "get_project_cards",
    "get_card_by_id",
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
    "get_dataset_from_model_id",
    "get_dataset",
    "calculate_rmse",
    "calculate_rmse_by_api",
    # Modules
    "pad_analysis",
    "pad_helper",
    "fileManagement",
    "intensityFind",
    "pixelProcessing",
    "regionRoutine",
]
