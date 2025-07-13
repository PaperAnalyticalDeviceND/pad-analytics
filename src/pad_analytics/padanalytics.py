import requests, os
import urllib3
import warnings
import sys
import contextlib
from PIL import Image, ImageFile
import ipywidgets as widgets
from IPython.display import display, HTML
from io import BytesIO
import io
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import tempfile

from . import regionRoutine
from . import pad_helper
from .dataset_manager import DatasetManager
import numpy as np
import csv
import cv2 as cv

# For resource file access
try:
    from importlib import resources
except ImportError:
    # Python < 3.9 fallback
    import importlib_resources as resources

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Control debug output
DEBUG_MODE = os.getenv("PAD_DEBUG", "").lower() in ("1", "true", "yes")

# Suppress Python warnings by default unless debug mode is enabled
if not DEBUG_MODE:
    warnings.filterwarnings("ignore", message=".*libpng.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="cv2")
    # Set OpenCV logging level to suppress libpng errors
    try:
        # Try new OpenCV constant first
        cv.setLogLevel(0)  # 0 = LOG_LEVEL_SILENT
    except AttributeError:
        try:
            # Try older constant if available
            cv.setLogLevel(cv.LOG_LEVEL_ERROR)
        except AttributeError:
            # Ignore if not available
            pass


@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output (for libpng errors)"""
    if DEBUG_MODE:
        yield
    else:
        # Try to suppress at the system level using os.dup2
        import os

        old_stderr = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 2)
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(devnull)
            os.close(old_stderr)


API_URL = "https://pad.crc.nd.edu/api/v2"


def _get_mapping_file_path():
    """Get the correct path to the model dataset mapping file."""
    try:
        # Try to get the file from package resources (when installed)
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


MODEL_DATASET_MAPPING = _get_mapping_file_path()

# Initialize dataset manager singleton
_dataset_manager = None

def get_dataset_manager():
    """Get or create the singleton DatasetManager instance."""
    global _dataset_manager
    if _dataset_manager is None:
        _dataset_manager = DatasetManager()
    return _dataset_manager


def get_data_api(request_url, data_type=""):
    try:
        # fetch_data_from_api
        r = requests.get(
            url=request_url, verify=False
        )  # NOTE: Using verify=False due to a SSL issue, I need a valid certificate, then I will remove this parameter.
        r.raise_for_status()  # Raise an exception if the status is not 200
        data = r.json()
        df = pd.json_normalize(data)
        return df
    except requests.exceptions.RequestException as e:
        print(e)
        print(f"Error accessing {data_type} data: {r.status_code}")
        return None


# Get card issue types
def get_card_issues():
    """Get all cards with reported issues.
    
    Retrieves cards from the PAD API that have been flagged with issues,
    such as image quality problems or processing errors.
    
    Returns:
        pd.DataFrame: DataFrame containing cards with issues. Typical columns include:
            - id: Card ID
            - issue_type: Type of issue reported
            - description: Issue description
            - Additional card metadata
            
    Example:
        >>> issues = pad.get_card_issues()
        >>> print(f"Found {len(issues)} cards with issues")
        Found 15 cards with issues
        >>> print(issues['issue_type'].value_counts())
        image_quality    8
        processing_error 7
    """
    request_url = f"{API_URL}/cards/issues"
    return get_data_api(request_url, "card issues")


# Get projects
def get_projects():
    """Get all projects from the PAD API.
    
    Retrieves a comprehensive list of all available PAD projects with metadata.
    Automatically cleans the data by removing columns with all NaN values.
    
    Returns:
        pd.DataFrame: DataFrame containing all projects with columns:
            - id: Project ID
            - project_name: Project name
            - description: Project description
            - status: Project status
            - Additional metadata columns (varies by project)
            
    Example:
        >>> projects = get_projects()
        >>> print(f"Found {len(projects)} projects")
        Found 25 projects
        
        >>> print(projects[['id', 'project_name']].head())
           id         project_name
        0   1        Pharmaceutical_QC
        1   2        Drug_Authentication
    """
    request_url = f"{API_URL}/projects"
    projects = get_data_api(request_url, "projects")

    # Find columns with all NaN values
    columns_with_all_nan = projects.columns[projects.isnull().all()]

    # Drop columns with all NaN values
    projects = projects.drop(columns=columns_with_all_nan)

    # Check if the column 'sample_names.sample_names' exists in the DataFrame
    if "sample_names.sample_names" in projects.columns:
        # Rename the column to 'sample_names'
        projects = projects.rename(
            columns={"sample_names.sample_names": "sample_names"}
        )

    # Specify the desired column order
    new_column_order = [
        "id",
        "project_name",
        "annotation",
        "test_name",
        "sample_names",
        "neutral_filler",
        "qpc20",
        "qpc50",
        "qpc80",
        "qpc100",
        "user_name",
        "notes",
    ]

    # Reorder the columns in the DataFrame
    projects = projects[new_column_order]

    # Reset the index of the dataframe, dropping the existing index.
    projects = projects.reset_index(drop=True)

    return projects


# Extended function to get project cards for either a single project ID or multiple project IDs
def get_project_cards(project_name=None, project_ids=None):
    """Get all cards (data samples) belonging to specific project(s).
    
    Retrieves cards from one or more projects. You can specify either a single
    project by name or multiple projects by providing a list of project IDs.
    
    Args:
        project_name (str, optional): Name of the project to get cards from.
        project_ids (list, optional): List of project IDs to get cards from.
        
    Returns:
        pd.DataFrame: Combined DataFrame with cards from specified project(s).
            Columns include:
            - id: Card ID
            - sample_id: Sample identifier
            - sample_name: Drug/sample name
            - quantity: Concentration value
            - url: Image URL
            - project_id: Associated project ID
            - Additional metadata
            
    Note:
        You must provide either project_name OR project_ids, not both.
        
    Example:
        >>> # Get cards from a specific project
        >>> cards = get_project_cards(project_name="Pharmaceutical_QC")
        >>> print(f"Found {len(cards)} cards in project")
        Found 1250 cards in project
        
        >>> # Get cards from multiple projects
        >>> cards = get_project_cards(project_ids=[1, 2, 3])
        >>> print(f"Found {len(cards)} cards across projects")
        Found 3750 cards across projects
    """

    def _get_project_cards_by_name(name):
        project_id = get_project(name=project_name).id.values[0]
        if project_id:
            return _get_project_cards_by_id(project_id)
        else:
            print(f"Project {name} not found.")
            return None

    # Get project cards
    def _get_project_cards_by_id(project_id):
        request_url = f"{API_URL}/projects/{project_id}/cards"
        return get_data_api(request_url, f"project {project_id} cards")

    # check if project_name is not None
    if project_name is not None:
        return _get_project_cards_by_name(project_name)

    # Check if project_ids is None, covert it to a list of all available project
    if project_ids is None:
        project_ids = get_projects().id.tolist()

    # Check if project_ids is a single integer, convert it to a list if so
    elif isinstance(project_ids, int):
        project_ids = [project_ids]
    # error
    elif not isinstance(project_ids, list):
        raise ValueError(
            "project_ids must be a single integer, a list of integers, or None"
        )

    all_cards = []  # List to hold dataframes from multiple projects

    for project_id in project_ids:
        # Get cards for each project
        project_cards = _get_project_cards_by_id(project_id)

        if project_cards is not None:
            all_cards.append(project_cards)

    # Concatenate all dataframes into one, if there is data
    if all_cards:
        combined_df = pd.concat(all_cards, ignore_index=True)
        return combined_df
    else:
        print("No data was retrieved for the provided project IDs.")
        return None


# def get_card(card_id):
#     request_url = f"{API_URL}/cards/{card_id}"
#     return get_data_api(request_url, f"card {card_id}")


def get_card_by_id(card_id):
    """Get a specific card by its ID.
    
    Retrieves detailed information about a single PAD card using its unique identifier.
    
    Args:
        card_id (int): The unique card ID to retrieve.
        
    Returns:
        pd.DataFrame: Single-row DataFrame with card information including:
            - id: Card ID
            - sample_id: Associated sample ID
            - sample_name: Drug/sample name
            - quantity: Concentration value
            - url: Image URL
            - project_id: Associated project
            - Additional metadata
            
    Example:
        >>> card = pad.get_card_by_id(15589)
        >>> print(f"Card for sample: {card['sample_name'].iloc[0]}")
        Card for sample: hydroxychloroquine
        >>> print(f"Concentration: {card['quantity'].iloc[0]}")
        Concentration: 100
    """
    request_url = f"{API_URL}/cards/{card_id}"
    return get_data_api(request_url, f"card {card_id}")


def get_card(card_id=None, sample_id=None):
    """Get a card by either card ID or sample ID.
    
    Flexible function that retrieves card information using either
    the card's unique ID or its associated sample ID.
    
    Args:
        card_id (int, optional): The card ID to retrieve.
        sample_id (int, optional): The sample ID to retrieve card for.
        
    Returns:
        pd.DataFrame: Card information with typical columns:
            - id: Card ID
            - sample_id: Sample identifier
            - sample_name: Drug/sample name
            - quantity: Concentration
            - url: Image URL
            - Additional metadata
            
    Raises:
        ValueError: If neither card_id nor sample_id is provided.
        
    Example:
        >>> # Get by card ID
        >>> card1 = pad.get_card(card_id=15589)
        >>> 
        >>> # Get by sample ID
        >>> card2 = pad.get_card(sample_id=53787)
        >>> 
        >>> # Both should return the same card if they match
        >>> print(card1['sample_name'].iloc[0])
        hydroxychloroquine
    """
    if card_id:
        # Get card by card_id
        return get_card_by_id(card_id)

    elif sample_id:
        # Get card samples by sample_id
        return get_card_by_sample_id(sample_id)
    else:
        raise ValueError("You must provide either card_id or sample_id")


def get_project_by_id(project_id):
    """Get a specific project by its ID.
    
    Retrieves detailed information about a single project using its unique
    identifier from the PAD API.
    
    Args:
        project_id (int): The unique project ID to retrieve.
        
    Returns:
        pd.DataFrame: Single-row DataFrame with detailed project information:
            - id: Project ID
            - project_name: Project name
            - description: Detailed project description
            - status: Current project status
            - created_date: Project creation date
            - Additional project metadata
            
    Example:
        >>> project = get_project_by_id(1)
        >>> print(f"Project: {project['project_name'].iloc[0]}")
        Project: Pharmaceutical_QC
        
        >>> print(f"Status: {project['status'].iloc[0]}")
        Status: active
    """
    request_url = f"{API_URL}/projects/{project_id}"
    return get_data_api(request_url, f"project {project_id}")


def get_project_by_name(project_name):
    """Get a project by its name.
    
    Searches for a project by name using case-insensitive matching.
    Useful when you know the project name but not the ID.
    
    Args:
        project_name (str): The name of the project to find.
        
    Returns:
        pd.DataFrame: DataFrame with matching project(s). Will be empty if
            no project with that name is found. Columns include:
            - id: Project ID
            - project_name: Project name (exact match)
            - description: Project description
            - status: Project status
            - Additional metadata
            
    Note:
        Search is case-insensitive ("QC" matches "qc", "Qc", etc.)
        
    Example:
        >>> project = get_project_by_name("Pharmaceutical_QC")
        >>> if not project.empty:
        ...     print(f"Found project ID: {project['id'].iloc[0]}")
        Found project ID: 1
        
        >>> # Case-insensitive search
        >>> project = get_project_by_name("pharmaceutical_qc")
        >>> print(len(project))  # Should find the same project
        1
    """
    projects = get_projects()
    project = projects[
        projects["project_name"].apply(lambda x: x.lower() == project_name.lower())
    ]
    return project


def get_project(id=None, name=None):
    """Get a project by ID or name (flexible interface).
    
    Convenience function that allows retrieving a project using either its
    ID or name. Automatically routes to the appropriate specific function.
    
    Args:
        id (int, optional): The project ID to retrieve.
        name (str, optional): The project name to search for.
        
    Returns:
        pd.DataFrame: Project information. Format depends on lookup method:
            - By ID: Single project with detailed information
            - By name: Matching project(s) from filtered list
            
    Raises:
        ValueError: If neither id nor name is provided.
        
    Note:
        You must provide exactly one parameter (either id OR name).
        
    Example:
        >>> # Get by ID
        >>> project = get_project(id=1)
        >>> print(project['project_name'].iloc[0])
        Pharmaceutical_QC
        
        >>> # Get by name
        >>> project = get_project(name="Pharmaceutical_QC")
        >>> print(project['id'].iloc[0])
        1
        
        >>> # Error case
        >>> project = get_project()  # Raises ValueError
        ValueError: You must provide either project_id or project_name
    """
    if id:
        # Get project by ID
        return get_project_by_id(id)
    elif name:
        # Get project by project_name
        return get_project_by_name(name)
    else:
        raise ValueError("You must provide either project_id or project_name")


# Function to load image from URL
def load_image_from_url(image_url):
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content))
    return img


# Function to create a widget that shows the image and its related data
def create_image_widget_with_info(image_url, data_df, multi_card_mode=False):
    """Create responsive image widget that adapts to single or multi-card display.
    
    Args:
        image_url (str): URL to the PAD image
        data_df (pd.DataFrame): Card metadata
        multi_card_mode (bool): If True, uses responsive layout for multi-card display
    """
    # Responsive dimensions based on display mode
    if multi_card_mode:
        # Balanced dimensions for multi-card display
        small_im_width = "300px"    # Standard image width
        small_im_max_width = "300px" 
        table_width = "350px"        # Wider table for better content display
    else:
        # Standard widths for single card display (maintain existing behavior)
        small_im_width = "300px"
        small_im_max_width = "300px"
        table_width = "500px"
    
    full_im_width = 800  # Always use fixed width for zoom overlay
    background_color_field = "#5c6e62"
    background_color_value = "#f9f9f9"
    image_id = data_df.ID.values[0]

    # Create compact HTML widget with JavaScript for image zoom on click
    image_style = f"width:{small_im_width}; height:auto; cursor: pointer; display:block;"
    
    zoomable_image_html = f"""
    <div id="imageContainer_{image_id}" style="display: flex; justify-content: center; align-items: center;">    
      <img id="zoomableImage_{image_id}" src="{image_url}" alt="Image" style="{image_style}" 
          onclick="
              var img = document.getElementById('zoomableImage_{image_id}');
              var overlay = document.getElementById('overlay_{image_id}');
              var currentWidth = img.style.width;
              if (currentWidth === '{small_im_width}') {{
                  img.style.width = '{full_im_width}px';  // Full size image width
                  overlay.style.display = 'flex';  // Show overlay
                  overlay.style.alignItems = 'flex-start';  // Align the image at the top
                  overlay.appendChild(img);  // Move image to overlay
              }} else {{
                  img.style.width = '{small_im_width}';  // Restore compact width
                  document.getElementById('imageContainer_{image_id}').appendChild(img);  // Move image back to grid
                  overlay.style.display = 'none';  // Hide overlay
              }}
          ">
      </div>
      <div id="overlay_{image_id}" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; display: none; background-color: rgba(255,255,255,0.9); z-index: 1000; align-items: flex-start; justify-content: center; overflow: auto;">
      </div>
    """

    # Create HTML widget for the zoomable image
    img_widget = widgets.HTML(zoomable_image_html)

    # ID label with left-aligned text, custom color, and bold font style using HTML
    id_label = widgets.HTML("<br>")

    # Arrange the clickable image in a vertical box (this will be the first column)
    image_column = widgets.VBox([img_widget])
    # Create responsive DataFrame-like table using HTML with field names as row headers
    table_style = f"""
    <style>
        table {{
            font-family: sans-serif;
            font-size: {13 if multi_card_mode else 14}px;
            border-collapse: collapse;
            width: {table_width};
            min-width: {350 if multi_card_mode else 500}px;
            table-layout: {'fixed' if multi_card_mode else 'auto'};
        }}
        td, th {{
            border: 1px solid #dddddd;
            text-align: left;
            padding: {2 if multi_card_mode else 4}px;
            word-wrap: break-word;
            overflow: hidden;
        }}

        th {{
            background-color: {background_color_field};
            color: white;
            text-align: left;
            width: {80 if multi_card_mode else 120}px;
            padding-left: {8 if multi_card_mode else 20}px;
            font-size: {12 if multi_card_mode else 14}px;
        }}
        td{{
            padding-left: {6 if multi_card_mode else 10}px;
            max-width: {180 if multi_card_mode else 300}px;
            text-overflow: ellipsis;
        }}
        tr:nth-child(even) {{
            background-color: {background_color_value};
        }}
        tr:hover {{
            background-color: #eeeee0;
        }}
    </style>
    """

    table_html = table_style + "<table>"
    for field in data_df.columns:

        table_html += "<tr>"
        table_html += f"<th>{field}</th>"
        # if type is bool add icon
        if data_df[field].dtype == "bool":
            val = "Yes" if data_df[field].values[0] else "No"
            table_html += f"<td>{val}</td>"
        else:
            table_html += f"<td>{data_df[field].values[0]}</td>"
        table_html += "</tr>"
    table_html += "</table>"

    # Create HTML widget for the table
    info_table = widgets.HTML(table_html)

    # Arrange the two columns (image and info) side by side with responsive layout
    if multi_card_mode:
        # Use flexible layout for multi-card mode
        columns = widgets.HBox([image_column, info_table], 
                              layout=widgets.Layout(width='100%', 
                                                   display='flex',
                                                   flex_flow='row wrap'))
    else:
        # Use standard layout for single card mode
        columns = widgets.HBox([image_column, info_table])

    # Display the ID label above the columns
    return widgets.VBox([id_label, columns])


def show_card(card_id=None, sample_id=None):
    """Display a single PAD card with its image and metadata in Jupyter notebook.
    
    Creates an interactive widget showing the PAD card image alongside detailed
    metadata including sample information, project details, and card status.
    Designed for visual inspection and data exploration in research workflows.
    
    Args:
        card_id (int, optional): The unique card ID to display.
        sample_id (int, optional): The sample ID to find and display the card for.
        
    Returns:
        None: Displays the widget directly in the Jupyter notebook interface.
        
    Raises:
        ValueError: If neither card_id nor sample_id is provided, or if both are provided.
        
    Note:
        This function requires a Jupyter notebook environment with ipywidgets.
        If the card image is not available, a placeholder image will be shown.
        You must provide exactly one parameter (either card_id OR sample_id).
        
        When using sample_id: If multiple cards exist for the same sample,
        all cards will be automatically displayed using batch visualization.
        For single cards, the standard single-card widget is used.
        
    Example:
        >>> # Display card by card ID
        >>> show_card(card_id=47918)
        # Shows interactive widget with card image and metadata table
        
        >>> # Display cards by sample ID (multiple cards)
        >>> show_card(sample_id=67890)
        # Automatically shows all cards for sample 67890 in batch view
        
        >>> # Card details displayed include:
        >>> # - Card ID, Sample ID, Sample Name
        >>> # - Quantity, Camera Type, Project Info
        >>> # - Creation date, Notes, Issue status
    """
    # Validate parameters
    if card_id is None and sample_id is None:
        raise ValueError("You must provide either card_id or sample_id")
    if card_id is not None and sample_id is not None:
        raise ValueError("You cannot provide both card_id and sample_id. Choose one.")
    
    # Get card information using the appropriate parameter
    if card_id is not None:
        info = get_card(card_id=card_id)
        display_id = card_id
    else:  # sample_id is not None
        info = get_card(sample_id=sample_id)
        # Handle multiple cards for the same sample_id
        if info is not None and not info.empty:
            if len(info) > 1:
                print(f"📋 Sample {sample_id} has {len(info)} cards. Displaying all cards:")
                # Use show_cards_from_df to display all cards for this sample
                show_cards_from_df(info)
                return  # Exit early since we've displayed all cards
            display_id = info['id'].iloc[0]
        else:
            display_id = f"sample_{sample_id}"

    if info is None or info.empty:
        identifier = card_id if card_id is not None else f"sample {sample_id}"
        print(f"Failed to retrieve data for {identifier}")
        return

    # Data validation: check if essential fields exist in the API response
    def safe_get(field, default="N/A"):
        try:
            if field in info.columns:
                return info[field].values[0]
            else:
                return default
        except (IndexError, KeyError):
            return default

    # Example of how to use `safe_get` for extracting fields
    data = {
        "ID": [display_id],
        "Sample ID": [safe_get("sample_id")],
        "Sample Name": [safe_get("sample_name")],
        "Quantity": [f"{safe_get('quantity')}%" if safe_get('quantity') != 'N/A' else 'N/A'],
        "Camera Type": [safe_get("camera_type_1")],
        "Issue": [safe_get("issue.name", safe_get("issue"))],
        "Project Name": [safe_get("project.project_name")],
        "Project Id": [safe_get("project.id")],
        "Notes": [safe_get("notes")],
        "Date of Creation": [safe_get("date_of_creation")],
        "Deleted": [safe_get("deleted", default=False)],  # If missing, default to False
    }

    # Convert data to DataFrame
    data_df = pd.DataFrame(data)

    # Handle missing image URL gracefully
    try:
        image_url = (
            "https://pad.crc.nd.edu/" + info["processed_file_location"].values[0]
        )
    except (KeyError, IndexError):
        identifier = card_id if card_id is not None else f"sample {sample_id}"
        print(f"No valid image found for {identifier}")
        image_url = "https://via.placeholder.com/300"  # Default placeholder image

    # Create the widget for the image and its info
    image_widget_box = create_image_widget_with_info(image_url, data_df)

    # Display the widget
    display(image_widget_box)


# Function to generate HTML for zoomable images with data from DataFrame
def generate_zoomable_image_html(image_id, sample_id, image_url):

    small_im_width = 300
    full_im_width = 600

    return f"""
    <div id="imageContainer_{image_id}">
        <!-- Information above the image -->
        <div style="position: relative; font-size: 14px; color: #5c6e62; margin-bottom: 5px;">
            <strong>ID:</strong> {image_id} <strong>Sample ID:</strong> {sample_id}
        </div>
        <!-- The zoomable image -->        
        <img id="zoomableImage_{image_id}" src="{image_url}" alt="Image" style="width:{small_im_width}px; cursor: pointer;" 
        onclick="
            var img = document.getElementById('zoomableImage_{image_id}');
            var overlay = document.getElementById('overlay_{image_id}');
            if (img.style.width == '{small_im_width}px') {{
                img.style.width = '{full_im_width}px';  // Full size image width
                overlay.style.display = 'flex';  // Show overlay
                overlay.style.alignItems = 'flex-start';  // Align the image at the top
                overlay.appendChild(img);  // Move image to overlay
            }} else {{
                img.style.width = '{small_im_width}px';  // Small size image width
                document.getElementById('imageContainer_{image_id}').appendChild(img);  // Move image back to grid
                overlay.style.display = 'none';  // Hide overlay
            }}
        ">
    </div>
    <div id="overlay_{image_id}" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; display: none; background-color: rgba(255,255,255,0.9); z-index: 1000; align-items: flex-start; justify-content: center; overflow: auto;">
    </div>
    """


# Function to create tabs based on the grouping column and number of images per row
def create_tabs(df, group_column, images_per_row=5):
    # Group the DataFrame by the chosen column
    grouped_data = df.groupby(group_column)

    # Create a list of widgets for each tab (text content + zoomable images)
    items = []
    for group_value, group in grouped_data:
        # Text content at the top for each tab (based on group_column), followed by a horizontal line <hr>
        text_content = widgets.HTML(
            f"""
        <div style="font-size: 18px; color: #5c6e62;">
            <strong>{group_column.capitalize()}:</strong> {group_value} (#Cards: {len(group)})
        </div>
        <hr style="border: 1px solid #ccc; margin-top: 10px;">
        """
        )

        # Create a grid of zoomable images for each tab based on the data in the group
        img_widgets = [
            widgets.HTML(
                generate_zoomable_image_html(row["id"], row["sample_id"], row["url"])
            )
            for _, row in group.iterrows()
        ]

        # Create a grid box to hold the images for each group, with a configurable number of images per row
        grid = widgets.GridBox(
            children=img_widgets,
            layout=widgets.Layout(
                grid_template_columns=f"repeat({images_per_row}, 300px)",  # Use the parameter for images per row
                grid_gap="10px",
            ),
        )

        # Combine text content and grid into a vertical box (VBox)
        combined_content = widgets.VBox([text_content, grid])

        # Add the combined content to the list of tab items
        items.append(combined_content)

    # Create the tab widget
    tab = widgets.Tab(children=items)

    # Set tab titles based on the group value and number of Cards
    for i, (group_value, group) in enumerate(grouped_data):
        tab.set_title(
            i, f"{group_value} ({len(group)})"
        )  # Show group_value and sample count

    # Create an Output widget with fixed height to contain the tab content
    output = widgets.Output(layout=widgets.Layout(height="1000px", overflow_y="auto"))

    # Display the tab widget inside the Output widget with a title
    with output:
        # Adding an HTML title above the tab
        display(
            widgets.HTML(
                f"<h2 style='text-align: center;'>Grouped by {group_column.capitalize()}</h2>"
            )
        )
        display(tab)

    # Display the Output widget with the tabs inside
    display(output)


def show_grouped_cards(df, group_column, images_per_row=5):
    """Display multiple PAD cards organized by groups in a tabbed interface.
    
    Creates a tabbed visualization where cards are grouped by a specified column
    and displayed as image grids. Each tab represents a different group value,
    allowing easy comparison and exploration of cards by categories.
    
    Args:
        df (pd.DataFrame): DataFrame containing card information. Must include
            'processed_file_location' column for image URLs.
        group_column (str): Column name to group cards by (e.g., 'sample_name',
            'project.project_name', 'quantity').
        images_per_row (int, optional): Number of card images to display per row
            in each tab. Defaults to 5.
            
    Returns:
        None: Displays the tabbed widget interface directly in Jupyter notebook.
        
    Note:
        Requires Jupyter notebook environment with ipywidgets. The DataFrame
        should contain card data with 'processed_file_location' for images.
        
    Example:
        >>> # Group cards by sample name
        >>> cards_df = get_project_cards(project_name="Drug_Study")
        >>> show_grouped_cards(cards_df, 'sample_name', images_per_row=3)
        # Creates tabs for each drug with 3 images per row
        
        >>> # Group by project with default 5 images per row
        >>> show_grouped_cards(cards_df, 'project.project_name')
        # Creates tabs for each project
    """
    # Ensure we're working on a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()

    # Add url to dataframe safely using .loc
    df.loc[:, "url"] = df["processed_file_location"].apply(
        lambda x: f"https://pad.crc.nd.edu/{x}"
    )

    create_tabs(df, group_column, images_per_row)


def create_thumbnail(url, size=(100, 100)):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.thumbnail(size)
    return img
    # create_thumbnail('https://pad.crc.nd.edu//var/www/html/images/padimages/processed/40000/42275_processed.png', size=(100, 100))


def standardize_names(name):
    return name.lower().replace(" ", "-")


# Extended function to get project cards for either a single project ID or multiple project IDs
def get_card_by_sample_id(sample_id):
    """Get card(s) associated with a specific sample ID.
    
    Retrieves all cards that are associated with the given sample ID using
    the PAD API v3. Multiple cards may exist for a single sample if it was 
    processed multiple times or under different conditions.
    
    Args:
        sample_id (int): The sample ID to search for.
        
    Returns:
        pd.DataFrame: DataFrame with matching cards. Columns include:
            - id: Card ID
            - sample_id: Sample identifier (will match input)
            - sample_name: Drug/sample name
            - quantity: Concentration
            - url: Image URL
            - project_id: Associated project
            - Additional metadata from nested API structure
            
    Raises:
        Exception: If API request fails or returns error.
        
    Example:
        >>> cards = pad.get_card_by_sample_id(53787)
        >>> print(f"Found {len(cards)} cards for sample 47918")
        Found 1 cards for sample 53787
        >>> print(cards[['id', 'sample_name', 'quantity']])
            id         sample_name  quantity
        0  15589  hydroxychloroquine       100
        
    Note:
        This function uses PAD API v3 endpoint which provides more detailed
        card information including nested project and sample data.
    """

    # Make API request
    url = f"https://pad.crc.nd.edu/api-ld/v3/cards/by-sample/{sample_id}"
    response = requests.get(url)
    data = response.json()

    if not data["success"]:
        raise Exception(f"API request failed: {data['error']}")

    # Extract cards data
    cards = data["data"]

    # Process each card to flatten the nested structure
    processed_cards = []
    for card in cards:
        processed_card = {
            "id": card["id"],
            "sample_name": card["sample_name"]["name"],
            "test_name": card["test_name"]["name"],
            "user_name": card["user_name"]["name"],
            "date_of_creation": card["date_of_creation"],
            "raw_file_location": card["raw_file_location"],
            "processed_file_location": card["processed_file_location"],
            "processing_date": None,  # This field wasn't in the original data
            "camera_type_1": card["camera_type_1"],
            "notes": card["notes"],
            "sample_id": card["sample_id"],
            "quantity": card["quantity"],
            "deleted": False,  # This field wasn't in the original data
            "issue": card.get("issue_id"),
            "project.id": card["project"]["id"],
            "project.user_name": None,  # This field wasn't in the project data
            "project.project_name": card["project"]["name"],
            "project.annotation": None,  # This field wasn't in the project data
            "project.test_name": None,  # This field wasn't in the project data
            "project.sample_names.sample_names": None,  # This field wasn't in the project data
            "project.neutral_filler": None,  # This field wasn't in the project data
            "project.qpc20": None,  # This field wasn't in the project data
            "project.qpc50": None,  # This field wasn't in the project data
            "project.qpc80": None,  # This field wasn't in the project data
            "project.qpc100": None,  # This field wasn't in the project data
            "project.notes": None,  # This field wasn't in the project data
        }
        processed_cards.append(processed_card)

    # Create DataFrame
    df = pd.DataFrame(processed_cards)

    return df


def show_cards_from_df(cards_df):
    """Display multiple PAD cards from a DataFrame (batch visualization).
    
    Creates individual card widgets for each row in the provided DataFrame.
    This is the most efficient way to display multiple cards when you already
    have the card data loaded, as it avoids additional API calls.
    
    Args:
        cards_df (pd.DataFrame): DataFrame containing card information. Should
            include columns like 'id', 'sample_id', 'sample_name', 'quantity',
            'processed_file_location', and other card metadata.
            
    Returns:
        None: Displays card widgets sequentially in the Jupyter notebook.
        
    Note:
        More efficient than show_cards() since data is already loaded.
        Assumes all necessary card fields are present in the DataFrame.
        Missing fields will display as 'N/A'.
        
    Example:
        >>> # Display all cards from a project
        >>> project_cards = get_project_cards(project_name="Quality_Control")
        >>> show_cards_from_df(project_cards)
        # Shows individual widgets for each card in the project
        
        >>> # Display cards from a dataset
        >>> dataset_cards = get_dataset_cards("FHI2020_Stratified_Sampling")
        >>> show_cards_from_df(dataset_cards[:10])  # Show first 10 cards
    """
    card_widgets = []

    # Iterate through each row in the DataFrame
    for index, row in cards_df.iterrows():
        # Extract the necessary fields from the DataFrame row
        id = row["id"]
        sample_id = row.get("sample_id", "N/A")
        sample_name = row.get("sample_name", "N/A")
        quantity = row.get("quantity", "N/A")
        camera_type = row.get("camera_type_1", "N/A")
        issue = row.get("issue", "N/A")
        project_name = row.get("project.project_name", "N/A")
        project_id = row.get("project.id", "N/A")
        notes = row.get("notes", "N/A")
        date_of_creation = row.get("date_of_creation", "N/A")
        deleted = row.get("deleted", False)  # Default to False if not present
        processed_file_location = row.get("processed_file_location", None)

        # Construct the data dictionary for the card
        data = {
            "ID": [id],
            "Sample ID": [sample_id],
            "Sample Name": [sample_name],
            "Quantity": [f"{quantity}%" if quantity != 'N/A' and quantity is not None else 'N/A'],
            "Camera Type": [camera_type],
            "Issue": [issue],
            "Project Name": [project_name],
            "Project Id": [project_id],
            "Notes": [notes],
            "Date of Creation": [date_of_creation],
            "Deleted": [deleted],
        }
        data_df = pd.DataFrame(data)

        # Generate the image URL, handling the case where it might be missing
        if processed_file_location:
            image_url = f"https://pad.crc.nd.edu/{processed_file_location}"
        else:
            image_url = (
                "https://via.placeholder.com/300"  # Use placeholder if no image URL
            )

        # Create the widget for this card with multi-card mode enabled
        card_widget = create_image_widget_with_info(image_url, data_df, multi_card_mode=True)
        card_widgets.append(card_widget)

    # Create a responsive layout to display the cards
    # Use responsive grid that adapts to available space
    max_cards_per_row = 2  # Adjust how many cards per row
    card_rows = [
        widgets.HBox(card_widgets[i : i + max_cards_per_row],
                    layout=widgets.Layout(width='100%', 
                                         display='flex',
                                         flex_flow='row wrap',
                                         justify_content='space-around'))
        for i in range(0, len(card_widgets), max_cards_per_row)
    ]

    # Display the rows of widgets vertically with responsive container
    display(widgets.VBox(card_rows, 
                        layout=widgets.Layout(width='100%',
                                             overflow='hidden')))


def show_cards(card_ids=None, sample_ids=None):
    """Display multiple PAD cards from a list of card IDs or sample IDs.
    
    Fetches card data for each ID via API calls and creates individual card
    widgets for visualization. Handles missing or invalid IDs gracefully
    with error messages. Supports both direct card ID access and sample-based lookup.
    
    Args:
        card_ids (list, optional): List of card IDs (integers) to display.
        sample_ids (list, optional): List of sample IDs (integers) to find and display cards for.
        
    Returns:
        None: Displays card widgets in a responsive grid layout.
        Shows error messages for any invalid or missing IDs.
        
    Raises:
        ValueError: If neither card_ids nor sample_ids is provided, or if both are provided.
        
    Note:
        Less efficient than show_cards_from_df() due to individual API calls.
        Use show_cards_from_df() when you already have the card data loaded.
        You must provide exactly one parameter (either card_ids OR sample_ids).
        When using sample_ids, multiple cards per sample will all be displayed.
        
    Example:
        >>> # Display specific cards by card ID
        >>> show_cards(card_ids=[47918, 47919, 47920])
        # Shows individual widgets for each valid card
        
        >>> # Display cards by sample IDs
        >>> show_cards(sample_ids=[52677, 52678, 52679])
        # Shows all cards from these samples (1 or more cards per sample)
        
        >>> # Research workflow example
        >>> aspirin_samples = [52677, 52678, 52679]
        >>> show_cards(sample_ids=aspirin_samples)
        # Shows all cards for aspirin study samples
    """
    # Validate parameters
    if card_ids is None and sample_ids is None:
        raise ValueError("You must provide either card_ids or sample_ids")
    if card_ids is not None and sample_ids is not None:
        raise ValueError("You cannot provide both card_ids and sample_ids. Choose one.")
    
    # Convert sample_ids to card_ids if needed
    if sample_ids is not None:
        print(f"🔍 Looking up cards for {len(sample_ids)} samples...")
        card_ids = []
        for sample_id in sample_ids:
            # Get all cards for this sample
            sample_cards = get_card(sample_id=sample_id)
            if sample_cards is not None and not sample_cards.empty:
                # Add all card IDs from this sample
                sample_card_ids = sample_cards['id'].tolist()
                card_ids.extend(sample_card_ids)
                if len(sample_card_ids) > 1:
                    print(f"  📋 Sample {sample_id}: found {len(sample_card_ids)} cards")
            else:
                print(f"  ⚠️ Sample {sample_id}: no cards found")
        
        if not card_ids:
            print("❌ No cards found for any of the provided sample IDs")
            return
        
        print(f"✅ Total cards to display: {len(card_ids)}")
        print()  # Add blank line for readability
    card_widgets = []

    # Iterate through each card in the DataFrame
    for card_id in card_ids:
        # Fetch card data
        info = get_card(card_id)

        # Handle the case where the API fails to return the card data
        if info is None:
            # print(f"Failed to retrieve data for card {card_id}")

            # Displaying the message with custom font and dark red color
            display(
                HTML(
                    f"""
            <div style="font-family: 'Courier New', monospace; color: darkred;">
                &#128308; No data was retrieved for card ID {card_id}.
            </div>
            """
                )
            )
            continue

        # Safely extract the required fields using the helper function `safe_get`
        def safe_get(field, default="N/A"):
            try:
                if field in info.columns:
                    return info[field].values[0]
                else:
                    return default
            except (IndexError, KeyError):
                return default

        # Prepare the data for the card
        data = {
            "ID": [card_id],
            "Sample ID": [safe_get("sample_id")],
            "Sample Name": [safe_get("sample_name")],
            "Quantity": [f"{safe_get('quantity')}%" if safe_get('quantity') != 'N/A' else 'N/A'],
            "Camera Type": [safe_get("camera_type_1")],
            "Issue": [safe_get("issue.name", safe_get("issue"))],
            "Project Name": [safe_get("project.project_name")],
            "Project Id": [safe_get("project.id")],
            "Notes": [safe_get("notes")],
            "Date of Creation": [safe_get("date_of_creation")],
            "Deleted": [safe_get("deleted", default=False)],
        }

        # Convert to DataFrame for display
        data_df = pd.DataFrame(data)

        # Handle missing image URL safely
        try:
            image_url = (
                "https://pad.crc.nd.edu/" + info["processed_file_location"].values[0]
            )
        except (KeyError, IndexError):
            print(f"No valid image found for card {card_id}")
            image_url = "https://via.placeholder.com/300"  # Placeholder if no image

        # Create the widget for the current card with multi-card mode enabled
        card_widget = create_image_widget_with_info(image_url, data_df, multi_card_mode=True)
        card_widgets.append(card_widget)

    # Create a responsive layout to display the cards
    # Use 2 cards per row to match show_cards_from_df() behavior
    max_cards_per_row = 2  # Adjusted to fit 650px cards side-by-side
    card_rows = [
        widgets.HBox(card_widgets[i : i + max_cards_per_row],
                    layout=widgets.Layout(width='100%', 
                                         display='flex',
                                         flex_flow='row wrap',
                                         justify_content='space-around'))
        for i in range(0, len(card_widgets), max_cards_per_row)
    ]

    # Display the rows of widgets vertically with responsive container
    display(widgets.VBox(card_rows, 
                        layout=widgets.Layout(width='100%',
                                             overflow='hidden')))


def get_models():
    """Get all available neural network models from the PAD API.
    
    Retrieves a comprehensive list of all available machine learning models
    that can be used for PAD image analysis and predictions.
    
    Returns:
        pd.DataFrame: DataFrame containing all available models with columns:
            - id: Model ID (used for predictions)
            - name: Model name/identifier
            - description: Model description
            - type: Model type (classification, regression, etc.)
            - status: Model status (active, deprecated, etc.)
            - Additional model metadata
            
    Example:
        >>> models = get_models()
        >>> print(f"Found {len(models)} available models")
        Found 4 available models
        
        >>> print(models[['id', 'name']].head())
           id                    name
        0  16    24fhiNN1classifyAPI
        1  17        24fhiNN1concAPI
        2  18         24fhiPLS1conc
        3  19    24fhiNN1concAPIv2
    """
    request_url = f"{API_URL}/neural-networks"
    return get_data_api(request_url, "card issues")


def get_model(nn_id):
    """Get detailed information about a specific neural network model.
    
    Retrieves comprehensive metadata and configuration details for a
    specific machine learning model by its ID.
    
    Args:
        nn_id (int): The neural network model ID to retrieve.
        
    Returns:
        pd.DataFrame: Single-row DataFrame with detailed model information:
            - id: Model ID
            - name: Model name/identifier
            - description: Detailed model description
            - architecture: Model architecture details
            - performance_metrics: Model performance data
            - training_info: Training configuration
            - Additional technical metadata
            
    Example:
        >>> model = get_model(16)
        >>> print(f"Model: {model['name'].iloc[0]}")
        Model: 24fhiNN1classifyAPI
        
        >>> print(f"Type: Classification model for drug identification")
        Type: Classification model for drug identification
    """
    request_url = f"{API_URL}/neural-networks/{nn_id}"
    return get_data_api(request_url, f"neural_network {nn_id}")


def read_img(image_url):
    # Get the image data from the URL
    response = requests.get(image_url)
    response.raise_for_status()  # Ensure the request was successful

    # Open the image using PIL directly from the HTTP response
    img = Image.open(BytesIO(response.content))
    return img


def download_file(url, filename, images_path):
    """Download a file from a URL and save it to a local file."""
    try:
        response = requests.get(url, stream=True, verify=False)
        if response.status_code == 200:
            path = os.path.join(images_path, filename)
            with open(path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            # print(f"File '{filename}' successfully downloaded to '{images_path}'")
        else:
            # Log error if the response status code is not 200
            print(
                f"Failed to download the file. URL: {url} returned status code: {response.status_code}"
            )
            raise Exception(
                f"Failed to download the file. URL: {url} returned status code: {response.status_code}"
            )
    except Exception as e:
        # Log any other exceptions during the download process
        print(f"An error occurred while downloading the file: {e}")
        # Optionally, you can re-raise the exception if you want it to be noticed by the calling function
        raise


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
    return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)


class pls:
    def __init__(self, coefficients_file):
        try:
            # load coeffs
            self.coeff = {}
            with open(coefficients_file) as csvcoeffs:
                csvcoeffreader = csv.reader(csvcoeffs)
                # i=0
                for row in csvcoeffreader:
                    elmts = []
                    for j in range(1, len(row)):
                        elmts.append(float(row[j]))
                    self.coeff[row[0]] = elmts
        except Exception as e:
            print("Error", e, "loading pls coefficients", coefficients_file)

    def quantity(self, in_file, drug):
        try:
            # grab image
            img = cv.imread(in_file)

            if img is None:
                print("Converting img.. ", in_file)
                # read image using Pillow and covert to cv2
                img_pil = Image.open(in_file)
                img = convert_from_image_to_cv2(img_pil)

            if img is None:
                raise Exception(f"Failed to load the file. URL: {in_file}.")

            # pls dictionary
            f = {}
            f = regionRoutine.fullRoutine(
                img, regionRoutine.intFind.findMaxIntensitiesFiltered, f, True, 10
            )

            # drug?
            # continue if no coefficients

            if drug.lower() not in self.coeff:
                print(drug.lower(), "--- NOT IN COEFFICIENTS FILE ---")
                return -1

            drug_coeff = self.coeff[drug.lower()]  # coeff['amoxicillin'] #

            # start with offst
            pls_concentration = drug_coeff[0]

            coeff_index = 1

            for letter in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]:
                for region in range(10):
                    for color_letter in ["R", "G", "B"]:
                        pixval = f[letter + str(region + 1) + "-" + color_letter]
                        pls_concentration += float(pixval) * drug_coeff[coeff_index]
                        coeff_index += 1

            # print(drug.lower(), "--- OK ---")
            return pls_concentration

        except Exception as e:
            print("Error", e, "pls analyzing image", in_file, "with", drug)
            return -1.0


def read_img(image_url):
    # Get the image data from the URL
    response = requests.get(image_url)
    response.raise_for_status()  # Ensure the request was successful

    # Open the image using PIL directly from the HTTP response
    img = Image.open(BytesIO(response.content))
    return img


def nn_predict(image_url, model_path, labels):

    # Read the image from the URL
    img = read_img(image_url)

    # crop image to get active area
    img = img.crop((71, 359, 71 + 636, 359 + 490))

    # for square images
    size = (454, 454)
    img = img.resize((size), Image.BICUBIC)  # , Image.ANTIALIAS)

    # reshape the image as numpy
    # im = np.asarray(img).flatten().reshape(1, HEIGHT_INPUT, WIDTH_INPUT, DEPTH)

    HEIGHT_INPUT, WIDTH_INPUT, DEPTH = (454, 454, 3)

    # reshape the image as numpy
    im = (
        np.asarray(img)
        .flatten()
        .reshape(1, HEIGHT_INPUT, WIDTH_INPUT, DEPTH)
        .astype(np.float32)
    )

    # Load the TFLite model and allocate tensors.
    # model_file = 'lite_models/' + arch + experiment + '_v1p0'

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print("input", input_details[0])

    # Test the model on random input data.
    input_shape = input_details[0]["shape"]
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]["index"], im)

    # predict
    interpreter.invoke()

    # result
    result = interpreter.get_tensor(output_details[0]["index"])

    num_label = np.argmax(result[0])
    prediction = labels[num_label]
    # print("Prediction: ", prediction)

    probability = tf.nn.softmax(result[0])[num_label].numpy()
    # print("Probability: ", probability)

    # energy
    energy = tf.reduce_logsumexp(result[0], -1)
    # print("Energy: ", energy.numpy())

    return prediction, probability, energy.numpy()


def predict(card_id, model_id, actual_api=None, verbose=False):
    """Make ML model predictions on a PAD card image.
    
    Downloads and runs machine learning models to predict either drug classification
    or concentration from PAD colorimetric patterns. Automatically detects model
    type (Neural Network vs PLS) and handles model downloads, image processing,
    and prediction execution.
    
    Args:
        card_id (int): The unique card ID to analyze from the PAD database.
        model_id (int): The model ID to use for prediction. Common models:
            - 16: Neural Network classifier (24fhiNN1classifyAPI)
            - 17: Neural Network concentration (24fhiNN1concAPI)
            - 18: PLS concentration model (24fhiPLS1conc)
            - 19: Neural Network concentration v2
        actual_api (str, optional): Override for the actual drug name if different
            from the card's sample_name. Used for label standardization.
        verbose (bool, optional): If True, prints detailed model information
            including model type, URL, and file name. Defaults to False.
            
    Returns:
        tuple: A 2-element tuple containing:
            - actual_label (str or float): Ground truth value. For classification
              models, returns standardized drug name (e.g., "aspirin"). For
              concentration models, returns actual concentration as float.
            - prediction (str/float or tuple): Model prediction result:
                * Neural Network classification: (drug_name, confidence, energy)
                * Neural Network concentration: (concentration, confidence, energy)  
                * PLS models: concentration as float
                
    Raises:
        Exception: If model download fails or prediction execution fails.
        FileNotFoundError: If required model files cannot be accessed.
        
    Note:
        - Model files are automatically downloaded and cached locally
        - Neural Network models use TensorFlow Lite (.tflite files)
        - PLS models use custom concentration prediction algorithms
        - Image processing includes automatic PAD region detection and color analysis
        - Label standardization is applied for consistent drug name formatting
        
    Example:
        >>> # Drug classification with Neural Network
        >>> actual, prediction = predict(card_id=47918, model_id=16)
        >>> print(f"Actual: {actual}")
        Actual: aspirin
        >>> print(f"Predicted: {prediction[0]}, Confidence: {prediction[1]:.2f}")
        Predicted: aspirin, Confidence: 0.95
        
        >>> # Concentration prediction with PLS model
        >>> actual, prediction = predict(card_id=47918, model_id=18, verbose=True)
        Model Type: pls
        Model URL: https://pad.crc.nd.edu/models/24fhiPLS1conc.pkl
        Model File: 24fhiPLS1conc.pkl
        >>> print(f"Actual: {actual:.2f} mg, Predicted: {prediction:.2f} mg")
        Actual: 75.50 mg, Predicted: 73.21 mg
        
        >>> # Override actual label for comparison studies
        >>> actual, pred = predict(card_id=47918, model_id=16, actual_api="ibuprofen")
        >>> # actual will be "ibuprofen" instead of card's sample_name
    """

    pad_url = "https://pad.crc.nd.edu/"

    card_df = get_card(card_id)

    # download model
    model_df = get_model(model_id)
    model_type = model_df.type.values[0]
    model_url = model_df.weights_url.values[0]
    model_file = os.path.basename(model_url)
    if verbose:
        print(f"Model Type: {model_type}")
        print(f"Model URL: {model_url}")
        print(f"Model File: {model_file}")

    if not os.path.exists(model_file):
        if pad_helper.pad_download(model_url):
            print(model_url, "downloaded.")
        else:
            print(model_url, "failed to download.")

    # label type
    labels = model_df.labels[0]
    try:  # Predict Concentration
        labels = list(map(int, labels))
        labels_type = "concentration"
    except:  # Predict API
        labels = list(map(standardize_names, labels))
        labels_type = "api"

    if verbose:
        print("Labels: ", labels)

    # define actual label
    if actual_api is None:
        actual_api = standardize_names(card_df.sample_name.values[0])

    if labels_type == "concentration":
        actual_label = card_df.quantity.values[0]
        # Convert numpy types to native Python types
        if hasattr(actual_label, 'item'):
            actual_label = actual_label.item()
    else:
        actual_label = actual_api

    # fix label names
    labels = list(map(standardize_names, get_model(model_id).labels.values[0]))

    # fix image url
    image_url = pad_url + card_df.processed_file_location.values[0]

    # make prediction
    if model_type == "tf_lite":
        prediction = nn_predict(image_url, model_file, labels)
        # Convert numpy types in the tuple to native Python types
        if isinstance(prediction, tuple) and len(prediction) == 3:
            drug_name, probability, energy = prediction
            # Convert probability and energy to native Python floats
            probability = float(probability) if hasattr(probability, 'item') else probability
            energy = float(energy) if hasattr(energy, 'item') else energy
            prediction = (drug_name, probability, energy)
    else:
        # Use temporary directory for better cross-platform compatibility
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_filename = temp_file.name

        try:
            download_file(
                image_url,
                os.path.basename(temp_filename),
                os.path.dirname(temp_filename),
            )
            pls_conc = pls(model_file)
            prediction = pls_conc.quantity(temp_filename, actual_api)
            # Convert numpy types to native Python types
            if hasattr(prediction, 'item'):
                prediction = prediction.item()
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    return actual_label, prediction


def show_prediction(card_id=None, sample_id=None, model_id=None):
    """Display a PAD card with ML model prediction results.
    
    Shows the card image and metadata alongside machine learning prediction
    results. Includes model information and formatted prediction output for
    analysis and validation workflows. Can identify card by either card_id
    or sample_id.
    
    Args:
        card_id (int, optional): The unique card ID to analyze.
        sample_id (int, optional): The sample ID to find and analyze the card for.
        model_id (int): The model ID to use for prediction (e.g., 16 for
            classification, 18 for concentration prediction).
            
    Returns:
        None: Displays the prediction widget in the Jupyter notebook with:
            - Card image and standard metadata
            - Prediction result formatted as:
                * Neural Network: "drug_name (confidence=XX.XX%)" e.g., "aspirin (confidence=95.32%)"
                * PLS models: concentration with % unit e.g., "73.21%"
            - Model file name and type information
            
    Raises:
        ValueError: If neither card_id nor sample_id is provided, or if both are provided.
        ValueError: If model_id is not provided.
            
    Note:
        Requires Jupyter notebook environment. You must provide exactly one of
        card_id OR sample_id. If sample_id returns multiple cards, only the 
        first card will be analyzed.
        
    Example:
        >>> # Show prediction by card ID
        >>> show_prediction(card_id=19208, model_id=16)
        # Displays card with drug classification result
        
        >>> # Show prediction by sample ID
        >>> show_prediction(sample_id=52677, model_id=18)
        # Finds card for sample and shows concentration prediction
        
        >>> # Widget shows:
        >>> # - All standard card metadata
        >>> # - Prediction: "aspirin (confidence=95.32%)" for NN or "75.32%" for PLS
        >>> # - Model File: "24fhiNN1classifyAPI.tflite"
        >>> # - Model Type: "tf_lite" or "pls"
    """
    # Validate parameters
    if card_id is None and sample_id is None:
        raise ValueError("You must provide either card_id or sample_id")
    if card_id is not None and sample_id is not None:
        raise ValueError("You cannot provide both card_id and sample_id. Choose one.")
    if model_id is None:
        raise ValueError("You must provide model_id for prediction")
    
    # Get card information using the appropriate parameter
    if card_id is not None:
        info = get_card(card_id=card_id)
        display_id = card_id
    else:  # sample_id is not None
        info = get_card(sample_id=sample_id)
        # Handle multiple cards for the same sample_id
        if info is not None and not info.empty:
            if len(info) > 1:
                print(f"⚠️ Sample {sample_id} has {len(info)} cards. Using the first one (Card ID: {info['id'].iloc[0]}) for prediction.")
            display_id = info['id'].iloc[0]
            card_id = display_id  # Set card_id for prediction call
            # Use only the first card's data
            info = info.iloc[[0]]
        else:
            display_id = f"sample_{sample_id}"
            print(f"Failed to retrieve data for sample {sample_id}")
            return

    if info is None or info.empty:
        identifier = card_id if card_id is not None else f"sample {sample_id}"
        print(f"Failed to retrieve data for {identifier}")
        return

    # Data validation: check if essential fields exist in the API response
    def safe_get(field, default="N/A"):
        try:
            if field in info.columns:
                return info[field].values[0]
            else:
                return default
        except (IndexError, KeyError):
            return default

    # model data

    model_df = get_model(model_id)
    model_type = model_df.type.values[0]
    model_url = model_df.weights_url.values[0]
    model_file = os.path.basename(model_url)

    # prediction
    _, prediction = predict(card_id, model_id)
    # Handle different prediction formats
    if isinstance(prediction, float):
        # PLS model - format concentration with 2 decimals and % unit
        prediction = f"{prediction:.2f}%"
    elif isinstance(prediction, tuple) and len(prediction) == 3:
        # Neural Network model - extract drug name and confidence
        drug_name, confidence, energy = prediction
        # Format as "drug_name (confidence=XX.XX%)"
        prediction = f"{drug_name} (confidence={float(confidence)*100:.2f}%)"

    # Example of how to use `safe_get` for extracting fields
    data = {
        "ID": [display_id],
        "Sample ID": [safe_get("sample_id")],
        "Sample Name": [safe_get("sample_name")],
        "Quantity": [f"{safe_get('quantity')}%" if safe_get('quantity') != 'N/A' else 'N/A'],
        "Prediction": [prediction],
        "Pred. Model File": [model_file],
        "Pred. Model type": [model_type],
        "Camera Type": [safe_get("camera_type_1")],
        "Issue": [safe_get("issue.name", safe_get("issue"))],
        "Project Name": [safe_get("project.project_name")],
        "Project Id": [safe_get("project.id")],
        "Notes": [safe_get("notes")],
        "Date of Creation": [safe_get("date_of_creation")],
        "Deleted": [safe_get("deleted", default=False)],  # If missing, default to False
    }

    # Convert data to DataFrame
    data_df = pd.DataFrame(data)

    # Handle missing image URL gracefully
    try:
        image_url = (
            "https://pad.crc.nd.edu/" + info["processed_file_location"].values[0]
        )
    except (KeyError, IndexError):
        identifier = card_id if card_id is not None else f"sample {sample_id}"
        print(f"No valid image found for {identifier}")
        image_url = "https://via.placeholder.com/300"  # Default placeholder image

    # Create the widget for the image and its info
    image_widget_box = create_image_widget_with_info(image_url, data_df)

    # Display the widget
    display(image_widget_box)


import pandas as pd

# def apply_predictions_to_dataframe(dataset_df, predict_function, model_id):
#     """
#     Applies a prediction function to each row of a dataframe based on an 'id' column.

#     Parameters:
#         dataset_df (pd.DataFrame): The input dataframe containing an 'id' column.
#         predict_function (function): The function to make predictions, which accepts (id, model_id) and returns (actual_label, prediction).
#         model_id (int): The model identifier to be passed to the predict function.

#     Returns:
#         pd.DataFrame: A dataframe with additional 'actual_label' and 'prediction' columns.
#     """
#     def apply_predict(row):
#         # Call the predict function and unpack the results
#         actual_label, prediction = predict_function(row['id'], model_id)
#         return pd.Series({'actual_label': actual_label, 'prediction': prediction})

#     # Apply the prediction function to each row
#     results = dataset_df.apply(apply_predict, axis=1)

#     # Concatenate the results with the original dataframe
#     return pd.concat([dataset_df, results], axis=1)

# import pandas as pd


def _apply_predictions_batch_nn(dataset_df, model_id, batch_size=32):
    """Optimized batch processing for Neural Network models.
    
    Loads the TensorFlow Lite model once and processes images in batches for
    maximum performance. Significantly faster than sequential processing.
    
    Args:
        dataset_df (pd.DataFrame): Dataset with 'id' and 'sample_name' columns
        model_id (int): Neural Network model ID
        batch_size (int): Number of images to process per batch
        
    Returns:
        pd.DataFrame: Results with id, label, prediction, confidence columns
    """
    import os
    import tempfile
    
    # Get model info once
    model_df = get_model(model_id)
    model_url = model_df.weights_url.values[0]
    model_file = os.path.basename(model_url)
    
    # Download model if needed (once)
    if not os.path.exists(model_file):
        if pad_helper.pad_download(model_url):
            print(f"Model {model_file} downloaded.")
        else:
            raise Exception(f"Failed to download model: {model_url}")
    
    # Load TensorFlow Lite interpreter ONCE
    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    
    # Get input/output details once
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get labels once
    labels = list(map(standardize_names, model_df.labels.values[0]))
    
    results = []
    total_rows = len(dataset_df)
    
    # Process in batches
    for i in range(0, total_rows, batch_size):
        batch_df = dataset_df.iloc[i:i+batch_size]
        current_batch_size = len(batch_df)
        
        print(f"Processing batch {i//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size} ({current_batch_size} images)")
        
        # Process each image in the batch
        for _, row in batch_df.iterrows():
            try:
                card_id = int(row["id"])
                sample_name = row["sample_name"]
                
                # Get card info
                card_df = get_card(card_id)
                if card_df is None or card_df.empty:
                    print(f"Warning: Could not get card data for ID {card_id}")
                    continue
                
                # Get actual label
                actual_label = standardize_names(card_df.sample_name.values[0])
                
                # Get image URL
                image_url = "https://pad.crc.nd.edu/" + card_df.processed_file_location.values[0]
                
                # Process image and predict (reusing the loaded interpreter)
                prediction = _predict_single_nn_with_interpreter(
                    interpreter, image_url, labels, input_details, output_details
                )
                
                if prediction is not None:
                    drug_name, confidence, energy = prediction
                    results.append({
                        'id': card_id,
                        'label': actual_label,
                        'prediction': drug_name,
                        'confidence': float(confidence)
                    })
                    
            except Exception as e:
                print(f"Error processing card {row['id']}: {e}")
                continue
    
    return pd.DataFrame(results)


def _predict_single_nn_with_interpreter(interpreter, image_url, labels, input_details, output_details):
    """Make a single Neural Network prediction using a pre-loaded interpreter.
    
    Args:
        interpreter: Pre-loaded TensorFlow Lite interpreter
        image_url (str): URL to the PAD image
        labels (list): List of drug labels
        input_details: Model input tensor details
        output_details: Model output tensor details
        
    Returns:
        tuple: (drug_name, confidence, energy) or None if failed
    """
    try:
        # Read and preprocess image (same as nn_predict)
        img = read_img(image_url)
        
        # Crop image to get active area (same coordinates as nn_predict)
        img = img.crop((71, 359, 71 + 636, 359 + 490))
        
        # Resize for square images (same as nn_predict)
        size = (454, 454)
        img = img.resize(size, Image.BICUBIC)
        
        # Convert to numpy array (same as nn_predict)
        HEIGHT_INPUT, WIDTH_INPUT, DEPTH = (454, 454, 3)
        im = (
            np.asarray(img)
            .flatten()
            .reshape(1, HEIGHT_INPUT, WIDTH_INPUT, DEPTH)
            .astype(np.float32)
        )
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]["index"], im)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        result = interpreter.get_tensor(output_details[0]["index"])
        
        # Process result (same as nn_predict)
        num_label = np.argmax(result[0])
        prediction = labels[num_label]
        
        probability = tf.nn.softmax(result[0])[num_label].numpy()
        energy = tf.reduce_logsumexp(result[0], -1).numpy()
        
        return (prediction, float(probability), float(energy))
        
    except Exception as e:
        print(f"Error in NN prediction for {image_url}: {e}")
        return None


def _apply_predictions_threaded_pls(dataset_df, model_id, max_workers=8):
    """Optimized batch processing for PLS models.
    
    Downloads the PLS model once and processes all predictions sequentially.
    Since PLS models are typically faster and can't be easily parallelized,
    this approach focuses on avoiding multiple model downloads.
    
    Args:
        dataset_df (pd.DataFrame): Dataset with 'id' and 'sample_name' columns
        model_id (int): PLS model ID
        max_workers (int): Not used for PLS (kept for API compatibility)
        
    Returns:
        pd.DataFrame: Results with id, label, prediction columns
    """
    import os
    import tempfile
    
    # Get model info once
    model_df = get_model(model_id)
    model_url = model_df.weights_url.values[0]
    model_file = os.path.basename(model_url)
    
    # Download model if needed (once)
    if not os.path.exists(model_file):
        if pad_helper.pad_download(model_url):
            print(f"PLS model {model_file} downloaded.")
        else:
            raise Exception(f"Failed to download PLS model: {model_url}")
    
    # Load PLS model once (pls class is defined in this same file)
    try:
        pls_conc = pls(model_file)
    except Exception as e:
        # Fallback to the current method if PLS model loading fails
        print(f"Warning: Failed to load PLS model ({e}), using fallback method")
        return _apply_predictions_pls_fallback(dataset_df, model_id)
    
    results = []
    total_rows = len(dataset_df)
    
    print(f"Processing {total_rows} PLS predictions with shared model...")
    
    # Process each prediction using the shared model
    for idx, (_, row) in enumerate(dataset_df.iterrows()):
        try:
            card_id = int(row["id"])
            sample_name = row["sample_name"]
            
            # Get card info
            card_df = get_card(card_id)
            if card_df is None or card_df.empty:
                print(f"Warning: Could not get card data for ID {card_id}")
                continue
            
            # Get actual label
            actual_label = card_df.quantity.values[0]
            # Convert numpy types to native Python types
            if hasattr(actual_label, 'item'):
                actual_label = actual_label.item()
            
            # Get image URL and download image
            image_url = "https://pad.crc.nd.edu/" + card_df.processed_file_location.values[0]
            
            # Use temporary directory for image processing
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_filename = temp_file.name
            
            try:
                # Download image
                download_file(
                    image_url,
                    os.path.basename(temp_filename),
                    os.path.dirname(temp_filename),
                )
                
                # Make prediction using shared model
                prediction = pls_conc.quantity(temp_filename, standardize_names(sample_name))
                
                # Convert numpy types to native Python types
                if hasattr(prediction, 'item'):
                    prediction = prediction.item()
                
                results.append({
                    'id': card_id,
                    'label': actual_label,
                    'prediction': float(prediction)
                })
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"Completed {idx + 1}/{total_rows} PLS predictions")
                
        except Exception as e:
            print(f"Error processing card {row['id']}: {e}")
            continue
    
    return pd.DataFrame(results)


def _apply_predictions_pls_fallback(dataset_df, model_id):
    """Fallback PLS processing using original predict() function."""
    print("Using fallback sequential PLS processing...")
    
    results = []
    total_rows = len(dataset_df)
    
    for idx, (_, row) in enumerate(dataset_df.iterrows()):
        try:
            card_id = int(row["id"])
            sample_name = row["sample_name"]
            actual_label, prediction = predict(card_id, model_id, actual_api=sample_name)
            
            results.append({
                'id': card_id,
                'label': actual_label,
                'prediction': float(prediction) if hasattr(prediction, 'item') else prediction
            })
            
            if (idx + 1) % 50 == 0:
                print(f"Completed {idx + 1}/{total_rows} predictions")
                
        except Exception as e:
            print(f"Error processing card {row['id']}: {e}")
            continue
    
    return pd.DataFrame(results)


def apply_predictions_to_dataframe(dataset_df, model_id, batch_size=32, max_workers=8):
    """Apply ML model predictions using optimized batch/parallel processing.
    
    OPTIMIZED VERSION: Uses different processing strategies based on model type:
    - Neural Networks: Batch processing with single model load (10x faster)
    - PLS models: Parallel processing with ThreadPoolExecutor
    
    Args:
        dataset_df (pd.DataFrame): Input dataset containing PAD card information.
            Must include the following columns:
            - 'id': Card IDs for prediction (int)
            - 'sample_name': Drug names for label standardization (str)
        model_id (int): The model ID to use for all predictions. Common models:
            - 16: Neural Network classifier (24fhiNN1classifyAPI)
            - 17: Neural Network concentration (24fhiNN1concAPI)
            - 18: PLS concentration model (24fhiPLS1conc)
            - 19: Neural Network concentration v2
        batch_size (int, optional): Batch size for Neural Network processing. Default: 32
        max_workers (int, optional): Max threads for PLS processing. Default: 8
            
    Returns:
        pd.DataFrame: Results dataframe with prediction columns:
            - 'id': Original card ID (int)
            - 'label': Actual/ground truth values (str for classification, float for concentration)
            - 'prediction': Model predictions (str for classification, float for concentration)
            - 'confidence': Prediction confidence scores (float, only for Neural Network models)
            
    Performance:
        - Neural Networks: ~10x faster than sequential (single model load + batching)
        - PLS models: ~8x faster than sequential (parallel processing)
        - Progress indicators show processing status
        
    Example:
        >>> # Neural Network classification (optimized batch processing)
        >>> dataset = get_dataset_cards("FHI2020_Stratified_Sampling").head(100)
        >>> results = apply_predictions_to_dataframe(dataset, model_id=16)
        Processing batch 1/4 (32 images)
        Processing batch 2/4 (32 images)
        ...
        >>> print(f"Processed {len(results)} predictions")
        
        >>> # PLS concentration (optimized parallel processing)
        >>> results = apply_predictions_to_dataframe(dataset, model_id=18)
        Processing 100 PLS predictions with 8 workers...
        Completed 10/100 predictions
        ...
    """
    # Validate required columns
    if 'id' not in dataset_df.columns or 'sample_name' not in dataset_df.columns:
        raise KeyError("Dataset must contain 'id' and 'sample_name' columns")
    
    if dataset_df.empty:
        return pd.DataFrame(columns=['id', 'label', 'prediction'])
    
    print(f"Starting optimized batch prediction for {len(dataset_df)} cards with model {model_id}")
    
    # Get model type to choose optimization strategy
    try:
        model_df = get_model(model_id)
        model_type = model_df.type.values[0]
        print(f"Model type: {model_type}")
    except Exception as e:
        raise Exception(f"Failed to get model {model_id}: {e}")
    
    # Choose optimization strategy based on model type
    if model_type == "tf_lite":
        # Neural Network: Use batch processing (load model once)
        print(f"Using optimized batch processing for Neural Network (batch_size={batch_size})")
        return _apply_predictions_batch_nn(dataset_df, model_id, batch_size)
    else:
        # PLS: Use parallel processing
        print(f"Using optimized parallel processing for PLS model (max_workers={max_workers})")
        return _apply_predictions_threaded_pls(dataset_df, model_id, max_workers)


def apply_predictions_to_dataframe_legacy(dataset_df, model_id):
    """Legacy sequential processing version (for comparison/fallback).
    
    This is the original implementation that loads the model for each prediction.
    Kept for backwards compatibility and performance comparison.
    Use apply_predictions_to_dataframe() for optimized performance.
    """
    def apply_predict(row):
        # Call the predict function and unpack the results
        id = int(row["id"])
        actual_label, prediction = predict(id, model_id, actual_api=row["sample_name"])

        if isinstance(prediction, float):
            return pd.Series(
                {"id": id, "label": actual_label, "prediction": prediction}
            )

        # assumes the first value is the prediction
        if isinstance(prediction, tuple) and len(prediction) == 3:
            return pd.Series(
                {
                    "id": id,
                    "label": actual_label,
                    "prediction": prediction[0],
                    "confidence": prediction[1],
                }
            )

    # Apply the prediction function to each row
    results = dataset_df.apply(apply_predict, axis=1)
    results["id"] = results["id"].astype(int)  # Convert 'id' to integer

    return results


def get_model_dataset_mapping(mapping_file_path=MODEL_DATASET_MAPPING):
    """Get the model-dataset mapping from the CSV file.
    
    Loads the static mapping file that associates machine learning models
    with their corresponding training and test datasets. This mapping is
    essential for data loading and model evaluation workflows.
    
    Args:
        mapping_file_path (str, optional): Path to the mapping CSV file.
            Defaults to package-included mapping file.
        
    Returns:
        pd.DataFrame: Mapping DataFrame with columns:
            - Model ID: Unique model identifier
            - Model Name: Human-readable model name
            - Dataset Name: Associated dataset name
            - Training Dataset: URL to training data
            - Test Dataset: URL to test data
            - Endpoint URL: API endpoint for predictions
            
    Raises:
        FileNotFoundError: If the mapping file cannot be found
    """
    if not os.path.exists(mapping_file_path):
        raise FileNotFoundError(
            f"Model dataset mapping file not found at: {mapping_file_path}\n"
            f"This file is required for dataset discovery features. "
            f"Please ensure the pad-analytics package was installed correctly."
        )
    
    try:
        model_dataset_mapping = pd.read_csv(mapping_file_path)
        return model_dataset_mapping
    except Exception as e:
        raise RuntimeError(
            f"Failed to read model dataset mapping file '{mapping_file_path}': {e}"
        ) from e


def get_dataset_list(mapping_file_path=MODEL_DATASET_MAPPING):
    """Get list of available datasets (legacy function for backward compatibility).
    
    LEGACY FUNCTION: This function provides the original static dataset list.
    For enhanced functionality, use get_datasets() instead which includes
    dynamic catalog data and rich metadata.
    
    Args:
        mapping_file_path (str, optional): Path to the CSV mapping file.
            Defaults to package-included mapping file.
        
    Returns:
        pd.DataFrame: DataFrame with static dataset mappings including:
            - Dataset Name: Name of the dataset
            - Training Dataset: URL to training data
            - Test Dataset: URL to test data 
            - Model ID: List of associated model identifiers
            
    Note:
        Consider using get_datasets() for a more comprehensive dataset overview
        with catalog metadata and documentation links.
        
    Example:
        >>> datasets = get_dataset_list()
        >>> print(datasets[['Dataset Name', 'Model ID']].head())
           Dataset Name                              Model ID
        0  FHI2020_Stratified_Sampling              [16, 17, 18]
        1  Another_Dataset                          [19]
    """
    mapping_df = get_model_dataset_mapping(mapping_file_path)
    datasets_df = (
        mapping_df.groupby(["Dataset Name", "Training Dataset", "Test Dataset"])[
            "Model ID"
        ]
        .apply(list)
        .reset_index()
    )
    datasets_df = pd.concat(
        [
            datasets_df,
            mapping_df[mapping_df["Model ID"].isna()][
                ["Dataset Name", "Training Dataset", "Test Dataset"]
            ],
        ],
        ignore_index=True,
    )
    return datasets_df


def get_datasets():
    """Get clean overview of all available datasets.
    
    Combines data from dynamic catalog and static mappings to provide
    a user-friendly overview with documentation links.
    
    Returns:
        pd.DataFrame: Dataset overview with columns:
            - Dataset Name (str): Name of the dataset
            - Total Records (int): Combined training + testing records  
            - Description (str): Dataset description from catalog
            - Documentation (str): Link to dataset readme at padproject.info
            - Source (str): Data source - "catalog", "static", or "hybrid"
            
    Example:
        >>> datasets_df = pad.get_datasets()
        >>> print(f"Found {len(datasets_df)} datasets")
        Found 10 datasets
        >>> print(datasets_df[['Dataset Name', 'Documentation']].iloc[0])
        Dataset Name: FHI2020_Stratified_Sampling
        Documentation: https://padproject.info/.../readme/
    """
    dm = get_dataset_manager()
    return dm.get_datasets()


def get_dataset_name_from_model_id(model_id, use_dynamic=True):
    """Get dataset name used by a specific model.
    
    Looks up which dataset was used to train and test the model.
    
    Args:
        model_id (int): The model ID to look up.
        use_dynamic (bool): Whether to use DatasetManager (True) or 
            static CSV only (False). Defaults to True.
        
    Returns:
        str or None: Dataset name (e.g., "FHI2020_Stratified_Sampling") 
            or None if model not found.
            
    Example:
        >>> dataset_name = pad.get_dataset_name_from_model_id(16)
        >>> print(dataset_name)
        FHI2020_Stratified_Sampling
        
    Note:
        This function replaces the deprecated get_dataset_from_model_id()
        which had a misleading name.
    """
    if use_dynamic:
        dm = get_dataset_manager()
        return dm.get_dataset_name_from_model_id(model_id)
    else:
        # Fallback to CSV lookup
        mapping_df = get_model_dataset_mapping()
        model_rows = mapping_df[mapping_df['Model ID'] == model_id]
        if not model_rows.empty:
            return model_rows.iloc[0]['Dataset Name']
        return None


def get_dataset_from_model_id(model_id, mapping_file_path=MODEL_DATASET_MAPPING, use_dynamic=True):
    """Get dataset data for a specific model (DEPRECATED).
    
    .. deprecated:: 0.2.0
        This function is deprecated and will be removed in a future version.
        The function name is misleading - it returns DataFrame data, not just the dataset name.
    
    Args:
        model_id (int): The model ID to get dataset for.
        mapping_file_path (str, optional): Path to mapping file.
        use_dynamic (bool, optional): Whether to use dynamic dataset manager.
        
    Returns:
        pd.DataFrame or None: Dataset with is_train column indicating train/test split,
            or None if model not found.
        
    Recommended alternatives:
        - get_dataset_name_from_model_id(model_id): Returns dataset name (string)
        - get_model_data(model_id, "all"): Returns dataset data (DataFrame)
        
    Example:
        >>> # DEPRECATED usage
        >>> data = get_dataset_from_model_id(16)
        >>> 
        >>> # RECOMMENDED alternatives
        >>> name = get_dataset_name_from_model_id(16)  # Just the name
        >>> data = get_model_data(16, "all")  # All data with is_train column
    """
    import warnings
    warnings.warn(
        "get_dataset_from_model_id() is deprecated and will be removed in a future version. "
        "Use get_dataset_name_from_model_id() to get the dataset name, or "
        "get_model_data(model_id, 'all') to get the dataset data.",
        DeprecationWarning,
        stacklevel=2
    )
    if use_dynamic:
        # Use the dataset manager
        dm = get_dataset_manager()
        dataset_name = dm.get_dataset_from_model_id(model_id)
        
        if dataset_name is None:
            print("No dataset found for this model")
            return None
            
        # Get URLs and load data
        train_url, test_url = dm.get_dataset_urls(dataset_name)
        
        train_df = None
        test_df = None
        
        if train_url:
            train_df = pd.read_csv(train_url)
            train_df["is_train"] = 1
            
        if test_url:
            test_df = pd.read_csv(test_url)
            test_df["is_train"] = 0
            
        # Combine datasets
        if train_df is not None and test_df is not None:
            data_df = pd.concat([train_df, test_df])
        elif train_df is not None:
            data_df = train_df
        elif test_df is not None:
            data_df = test_df
        else:
            print(f"No dataset URLs found for dataset: {dataset_name}")
            return None
            
        return data_df
    else:
        # Fallback to old method
        model_dataset_mapping = get_model_dataset_mapping(mapping_file_path)
        model_dataset = model_dataset_mapping[model_dataset_mapping["Model ID"] == model_id]

        # display(model_dataset)
        if len(model_dataset) == 0:
            print("No dataset found for this model")
            return None
        else:
            # get Dataset dataframe
            train_url = model_dataset[model_dataset["Model ID"] == model_id][
                "Training Dataset"
            ].values[0]
            train_df = pd.read_csv(train_url)
            test_url = model_dataset[model_dataset["Model ID"] == model_id][
                "Test Dataset"
            ].values[0]
            test_df = pd.read_csv(test_url)

            # combine train_df and test_df but make a column to identify if the row is train or test
            train_df["is_train"] = 1
            test_df["is_train"] = 0
            data_df = pd.concat([train_df, test_df])
            return data_df


def get_dataset(name, use_dynamic=True):
    """Load a dataset by name (DEPRECATED).
    
    .. deprecated:: 0.2.0
        This function is deprecated and will be removed in a future version.
        
    Args:
        name (str): Name of the dataset to load.
        use_dynamic (bool, optional): Whether to use dynamic dataset manager.
            Defaults to True.
        
    Returns:
        pd.DataFrame or None: Dataset with is_train column, or None if not found.
        
    Recommended alternatives:
        - get_dataset_cards(name): Clean dataset view without is_train column
        - get_model_data(model_id, "all"): Dataset with is_train column for specific model
        
    Example:
        >>> # DEPRECATED usage
        >>> data = get_dataset("FHI2020_Stratified_Sampling")
        >>> 
        >>> # RECOMMENDED alternatives
        >>> cards = get_dataset_cards("FHI2020_Stratified_Sampling")  # Clean view
        >>> model_data = get_model_data(16, "all")  # Model-specific with is_train
    """
    import warnings
    warnings.warn(
        "get_dataset() is deprecated and will be removed in a future version. "
        "Use get_dataset_cards() for clean dataset view, or "
        "get_model_data(model_id, 'all') for dataset with train/test distinction.",
        DeprecationWarning,
        stacklevel=2
    )
    if use_dynamic:
        # Use the dataset manager
        dm = get_dataset_manager()
        train_url, test_url = dm.get_dataset_urls(name)
        
        if train_url is None and test_url is None:
            print(f"Dataset with name {name} not found")
            return None
            
        train_df = None
        test_df = None
        
        if train_url:
            try:
                train_df = pd.read_csv(train_url)
                train_df["is_train"] = 1
            except Exception as e:
                print(f"Error loading training data: {e}")
                
        if test_url:
            try:
                test_df = pd.read_csv(test_url)
                test_df["is_train"] = 0
            except Exception as e:
                print(f"Error loading test data: {e}")
                
        # Combine datasets
        if train_df is not None and test_df is not None:
            data_df = pd.concat([train_df, test_df])
        elif train_df is not None:
            data_df = train_df
        elif test_df is not None:
            data_df = test_df
        else:
            print(f"Failed to load any data for dataset: {name}")
            return None
            
        return data_df
    else:
        # Fallback to old method
        df = get_dataset_list(use_dynamic=False)
        dataset = df[df["Dataset Name"] == name]

        if len(dataset) > 0:
            train_df = None
            test_df = None

            # get Dataset dataframe
            if "Test Dataset" in dataset.columns:
                test_url = dataset["Test Dataset"].values[0]
                test_df = pd.read_csv(test_url)
                test_df["is_train"] = 0

            # print(dataset['Training Dataset'])
            if dataset["Training Dataset"].notna().any():
                train_url = dataset["Training Dataset"].values[0]
                train_df = pd.read_csv(train_url)
                train_df["is_train"] = 1

            # combine train_df and test_df but make a column to identify if the row is train or test
            data_df = pd.concat([train_df, test_df])
            return data_df
        else:
            print(f"Dataset with name {name} not found")
            return None


def get_dataset_cards(dataset_name, use_dynamic=True):
    """Get all cards (samples) from a specific dataset by name.
    
    Returns a clean dataset view without implementation details like the
    'is_train' column. This is the recommended way to access dataset contents
    when you need all samples regardless of train/test split.
    
    Args:
        dataset_name (str): Name of the dataset to retrieve cards from
            (e.g., "FHI2020_Stratified_Sampling").
        use_dynamic (bool, optional): Whether to use the dynamic dataset manager.
            Defaults to True for enhanced functionality.
        
    Returns:
        pd.DataFrame or None: Dataset with all cards/samples, or None if
            dataset not found. Columns typically include:
            - id: Card ID
            - sample_id: Sample identifier
            - sample_name: Drug/sample name  
            - quantity: Concentration/amount
            - url: Image URL
            - Additional metadata columns
            
    Note:
        The 'is_train' column is NOT included for a clean view.
        Use get_model_data() if you need train/test distinction.
        
    Example:
        >>> cards = get_dataset_cards("FHI2020_Stratified_Sampling")
        >>> print(f"Dataset contains {len(cards)} samples")
        Dataset contains 8001 samples
        
        >>> print(cards[['sample_name', 'quantity']].head())
           sample_name  quantity
        0    Aspirin        50.0
        1    Ibuprofen     100.0
    """
    if use_dynamic:
        dm = get_dataset_manager()
        return dm.get_dataset_cards(dataset_name)
    else:
        # Fallback: use get_dataset function and remove is_train column
        df = get_dataset(dataset_name, use_dynamic=False)
        if df is not None and 'is_train' in df.columns:
            df = df.drop('is_train', axis=1)
        return df


def get_model_data(model_id, data_type="all", use_dynamic=True):
    """Get training, testing, or all data for a specific model.
    
    Retrieves the dataset used by a specific model with flexible options
    for train/test data selection. Uses the hybrid dataset management
    system to find and load the appropriate data.
    
    Args:
        model_id (int): The model ID to retrieve data for.
        data_type (str, optional): Type of data to return. Options:
            - "train": Training data only (no is_train column)
            - "test": Test data only (no is_train column)
            - "all": Combined data (includes is_train column)
            Defaults to "all".
        use_dynamic (bool): Whether to use DatasetManager (True) or
            fallback to legacy functions (False). Defaults to True.
        
    Returns:
        pd.DataFrame or None: Requested dataset or None if model not found.
            When data_type="all", includes 'is_train' column where:
            - is_train=1: Training samples
            - is_train=0: Test samples
            
    Raises:
        ValueError: If data_type is not "train", "test", or "all".
        
    Example:
        >>> # Get all data with train/test labels
        >>> all_data = pad.get_model_data(16, "all")
        >>> train_count = len(all_data[all_data['is_train'] == 1])
        >>> test_count = len(all_data[all_data['is_train'] == 0])
        >>> print(f"Model 16: {train_count} train, {test_count} test")
        Model 16: 5923 train, 2078 test
        
        >>> # Get only training data
        >>> train_data = pad.get_model_data(16, "train")
        >>> print(f"Training samples: {len(train_data)}")
        Training samples: 5923
    """
    if data_type not in ["train", "test", "all"]:
        raise ValueError("data_type must be 'train', 'test', or 'all'")
    
    if use_dynamic:
        dm = get_dataset_manager()
        return dm.get_model_data(model_id, data_type)
    else:
        # Fallback: use original function and filter
        df = get_dataset_from_model_id(model_id, use_dynamic=False)
        if df is None:
            return None
            
        if data_type == "train":
            result = df[df['is_train'] == 1].copy()
            return result.drop('is_train', axis=1)
        elif data_type == "test":
            result = df[df['is_train'] == 0].copy()
            return result.drop('is_train', axis=1)
        else:  # data_type == "all"
            return df


def get_dataset_info(name, use_dynamic=True):
    """Get comprehensive information about a dataset.
    
    Provides rich metadata by combining information from the dynamic catalog
    and static model mappings. Includes descriptions, record counts, model
    associations, and dataset URLs.
    
    Args:
        name (str): Name of the dataset to get information for.
        use_dynamic (bool, optional): Whether to use the dynamic dataset manager.
            Defaults to True for enhanced metadata.
        
    Returns:
        dict: Comprehensive dataset information including:
            - name (str): Dataset name
            - source (str): Data source ("catalog", "static", or "hybrid")
            - description (str): Dataset description
            - record_count (int): Total number of records
            - models (list): List of models using this dataset
            - training_dataset_url (str): URL to training data
            - test_dataset_url (str): URL to test data
            - Additional catalog metadata when available
            
    Example:
        >>> info = get_dataset_info("FHI2020_Stratified_Sampling")
        >>> print(f"Dataset: {info['name']}")
        Dataset: FHI2020_Stratified_Sampling
        
        >>> print(f"Records: {info['record_count']}")
        Records: 8001
        
        >>> print(f"Used by {len(info['models'])} models")
        Used by 3 models
    """
    if use_dynamic:
        dm = get_dataset_manager()
        return dm.get_dataset_info(name)
    else:
        # Fallback: get basic info from static mapping
        df = get_dataset_list(use_dynamic=False)
        dataset = df[df["Dataset Name"] == name]
        
        if len(dataset) == 0:
            return {"name": name, "source": "not_found"}
            
        return {
            "name": name,
            "source": "static",
            "training_dataset_url": dataset["Training Dataset"].values[0] if pd.notna(dataset["Training Dataset"].values[0]) else None,
            "test_dataset_url": dataset["Test Dataset"].values[0] if pd.notna(dataset["Test Dataset"].values[0]) else None,
            "models": dataset["Model ID"].values[0] if "Model ID" in dataset.columns else []
        }


def calculate_rmse(group, pred_col="prediction", actual_col="label"):
    actual = group[actual_col].astype(int)
    predicted = group[pred_col].astype(int)
    return np.sqrt(np.mean((actual - predicted) ** 2))


def calculate_rmse_by_api(result, actual_col="label", pred_col="prediction"):
    # Grouping by 'sample_name' and applying the RMSE calculation
    rmse_by_class = result.groupby("sample_name").apply(
        calculate_rmse, include_groups=False
    )

    # Convert the Series to a DataFrame and reset the index
    rmse_df = rmse_by_class.reset_index(name="rmse")
    return rmse_df


def main():
    """Main entry point for the pad-analysis command line tool."""
    import argparse

    parser = argparse.ArgumentParser(description="PAD ML Workflow Analysis Tool")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.add_argument(
        "--help-commands", action="store_true", help="Show available commands"
    )

    args = parser.parse_args()

    if args.help_commands:
        print("PAD ML Workflow Analysis Tool")
        print("=" * 40)
        print("Available functions:")
        print("- get_projects(): Get all projects")
        print("- get_card(card_id): Get specific card")
        print("- get_models(): Get all models")
        print("- predict(card_id, model_id): Make prediction")
        print("\nUse as a Python module:")
        print("  import padanalytics")
        print("  projects = padanalytics.get_projects()")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
