"""
Cache utilities for handling API calls when padanalytics is not available
"""

import requests
import pandas as pd
from typing import Optional
import warnings

def get_card_direct(card_id: int) -> Optional[pd.DataFrame]:
    """
    Direct API call to get card data when padanalytics is not available.
    
    Args:
        card_id: The PAD card ID
        
    Returns:
        DataFrame with card information or None if failed
    """
    try:
        # Direct API call to get card data
        url = f"https://pad.crc.nd.edu/api/v2/cards/{card_id}"
        
        response = requests.get(url, verify=False, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to DataFrame format similar to padanalytics
        card_data = {
            'id': [data.get('id', card_id)],
            'sample_name': [data.get('sample_name', '')],
            'sample_id': [data.get('sample_id', 0)],
            'quantity': [data.get('quantity', None)],
            'processed_file_location': [data.get('processed_file_location', '')]
        }
        
        return pd.DataFrame(card_data)
        
    except Exception as e:
        warnings.warn(f"Failed to get card {card_id} via direct API: {e}")
        return None