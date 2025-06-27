#!/usr/bin/env python3
"""Minimal test to check predict function by extracting just what we need."""

import os
import sys
import requests
import pandas as pd
import numpy as np

# Minimal implementation to test predict without importing the full module
API_URL = "https://pad.crc.nd.edu/api/v2"

def get_data_api(request_url, data_type=""):
    try:
        r = requests.get(url=request_url, verify=False)
        r.raise_for_status()
        data = r.json()
        df = pd.json_normalize(data)
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error accessing {data_type} data: {e}")
        return None

def get_card(card_id):
    request_url = f"{API_URL}/cards/{card_id}"
    return get_data_api(request_url, f"card {card_id}")

def get_model(model_id):
    request_url = f"{API_URL}/neural-networks/{model_id}"
    return get_data_api(request_url, f"neural_network {model_id}")

print("Testing card 19208 with model 18")
print("=" * 50)

# Get card info
print("\nFetching card 19208...")
card_df = get_card(19208)
if card_df is not None:
    print(f"Card sample_name: {card_df['sample_name'].values[0] if 'sample_name' in card_df.columns else 'N/A'}")
    print(f"Card quantity: {card_df['quantity'].values[0] if 'quantity' in card_df.columns else 'N/A'}")
else:
    print("Failed to fetch card data")

# Get model info
print("\nFetching model 18...")
model_df = get_model(18)
if model_df is not None:
    print(f"Model type: {model_df['type'].values[0] if 'type' in model_df.columns else 'N/A'}")
    print(f"Model labels: {model_df['labels'].values[0] if 'labels' in model_df.columns else 'N/A'}")
    print(f"Model weights_url: {model_df['weights_url'].values[0] if 'weights_url' in model_df.columns else 'N/A'}")
    
    # Determine what type of prediction this would be
    model_type = model_df['type'].values[0] if 'type' in model_df.columns else None
    
    print("\n" + "-" * 50)
    if model_type == 'tf_lite':
        print("This is a NEURAL NETWORK model")
        print("The prediction would return a tuple: (predicted_label, probability, energy)")
    else:
        print("This is a PLS model")
        print("The prediction would return a float value (concentration)")
else:
    print("Failed to fetch model data")

print("\n" + "=" * 50)