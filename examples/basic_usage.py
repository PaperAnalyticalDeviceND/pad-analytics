#!/usr/bin/env python3
"""
Basic usage example for pad-analytics package.

This example demonstrates how to:
1. Connect to the PAD API
2. Explore available data
3. Make predictions using pre-trained models
"""

from pad_analytics import padanalytics as pad
import pandas as pd

def main():
    print("=== PAD Analytics Basic Usage Example ===\n")
    
    # 1. Get available projects
    print("1. Fetching available projects...")
    projects = pad.get_projects()
    if not projects.empty:
        print(f"   Found {len(projects)} projects")
        print(f"   First project: {projects.iloc[0]['name'] if 'name' in projects.columns else 'N/A'}")
    else:
        print("   No projects found or API unavailable")
    
    # 2. Get a specific card
    print("\n2. Fetching PAD card data...")
    card_id = 19208  # Example card ID
    try:
        card_data = pad.get_card(card_id)
        if not card_data.empty:
            print(f"   Card ID: {card_id}")
            print(f"   Sample: {card_data['sample_name'].values[0] if 'sample_name' in card_data.columns else 'N/A'}")
            print(f"   Concentration: {card_data['quantity'].values[0] if 'quantity' in card_data.columns else 'N/A'} µg/mL")
    except Exception as e:
        print(f"   Error fetching card: {e}")
    
    # 3. Make a prediction
    print("\n3. Making predictions...")
    model_id = 18  # PLS model for concentration
    try:
        actual, prediction = pad.predict(card_id, model_id)
        print(f"   Model ID: {model_id} (PLS)")
        print(f"   Actual concentration: {actual} µg/mL")
        print(f"   Predicted concentration: {prediction:.2f} µg/mL")
        print(f"   Error: {abs(actual - prediction):.2f} µg/mL")
    except Exception as e:
        print(f"   Error making prediction: {e}")
    
    # 4. Get available models
    print("\n4. Available models...")
    try:
        models = pad.get_models()
        if not models.empty:
            print(f"   Found {len(models)} models")
            for idx, model in models.head(3).iterrows():
                print(f"   - Model {model.get('id', 'N/A')}: {model.get('name', 'N/A')}")
    except Exception as e:
        print(f"   Error fetching models: {e}")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()