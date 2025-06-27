#!/usr/bin/env python3
"""
Batch prediction example for pad-analytics package.

This example shows how to:
1. Load a dataset
2. Apply predictions to multiple samples
3. Calculate performance metrics
"""

from pad_analytics import padanalytics as pad
import pandas as pd
import numpy as np

def main():
    print("=== PAD Analytics Batch Predictions Example ===\n")
    
    # 1. Load a dataset (example with synthetic data)
    print("1. Creating example dataset...")
    # In real usage, you would load from pad.get_dataset() or a CSV file
    example_data = pd.DataFrame({
        'card_id': [19208, 19209, 19210, 19211, 19212],
        'sample_name': ['Amoxicillin', 'Amoxicillin', 'Ciprofloxacin', 'Ciprofloxacin', 'Amoxicillin'],
        'quantity': [50.0, 80.0, 30.0, 100.0, 0.0],
        'api': ['Amoxicillin', 'Amoxicillin', 'Ciprofloxacin', 'Ciprofloxacin', 'Amoxicillin']
    })
    print(f"   Dataset has {len(example_data)} samples")
    print(f"   APIs: {example_data['api'].unique()}")
    
    # 2. Apply predictions to the dataset
    print("\n2. Applying model predictions...")
    model_id = 18  # PLS concentration model
    try:
        # Note: This will make API calls for each card_id
        # In real usage, ensure card_ids exist in the PAD database
        results = pad.apply_predictions_to_dataframe(example_data, model_id)
        
        if 'prediction' in results.columns:
            print("   Predictions added successfully")
            print("\n   Results preview:")
            print(results[['card_id', 'sample_name', 'quantity', 'prediction']].head())
    except Exception as e:
        print(f"   Error in batch prediction: {e}")
        # Create mock predictions for demonstration
        results = example_data.copy()
        results['prediction'] = results['quantity'] + np.random.normal(0, 5, len(results))
    
    # 3. Calculate RMSE by API
    print("\n3. Calculating performance metrics...")
    if 'prediction' in results.columns:
        try:
            rmse_results = pad.calculate_rmse_by_api(results)
            print("\n   RMSE by API:")
            for _, row in rmse_results.iterrows():
                print(f"   - {row['api']}: {row['rmse']:.2f} µg/mL")
            
            # Overall metrics
            overall_rmse = np.sqrt(np.mean((results['quantity'] - results['prediction'])**2))
            print(f"\n   Overall RMSE: {overall_rmse:.2f} µg/mL")
            
            # Mean Absolute Error
            mae = np.mean(np.abs(results['quantity'] - results['prediction']))
            print(f"   Mean Absolute Error: {mae:.2f} µg/mL")
            
        except Exception as e:
            print(f"   Error calculating metrics: {e}")
    
    # 4. Export results
    print("\n4. Exporting results...")
    output_file = "batch_prediction_results.csv"
    try:
        results.to_csv(output_file, index=False)
        print(f"   Results saved to: {output_file}")
    except Exception as e:
        print(f"   Error saving results: {e}")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()