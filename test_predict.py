#!/usr/bin/env python3
"""Test script to check the predict function output for card 19208 with model 18."""

import sys
import os

# Add src to path since we're testing the local code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Import the module
    import padanalytics as pad
    
    print("Testing predict function")
    print("=" * 50)
    print("Command: actual_label, prediction = pad.predict(19208, 18)")
    print()
    
    # Call the predict function
    actual_label, prediction = pad.predict(19208, 18)
    
    print("Results:")
    print("-" * 50)
    print(f"actual_label: {actual_label}")
    print(f"actual_label type: {type(actual_label)}")
    print()
    print(f"prediction: {prediction}")
    print(f"prediction type: {type(prediction)}")
    
    # Check if prediction is a tuple (neural network) or float (PLS)
    if isinstance(prediction, tuple):
        print("\nThis is a neural network model (tf_lite)")
        if len(prediction) == 3:
            print(f"  Predicted label: {prediction[0]}")
            print(f"  Probability: {prediction[1]}")
            print(f"  Energy: {prediction[2]}")
    elif isinstance(prediction, float):
        print("\nThis is a PLS model")
        print(f"  Concentration: {prediction}")
    else:
        print(f"\nUnexpected prediction type: {type(prediction)}")
    
    print("\n" + "=" * 50)
    print("Test completed successfully!")
    
except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)