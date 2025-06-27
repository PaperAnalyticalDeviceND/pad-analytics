#!/usr/bin/env python3
"""Test the predict function using the installed package."""

try:
    # Import from the installed package
    from pad_ml_workflow import padanalytics as pad
    
    print("Testing predict function with installed package")
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
    elif isinstance(prediction, (int, np.integer)):
        print("\nThis is a PLS model (returned as integer)")
        print(f"  Concentration: {prediction}")
    else:
        print(f"\nUnexpected prediction type: {type(prediction)}")
    
    print("\n" + "=" * 50)
    print("Test completed successfully!")
    
except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()