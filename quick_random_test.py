#!/usr/bin/env python3
import pad_analytics as pad
import warnings
warnings.filterwarnings("ignore")

print("Testing specific random_state values for failed predictions...\n")

# Test specific random states that often produce errors
test_states = [42, 123, 7, 13, 99, 101, 256, 500, 1234, 2024]

test_data = pad.get_model_data(16, data_type="test")
print(f"Test data size: {len(test_data)} cards\n")

for random_state in test_states:
    try:
        sample = test_data.sample(200, random_state=random_state)
        results = pad.apply_predictions_to_dataframe(sample, model_id=16, batch_size=64)
        
        # Count failures
        failed = results[results["label"] != results["prediction"]]
        num_failures = len(failed)
        
        if num_failures >= 4:
            print(f"✓ random_state={random_state}: {num_failures} failures")
        else:
            print(f"  random_state={random_state}: {num_failures} failures (too few)")
            
    except Exception as e:
        print(f"✗ random_state={random_state}: Error")