#!/usr/bin/env python3
import pad_analytics as pad
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("Searching for random_state with at least 4 failed predictions...")
print("Testing model 16 on 200-sample batches\n")

# Get test data once
test_data = pad.get_model_data(16, data_type="test")
print(f"Total test data size: {len(test_data)} cards\n")

# Try different random states
found_states = []
for random_state in range(1, 100):  # Test first 100 random states
    try:
        # Sample 200 cards
        sample = test_data.sample(200, random_state=random_state)
        
        # Apply predictions
        results = pad.apply_predictions_to_dataframe(sample, model_id=16, batch_size=64)
        
        # Find failed predictions
        failed_results = results[results["label"] != results["prediction"]]
        num_failures = len(failed_results)
        
        if num_failures >= 4:
            accuracy = (len(results) - num_failures) / len(results) * 100
            found_states.append((random_state, num_failures, accuracy))
            print(f"✓ random_state={random_state}: {num_failures} failures (accuracy: {accuracy:.1f}%)")
            
            # If we found 10 good options, stop
            if len(found_states) >= 10:
                break
                
    except Exception as e:
        print(f"✗ random_state={random_state}: Error - {str(e)[:50]}...")
        continue

print(f"\nFound {len(found_states)} suitable random states")
if found_states:
    print("\nBest options (sorted by number of failures):")
    found_states.sort(key=lambda x: x[1], reverse=True)
    for state, failures, acc in found_states[:10]:
        print(f"  random_state={state}: {failures} failures (accuracy: {acc:.1f}%)")
        
    print(f"\nRecommended: random_state={found_states[0][0]} with {found_states[0][1]} failures")