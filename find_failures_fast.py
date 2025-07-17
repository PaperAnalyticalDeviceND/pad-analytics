#!/usr/bin/env python3
import pad_analytics as pad
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

print("Aggressively searching for random_state with at least 3 failures...")
print("Testing smaller batches first to find promising candidates\n")

# Get test data
test_data = pad.get_model_data(16, data_type="test")
print(f"Test data size: {len(test_data)} cards\n")

# Phase 1: Quick scan with tiny samples
print("Phase 1: Quick scan with 20-sample batches...")
promising_states = []

for rs in range(1, 200, 3):  # Test every 3rd number up to 200
    try:
        # Test with just 20 samples for speed
        tiny_sample = test_data.sample(20, random_state=rs)
        tiny_results = pad.apply_predictions_to_dataframe(tiny_sample, model_id=16, batch_size=20)
        tiny_failed = tiny_results[tiny_results["label"] != tiny_results["prediction"]]
        
        if len(tiny_failed) >= 1:  # If even 1 failure in 20, it's promising
            promising_states.append(rs)
            print(f"  Promising: random_state={rs} ({len(tiny_failed)} failures in 20 samples)")
            
        if len(promising_states) >= 20:  # Found enough candidates
            break
            
    except:
        pass

print(f"\nPhase 2: Testing {len(promising_states)} promising states with 200 samples...")

# Phase 2: Test promising states with full 200 samples
best_states = []
for rs in promising_states[:10]:  # Test top 10 most promising
    try:
        sample = test_data.sample(200, random_state=rs)
        results = pad.apply_predictions_to_dataframe(sample, model_id=16, batch_size=64)
        failed = results[results["label"] != results["prediction"]]
        num_failures = len(failed)
        
        if num_failures >= 3:
            accuracy = (200 - num_failures) / 200 * 100
            best_states.append((rs, num_failures, accuracy))
            print(f"✓ random_state={rs}: {num_failures} failures (accuracy: {accuracy:.1f}%)")
            
            # Show confusion details
            if num_failures >= 5:
                print(f"  Confusion examples:")
                for idx in failed.index[:3]:
                    row = failed.loc[idx]
                    print(f"    {row['label']} → {row['prediction']} (conf: {row['confidence']:.1%})")
                    
    except Exception as e:
        print(f"Error with state {rs}: {str(e)[:50]}")

if best_states:
    print(f"\n=== RESULTS: Found {len(best_states)} suitable random states ===")
    best_states.sort(key=lambda x: x[1], reverse=True)
    print("\nBest options:")
    for rs, failures, acc in best_states[:5]:
        print(f"  random_state={rs}: {failures} failures (accuracy: {acc:.1f}%)")
    
    print(f"\n🎯 RECOMMENDED: random_state={best_states[0][0]} with {best_states[0][1]} failures")
else:
    print("\n❌ No suitable random states found in this range")