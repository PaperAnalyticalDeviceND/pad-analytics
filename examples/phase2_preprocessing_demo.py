#!/usr/bin/env python3
"""
PAD Analytics Phase 2 - Preprocessing Pipeline Demo

This script demonstrates the Phase 2 preprocessing pipeline functionality:
- Unified preprocessing interface for different model types
- Automatic model type detection and preprocessor selection
- Preprocessing cache integration
- Batch processing capabilities

Usage:
    python examples/phase2_preprocessing_demo.py
"""

import sys
import os
from pathlib import Path
import time

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pad_analytics as pad


def main():
    """Main demo function."""
    print("=" * 70)
    print("PAD Analytics Phase 2 - Preprocessing Pipeline Demo")
    print("=" * 70)
    
    # Test 1: Basic preprocessing pipeline creation
    print("\n1. 📦 Testing Preprocessing Pipeline Creation")
    print("-" * 50)
    
    try:
        # Create pipelines for different model types
        nn_pipeline = pad.PreprocessingPipeline(model_id=16)  # Neural Network
        pls_pipeline = pad.PreprocessingPipeline(model_id=18)  # PLS
        
        print(f"✅ Neural Network pipeline: {nn_pipeline}")
        print(f"✅ PLS pipeline: {pls_pipeline}")
        
        # Show pipeline information
        print(f"\nNN Pipeline Info:")
        nn_info = nn_pipeline.get_preprocessing_info()
        for key, value in nn_info.items():
            print(f"   • {key}: {value}")
        
        print(f"\nPLS Pipeline Info:")
        pls_info = pls_pipeline.get_preprocessing_info()
        for key, value in pls_info.items():
            print(f"   • {key}: {value}")
            
    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")
        return
    
    # Test 2: Feature name extraction
    print("\n2. 🔍 Testing Feature Names")
    print("-" * 50)
    
    try:
        nn_features = nn_pipeline.get_feature_names()
        pls_features = pls_pipeline.get_feature_names()
        
        print(f"NN features: {len(nn_features)} total")
        print(f"   First 5: {nn_features[:5]}")
        print(f"   Last 5: {nn_features[-5:]}")
        
        print(f"\nPLS features: {len(pls_features)} total")
        print(f"   First 5: {pls_features[:5]}")
        print(f"   Last 5: {pls_features[-5:]}")
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
    
    # Test 3: Dataset preprocessing
    print("\n3. 📊 Testing Dataset Preprocessing")
    print("-" * 50)
    
    try:
        # Load a dataset
        dataset = pad.CachedDataset("FHI2020_Stratified_Sampling")
        metadata = dataset.load_dataset_metadata()
        
        print(f"Dataset loaded: {len(metadata)} cards")
        
        # Test preprocessing with small batch
        print("\n🔄 Testing NN preprocessing (3 cards)...")
        start_time = time.time()
        
        # Get first 3 cards for testing
        test_cards = metadata.head(3).to_dict('records')
        
        # Preprocess with NN pipeline
        nn_result = nn_pipeline.preprocess_batch(test_cards)
        nn_time = time.time() - start_time
        
        print(f"✅ NN preprocessing completed in {nn_time:.3f}s")
        print(f"   Result shape: {nn_result.shape}")
        print(f"   Columns: {list(nn_result.columns)}")
        
        # Test PLS preprocessing
        print("\n🔄 Testing PLS preprocessing (3 cards)...")
        start_time = time.time()
        
        pls_result = pls_pipeline.preprocess_batch(test_cards)
        pls_time = time.time() - start_time
        
        print(f"✅ PLS preprocessing completed in {pls_time:.3f}s")
        print(f"   Result shape: {pls_result.shape}")
        print(f"   Columns: {list(pls_result.columns)}")
        
    except Exception as e:
        print(f"❌ Dataset preprocessing failed: {e}")
    
    # Test 4: Convenience functions
    print("\n4. 🛠️ Testing Convenience Functions")
    print("-" * 50)
    
    try:
        # Test convenience functions
        nn_conv_pipeline = pad.create_neural_network_pipeline(16)
        pls_conv_pipeline = pad.create_pls_pipeline(18)
        
        print(f"✅ NN convenience pipeline: {nn_conv_pipeline}")
        print(f"✅ PLS convenience pipeline: {pls_conv_pipeline}")
        
        # Test error handling
        try:
            # This should fail - model 18 is PLS, not NN
            bad_pipeline = pad.create_neural_network_pipeline(18)
            print("❌ Error handling failed - should have thrown exception")
        except ValueError as e:
            print(f"✅ Error handling works: {e}")
            
    except Exception as e:
        print(f"❌ Convenience functions failed: {e}")
    
    # Test 5: Configuration testing
    print("\n5. ⚙️ Testing Configuration")
    print("-" * 50)
    
    try:
        # Test configuration modification
        pipeline = pad.PreprocessingPipeline(model_id=16)
        
        # Show default config
        default_config = pipeline.get_config()
        print(f"Default config keys: {list(default_config.keys())}")
        
        # Update configuration
        custom_config = {
            'target_size': (224, 224),  # Different size
            'normalize': False
        }
        pipeline.set_config(custom_config)
        
        # Show updated config
        updated_config = pipeline.get_config()
        print(f"Updated target_size: {updated_config.get('target_size')}")
        print(f"Updated normalize: {updated_config.get('normalize')}")
        
    except Exception as e:
        print(f"❌ Configuration testing failed: {e}")
    
    # Test 6: Cache integration
    print("\n6. 💾 Testing Cache Integration")
    print("-" * 50)
    
    try:
        # Create pipeline with cache manager
        cache_mgr = pad.CacheManager()
        cached_pipeline = pad.PreprocessingPipeline(model_id=16, cache_manager=cache_mgr)
        
        print(f"✅ Pipeline with cache: {cached_pipeline}")
        print(f"   Cache enabled: {cached_pipeline.get_preprocessing_info()['cache_enabled']}")
        
        # Show cache stats
        cache_stats = cache_mgr.get_cache_stats()
        print(f"   Cache stats: {cache_stats['num_preprocessed']} preprocessed items")
        
    except Exception as e:
        print(f"❌ Cache integration failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 Phase 2 Preprocessing Pipeline Demo Complete!")
    print("=" * 70)
    
    print("\n📝 What was demonstrated:")
    print("   ✅ Automatic model type detection")
    print("   ✅ Unified preprocessing interface")
    print("   ✅ Neural Network and PLS preprocessing")
    print("   ✅ Batch processing capabilities")
    print("   ✅ Feature name extraction")
    print("   ✅ Configuration management")
    print("   ✅ Cache integration")
    print("   ✅ Error handling and validation")
    
    print("\n🚀 Ready for Phase 3: Model Adapter Interface!")
    print("\n💡 Next steps:")
    print("   • Try: pipeline.preprocess_dataset(dataset, max_images=10)")
    print("   • Try: pipeline.set_config({'normalize': False})")
    print("   • Try: pad.create_neural_network_pipeline(16)")
    print("   • Explore preprocessing cache in ~/.pad_cache/preprocessed/")


if __name__ == "__main__":
    main()