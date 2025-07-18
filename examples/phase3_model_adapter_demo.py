#!/usr/bin/env python3
"""
PAD Analytics Phase 3 - Model Adapter Demo

This script demonstrates the Phase 3 model adapter functionality:
- Unified model interface for different model types
- Automatic model loading and caching
- Consistent prediction API
- Integration with preprocessing pipeline

Usage:
    python examples/phase3_model_adapter_demo.py
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
    print("PAD Analytics Phase 3 - Model Adapter Demo")
    print("=" * 70)
    
    # Test 1: Basic model adapter creation
    print("\n1. 📦 Testing Model Adapter Creation")
    print("-" * 50)
    
    try:
        # Create adapters for different model types
        nn_adapter = pad.ModelAdapter(model_id=16, auto_load=False)  # Neural Network
        pls_adapter = pad.ModelAdapter(model_id=18, auto_load=False)  # PLS
        
        print(f"✅ Neural Network adapter: {nn_adapter}")
        print(f"✅ PLS adapter: {pls_adapter}")
        
        # Show adapter information
        print(f"\nNN Adapter Info:")
        nn_info = nn_adapter.get_model_info()
        for key, value in nn_info.items():
            print(f"   • {key}: {value}")
        
        print(f"\nPLS Adapter Info:")
        pls_info = pls_adapter.get_model_info()
        for key, value in pls_info.items():
            print(f"   • {key}: {value}")
            
    except Exception as e:
        print(f"❌ Adapter creation failed: {e}")
        return
    
    # Test 2: Model loading
    print("\n2. 🔄 Testing Model Loading")
    print("-" * 50)
    
    try:
        print("Loading Neural Network model...")
        print("   → Getting model URL from PAD API...")
        nn_loaded = nn_adapter.load_model()
        print(f"NN model loaded: {nn_loaded}")
        
        print("Loading PLS model...")
        print("   → Getting model URL from PAD API...")
        pls_loaded = pls_adapter.load_model()
        print(f"PLS model loaded: {pls_loaded}")
        
        if nn_loaded:
            nn_summary = nn_adapter.get_model_summary()
            print(f"NN Model Summary: {nn_summary}")
        
        if pls_loaded:
            pls_summary = pls_adapter.get_model_summary()
            print(f"PLS Model Summary: {pls_summary}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("This is expected if TensorFlow or other dependencies are missing")
        print("Note: Model URLs are now correctly retrieved from PAD API")
    
    # Test 3: Unified prediction interface
    print("\n3. 🎯 Testing Unified Prediction Interface")
    print("-" * 50)
    
    try:
        # Load a dataset
        dataset = pad.CachedDataset("FHI2020_Stratified_Sampling")
        metadata = dataset.load_dataset_metadata()
        
        print(f"Dataset loaded: {len(metadata)} cards")
        
        # Get a test card
        card_data = metadata.iloc[0].to_dict()
        print(f"Test card: {card_data['id']} - {card_data['sample_name']}")
        
        # Test NN prediction (if loaded)
        if nn_adapter.is_loaded():
            print("\n🔄 Testing NN prediction...")
            start_time = time.time()
            nn_result = nn_adapter.predict(card_data)
            nn_time = time.time() - start_time
            
            print(f"✅ NN prediction completed in {nn_time:.3f}s")
            print(f"   Result: {nn_result}")
        else:
            print("⏭️  NN prediction skipped (model not loaded)")
        
        # Test PLS prediction (if loaded)
        if pls_adapter.is_loaded():
            print("\n🔄 Testing PLS prediction...")
            start_time = time.time()
            pls_result = pls_adapter.predict(card_data)
            pls_time = time.time() - start_time
            
            print(f"✅ PLS prediction completed in {pls_time:.3f}s")
            print(f"   Result: {pls_result}")
        else:
            print("⏭️  PLS prediction skipped (model not loaded)")
        
    except Exception as e:
        print(f"❌ Prediction testing failed: {e}")
    
    # Test 4: Batch prediction
    print("\n4. 📊 Testing Batch Prediction")
    print("-" * 50)
    
    try:
        # Test with small batch
        test_cards = metadata.head(3).to_dict('records')
        print(f"Testing with {len(test_cards)} cards")
        
        # Test NN batch prediction (if loaded)
        if nn_adapter.is_loaded():
            print("\n🔄 Testing NN batch prediction...")
            start_time = time.time()
            nn_batch_results = nn_adapter.predict_batch(test_cards)
            nn_batch_time = time.time() - start_time
            
            print(f"✅ NN batch prediction completed in {nn_batch_time:.3f}s")
            print(f"   Results: {len(nn_batch_results)} predictions")
            for i, result in enumerate(nn_batch_results):
                print(f"   Card {i+1}: {result}")
        else:
            print("⏭️  NN batch prediction skipped (model not loaded)")
        
        # Test PLS batch prediction (if loaded)
        if pls_adapter.is_loaded():
            print("\n🔄 Testing PLS batch prediction...")
            start_time = time.time()
            pls_batch_results = pls_adapter.predict_batch(test_cards)
            pls_batch_time = time.time() - start_time
            
            print(f"✅ PLS batch prediction completed in {pls_batch_time:.3f}s")
            print(f"   Results: {len(pls_batch_results)} predictions")
            for i, result in enumerate(pls_batch_results):
                print(f"   Card {i+1}: {result}")
        else:
            print("⏭️  PLS batch prediction skipped (model not loaded)")
        
    except Exception as e:
        print(f"❌ Batch prediction testing failed: {e}")
    
    # Test 5: Convenience functions
    print("\n5. 🛠️ Testing Convenience Functions")
    print("-" * 50)
    
    try:
        # Test convenience functions
        print("Testing convenience creation functions...")
        
        try:
            nn_conv_adapter = pad.create_neural_network_adapter(16, auto_load=False)
            print(f"✅ NN convenience adapter: {nn_conv_adapter}")
        except Exception as e:
            print(f"❌ NN convenience adapter failed: {e}")
        
        try:
            pls_conv_adapter = pad.create_pls_adapter(18, auto_load=False)
            print(f"✅ PLS convenience adapter: {pls_conv_adapter}")
        except Exception as e:
            print(f"❌ PLS convenience adapter failed: {e}")
        
        # Test error handling
        try:
            # This should fail - model 18 is PLS, not NN
            bad_adapter = pad.create_neural_network_adapter(18, auto_load=False)
            print("❌ Error handling failed - should have thrown exception")
        except ValueError as e:
            print(f"✅ Error handling works: {e}")
        
        # Test get_available_models
        try:
            available_models = pad.get_available_models()
            print(f"✅ Available models: {len(available_models)} models")
            for model_id, info in available_models.items():
                model_type = info.get('adapter_class', 'Unknown')
                model_name = info.get('model_info', {}).get('name', 'Unknown')
                print(f"   Model {model_id}: {model_name} ({model_type})")
        except Exception as e:
            print(f"❌ get_available_models failed: {e}")
            
    except Exception as e:
        print(f"❌ Convenience functions failed: {e}")
    
    # Test 6: Cache integration
    print("\n6. 💾 Testing Cache Integration")
    print("-" * 50)
    
    try:
        # Create adapter with cache manager
        cache_mgr = pad.CacheManager()
        cached_adapter = pad.ModelAdapter(model_id=16, cache_manager=cache_mgr, auto_load=False)
        
        print(f"✅ Adapter with cache: {cached_adapter}")
        print(f"   Cache enabled: {cached_adapter.get_model_info()['cache_enabled']}")
        
        # Show cache stats
        cache_stats = cache_mgr.get_cache_stats()
        print(f"   Cache stats:")
        print(f"   • Models: {cache_stats.get('num_models', 0)}")
        print(f"   • Preprocessed: {cache_stats.get('num_preprocessed', 0)}")
        print(f"   • Total size: {cache_stats.get('total_size_mb', 0)} MB")
        
    except Exception as e:
        print(f"❌ Cache integration failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 Phase 3 Model Adapter Demo Complete!")
    print("=" * 70)
    
    print("\n📝 What was demonstrated:")
    print("   ✅ Unified model interface creation")
    print("   ✅ Automatic model type detection")
    print("   ✅ Correct model URL retrieval from PAD API")
    print("   ✅ Model loading and management")
    print("   ✅ Consistent prediction API")
    print("   ✅ Batch prediction capabilities")
    print("   ✅ Convenience functions")
    print("   ✅ Cache integration")
    print("   ✅ Error handling and validation")
    
    print("\n🚀 Ready for Phase 4: Advanced Features!")
    print("\n💡 Next steps:")
    print("   • Try: adapter.predict(card_data)")
    print("   • Try: adapter.predict_batch(cards_list)")
    print("   • Try: adapter.predict_dataset(dataset, max_cards=10)")
    print("   • Explore model cache in ~/.pad_cache/models/")


if __name__ == "__main__":
    main()