#!/usr/bin/env python3
"""
PAD Analytics Data Caching Demo - Adapted for Enhanced API

This demonstrates the Phase 1 caching system working with the enhanced API functions.
"""

import time
import sys
import os

# Add src to path for local testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pad_analytics as pad


def demo_caching_with_enhanced_api():
    """Demonstrate caching integration with enhanced API."""
    print("=" * 60)
    print("🚀 PAD Analytics Data Caching - Enhanced API Integration")
    print("=" * 60)
    
    # 1. Test enhanced API directly
    print("\n1. Testing enhanced API...")
    start_time = time.time()
    dataset_cards = pad.get_dataset_cards("FHI2020_Stratified_Sampling")
    api_time = time.time() - start_time
    print(f"   ✅ Enhanced API: {len(dataset_cards)} cards in {api_time:.2f}s")
    
    # 2. Test caching system
    print("\n2. Testing caching system...")
    try:
        # First load (will cache)
        print("   📡 First load (caching)...")
        cached_dataset = pad.CachedDataset("FHI2020_Stratified_Sampling")
        start_time = time.time()
        metadata1 = cached_dataset.load_dataset_metadata()
        cache_time1 = time.time() - start_time
        print(f"   ✅ Cached: {len(metadata1)} cards in {cache_time1:.2f}s")
        
        # Second load (from cache)
        print("   💾 Second load (from cache)...")
        cached_dataset2 = pad.CachedDataset("FHI2020_Stratified_Sampling")
        start_time = time.time()
        metadata2 = cached_dataset2.load_dataset_metadata()
        cache_time2 = time.time() - start_time
        print(f"   ✅ From cache: {len(metadata2)} cards in {cache_time2:.2f}s")
        
        # Performance comparison
        if cache_time2 < cache_time1:
            speedup = cache_time1 / cache_time2
            print(f"   🚀 Cache speedup: {speedup:.1f}x faster!")
        
    except Exception as e:
        print(f"   ✗ Caching failed: {e}")
        return
    
    # 3. Cache management
    print("\n3. Cache management...")
    try:
        cache_mgr = pad.CacheManager()
        print(f"   📁 Cache directory: {cache_mgr.cache_dir}")
        
        # Check cache size
        cache_size = cache_mgr._get_cache_size_bytes() / (1024 * 1024)  # MB
        print(f"   💾 Cache size: {cache_size:.1f} MB")
        
    except Exception as e:
        print(f"   ⚠️  Cache management check failed: {e}")
    
    print("\n4. Benefits summary:")
    print("   ✅ Eliminates redundant downloads")
    print("   ✅ Enables offline research workflows") 
    print("   ✅ Faster iteration on cached datasets")
    print("   ✅ Works seamlessly with enhanced API")
    
    print(f"\n🎉 Phase 1 caching successfully integrated!")


if __name__ == "__main__":
    demo_caching_with_enhanced_api()