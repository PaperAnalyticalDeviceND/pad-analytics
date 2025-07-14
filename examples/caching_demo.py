"""
PAD Analytics Data Caching Demo - Phase 1

This example demonstrates the new caching functionality that eliminates
redundant downloads and enables offline research workflows.

Features demonstrated:
- Automatic image caching and reuse
- Offline dataset preparation
- Performance improvements
- Cache management utilities
"""

import time
import pad_analytics as pad


def demo_basic_caching():
    """Demonstrate basic caching functionality."""
    print("=" * 60)
    print("🚀 PAD Analytics Data Caching Demo - Phase 1")
    print("=" * 60)
    
    # 1. Create a cached dataset
    print("\n1. Creating cached dataset...")
    dataset = pad.CachedDataset("FHI2020_Stratified_Sampling")
    print(f"   Dataset: {dataset}")
    
    # 2. Load dataset metadata (cached automatically)
    print("\n2. Loading dataset metadata...")
    start_time = time.time()
    dataset.load_dataset_metadata()
    load_time = time.time() - start_time
    print(f"   ✅ Loaded {len(dataset)} cards in {load_time:.2f}s")
    
    # 3. Check current cache coverage
    print("\n3. Checking cache coverage...")
    coverage = dataset.get_cache_coverage()
    print(f"   📊 Cache coverage: {coverage['estimated_coverage_percent']}%")
    print(f"   📦 Sample: {coverage['sample_cached']}/{coverage['sample_size']} images cached")
    
    return dataset


def demo_image_caching(dataset, max_images=20):
    """Demonstrate image caching with progress tracking."""
    print(f"\n4. Caching {max_images} images (for demo)...")
    
    # Download and cache images
    start_time = time.time()
    stats = dataset.download_and_cache_images(
        max_images=max_images,
        max_workers=4  # Moderate parallelism for demo
    )
    cache_time = time.time() - start_time
    
    print(f"\n   📈 Caching Performance:")
    print(f"   • Total time: {cache_time:.1f}s")
    print(f"   • New images cached: {stats['cached_new']}")
    print(f"   • Already cached: {stats['already_cached']}")
    
    if stats['cached_new'] > 0:
        avg_time = cache_time / stats['cached_new']
        print(f"   • Avg time per new image: {avg_time:.2f}s")


def demo_cached_predictions(dataset):
    """Demonstrate cache-aware predictions."""
    print("\n5. Testing cache-aware predictions...")
    
    # Get a small sample for testing
    sample_cards = dataset.dataset_df.head(5)
    
    print(f"   Testing predictions on {len(sample_cards)} cards...")
    
    for i, (_, card) in enumerate(sample_cards.iterrows()):
        card_id = int(card['id'])
        
        print(f"\n   Card {i+1}/{len(sample_cards)} (ID: {card_id}):")
        
        # Test cache-aware prediction
        start_time = time.time()
        try:
            actual, prediction = pad.predict_with_cache(
                card_id=card_id, 
                model_id=16,  # Neural Network classifier
                verbose=True
            )
            pred_time = time.time() - start_time
            
            if isinstance(prediction, tuple):
                drug, confidence, energy = prediction
                print(f"   ✅ Prediction: {drug} (confidence: {confidence*100:.1f}%)")
            else:
                print(f"   ✅ Prediction: {prediction}")
            
            print(f"   ⏱️  Prediction time: {pred_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


def demo_cache_management():
    """Demonstrate cache management utilities."""
    print("\n6. Cache Management...")
    
    # Get cache status
    status = pad.get_cache_status()
    print(f"   📊 Cache Status:")
    print(f"   • Directory: {status['cache_directory']}")
    print(f"   • Total size: {status['total_size_mb']:.1f} MB")
    print(f"   • Cached images: {status['num_cached_images']}")
    print(f"   • Cached datasets: {status['num_cached_datasets']}")
    print(f"   • Status: {status['status']}")
    
    # Cache manager for detailed stats
    cache_manager = pad.CacheManager()
    detailed_stats = cache_manager.get_cache_stats()
    
    if detailed_stats['oldest_entry']:
        import datetime
        oldest = datetime.datetime.fromtimestamp(detailed_stats['oldest_entry'])
        newest = datetime.datetime.fromtimestamp(detailed_stats['newest_entry'])
        print(f"   • Oldest entry: {oldest.strftime('%Y-%m-%d %H:%M')}")
        print(f"   • Newest entry: {newest.strftime('%Y-%m-%d %H:%M')}")


def demo_performance_comparison():
    """Demonstrate performance improvements with caching."""
    print("\n7. Performance Comparison (Simulated)...")
    
    print("   🐌 Without caching (traditional):")
    print("      • 10 predictions: ~20-30 seconds")
    print("      • 100 predictions: ~200-300 seconds") 
    print("      • Every run downloads images again")
    print("      • Requires internet connection")
    
    print("\n   🚀 With caching (Phase 1):")
    print("      • First run: Similar time (downloading + caching)")
    print("      • Subsequent runs: 50-80% faster")
    print("      • Offline capability after caching")
    print("      • No redundant downloads")


def demo_offline_workflow():
    """Demonstrate offline research workflow."""
    print("\n8. Offline Research Workflow...")
    
    print("   📋 Recommended workflow:")
    print("   1. Create cached dataset: dataset = pad.CachedDataset('my_dataset')")
    print("   2. Download all images: dataset.download_and_cache_images()")
    print("   3. Verify coverage: dataset.get_cache_coverage()")
    print("   4. Work offline: Use pad.predict_with_cache() or cached predictions")
    print("   5. Share cache: Copy ~/.pad_cache to collaborators")
    
    print("\n   ✅ Benefits:")
    print("   • Fast iteration on model development")
    print("   • Reproducible results (same images every time)")
    print("   • Field research capability (offline)")
    print("   • Reduced server load")


def main():
    """Run the complete caching demo."""
    try:
        # Basic setup and caching
        dataset = demo_basic_caching()
        
        # Image caching demo
        demo_image_caching(dataset, max_images=10)  # Small demo
        
        # Prediction testing
        demo_cached_predictions(dataset)
        
        # Cache management
        demo_cache_management()
        
        # Performance info
        demo_performance_comparison()
        
        # Workflow guidance
        demo_offline_workflow()
        
        print("\n" + "=" * 60)
        print("✅ Caching Demo Complete!")
        print("=" * 60)
        
        print("\nNext steps:")
        print("• Try: dataset.download_and_cache_images() for full dataset")
        print("• Try: pad.apply_predictions_to_dataframe_cached() for batch processing")
        print("• Try: cache_manager.cleanup_old_cache() for maintenance")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure you have internet connection and PAD API access")


if __name__ == "__main__":
    main()