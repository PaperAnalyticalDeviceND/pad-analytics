#!/usr/bin/env python3
"""
PAD Analytics Phase 4 - Advanced Features Demo

This script demonstrates the Phase 4 advanced features:
- Batch processing optimizations
- Async/parallel processing capabilities
- Performance monitoring and metrics
- Configuration management
- Smart caching strategies
- Comprehensive error handling

Usage:
    python examples/phase4_advanced_features_demo.py
"""

import sys
import os
from pathlib import Path
import time
import asyncio

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pad_analytics as pad


async def main():
    """Main demo function."""
    print("=" * 80)
    print("PAD Analytics Phase 4 - Advanced Features Demo")
    print("=" * 80)
    
    # Test 1: Configuration Management
    print("\n1. 🔧 Configuration Management System")
    print("-" * 60)
    
    try:
        # Get global configuration
        config = pad.get_config()
        print(f"✅ Global configuration loaded")
        print(f"   • Cache enabled: {config.cache.enabled}")
        print(f"   • Performance monitoring: {config.performance.enabled}")
        print(f"   • Batch size: {config.preprocessing.batch_size}")
        print(f"   • Parallel processing: {config.preprocessing.parallel_processing}")
        
        # Show configuration summary
        config_mgr = pad.get_global_config_manager()
        summary = config_mgr.get_config_summary()
        print(f"\n📊 Configuration Summary:")
        for section, settings in summary.items():
            print(f"   {section}:")
            for key, value in settings.items():
                print(f"     • {key}: {value}")
        
        # Update some configuration
        print(f"\n🔄 Updating configuration...")
        pad.update_global_config({
            'preprocessing': {'batch_size': 64, 'num_workers': 8},
            'performance': {'detailed_monitoring': True}
        })
        
        updated_config = pad.get_config()
        print(f"   • New batch size: {updated_config.preprocessing.batch_size}")
        print(f"   • New num workers: {updated_config.preprocessing.num_workers}")
        
    except Exception as e:
        print(f"❌ Configuration management failed: {e}")
    
    # Test 2: Performance Monitoring Setup
    print("\n2. 📊 Performance Monitoring System")
    print("-" * 60)
    
    try:
        # Get performance monitor
        monitor = pad.get_global_monitor()
        print(f"✅ Performance monitor initialized")
        
        # Show system information
        system_info = pad.get_system_info()
        print(f"💻 System Information:")
        print(f"   • CPU cores: {system_info.get('cpu_count', 'unknown')}")
        print(f"   • CPU usage: {system_info.get('cpu_percent', 'unknown')}%")
        print(f"   • Memory total: {system_info.get('memory_total_gb', 'unknown'):.1f} GB")
        print(f"   • Memory available: {system_info.get('memory_available_gb', 'unknown'):.1f} GB")
        
        # Test performance monitoring context manager
        with monitor.monitor_operation("demo_operation", batch_size=10) as op_id:
            # Simulate some work
            time.sleep(0.1)
            
        # Get metrics summary
        metrics_summary = monitor.get_metrics_summary()
        if metrics_summary:
            print(f"📈 Performance Metrics:")
            print(f"   • Total operations: {metrics_summary.get('total_operations', 0)}")
            print(f"   • Average duration: {metrics_summary.get('duration_stats', {}).get('avg', 0):.3f}s")
            print(f"   • Memory usage: {metrics_summary.get('memory_stats', {}).get('avg_mb', 0):.1f} MB")
        
    except Exception as e:
        print(f"❌ Performance monitoring failed: {e}")
    
    # Test 3: Advanced Model Adapter with Performance Monitoring
    print("\n3. 🎯 Advanced Model Adapter Features")
    print("-" * 60)
    
    try:
        # Create model adapter with configuration
        config = pad.get_config()
        cache_mgr = pad.CacheManager() if config.cache.enabled else None
        
        adapter = pad.ModelAdapter(model_id=16, cache_manager=cache_mgr, auto_load=False)
        print(f"✅ Model adapter created: {adapter}")
        
        # Load model with performance monitoring
        print(f"🔄 Loading model with monitoring...")
        load_success = adapter.load_model()
        print(f"   Model loaded: {load_success}")
        
        if load_success:
            model_summary = adapter.get_model_summary()
            print(f"📝 Model Summary:")
            print(f"   • Type: {model_summary.get('model_type', 'unknown')}")
            print(f"   • Labels: {len(model_summary.get('labels', []))} classes")
            print(f"   • Input shape: {model_summary.get('input_shape', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Model adapter setup failed: {e}")
        print("   This is expected if TensorFlow or models are not available")
        adapter = None
    
    # Test 4: Optimized Batch Processing
    print("\n4. ⚡ Optimized Batch Processing")
    print("-" * 60)
    
    try:
        # Load a small dataset for testing
        dataset = pad.CachedDataset("FHI2020_Stratified_Sampling")
        metadata = dataset.load_dataset_metadata()
        
        print(f"📦 Dataset loaded: {len(metadata)} cards")
        
        # Get test data
        test_cards = metadata.head(5).to_dict('records')
        print(f"🧪 Using {len(test_cards)} cards for testing")
        
        if adapter and adapter.is_loaded():
            # Test sequential vs parallel processing
            print(f"\n🔄 Testing batch processing modes...")
            
            # Sequential processing
            start_time = time.time()
            sequential_results = adapter.predict_batch(test_cards, parallel=False)
            sequential_time = time.time() - start_time
            
            print(f"   • Sequential: {sequential_time:.3f}s ({len(sequential_results)} predictions)")
            
            # Parallel processing
            start_time = time.time() 
            parallel_results = adapter.predict_batch(test_cards, parallel=True)
            parallel_time = time.time() - start_time
            
            print(f"   • Parallel: {parallel_time:.3f}s ({len(parallel_results)} predictions)")
            
            # Compare results
            if sequential_results and parallel_results:
                results_match = sequential_results == parallel_results
                print(f"   • Results match: {results_match}")
                
        else:
            print("⏭️  Batch processing skipped (model not loaded)")
            
    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
    
    # Test 5: Asynchronous Processing
    print("\n5. 🔄 Asynchronous Processing")
    print("-" * 60)
    
    try:
        if adapter and adapter.is_loaded():
            # Test async single prediction
            print(f"🔄 Testing async single prediction...")
            start_time = time.time()
            async_result = await adapter.predict_async(test_cards[0])
            async_time = time.time() - start_time
            
            print(f"   • Async prediction: {async_time:.3f}s")
            print(f"   • Result: {async_result}")
            
            # Test async batch prediction
            print(f"🔄 Testing async batch prediction...")
            start_time = time.time()
            async_batch_results = await adapter.predict_batch_async(test_cards[:3])
            async_batch_time = time.time() - start_time
            
            print(f"   • Async batch: {async_batch_time:.3f}s ({len(async_batch_results)} predictions)")
            
            # Test async dataset prediction with progress
            def progress_callback(batch_num, total_batches, batch_size):
                print(f"     Progress: Batch {batch_num}/{total_batches} ({batch_size} items)")
            
            print(f"🔄 Testing async dataset prediction with progress...")
            start_time = time.time()
            async_dataset_results = await adapter.predict_dataset_async(
                metadata.head(10), 
                batch_size=3,
                progress_callback=progress_callback
            )
            async_dataset_time = time.time() - start_time
            
            print(f"   • Async dataset: {async_dataset_time:.3f}s ({len(async_dataset_results)} total)")
            
        else:
            print("⏭️  Async processing skipped (model not loaded)")
            
    except Exception as e:
        print(f"❌ Async processing test failed: {e}")
    
    # Test 6: Performance Metrics Analysis
    print("\n6. 📈 Performance Metrics Analysis")
    print("-" * 60)
    
    try:
        monitor = pad.get_global_monitor()
        
        # Get comprehensive metrics summary
        full_summary = monitor.get_metrics_summary()
        
        if full_summary:
            print(f"📊 Comprehensive Performance Analysis:")
            print(f"   • Total operations: {full_summary.get('total_operations', 0)}")
            print(f"   • Error rate: {full_summary.get('error_rate', 0):.1%}")
            
            duration_stats = full_summary.get('duration_stats', {})
            print(f"   • Duration stats:")
            print(f"     - Min: {duration_stats.get('min', 0):.3f}s")
            print(f"     - Max: {duration_stats.get('max', 0):.3f}s") 
            print(f"     - Avg: {duration_stats.get('avg', 0):.3f}s")
            print(f"     - Total: {duration_stats.get('total', 0):.3f}s")
            
            memory_stats = full_summary.get('memory_stats', {})
            print(f"   • Memory stats:")
            print(f"     - Min: {memory_stats.get('min_mb', 0):.1f} MB")
            print(f"     - Max: {memory_stats.get('max_mb', 0):.1f} MB")
            print(f"     - Avg: {memory_stats.get('avg_mb', 0):.1f} MB")
            
            throughput_stats = full_summary.get('throughput_stats', {})
            if throughput_stats:
                print(f"   • Throughput stats:")
                print(f"     - Min: {throughput_stats.get('min_items_per_sec', 0):.1f} items/sec")
                print(f"     - Max: {throughput_stats.get('max_items_per_sec', 0):.1f} items/sec")
                print(f"     - Avg: {throughput_stats.get('avg_items_per_sec', 0):.1f} items/sec")
            
            operation_breakdown = full_summary.get('operation_breakdown', {})
            if operation_breakdown:
                print(f"   • Operation breakdown:")
                for op_name, count in operation_breakdown.items():
                    print(f"     - {op_name}: {count} times")
        else:
            print("📊 No performance metrics available yet")
    
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")
    
    # Test 7: Cache Performance Analysis
    print("\n7. 💾 Cache Performance Analysis")
    print("-" * 60)
    
    try:
        if cache_mgr:
            cache_stats = cache_mgr.get_cache_stats()
            print(f"💾 Cache Statistics:")
            print(f"   • Images cached: {cache_stats.get('num_images', 0)}")
            print(f"   • Metadata cached: {cache_stats.get('num_metadata', 0)}")
            print(f"   • Datasets cached: {cache_stats.get('num_datasets', 0)}")
            print(f"   • Models cached: {cache_stats.get('num_models', 0)}")
            print(f"   • Preprocessed cached: {cache_stats.get('num_preprocessed', 0)}")
            print(f"   • Total cache size: {cache_stats.get('total_size_mb', 0):.1f} MB")
            
            # Show cache directory structure
            cache_info = cache_mgr.get_cache_info()
            print(f"📁 Cache Directory: {cache_info.get('cache_dir', 'unknown')}")
            print(f"   • Cache enabled: {cache_info.get('enabled', False)}")
            print(f"   • Auto cleanup: {cache_info.get('auto_cleanup', False)}")
        else:
            print("💾 Cache manager not available")
    
    except Exception as e:
        print(f"❌ Cache analysis failed: {e}")
    
    # Test 8: Configuration Export
    print("\n8. 💾 Configuration and Metrics Export")
    print("-" * 60)
    
    try:
        config_mgr = pad.get_global_config_manager()
        monitor = pad.get_global_monitor()
        
        # Export configuration
        config_path = Path("pad_analytics_config_demo.yml")
        config_mgr.save_config(config_path, format='yaml')
        print(f"✅ Configuration exported to: {config_path}")
        
        # Export performance metrics
        metrics_path = Path("pad_analytics_metrics_demo.json")
        monitor.export_metrics(metrics_path)
        print(f"✅ Performance metrics exported to: {metrics_path}")
        
        # Show file sizes
        if config_path.exists():
            config_size = config_path.stat().st_size
            print(f"   • Config file size: {config_size} bytes")
        
        if metrics_path.exists():
            metrics_size = metrics_path.stat().st_size
            print(f"   • Metrics file size: {metrics_size} bytes")
            
    except Exception as e:
        print(f"❌ Export failed: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("🎉 Phase 4 Advanced Features Demo Complete!")
    print("=" * 80)
    
    print("\n📝 Features Demonstrated:")
    print("   ✅ Configuration management system")
    print("   ✅ Performance monitoring and metrics")
    print("   ✅ Advanced model adapter with auto-detection")
    print("   ✅ Optimized batch processing (sequential vs parallel)")
    print("   ✅ Asynchronous processing capabilities")
    print("   ✅ Progress tracking for long operations")
    print("   ✅ Comprehensive performance analysis")
    print("   ✅ Cache performance monitoring")
    print("   ✅ Configuration and metrics export")
    
    print("\n🚀 Phase 4 Advanced Features Summary:")
    print("   • ⚡ Batch Processing: Vectorized operations, chunked processing")
    print("   • 🔄 Async/Parallel: ThreadPoolExecutor, async/await support")
    print("   • 📊 Monitoring: Timing, memory, CPU, throughput metrics")
    print("   • 🔧 Configuration: YAML/JSON files, environment variables")
    print("   • 💾 Smart Caching: Hierarchical caching with TTL")
    print("   • 🛡️  Error Handling: Graceful degradation, retry logic")
    
    print("\n💡 Advanced Usage Examples:")
    print("   • adapter.predict_batch(cards, parallel=True, max_workers=8)")
    print("   • await adapter.predict_dataset_async(dataset, progress_callback=callback)")
    print("   • pad.update_global_config({'preprocessing': {'batch_size': 128}})")
    print("   • monitor.export_metrics('metrics.json', operation_filter='predict')")
    print("   • config_manager.save_config('my_config.yml')")


if __name__ == "__main__":
    asyncio.run(main())