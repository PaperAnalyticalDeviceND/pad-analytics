# Quick Phase 1 Testing Instructions

## Option A: Run the Full Test Suite
```bash
python test_phase1_manual.py
```
This will guide you through all tests step by step.

## Option B: Interactive Python Testing

Open a Python interpreter in your project directory:

```bash
cd /mnt/slow_data/TAI/Users/pmoreira/pad-ml-workflow-v2
python3
```

Then run these commands one by one:

### Step 1: Basic Import Test
```python
import sys
sys.path.insert(0, 'src')
import pad_analytics as pad

# Check version and imports
print(f"Version: {pad.__version__}")
print(f"CacheManager available: {hasattr(pad, 'CacheManager')}")
print(f"CachedDataset available: {hasattr(pad, 'CachedDataset')}")
```

### Step 2: Create Cache Manager
```python
cache_mgr = pad.CacheManager()
print(f"Cache directory: {cache_mgr.cache_dir}")

# Check cache stats
stats = cache_mgr.get_cache_stats()
print(f"Cache stats: {stats}")
```

### Step 3: Test Dataset Caching
```python
# Create cached dataset
dataset = pad.CachedDataset("FHI2020_Stratified_Sampling")
print(f"Dataset created: {dataset}")

# Load metadata (uses cache if available)
metadata = dataset.load_dataset_metadata()
print(f"Loaded {len(metadata)} records")

# Check coverage
coverage = dataset.get_cache_coverage()
print(f"Cache coverage: {coverage}")
```

### Step 4: Test Small Image Caching (Optional)
```python
# Only if you want to test actual image downloads
# This requires internet and may take a few minutes
stats = dataset.download_and_cache_images(max_images=5, max_workers=2)
print(f"Caching results: {stats}")
```

## Option C: Use Existing Demo
```bash
python examples/caching_demo_simple.py
```

## What to Look For

### ✅ Success Indicators:
- No import errors
- Cache directory created in ~/.pad_cache
- Dataset loads from cache quickly (< 1 second)
- Coverage percentage shows cached data
- Image caching works without errors

### ❌ Common Issues:
- Import warnings about ipywidgets/tensorflow (NORMAL - caching still works)
- Network errors during image caching (check internet)
- Permission errors (check ~/.pad_cache permissions)

### 📊 Expected Results:
- Dataset: ~8000 records from FHI2020_Stratified_Sampling
- Cache size: ~1-2 MB initially
- Load time: Very fast from cache (< 0.1s)
- First image download: ~2-5 seconds per image
- Subsequent loads: Instant from cache

## Cache Location
Your cache is stored at: `~/.pad_cache/`
```bash
ls -la ~/.pad_cache/
# Should show: datasets/, raw_images/, metadata/, cache_info.json
```

## Reset Cache (if needed)
```python
cache_mgr = pad.CacheManager()
cache_mgr.clear_cache(confirm=True)  # Clears everything
```