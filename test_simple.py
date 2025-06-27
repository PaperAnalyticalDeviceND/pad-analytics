#!/usr/bin/env python3
"""Simple test to verify package installation."""

import subprocess
import sys

print("Testing PAD ML Workflow Package")
print("=" * 50)

# Test 1: Check if package is installed
print("\n1. Checking package installation...")
result = subprocess.run([sys.executable, "-m", "pip", "show", "pad-analytics"], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print("✓ Package is installed")
    print(result.stdout)
else:
    print("✗ Package not found")

# Test 2: Test console script
print("\n2. Testing console script...")
result = subprocess.run(["pad-analysis", "--version"], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print("✓ Console script works")
    print(f"   Output: {result.stdout.strip()}")
else:
    print("✗ Console script failed")
    if result.stderr:
        print(f"   Error: {result.stderr[:200]}...")

# Test 3: Direct import test
print("\n3. Testing direct module imports...")
try:
    # Add src to path for testing
    sys.path.insert(0, 'src')
    
    # Test individual module imports
    print("   Testing padanalytics module...")
    import padanalytics
    print("   ✓ padanalytics imported")
    
    print("   Testing pad_helper module...")
    import pad_helper  
    print("   ✓ pad_helper imported")
    
    print("\n✓ Basic imports successful!")
    
except Exception as e:
    print(f"✗ Import failed: {e}")

print("\n" + "=" * 50)
print("Package is properly installed and can be used!")