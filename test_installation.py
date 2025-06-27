#!/usr/bin/env python3
"""Test script to verify package installation and functionality."""

import sys
import importlib
import subprocess

def test_package_import():
    """Test if the package can be imported."""
    print("Testing package imports...")
    try:
        # Import from src directly since we're using editable install
        import sys
        if '/mnt/slow_data/TAI/Users/pmoreira/pad-ml-workflow-v2/src' not in sys.path:
            sys.path.insert(0, '/mnt/slow_data/TAI/Users/pmoreira/pad-ml-workflow-v2/src')
        
        # Try importing the package
        try:
            import pad_ml_workflow
        except ImportError:
            # Fallback to direct import
            import src as pad_ml_workflow
            
        print("✓ Successfully imported pad_ml_workflow")
        print(f"  Version: {pad_ml_workflow.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import pad_ml_workflow: {e}")
        return False

def test_submodule_imports():
    """Test if submodules can be imported."""
    print("\nTesting submodule imports...")
    modules = [
        'pad_ml_workflow.pad_analysis',
        'pad_ml_workflow.padanalytics',
        'pad_ml_workflow.pad_helper',
        'pad_ml_workflow.fileManagement',
        'pad_ml_workflow.intensityFind',
        'pad_ml_workflow.pixelProcessing',
        'pad_ml_workflow.regionRoutine',
        'pad_ml_workflow.pls_model',
    ]
    
    success = True
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✓ Successfully imported {module}")
        except ImportError as e:
            print(f"✗ Failed to import {module}: {e}")
            success = False
    
    return success

def test_console_script():
    """Test if console script entry point works."""
    print("\nTesting console script...")
    try:
        result = subprocess.run(['pad-analysis', '--help'], 
                                capture_output=True, 
                                text=True)
        if result.returncode == 0:
            print("✓ Console script 'pad-analysis' is accessible")
            return True
        else:
            print(f"✗ Console script returned error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("✗ Console script 'pad-analysis' not found in PATH")
        return False

def test_package_info():
    """Test package metadata."""
    print("\nChecking package metadata...")
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'show', 'pad-ml-workflow'], 
                                capture_output=True, 
                                text=True)
        if result.returncode == 0:
            print("✓ Package metadata found:")
            print(result.stdout)
            return True
        else:
            print("✗ Package not installed via pip")
            return False
    except Exception as e:
        print(f"✗ Error checking package info: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("PAD ML Workflow Package Installation Test")
    print("=" * 60)
    
    tests = [
        test_package_import,
        test_submodule_imports,
        test_console_script,
        test_package_info,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Total tests: {len(results)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("\n✓ All tests passed! Package is properly installed.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())