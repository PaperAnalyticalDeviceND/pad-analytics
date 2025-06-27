#!/usr/bin/env python3
"""Test runner for PAD ML Workflow package."""

import subprocess
import sys
import os


def run_tests():
    """Run all tests using pytest."""
    print("Running PAD ML Workflow Tests")
    print("=" * 50)
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # Run pytest with coverage if available
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v",  # verbose output
            "--tb=short",  # shorter traceback format
            "--color=yes"  # colored output
        ], check=False)
        
        if result.returncode == 0:
            print("\n✓ All tests passed!")
        else:
            print("\n✗ Some tests failed!")
            return False
            
    except FileNotFoundError:
        print("pytest not found. Running basic import tests...")
        
        # Fallback to basic import tests
        try:
            sys.path.insert(0, 'src')
            
            print("Testing basic imports...")
            import padanalytics
            import regionRoutine
            import pixelProcessing
            import fileManagement
            
            print("✓ All basic imports successful!")
            
        except Exception as e:
            print(f"✗ Import test failed: {e}")
            return False
    
    return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)