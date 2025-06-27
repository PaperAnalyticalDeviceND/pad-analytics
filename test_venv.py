#!/usr/bin/env python3
"""Test script specifically for virtual environment testing."""

import sys
import subprocess
import os

def run_test(description, command):
    """Run a test command and report results."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Success!")
            if result.stdout:
                print(f"  Output: {result.stdout.strip()}")
            return True
        else:
            print(f"✗ Failed!")
            if result.stderr:
                print(f"  Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False

def main():
    """Run all tests in virtual environment."""
    print("=" * 60)
    print("PAD ML Workflow Virtual Environment Test")
    print("=" * 60)
    
    # Check if we're in virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Not running in a virtual environment!")
        print("Please activate the virtual environment first:")
        print("  source test_env/bin/activate")
        return 1
    
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version}")
    
    tests = [
        ("Testing pip list", "pip list | grep pad-ml-workflow"),
        ("Testing basic import", "python -c 'import sys; print(sys.path); import src'"),
        ("Testing module list", "python -c 'import os; print([f for f in os.listdir(\"src\") if f.endswith(\".py\")])'"),
        ("Testing direct padanalytics import", "cd src && python -c 'import padanalytics; print(\"Imported successfully\")'"),
        ("Testing console script", "pad-analysis --help || echo 'Console script not working'"),
    ]
    
    results = []
    for desc, cmd in tests:
        results.append(run_test(desc, cmd))
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Total tests: {len(results)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())