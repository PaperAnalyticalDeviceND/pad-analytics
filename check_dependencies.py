#!/usr/bin/env python3
"""
Check dependencies for PAD Analytics Phase 4 features
"""

import sys
import subprocess

def check_dependencies():
    print('🔍 Checking PAD Analytics Phase 4 Dependencies')
    print('=' * 50)

    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print('✅ Virtual environment detected')
        print(f'📁 Python path: {sys.prefix}')
    else:
        print('⚠️  No virtual environment detected (using system Python)')

    print(f'🐍 Python version: {sys.version}')
    print()

    # Required dependencies for Phase 4
    dependencies = {
        'Core Dependencies': [
            ('numpy', 'numpy'),
            ('pandas', 'pandas'), 
            ('requests', 'requests'),
            ('PIL', 'pillow'),
        ],
        'Phase 4 Advanced Features': [
            ('psutil', 'psutil'),        # Performance monitoring
            ('yaml', 'pyyaml'),          # Configuration management
        ],
        'Optional (for full functionality)': [
            ('tensorflow', 'tensorflow'),    # Neural network models
            ('cv2', 'opencv-python'),        # OpenCV - Image processing
            ('nest_asyncio', 'nest-asyncio'),  # Jupyter async support
        ],
        'Existing Dependencies': [
            ('matplotlib', 'matplotlib'),    # Visualization
            ('ipywidgets', 'ipywidgets'),    # Jupyter widgets
        ]
    }

    all_installed = True
    missing_deps = []

    for category, deps in dependencies.items():
        print(f'📦 {category}:')
        for import_name, pip_name in deps:
            try:
                if import_name == 'PIL':
                    import PIL
                elif import_name == 'yaml':
                    import yaml
                elif import_name == 'cv2':
                    import cv2
                else:
                    __import__(import_name)
                print(f'   ✅ {import_name}')
            except ImportError:
                print(f'   ❌ {import_name} - NOT INSTALLED')
                all_installed = False
                missing_deps.append(pip_name)
        print()

    print('📋 Summary:')
    if all_installed:
        print('🎉 All dependencies are installed!')
        print('✅ You can use all Phase 4 advanced features!')
    else:
        print(f'⚠️  Missing {len(missing_deps)} dependencies')
        print('📋 To install missing dependencies:')
        if missing_deps:
            print(f'   pip install {" ".join(missing_deps)}')
        
        # Show what features need what dependencies
        print()
        print('📋 What each dependency enables:')
        feature_map = {
            'psutil': 'Performance monitoring (CPU, memory tracking)',
            'pyyaml': 'Configuration management (YAML config files)',
            'tensorflow': 'Neural network model predictions',
            'opencv-python': 'Advanced image processing',
            'nest-asyncio': 'Async support in Jupyter notebooks',
            'pillow': 'Image loading and processing',
        }
        
        for dep in missing_deps:
            if dep in feature_map:
                print(f'   • {dep}: {feature_map[dep]}')

    return all_installed, missing_deps

if __name__ == "__main__":
    check_dependencies()