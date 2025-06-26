from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pad-ml-workflow",
    version="0.1.0",
    author="PAD ML Team",
    author_email="",
    description="A complete workflow for machine learning models using data from the PAD API v2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pad-ml-workflow-v2",
    packages=find_packages(where="src"),
    package_dir={"pad_ml_workflow": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "keras>=2.13.0",
        "tensorflow>=2.13.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "requests>=2.25.0",
        "pillow>=8.0.0",
        "opencv-python>=4.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=6.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pad-analysis=pad_ml_workflow.padanalytics:main",
        ],
    },
)