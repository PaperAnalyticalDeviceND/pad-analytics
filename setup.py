from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pad-analytics",
    version="0.2.1",
    author="Paper Analytical Device Project Team",
    author_email="pad-project@nd.edu",
    description="Python tools for analyzing Paper Analytical Devices (PADs) to detect and quantify pharmaceutical compounds through colorimetric analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PaperAnalyticalDeviceND/pad-analytics",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.13.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0,<2.0.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "requests>=2.25.0",
        "pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "ipywidgets>=7.6.0",
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
    keywords="paper analytical device PAD colorimetric analysis pharmaceutical quality machine learning drug detection",
    project_urls={
        "Documentation": "https://pad.crc.nd.edu/docs",
        "Source": "https://github.com/PaperAnalyticalDeviceND/pad-analytics",
        "Tracker": "https://github.com/PaperAnalyticalDeviceND/pad-analytics/issues",
        "Homepage": "https://padproject.nd.edu",
    },
    entry_points={
        "console_scripts": [
            "pad-analytics=pad_analytics.padanalytics:main",
        ],
    },
    include_package_data=True,
)