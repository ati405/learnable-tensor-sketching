#!/usr/bin/env python3
"""
Setup script for Learnable Tensor Sketching framework.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="learnable-tensor-sketching",
    version="1.0.0",
    author="Anonymous",
    author_email="anonymous@anonymous.org",
    description="A hierarchical learnable tensor sketching framework for genomic sequence similarity estimation",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/ati405/learnable-tensor-sketching",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "jupyter>=1.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
        "gpu": [
            "torch>=1.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tensor-sketch=learnable.working_phase2_sketch:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.csv", "*.json"],
    },
    keywords=[
        "tensor-sketching",
        "genomics", 
        "bioinformatics",
        "machine-learning",
        "sequence-analysis",
        "similarity-estimation",
        "deep-learning",
        "pytorch"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/learnable-tensor-sketching/issues",
        "Source": "https://github.com/yourusername/learnable-tensor-sketching",
        "Documentation": "https://github.com/yourusername/learnable-tensor-sketching/blob/main/docs/",
        "Paper": "https://doi.org/TBD",  # Will be updated upon publication
    },
    zip_safe=False,
)