#!/usr/bin/env python3
"""
Setup script for SDH Subtitle Normalizer
"""

from setuptools import setup, find_packages

setup(
    name="sdh-subtitle-normalizer",
    version="1.0.0",
    description="AI-powered tool to normalize SDH (Signs and Descriptions for Hearing Impaired) subtitles",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    py_modules=[
        # Expose top-level training utility scripts
        'train_mps',
    ],
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.20.0",
        "tokenizers>=0.12.0",
        "accelerate>=0.12.0",
        "sentencepiece>=0.1.99",
    ],
    python_requires=">=3.7",
    entry_points={
        'console_scripts': [
            'analyze-srt=src.srt_analyzer:main',
            'generate-training-data=src.training_data_generator:main',
            'train-normalizer=src.train_model:main',
            'normalize-subtitles=src.normalize_subtitles:main',
            'train-mps=train_mps:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
