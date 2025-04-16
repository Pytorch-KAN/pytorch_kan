from setuptools import setup, find_packages
import os

# Read README.md for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pytorch-kan",
    version="0.1.0",
    description="Efficient and scalable implementation of Kolmogorov-Arnold Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/username/pytorch_kan",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8.1",
    install_requires=[
        "torch>=2.0.0,<3.0.0",
        "torchvision>=0.15.0,<0.22.0",
        "torchaudio>=2.0.0,<3.0.0",
        "numpy>=1.24.0,<3.0.0",
        "pandas>=1.5.0,<3.0.0",
        "matplotlib>=3.6.0,<4.0.0",
        "scipy>=1.10.0,<2.0.0",
        "scikit-learn>=1.2.0,<2.0.0",
    ],
    extras_require={
        "visualization": ["matplotlib>=3.6.0", "seaborn>=0.12.0", "plotly>=5.13.0"],
        "notebook": ["jupyter>=1.0.0", "notebook>=6.5.0"],
        "transformers": ["transformers>=4.30.0", "datasets>=2.10.0"],
        "all": [
            "matplotlib>=3.6.0",
            "seaborn>=0.12.0",
            "plotly>=5.13.0",
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
            "transformers>=4.30.0",
            "datasets>=2.10.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
