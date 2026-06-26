from setuptools import find_packages, setup

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="semantic-plagiarism-engine",
    version="0.1.0",
    author="Hirad2hhk",
    author_email="hirad2hhk@example.com",
    description="An educational implementation for detecting semantic plagiarism and near-duplicate content using advanced techniques like MinHash, SimHash, and TF-IDF weighting.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hirad2hhk/semantic-plagiarism-engine",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "click>=8.1",
        "pytest>=7.0",
    ],
    entry_points={
        "console_scripts": [
            "plagiarism-engine=plagiarism_engine.cli:cli",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
    },
)
