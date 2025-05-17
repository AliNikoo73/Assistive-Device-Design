from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gaitsim_assist",
    version="0.1.0",
    author="Assistive Device Design Team",
    author_email="your.email@example.com",
    description="A Python library for gait simulations and assistive device design",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gaitsim_assist",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "pandas>=1.2.0",
        "scipy>=1.6.0",
        "seaborn>=0.11.0",
        "pytest>=6.0.0",
        "jupyter>=1.0.0",
        "casadi>=3.5.5",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black",
            "flake8",
            "sphinx",
            "sphinx_rtd_theme",
        ],
    },
    include_package_data=True,
    package_data={
        "gaitsim_assist": ["models/*.osim", "data/*.csv", "data/*.json"],
    },
) 