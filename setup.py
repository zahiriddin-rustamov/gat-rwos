# setup.py

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="gat_rwos",
    version="0.1.0",
    author="Zahiriddin Rustamov",
    author_email="700043167@uaeu.ac.ae",
    description="GAT-RWOS: Oversampling via GAT and Random Walks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zahiriddin-rustamov/gat-rwos",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "optuna",
        "scikit-learn",
        "xgboost",
        "tqdm",
        "PyYAML",
        "scipy",
        "torch_geometric",
    ],
    entry_points={
        "console_scripts": [
            "gat-rwos=gat_rwos.cli:main",
            "gat_rwos=gat_rwos.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
