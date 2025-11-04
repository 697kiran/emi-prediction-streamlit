
from setuptools import setup, find_packages

setup(
    name="emipredict-ai",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="EMIPredict AI - Intelligent Financial Risk Assessment Platform",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.1",
        "pandas>=1.5.3",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.6",
        "plotly>=5.15.0",
        "mlflow>=2.5.0",
        "joblib>=1.3.1",
        "seaborn>=0.12.2",
        "matplotlib>=3.7.1",
        "imbalanced-learn>=0.11.0",
    ],
)
