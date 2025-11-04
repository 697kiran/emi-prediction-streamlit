# lightweight shim package for mlflow so import mlflow works
# This is NOT the real mlflow; it's a no-op stub so your app can import mlflow.* without installing it.
__all__ = ["sklearn", "xgboost"]

# optional metadata
__version__ = "0.0.0-shim"
