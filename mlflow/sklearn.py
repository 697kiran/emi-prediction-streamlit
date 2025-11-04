# minimal stub for mlflow.sklearn used only to satisfy imports.
# The real functions are not implemented — these are no-op placeholders.

def log_model(*args, **kwargs):
    """No-op log_model"""
    return None

def save_model(*args, **kwargs):
    """No-op save_model"""
    return None

def load_model(*args, **kwargs):
    """No-op load_model"""
    return None

def sklearn_log_model(*args, **kwargs):
    return None

# Provide an object so callers like mlflow.sklearn.log_model(...) work
