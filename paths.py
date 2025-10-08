import os

DATA_PATH = os.environ.get("DATA_PATH", "./data")
MODEL_WEIGHTS_PATH = os.environ.get("MODEL_PATH", "model_weights")
MODEL_REGISTRY_ENTRY_PATHS = os.environ.get("MODEL_REGISTRY_ENTRY_PATHS", "./model_registry_entries")
LOG_PATH = os.environ.get("LOG_PATH", "./logs")