import os

BACKEND_URL = os.environ.get("BACKEND_URL", "localhost:8000")
CELERY_URL = os.environ.get("CELERY_URL", "localhost:7001")

DATA_PATH = os.environ.get("DATA_PATH", "./data")
MODEL_WEIGHTS_PATH = os.environ.get("MODEL_PATH", "model_weights")
MODEL_REGISTRY_ENTRY_PATHS = os.environ.get("MODEL_REGISTRY_ENTRY_PATHS", "./model_registry_entries")
LOG_PATH = os.environ.get("LOG_PATH", "./logs")