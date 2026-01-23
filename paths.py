import os

BACKEND_URL = os.environ.get("BACKEND_URL", "localhost:8000")
CELERY_URL = os.environ.get("CELERY_URL", "localhost:7001")

DATA_PATH = os.environ.get("DATA_PATH", "./data")
TRAINED_MODEL_WEIGHTS_PATH = os.environ.get("TRAINED_MODEL_WEIGHTS_PATH", "trained_model_weights")
TRAINED_MODEL_INFO_PATHS = os.environ.get("TRAINED_MODEL_INFO_PATHS", "trained_model_infos")
LOG_PATH = os.environ.get("LOG_PATH", "./logs")