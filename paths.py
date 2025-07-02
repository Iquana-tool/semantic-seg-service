import os


DATA_PATH = os.environ.get("DATA_PATH", "./data")
MODEL_PATH = os.environ.get("MODEL_PATH", "./saved_models")
LOG_PATH = os.environ.get("LOG_PATH", "./logs")
JOBS_PATH = os.environ.get("JOBS_PATH", "./training_jobs")
