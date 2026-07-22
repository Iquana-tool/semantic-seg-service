from dotenv import load_dotenv
load_dotenv(override=True)

from celery import Celery
from paths import REDIS_URL

celery_app = Celery(
    "iquana_service_semantic_segmentation", # Must match the name in your other services
    broker=f"{REDIS_URL}/0",
    backend=f"{REDIS_URL}/1"
)

import training
import training.core
import training.dataloader
import training.metrics
import inference.core
import app.state