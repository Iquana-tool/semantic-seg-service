from celery import Celery
from redis import Redis

redis = Redis(host="localhost", port=6379, db=0)
celery = Celery(
    "semantic_seg",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)
