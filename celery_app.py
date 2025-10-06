from celery import Celery
from redis import Redis

redis = Redis(host="localhost", port=6379, db=0)
celery = Celery(
    "training_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

# Optional: Use Redis lock to prevent concurrent training
from celery.app.control import Inspect
def is_training_running():
    inspect = Inspect(app=celery)
    active_tasks = inspect.active()
    return any("training.tasks.train_model" in task for tasks in active_tasks.values() for task in tasks)
