# worker/tasks.py
from training.core import train_model_logic  # Use a distinct name
from celery import shared_task

@shared_task(bind=True, name="training")
def train_model_task(self, model, model_info):
    """
    This is the entry point for the Celery Worker.
    'self' gives us access to task metadata (like progress updates).
    """
    # Call the actual computation logic
    result = train_model_logic(self, model, model_info)
    return result
