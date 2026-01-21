# worker/tasks.py
from training.core import train_model_logic  # Use a distinct name
from celery import shared_task

@shared_task(bind=True, name="semantic.training")
def train_model_task(self, req_dict, model_registry_key):
    """
    This is the entry point for the Celery Worker.
    'self' gives us access to task metadata (like progress updates).
    """
    # Call the actual computation logic
    result = train_model_logic(self, req_dict, model_registry_key)
    return result
