from training.training import train_model
from celery import shared_task



@shared_task(bind=True)
def train_model(self, req_dict, model_registry_key):
    """ The celery task for training. Wraps the actual training function. """
    train_model(self, req_dict, model_registry_key)
