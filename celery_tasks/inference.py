from inference.core import inference_logic
from celery import shared_task


@shared_task(bind=True)
async def inference_task(self, file, model_registry_key, mask_id):
    await inference_logic(self, file, model_registry_key, mask_id)
