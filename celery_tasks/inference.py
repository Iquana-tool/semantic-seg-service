from inference.inference import inference
from celery import shared_task


@shared_task(bind=True)
async def inference_task(self, file, model_registry_key, mask_id):
    await inference(self, file, model_registry_key, mask_id)
