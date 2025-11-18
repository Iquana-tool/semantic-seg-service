from io import BytesIO
from logging import getLogger

from fastapi import UploadFile, File, APIRouter

from celery_tasks.inference import inference_task

router = APIRouter(prefix="/inference", tags=["inference"])
logger = getLogger(__name__)


@router.post("/start_inference/model={model_registry_key}&mask_id={mask_id}")
async def inference(
    model_registry_key: str,
    mask_id: int,
    file: UploadFile = File(...),
):
    """ Segment a single image with the specified model."""
    inference_task.delay(file, model_registry_key, mask_id)
    return {
        "success": True,
        "message": "Image inference has been queued."
    }