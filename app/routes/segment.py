from io import BytesIO
from logging import getLogger

from fastapi import UploadFile, File, APIRouter

from celery_tasks.inference import inference

router = APIRouter(prefix="/segment", tags=["segment"])
logger = getLogger(__name__)


@router.post("/segment_img/model={model_registry_key}&mask_id={mask_id}")
async def segment_batch(
    model_registry_key: str,
    mask_id: int,
    file: UploadFile = File(...),
):
    """ Segment a single image with the specified model."""
    inference.delay(file, model_registry_key, mask_id)
    return {
        "success": True,
        "message": "Image inference has been queued."
    }