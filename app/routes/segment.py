from fastapi.responses import StreamingResponse
from logging import getLogger
from app.state import MODEL_REGISTRY
from app.schemas.segment import B64SegmentationRequest
from app.util.image_conversions import *
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import torch
from models import load_model_from_id
from PIL import Image
from io import BytesIO
import zipfile
import os
from app.schemas.training_run import JobStatusEnum, TrainingRun

router = APIRouter(prefix="/segment", tags=["segment"])
logger = getLogger(__name__)


@router.post("/segment_img/model={model_registry_key}")
async def segment_batch(
    model_registry_key: str,
    file: UploadFile = File(...),
):
    """ Segment a single image with the specified model."""
    registry_entry = MODEL_REGISTRY.models[model_registry_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = registry_entry.loader.load_model()
        model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Could not load model: {e}")
        raise HTTPException(status_code=400, detail=f"Could not load model: {e}")

    # Read and preprocess all images into a batch
    try:
        img_bytes = await file.read()
        img_arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        processed_img_tensor = preprocess_image(img_arr, registry_entry.info.training_run.data_profile)
    except Exception as e:
        logger.error(f"Error reading {file.filename}: {e}")
        raise e

    logits = model(processed_img_tensor)  # [N, num_classes, H, W]
    #preds = torch.softmax(logits, dim=1)
    pred = torch.argmax(logits, dim=1).int()  # [N, H, W]
    masks_np = pred.cpu().numpy()  # shape: (N, H, W)

    return StreamingResponse(
        BytesIO(masks_np.tobytes()),
    )
