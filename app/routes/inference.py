from io import BytesIO
from logging import getLogger

from fastapi import UploadFile, File, APIRouter, Response

from inference.inference import inference

router = APIRouter(prefix="/inference", tags=["inference"])
logger = getLogger(__name__)


@router.post("/model={model_registry_key}&mask_id={mask_id}")
async def infer_image(
    model_registry_key: str,
    mask_id: int,
    file: UploadFile = File(...),
):
    """ Segment a single image with the specified model."""
    mask, score = await inference(model_registry_key, mask_id, file)
    # Convert the mask to raw bytes
    mask_bytes = mask.tobytes()

    # Return the raw bytes with metadata in headers
    return Response(
        content=mask_bytes,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": "attachment; filename=mask.bin",
            "X-Mask-Shape": f"{mask.shape[0]},{mask.shape[1]}",  # e.g., "256,256"
            "X-Mask-Dtype": str(mask.dtype),  # e.g., "uint8"
            "X-Score": str(score)  # Optional: Include the score
        }
    )