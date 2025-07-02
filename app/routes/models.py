from logging import getLogger
from fastapi import APIRouter, UploadFile, File
from models import MODEL_REGISTRY


router = APIRouter("/models", tags=["models"])
logger = getLogger(__name__)


@router.get("/get_trainable_models")
async def get_trainable_models():
    """Retrieve all available segmentation models."""
    logger.debug("Fetching available segmentation models.")
    # Get the models from the registry, excluding the 'getter' key
    models = {key:
        {k: v for k, v in value.items() if k != "getter"}
              for key, value in MODEL_REGISTRY.items()
              }
    return {"success": True, "models": models}
