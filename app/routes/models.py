import os
from collections import defaultdict
from logging import getLogger
from fastapi import APIRouter, UploadFile, File
from models import MODEL_REGISTRY
from paths import MODEL_PATH

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


@router.get("/get_trained_models")
async def get_trained_models():
    """Retrieve all trained segmentation models."""
    logger.debug("Fetching trained segmentation models.")
    # Get the trained models from the weights directory
    trained_models = os.listdir(MODEL_PATH)
    trained_models_dict = defaultdict(list)
    for weight in trained_models:
        name = weight.split(".")[0]
        identifier, model_id = name.split("_")
        trained_models_dict[identifier].append(model_id)
    return {"success": True,
            "message": f"Found {len(trained_models)} trained models.",
            "models": trained_models_dict}
