import json
import os
from collections import defaultdict
from logging import getLogger
from fastapi import APIRouter, UploadFile, File
from models import MODEL_REGISTRY
from paths import MODEL_PATH

router = APIRouter(prefix="/models", tags=["models"])
logger = getLogger(__name__)


@router.get("/get_trainable_base_models")
async def get_trainable_base_models():
    """Retrieve all available segmentation models."""
    logger.debug("Fetching available segmentation models.")
    # Get the models from the registry, excluding the 'getter' key
    models = {key:
        {k: v for k, v in value.items() if k != "getter"}
              for key, value in MODEL_REGISTRY.items()
              }
    return {"success": True, "models": models}


@router.get("/get_trained_models_of_dataset/{dataset_id}")
async def get_trained_models_of_dataset(dataset_id: int):
    """Retrieve all trained segmentation models."""
    logger.debug("Fetching trained segmentation models.")
    # Get the trained models from the weights directory
    trained_models = [file for file in os.listdir(MODEL_PATH) if not file.endswith(".json")]
    trained_models_result = []
    for weight in trained_models:
        name = weight.split(".")[0]
        identifier, model_id = name.split("_")
        meta_entry = json.load(open(os.path.join(MODEL_PATH, weight.rsplit(".", 1)[0] + ".json")))
        if not int(meta_entry["dataset_id"]) == dataset_id:
            continue
        registry_entry = MODEL_REGISTRY.get(identifier)
        del registry_entry["getter"]
        model_entry = {**registry_entry, **meta_entry}
        trained_models_result.append(model_entry)
    return {"success": True,
            "message": f"Found {len(trained_models)} trained models.",
            "models": trained_models_result}
