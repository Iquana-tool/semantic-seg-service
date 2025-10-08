import json
import os
import shutil
from collections import defaultdict
from logging import getLogger
from typing import Union, Literal

from absl.logging import LOG_DIR
from fastapi import APIRouter, UploadFile, File
from app.state import MODEL_REGISTRY
from paths import MODEL_WEIGHTS_PATH, LOG_PATH

router = APIRouter(prefix="/models", tags=["models"])
logger = getLogger(__name__)


@router.get("/get_models/type={type}&available={available}&dataset_id={dataset_id}")
async def get_models(type: Literal["base", "trained"], available: bool, dataset_id: int):
    """Retrieve all available segmentation models."""
    logger.debug("Fetching available segmentation models.")
    # Get the models from the registry, excluding the 'getter' key
    return {
        "success": True,
        "models": MODEL_REGISTRY.list_models(filter_type=type,
                                             filter_availablity=available,
                                             filter_dataset=dataset_id,
                                             return_as_json=True)
    }


@router.get("/get_model_metadata/{model_registry_key}")
async def get_model_metadata(model_registry_key: str):
    """Retrieve metadata for a specific model."""
    if model_registry_key not in MODEL_REGISTRY.models:
        return {"success": False,
                "message": f"Model {model_registry_key} not found."}
    return {
        "success": True,
        "message": f"Info for model {model_registry_key} retrieved.",
        "result": MODEL_REGISTRY.models[model_registry_key].info.model_dump_json()
    }


@router.delete("/delete_model/{model_registry_key}")
async def delete_model(model_registry_key: str):
    """Deletes a model based on its id."""
    logger.debug(f"Deleting model with id: {model_registry_key}.")
    result = MODEL_REGISTRY.delete_model(model_registry_key)

    return {
        "success": True,
        "message": f"Model with id {model_registry_key} successfully removed."
    }