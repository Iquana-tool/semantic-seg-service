import json
import os
from collections import defaultdict
from logging import getLogger
from fastapi import APIRouter, UploadFile, File
from models import MODEL_REGISTRY
from paths import MODEL_PATH, JOBS_PATH

router = APIRouter(prefix="/models", tags=["models"])
logger = getLogger(__name__)


@router.get("/get_trainable_base_models")
async def get_trainable_base_models():
    """Retrieve all available segmentation models."""
    logger.debug("Fetching available segmentation models.")
    # Get the models from the registry, excluding the 'getter' key
    result = []
    for key, entry in MODEL_REGISTRY.items():
        entry_copy = entry.copy()
        entry_copy.pop("getter")  # This does not need to be returned
        entry_copy["model_identifier"] = key
        result.append(entry_copy)
    return {"success": True, "models": result}


@router.get("/get_trained_models_of_dataset/{dataset_id}")
async def get_trained_models_of_dataset(dataset_id: int):
    """Retrieve all trained segmentation models."""
    logger.debug("Fetching trained segmentation models.")
    # Get the trained models from the weights directory
    trained_models = [file for file in os.listdir(MODEL_PATH) if file.endswith(".json")]
    trained_models_result = []
    for meta_json_file in trained_models:
        name = meta_json_file.split(".")[0]
        identifier, model_id = name.split("_")
        meta_entry = json.load(open(os.path.join(MODEL_PATH, meta_json_file)))
        if meta_entry is None:
            logger.warning(f"Metadata for model {name} is None, skipping.")
            continue
        if not int(meta_entry["dataset_id"]) == dataset_id:
            continue
        registry_entry = MODEL_REGISTRY.get(identifier).copy()
        registry_entry.pop("getter")
        registry_entry["model_identifier"] = model_id
        model_entry = {**registry_entry, **meta_entry}
        trained_models_result.append(model_entry)
    return {"success": True,
            "message": f"Found {len(trained_models)} trained models.",
            "models": trained_models_result}


@router.get("/get_training_models_of_dataset/{dataset_id}")
async def get_training_models():
    """Retrieve all models that are currently being trained."""
    logger.debug("Fetching currently training segmentation models.")
    # Get the training models from the weights directory
    training_models = [file for file in os.listdir(JOBS_PATH) if file.endswith(".json")]
    training_models_result = []

    for job_file in training_models:
        job_info = json.load(open(os.path.join(JOBS_PATH, job_file)))
        if job_info["status"] == "in progress":
            training_models_result.append(job_file.split(".")[0])  # Get the job ID without the .json extension
    return {"success": True,
            "message": f"Found {len(training_models)} training models.",
            "models": training_models_result}


@router.get("/get_model_metadata/{model_id}")
async def get_model_metadata(model_id: int):
    """Retrieve metadata for a specific model."""
    logger.debug(f"Fetching metadata for model ID {model_id}.")
    meta_files = [f for f in os.listdir(MODEL_PATH) if f.endswith(".json") and f.split("_")[-1].split(".")[0] == str(model_id)]
    if not meta_files:
        return {"success": False, "message": f"No trained model found with ID {model_id}."}
    if len(meta_files) > 1:
        logger.warning(f"Multiple metadata jsons found for model ID {model_id}. Using the first one: {meta_files[0]}")
    meta_file = os.path.join(MODEL_PATH, meta_files[0])
    with open(meta_file, "r") as f:
        metadata = json.load(f)
    return {"success": True, "metadata": metadata}
