from logging import getLogger

from fastapi import APIRouter

from app.state import MODEL_REGISTRY

router = APIRouter(prefix="/models", tags=["models"])
logger = getLogger(__name__)


@router.get("/all")
async def get_models():
    """Retrieve all segmentation models."""
    logger.debug("Fetching available segmentation models.")
    # Get the models from the registry, excluding the 'getter' key
    return {
        "success": True,
        "message": "Retrieved all models.",
        "result": MODEL_REGISTRY.list_models(only_return_available=False)
    }


@router.get("/all/available")
async def get_models():
    """Retrieve all available segmentation models."""
    logger.debug("Fetching available segmentation models.")
    # Get the models from the registry, excluding the 'getter' key
    return {
        "success": True,
        "message": "Retrieved all available models.",
        "result": MODEL_REGISTRY.list_models(only_return_available=True)
    }


@router.get("/{model_registry_key}")
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


@router.delete("/{model_registry_key}")
async def delete_model(model_registry_key: str):
    """Deletes a model based on its id."""
    logger.debug(f"Deleting model with id: {model_registry_key}.")
    result = MODEL_REGISTRY.delete_model(model_registry_key)

    return {
        "success": True,
        "message": f"Model with id {model_registry_key} successfully removed."
    }