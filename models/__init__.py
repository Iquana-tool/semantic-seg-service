import json
import os

import torch
from logging import getLogger
from paths import MODEL_PATH
from .unet import get_unet, get_unet_plusplus

logger = getLogger(__name__)

MODEL_REGISTRY = {
    "unet": {
        "getter": get_unet,  # Function to get the UNet model, absolutely required
        "Name": "UNet",
        "Description": "Simple Decoder Encoder Network with Skip Connections. UNet is simple but powerful.",
        "Training speed": "Medium",
        "Model size": "Small",
        "Automatically tuned": False,
        "Pre-trained": False
    },
    # Placeholders for other models, e.g. "nnunet": get_nnunet
}


def load_model_from_checkpoint_path(checkpoint_path: str, device: str = 'cpu', eval_mode=True) -> torch.nn.Module:
    """
    Load a model from a weights file.

    Args:
        checkpoint_path (str): Path to the model weights file.
        device (str): Device to load the model on, e.g., 'cpu' or 'cuda'.
        eval_mode (bool): Whether to set the model in evaluation or train mode (i.e. gradients are not computed).

    Returns:
        tuple: (model, checkpoint) where model is the loaded model and checkpoint is the state dictionary for further
        use.
    """
    config_path = checkpoint_path.rsplit(".", 1)[0] + ".json"
    if not os.path.exists(checkpoint_path) or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Model weights file {checkpoint_path} or config file {config_path} does not exist.")
    # Get the registration key and model ID from the file name
    reg_key, model_id = parse_weight_file_name(checkpoint_path)
    logger.info(f"Loading {reg_key} model with ID: {model_id}")
    # Load the model using the registry
    model_fn = MODEL_REGISTRY.get(reg_key, {}).get("getter")
    with open(config_path, "r") as f:
        config = json.load(f)
    model = model_fn(**config)
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if eval_mode:
        model.eval()
    return model, checkpoint


def load_model_from_id(model_id: int, device: str = 'cpu', eval_mode=True) -> torch.nn.Module:
    """
    Load a model from its ID.

    Args:
        model_id (int): The ID of the model to load.
        device (str): Device to load the model on, e.g., 'cpu' or 'cuda'.
        eval_mode (bool): Whether to set the model in evaluation or train mode (i.e. gradients are not computed).

    Returns:
        tuple: (model, checkpoint) where model is the loaded model and checkpoint is the state dictionary for further
        use.
    """
    weight_files = [f for f in os.listdir(MODEL_PATH) if f.endswith(".pt") and f.split("_")[-1].split(".")[0] == str(model_id)]
    if not weight_files:
        raise FileNotFoundError(f"No trained model found with ID {model_id}.")
    if len(weight_files) > 1:
        logger.warning(f"Multiple weight files found for model ID {model_id}. "
                       f"Using the first one: {weight_files[0]}")
    weight_file = os.path.join(MODEL_PATH, weight_files[0])
    logger.info(f"Loading {weight_file} model with ID: {model_id}")
    return load_model_from_checkpoint_path(weight_file, device=device, eval_mode=eval_mode)


def load_metadata_from_id(model_id: int) -> dict:
    """
    Load metadata from a model ID.

    Args:
        model_id (int): The ID of the model to load metadata for.

    Returns:
        dict: Metadata dictionary containing model identifier and other details.
    """
    meta_files = [f for f in os.listdir(MODEL_PATH) if f.endswith(".json") and
                    f.split("_")[-1].split(".")[0] == str(model_id)]
    if not meta_files:
        raise FileNotFoundError(f"No trained model found with ID {model_id}.")
    if len(meta_files) > 1:
        logger.warning(f"Multiple metadata jsons found for model ID {model_id}. "
                       f"Using the first one: {meta_files[0]}")
    meta_file = os.path.join(MODEL_PATH, meta_files[0])
    with open(meta_file, "r") as f:
        metadata = json.load(f)
    return metadata, meta_file


def parse_weight_file_name(file_name: str):
    """
    Parse the model identifier and ID from a weight file name.

    Args:
        file_name (str): The name of the weight file.

    Returns:
        tuple: (model_identifier, model_id)
    """
    base_name = os.path.basename(file_name).split(".")[0]
    reg_key, model_id = base_name.split("_")
    return reg_key, int(model_id)


def get_registry_key_from_id(model_id: int):
    # Get all files with the model id in them. This should be exactly one.
    result = [parse_weight_file_name(filename) for filename in os.listdir(MODEL_PATH) if filename.endswith(".json") and
             model_id == parse_weight_file_name(filename)[1]]
    return result[0]


def delete_model(model_id: int):
    """
    Delete a model by its ID.

    Args:
        model_id (int): The ID of the model to delete.
    """
    weight_files = [f for f in os.listdir(MODEL_PATH) if f.endswith(".pt") and f.split("_")[-1].split(".")[0] == str(model_id)]
    meta_files = [f for f in os.listdir(MODEL_PATH) if f.endswith(".json") and f.split("_")[-1].split(".")[0] == str(model_id)]

    for file in weight_files + meta_files:
        os.remove(os.path.join(MODEL_PATH, file))
        logger.info(f"Deleted model file: {file}")

