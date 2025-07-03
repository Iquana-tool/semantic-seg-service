import json
import os

import torch

from .unet import get_unet

MODEL_REGISTRY = {
    "unet": {
        "getter": get_unet,
        "description": "UNet model for image segmentation.",
        "speed": "Fast",
        "automatic tuning": False
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
    config_path = checkpoint_path.split(".")[0] + ".json"
    if not os.path.exists(checkpoint_path) or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Model weights file {checkpoint_path} or config file {config_path} does not exist.")
    # Get the registration key and model ID from the file name
    reg_key, model_id = parse_weight_file_name(checkpoint_path)
    # Load the model using the registry
    model_fn = MODEL_REGISTRY.get(reg_key, {}).get("getter")
    with open(config_path, "r") as f:
        config = json.load(f)
    model = model_fn(**config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if eval_mode:
        model.eval()
    return model, checkpoint


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
    return reg_key, model_id
