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
