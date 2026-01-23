from logging import getLogger

import cv2
import numpy as np
import torch
from torchvision.io import read_image
from torchvision.transforms import Resize
from app.state import MODEL_REGISTRY
from app.util.image_conversions import preprocess_image

logger = getLogger(__name__)


async def inference_logic(image_url, model_registry_key):
    model_info = MODEL_REGISTRY.get_model_info(model_registry_key)
    model_loader = MODEL_REGISTRY.get_model_loader(model_info)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = model_loader.load_model()
        model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Could not load model: {e}")
        raise e

    # Read and preprocess all images into a batch
    try:
        img_tensor = read_image(image_url).float() / 255.0
        img_tensor = Resize(size=model_info.training_req.image_size).forward(img_tensor)
        img_tensor = img_tensor.to(device)
    except Exception as e:
        logger.error(f"Error reading {image_url}: {e}")
        raise e
    logits = model(img_tensor)  # [N, num_classes, H, W]
    max_tensor = torch.max(logits, 1)[1].item()
    pred = max_tensor[1].int()  # [N, H, W]
    confidence = torch.mean(max_tensor[0])
    masks_np = pred.cpu().numpy()  # shape: (N, H, W)
    return masks_np, confidence
