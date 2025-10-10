import cv2
import numpy as np

from app.main_api.inference import post_mask
from app.state import MODEL_REGISTRY
import torch
from logging import getLogger

from app.util.image_conversions import preprocess_image

logger = getLogger(__name__)


async def inference(task, file, model_registry_key, mask_id):
    registry_entry = MODEL_REGISTRY.models[model_registry_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = registry_entry.loader.load_model()
        model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Could not load model: {e}")
        raise e

    # Read and preprocess all images into a batch
    try:
        img_bytes = await file.read()
        img_arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        processed_img_tensor = preprocess_image(img_arr, registry_entry.info.training_run.data_profile)
    except Exception as e:
        logger.error(f"Error reading {file.filename}: {e}")
        raise e

    logits = model(processed_img_tensor)  # [N, num_classes, H, W]
    pred = torch.argmax(logits, dim=1).int()  # [N, H, W]
    masks_np = pred.cpu().numpy()  # shape: (N, H, W)
    await post_mask(masks_np, mask_id)
