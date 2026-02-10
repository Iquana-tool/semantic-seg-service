from logging import getLogger

import torch
import torch.nn.functional as F
from iquana_toolbox.schemas.service_requests import MultiSemanticSegmentationRequest, SemanticSegmentationRequest
from torchvision.transforms import ToTensor

from app.state import MODEL_REGISTRY
from celery_app import celery_app

logger = getLogger(__name__)


async def inference_logic(request: SemanticSegmentationRequest):
    # 1. Load and prepare model
    model = MODEL_REGISTRY.load_model(request.model_registry_key)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # 2. Preprocess: Add batch dimension and move to device
    # Tensors are not moved in-place; you must assign the result
    image_tensor = ToTensor()(request.image).unsqueeze(0).to(device)

    # 3. Inference
    with torch.no_grad():
        outputs = model(image_tensor)  # Shape: [1, num_classes, H, W]

    # 4. Extract Mask and Confidence
    # Apply softmax to get probabilities across the class dimension (dim=1)
    probabilities = F.softmax(outputs, dim=1)

    # Get the highest probability (confidence) and the index (mask)
    confidence_map, semantic_mask = torch.max(probabilities, dim=1)

    # Convert to CPU/Numpy for the response
    return semantic_mask.squeeze(0).cpu().numpy(), torch.mean(confidence_map.squeeze(0)).cpu().item()


@celery_app.task(name="semantic_segmentation.inference.batched")
async def inference_multi(request: MultiSemanticSegmentationRequest):
    raise NotImplementedError("This method is not implemented.")
