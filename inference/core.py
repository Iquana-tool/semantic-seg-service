from logging import getLogger

import torch
from torchvision.io import read_image
from torchvision.transforms import Resize

from app.state import MODEL_REGISTRY

logger = getLogger(__name__)


async def inference_logic(images, model_registry_key):
    raise NotImplementedError("This method is not implemented.")
