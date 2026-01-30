import cv2
import numpy as np
import torchvision.transforms.functional as TF
import base64
import torch
from io import BytesIO
from PIL import Image

from app.schemas.data_profile import DataProfile


def b64_to_pil(img_b64: str) -> Image.Image:
    """Convert base64 image to Pillow Image."""
    image_data = base64.b64decode(img_b64)
    return Image.open(BytesIO(image_data)).convert('RGB')


def ndarray_to_base64(mask_np: np.ndarray) -> str:
    """Convert a numpy array to a base64 image."""
    mask_img = Image.fromarray(mask_np.astype(np.uint8))
    buf = BytesIO()
    mask_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf8')


def preprocess_image(img_arr, data_profile: DataProfile):
    """Preprocess the image for model input."""
    img = data_profile.preprocess(img_arr)
    img = torch.from_numpy(img_arr) # int, RGB CxHxW
    img = torch.permute(img, (2, 0, 1))
    img = torch.div(img, 255)  # Convert to float and normalize to [0, 1]
    img = TF.resize(img, list(data_profile.image_size), interpolation=TF.InterpolationMode.BILINEAR)
    return img.unsqueeze(0).to(torch.float32)  # Add batch dimension and convert to float32
