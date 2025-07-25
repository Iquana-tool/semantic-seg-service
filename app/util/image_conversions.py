import cv2
import numpy as np
import torchvision.transforms.functional as TF
import base64
import torch
from io import BytesIO
from PIL import Image


def b64_to_pil(img_b64: str) -> Image.Image:
    image_data = base64.b64decode(img_b64)
    return Image.open(BytesIO(image_data)).convert('RGB')


def mask_to_base64(mask_np: np.ndarray) -> str:
    mask_img = Image.fromarray(mask_np.astype(np.uint8))
    buf = BytesIO()
    mask_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf8')


def preprocess_image(img_arr, image_size=(256, 256)):
    """Preprocess the image for model input."""
    img = torch.from_numpy(img_arr).float()  # float32, CxHxW
    img = torch.permute(img, (2, 0, 1))
    img = TF.resize(img, image_size, interpolation=TF.InterpolationMode.BILINEAR)  # Resize to target size
    #img = TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    return img.unsqueeze(0)
