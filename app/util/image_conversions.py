import numpy as np
import torchvision.transforms.functional as TF
import base64
from io import BytesIO
from PIL import Image


def b64_to_pil(img_b64: str) -> Image.Image:
    image_data = base64.b64decode(img_b64)
    return Image.open(BytesIO(image_data)).convert('RGB')


def mask_to_base64(mask_np: np.ndarray) -> str:
    from PIL import Image
    mask_img = Image.fromarray(mask_np.astype(np.uint8))
    buf = BytesIO()
    mask_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf8')


def preprocess_image(pil_img, image_size=(256, 256)):
    img = TF.to_tensor(pil_img)  # float32, CxHxW
    img = TF.resize(img, image_size)
    img = TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    img = img.unsqueeze(0)  # [1, C, H, W]
    return img
