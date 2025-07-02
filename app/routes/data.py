import os.path
from logging import getLogger
from fastapi import APIRouter, UploadFile, File
from paths import DATA_PATH
import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from io import BytesIO

router = APIRouter("/data", tags=["data"])
logger = getLogger(__name__)


@router.post("/upload_file_to_dataset")
async def upload_file(dataset_id: int, is_image: bool, file: UploadFile = File(...), filename: str = None):
    """Upload a single image file to a dataset and save it as a PyTorch tensor."""

    # Compute target .pt path
    target_dir = os.path.join(DATA_PATH, str(dataset_id), "images" if is_image else "masks")
    os.makedirs(target_dir, exist_ok=True)

    base_filename = os.path.splitext(file.filename)[0] if not filename else filename
    tensor_path = os.path.join(target_dir, base_filename + ".pt")

    # Read file content into memory
    contents = await file.read()

    # Convert to NumPy using OpenCV from memory
    img_arr = cv.imdecode(np.frombuffer(contents, np.uint8), cv.IMREAD_UNCHANGED)
    if img_arr is None:
        return {"success": False, "message": "Could not decode image file."}

    # Convert to PyTorch tensor (channels first if image)
    tensor = torch.from_numpy(img_arr)
    if is_image and tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1)  # HWC → CHW

    # Save as .pt
    tensor = tensor.to(dtype=torch.long)
    torch.save(tensor, tensor_path)

    return {"success": True, "message": f"Tensor saved to {tensor_path}"}



@router.post("/upload_dataset")
async def upload_dataset(dataset_id: int, images: list[UploadFile] = File(...), masks: list[UploadFile] = File(...)):
    """Upload multiple image and mask files to a dataset and save them as PyTorch tensors."""
    img_dict = {file.filename.split()[0]: file for file in images}
    mask_dict = {file.filename.split()[0]: file for file in masks}
    uploaded_files = 0
    for i, (name, img) in enumerate(img_dict.items()):
        if name not in mask_dict:
            logger.warning(f"Could not find mask for image {name}. Skipping.")
            continue
        await upload_file(dataset_id, True, img, str(i))
        await upload_file(dataset_id, False, mask_dict[name], str(i))
        uploaded_files += 1
    return {
        "success": True,
        "message": f"Uploaded {uploaded_files} images and masks to dataset {dataset_id}."
    }
