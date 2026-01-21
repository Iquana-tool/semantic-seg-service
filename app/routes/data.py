import os.path
from logging import getLogger
from typing import Literal

import cv2 as cv
import numpy as np
import torch
from fastapi import APIRouter, UploadFile, File, Form

from paths import DATA_PATH

router = APIRouter(prefix="/data", tags=["data"])
logger = getLogger(__name__)


@router.post("/upload_file_to_dataset")
async def upload_file(
    dataset_id: int = Form(...),
    type: Literal["images", "masks"] = Form(...),
    filename: str = Form(None),
    file: UploadFile = File(...)
):
    """
        Upload a single image file to a dataset and save it as a PyTorch tensor.
        This progressively builds the dataset aka after a file is marked as fully annotated it is uploaded here.
        Uploading multiple files at once is barely needed.
        :params dataset_id: The id of the dataset. This names the local files.
        :params type: The type of the file. It can be either "images", "masks". This defines the type of the uploaded
            file.
        :params filename: The filename of the uploaded file. This is needed to match image and mask pairs, ie. they
            need to have the same name, else they cant be matched.
        :params file: The file to be uploaded.
    """
    try:
        is_image = type != "masks"
        # Compute target .pt path
        target_dir = os.path.join(DATA_PATH, str(dataset_id), type)
        os.makedirs(target_dir, exist_ok=True)

        base_filename = os.path.splitext(file.filename)[0] if not filename else filename
        tensor_path = os.path.join(target_dir, base_filename + ".pt")

        # Read file content into memory
        contents = await file.read()

        # Convert to NumPy using OpenCV from memory
        # Read masks as grayscale and images as normal
        flag = cv.IMREAD_COLOR_RGB if is_image else cv.IMREAD_GRAYSCALE
        img_arr = cv.imdecode(np.frombuffer(contents, np.uint8), flag)
        if img_arr is None:
            return {"success": False, "message": "Could not decode image file."}

        # Convert to PyTorch tensor (channels first if image)
        tensor = torch.from_numpy(img_arr)
        if is_image and tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)  # HWC → CHW
        # Save as .pt
        if is_image:
            tensor = torch.div(tensor, 255)
            tensor = tensor.to(dtype=torch.float32)
        else:
            tensor = tensor.to(dtype=torch.long)
        if not is_image:
            print(tensor.unique())
        torch.save(tensor, tensor_path)

        return {"success": True, "message": f"Tensor saved to {tensor_path}"}
    except Exception as e:
        raise e
