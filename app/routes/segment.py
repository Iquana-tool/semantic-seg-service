from fastapi.responses import StreamingResponse
from logging import getLogger
from models import load_metadata_from_id
from app.schemas.segment import B64SegmentationRequest
from app.util.image_conversions import *
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import torch
from models import load_model_from_id
from PIL import Image
from io import BytesIO
import zipfile
import os
from app.schemas.training_run import JobStatusEnum, TrainingRun

router = APIRouter(prefix="/segment", tags=["segment"])
logger = getLogger(__name__)


@router.post("/segment_batch/model={model_registry_key}")
async def segment_batch(
    model_registry_key: str,
    file_ids: list[int]
):
    """ Segment a batch of images using the specified model.
    Args:
        model_id (str): The ID of the model to use for segmentation.
        files (list[UploadFile]): List of image files to segment. These will be uploaded as multipart/form-data.

    Returns:
        StreamingResponse: A ZIP file containing the segmentation masks for each input image.
    """
    if len(files) > 10:
        logger.warning(f"Uploading {len(files)} files before segmentation. This can take a while. Consider smaller "
                       f"batches.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, chkpt = load_model_from_id(model_id, device, eval_mode=True)

        meta, info_save_path = load_metadata_from_id(model_id)
        model_info = TrainingRun()
        model_info.load(info_save_path)
        model_info.set_inference_status(JobStatusEnum.IN_PROGRESS)
        image_size = model_info.image_size
        model_info.save(info_save_path)
    except Exception as e:
        logger.error(f"Could not load model: {e}")
        raise HTTPException(status_code=400, detail=f"Could not load model: {e}")

    # Read and preprocess all images into a batch
    img_tensors = []
    filenames = []
    og_shapes = []
    for file in files:
        try:
            img_bytes = await file.read()
            img_arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            og_shapes.append(img_arr.shape[:2])
            processed_img_tensor = preprocess_image(img_arr, image_size=image_size)
            img_tensors.append(processed_img_tensor.to(dtype=torch.float32))
            filenames.append(file.filename)
        except Exception as e:
            logger.error(f"Error reading {file.filename}: {e}")
            raise e
            raise HTTPException(status_code=400, detail=f"Error reading {file.filename}: {e}")

    if not img_tensors:
        raise HTTPException(status_code=400, detail="No valid images provided.")

    batch = torch.cat(img_tensors, dim=0).to(device)  # [N, C, H, W]
    print(f"Batch shape: {batch.shape}, Device: {device}")
    logits = model(batch)  # [N, num_classes, H, W]
    #preds = torch.softmax(logits, dim=1)
    print(f"Logits shape: {logits.shape}")
    pred = torch.argmax(logits, dim=1).int()  # [N, H, W]
    print(f"Prediction shape: {pred.shape}")
    masks_np = pred.cpu().numpy()  # shape: (N, H, W)
    print(f"Masks shape: {masks_np.shape}")

    # Prepare ZIP in memory
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as mask_zip:
        for fname, mask_np, og_shape in zip(filenames, masks_np, og_shapes):
            logger.info(f"Processing file {fname} with original shape {og_shape} and mask shape {mask_np.shape}")
            mask_np = cv2.resize(mask_np, (og_shape[1], og_shape[0]), interpolation=cv2.INTER_NEAREST)
            success, encoded_img = cv2.imencode('.png', mask_np.astype(np.uint8))
            if not success:
                raise RuntimeError("cv2.imencode failed!")
            mask_zip.writestr(fname, encoded_img.tobytes())
    zip_buf.seek(0)
    model_info.set_inference_status(JobStatusEnum.FINISHED)
    model_info.save(info_save_path)
    # Return zip as file download
    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="masks.zip"'}
    )
