from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from logging import getLogger
from models import load_model_from_id, load_metadata_from_id
from app.schemas.segment import B64SegmentationRequest
from app.util.image_conversions import *
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import torch
from models import load_model_from_id
from PIL import Image
from io import BytesIO
import zipfile
import os


router = APIRouter()
logger = getLogger(__name__)


@router.post("/segment_base64")
async def segment_b64image(request: B64SegmentationRequest):
    logger.info("Segment image request received for model %s.", request.model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the model
    try:
        model= load_model_from_id(request.model_id, device, eval_mode=True, return_metadata=True)
        meta = load_metadata_from_id(request.model_id)
        # meta should include num_classes, image_size used for this model
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=400, detail=f"Could not load model: {e}")

    # 2. Decode and preprocess the image
    try:
        img = b64_to_pil(request.image_b64)
        image_size = meta.get("image_size", (256, 256))
        img_tensor = preprocess_image(img, image_size=image_size).to(device)
    except Exception as e:
        logger.error(f"Image decoding/preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail=f"Could not decode or preprocess image: {e}")

    # 3. Run inference
    with torch.no_grad():
        logits = model(img_tensor)
        pred = torch.argmax(logits, dim=1)
    mask_np = pred.squeeze(0).cpu().numpy()

    # 4. Encode mask as base64 PNG
    mask_b64 = mask_to_base64(mask_np)

    return {
        "mask_b64": mask_b64,
        "mask_shape": mask_np.shape,
        "message": "Segmentation completed."
    }


@router.post("/segment_batch")
async def segment_batch(
    model_id: str = Form(...),
    files: list[UploadFile] = File(...)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = load_model_from_id(model_id, device, eval_mode=True, return_metadata=True)
        meta = load_metadata_from_id(model_id)
        image_size = meta.get("image_size", (256,256))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load model: {e}")

    # Read and preprocess all images into a batch
    img_tensors = []
    filenames = []
    for file in files:
        try:
            img_bytes = await file.read()
            pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
            img_tensors.append(preprocess_image(pil_img, image_size=image_size))
            filenames.append(file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading {file.filename}: {e}")

    if not img_tensors:
        raise HTTPException(status_code=400, detail="No valid images provided.")

    batch = torch.cat(img_tensors, dim=0).to(device)  # [N, C, H, W]

    with torch.no_grad():
        logits = model(batch)  # [N, num_classes, H, W]
        pred = torch.argmax(logits, dim=1)  # [N, H, W]
    masks_np = pred.cpu().numpy()  # shape: (N, H, W)

    # Prepare ZIP in memory
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as mask_zip:
        for fname, mask_np in zip(filenames, masks_np):
            mask_img = Image.fromarray(mask_np.astype('uint8'))
            mask_fname = os.path.splitext(fname)[0] + "_mask.png"
            mask_bytes = BytesIO()
            mask_img.save(mask_bytes, format='PNG')
            mask_bytes.seek(0)
            mask_zip.writestr(mask_fname, mask_bytes.read())
    zip_buf.seek(0)

    # Return zip as file download
    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="masks.zip"'}
    )
