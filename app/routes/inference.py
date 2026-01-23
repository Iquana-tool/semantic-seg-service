import json
from logging import getLogger

import numpy as np
from fastapi import APIRouter, Body
from pycocotools import mask as maskUtils
from starlette.responses import StreamingResponse

from inference.core import inference_logic

router = APIRouter(prefix="/inference", tags=["inference"])
logger = getLogger(__name__)


@router.post("/{model_registry_key}")
async def inference(
    model_registry_key: str,
    image_url: str,
):
    """ Segment a single image with the specified model."""
    # 1. Run inference
    mask, score = await inference_logic(model_registry_key, image_url)

    # 2. Prepare for RLE
    # pycocotools requires Fortran-order (column-major) arrays
    mask_fortran = np.asfortranarray(mask.astype(np.uint8))

    rle_masks = {}
    unique_labels = np.unique(mask_fortran)

    for label in unique_labels:
        if label == 0: continue  # Skip background

        # Create a binary mask for just this class
        binary_mask = (mask_fortran == label).astype(np.uint8)

        # Encode to RLE
        encoded = maskUtils.encode(binary_mask)

        # Convert bytes to string so it's JSON serializable
        encoded['counts'] = encoded['counts'].decode('utf-8')
        rle_masks[int(label)] = encoded

    return {
        "model": model_registry_key,
        "mask": rle_masks,
        "confidence": float(score),  # Ensure float for JSON
    }


@router.post("/{model_registry_key}/stream_multi")
async def stream_multi_inference(
        model_registry_key: str,
        image_urls: list[str] = Body(...),
):
    async def generate_results():
        for url in image_urls:
            try:
                # 1. Run inference for one image
                mask, score = await inference_logic(model_registry_key, url)

                # 2. Process to RLE
                mask_fortran = np.asfortranarray(mask.astype(np.uint8))
                rle_masks = {}
                for label in np.unique(mask_fortran):
                    if label == 0: continue

                    binary_mask = (mask_fortran == label).astype(np.uint8)
                    encoded = maskUtils.encode(binary_mask)
                    encoded['counts'] = encoded['counts'].decode('utf-8')
                    rle_masks[int(label)] = encoded

                # 3. Create the payload for this specific image
                yield json.dumps({
                    "url": url,
                    "mask": rle_masks,
                    "confidence": float(score),
                    "status": "success"
                }) + "\n"  # Newline delimiter for the stream

            except Exception as e:
                yield json.dumps({
                    "url": url,
                    "error": str(e),
                    "status": "error"
                }) + "\n"

    return StreamingResponse(generate_results(), media_type="application/x-ndjson")
