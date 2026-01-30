import json
from logging import getLogger

from fastapi import APIRouter
from iquana_toolbox.schemas.contour_hierarchy import ContourHierarchy
from iquana_toolbox.schemas.service_requests import SemanticSegmentationRequest, MultiSemanticSegmentationRequest
from starlette.responses import StreamingResponse
from inference.core import inference_logic

router = APIRouter(tags=["inference"])
session_router = APIRouter(prefix="/annotation_session", tags=["session"])
logger = getLogger(__name__)


@session_router.post("/run")
async def inference(
    request: SemanticSegmentationRequest,
):
    """ Segment a single image with the specified model."""
    mask, score = await inference_logic(request.image, request.model_registry_key)
    return {
        "success": True,
        "message": "Successfully segmented image.",
        "result": ContourHierarchy.from_semantic_mask(
            mask,
            request.label_hierarchy,
            request.model_registry_key,
        )
    }


@router.post("/stream_multi")
async def stream_multi_inference(
        request: MultiSemanticSegmentationRequest
):
    async def generate_results():
        for req in request.images:
            try:
                mask, score = await inference_logic(req.image, request.model_registry_key)

                yield json.dumps({
                    "success": True,
                    "message": "Successfully segmented image.",
                    "result": ContourHierarchy.from_semantic_mask(
                        mask,
                        request.label_hierarchy,
                        request.model_registry_key,
                    ).model_dump()
                }) + "\n"  # Newline delimiter for the stream

            except Exception as e:
                yield json.dumps({
                    "success": False,
                    "message": str(e),
                    "result": None
                }) + "\n"

    return StreamingResponse(generate_results(), media_type="application/x-ndjson")
