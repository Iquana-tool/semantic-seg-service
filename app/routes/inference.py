import json
from logging import getLogger

from fastapi import APIRouter
from iquana_toolbox.schemas.contour_hierarchy import ContourHierarchy
from iquana_toolbox.schemas.service_requests import SemanticSegmentationRequest, MultiSemanticSegmentationRequest
from starlette.responses import StreamingResponse
from inference.core import inference

router = APIRouter(tags=["inference"])
session_router = APIRouter(prefix="/annotation_session", tags=["session"])
logger = getLogger(__name__)


@session_router.post("/run")
async def inference(
    request: SemanticSegmentationRequest,
):
    """ Segment a single image with the specified model."""
    mask, score = await inference(request)
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
    raise NotImplementedError()
