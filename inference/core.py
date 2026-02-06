from logging import getLogger
from iquana_toolbox.schemas.service_requests import MultiSemanticSegmentationRequest, SemanticSegmentationRequest
from celery_app import celery_app

logger = getLogger(__name__)


async def inference(request: SemanticSegmentationRequest):
    raise NotImplementedError()


@celery_app.task(name="semantic_segmentation.inference.batched")
async def inference_multi(request: MultiSemanticSegmentationRequest):
    raise NotImplementedError("This method is not implemented.")
