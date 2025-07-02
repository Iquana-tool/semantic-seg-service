import os
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Tuple
from models import MODEL_REGISTRY
from paths import MODEL_PATH


class SegmentationRequest(BaseModel):
    """ Model for validating the segmentation request. """
    image_b64: int = Field(..., description="Base64 encoded image to segment.")
    model_id: int = Field(..., description="Identifier of the segmentation model to use. Must be a trained model, whose "
                                           "weights are stored in this repo.")

    @field_validator('model_id')
    def validate_model(cls, value: str) -> str:
        model_weight_files = os.listdir(MODEL_PATH)
        model_ids = [int((file.split('.')[0]).split("_")[-1]) for file in model_weight_files]
        if value not in model_ids:
            raise ValueError(f"Model ID {value} is not a trained model. You can get all trained models by calling "
                             f"`GET /models/get_trained_models`.")