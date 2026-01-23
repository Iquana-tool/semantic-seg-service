import os
from typing import List

from pydantic import BaseModel, Field
from pydantic import field_validator

from app.schemas.augmentations import Augmentations
from app.schemas.hyperparams import HyperParams
from paths import DATA_PATH


class TrainingRequest(BaseModel):
    image_urls: List[str] = Field(..., description="List of image urls to train on.")
    mask_urls: List[str] = Field(..., description="List of mask urls to train on.")
    val_ratio: float = Field(default=0.1, description="Ratio of training data to validation data.")
    image_size: tuple = Field((224, 224), description="Image size.")
    model_registry_key: str = Field(default="unet", description="A key from the model registry")
    hyper_params: HyperParams = Field(default_factory=HyperParams, description="Hyperparameters")
    augmentations: Augmentations = Field(default_factory=Augmentations, description="Augmentations")
    num_epochs: int = Field(default=1, description="Number of epochs to train.")
