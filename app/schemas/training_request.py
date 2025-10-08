import os

from pydantic import field_validator

from app.schemas.augmentations import Augmentations
from app.schemas.data_profile import DataProfile
from app.schemas.hyperparams import HyperParams
from paths import DATA_PATH
from pydantic import BaseModel, Field


class TrainingRequest(BaseModel):
    dataset_id: int = Field(default=1, description="Dataset ID")
    model_registry_key: str = Field(default="unet", description="A key from the model registry")
    hyper_params: HyperParams = Field(default_factory=HyperParams, description="Hyperparameters")
    augmentations: Augmentations = Field(default_factory=Augmentations, description="Augmentations")
    data_profile: DataProfile = Field(default_factory=DataProfile, description="DataProfile")
    num_epochs: int = Field(default=1, description="Number of epochs to train.")

    @field_validator('dataset_id')
    def validate_dataset_id(cls, value):
        if str(value) not in os.listdir(DATA_PATH):
            raise ValueError(f"Dataset with ID {value} does not exist in the data path {DATA_PATH}."
                             f"Please make sure to upload the dataset first.")
        return value
