import os

from pydantic import BaseModel, Field, field_validator

from app.schemas.augmentations import Augmentations
from app.schemas.data_profile import DataProfile
from paths import DATA_PATH


class HyperParams(BaseModel):
    batch_size: int = Field(default=32, description="Batch size to use for training.")
    learning_rate: float = Field(default=0.001, description="Learning rate to use for training.")
    lr_scheduler: str = Field(default=None, description="Learning rate scheduler to use for training.")
    early_stopping_patience: int = Field(default=25, description="Number of epochs to wait for improvement. If this "
                                                                 "number is exceeded without the model improving, then "
                                                                 "training is stopped regardless of the amount of "
                                                                 "remaining epochs. If this value is smaller or equal "
                                                                 "to zero, no early stopping will be applied.")


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
