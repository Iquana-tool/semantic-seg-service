import os
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Tuple
from models import MODEL_REGISTRY
from paths import DATA_PATH, MODEL_PATH
from models import parse_weight_file_name
from typing import Union


class TrainingRequest(BaseModel):
    dataset_id: int = Field(default=1, description="Dataset ID")
    model_identifier: str = Field(default="unet", description="Identifier for the model to be trained. "
                                                      "Can either be a registry key, or an int representing a trained"
                                                                          " model")

    # Parameters
    epochs: int = Field(default=50, description="Number of epochs to train the model.")
    batch_size: int = Field(default=64, description="Batch size to train the model.")
    lr: float = Field(default=0.0001, description="Learning rate to train the model.")
    augment: bool = Field(default=True, description="Whether to augment the dataset. This should be done for small "
                                                    "datasets, but can be left out for bigger datasets.")
    image_size: Optional[Tuple[int, int]] = Field(default=(256, 256), description="Image size to use. Smaller values "
                                                                                  "may lead to faster training, "
                                                                                  "but may also lead to "
                                                                                  "loss of information.")
    num_classes: int = Field(default=2, description="Number of classes that are present on the masks. Corresponds to the "
                                                    "number of labels.")
    in_channels: int = 3
    early_stopping: bool = Field(default=True, description="Whether to use early stopping during training. "
                                                           "This will stop training if the validation loss "
                                                           "does not improve for 5 epochs.")

    @field_validator('model_identifier')
    def validate_model_identifier(cls, value):
        if not value:
            raise ValueError("Model identifier cannot be empty.")

        try:
            # Try to convert the value to an integer
            # This is important for proper job id handling
            new_value = int(value)
            return new_value
        except ValueError:
            return value

    @field_validator('dataset_id')
    def validate_dataset_id(cls, value):
        if str(value) not in os.listdir(DATA_PATH):
            raise ValueError(f"Dataset with ID {value} does not exist in the data path {DATA_PATH}."
                             f"Please make sure to upload the dataset first.")
        return value
