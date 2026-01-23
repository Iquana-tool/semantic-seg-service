import json
import os
from logging import getLogger
from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field, computed_field
from typing_extensions import Literal

from app.schemas.hyperparams import HyperParams
from app.schemas.training_progress import TrainingProgress
from app.schemas.training_request import TrainingRequest
from paths import TRAINED_MODEL_WEIGHTS_PATH, TRAINED_MODEL_INFO_PATHS

logger = getLogger(__name__)


class ModelInfo(BaseModel):
    identifier_str: str = Field(..., description="Model registry key.")
    name: str = Field(..., description="Human readable model name.")
    type: Literal["base", "trained"] = Field("base", description="Whether the model is just a base architecture or "
                                                                 "already trained.")
    description: str = Field(..., description="Free text model description.")
    tags: list[str] = Field(default_factory=list, description="List of tags associated with this model.")
    training_req: Optional[TrainingRequest] = Field(default=None,
                                     description="Class object to track hyperparameters.")
    training_progress: Optional[TrainingProgress] = Field(default=None,
                                                          description="Class object to track training progress.")

    def is_base_model(self) -> bool:
        return self.type == "base"

    @computed_field
    def model_path(self) -> str:
        if self.is_base_model():
            return "BaseModel - No path"
        else:
            return os.path.join(TRAINED_MODEL_WEIGHTS_PATH, self.identifier_str + ".pt")

    def save_to_disk(self) -> None:
        save_path = Path(os.path.join(TRAINED_MODEL_INFO_PATHS, str(self.identifier_str) + ".json"))
        save_path.write_text(self.model_dump_json(indent=4))
