from logging import getLogger
from typing import Union, Optional

from pydantic import BaseModel, Field

from app.schemas.training_run import TrainingRun
from typing_extensions import Literal

logger = getLogger(__name__)


class ModelInfo(BaseModel):
    identifier_str: str = Field(..., description="Model registry key.")
    name: str = Field(..., description="Human readable model name.")
    type: Literal["base", "trained"] = Field("base", description="Whether the model is just a base architecture or "
                                                                 "already trained.")
    description: str = Field(..., description="Free text model description.")
    tags: list[str] = Field(default_factory=list, description="List of tags associated with this model.")
    training_run: Optional[TrainingRun] = Field(default=None, description="A training run object associated with this model. "
                                                                  "None for base models.")

    def is_base_model(self) -> bool:
        return self.type == "base"
