from logging import getLogger
from typing import Union
from app.schemas.training_run import TrainingRun
from typing_extensions import Literal

logger = getLogger(__name__)


class ModelInfo:
    def __init__(self,
                 identifier_str: str,
                 name: str,
                 type: Literal["base", "trained"],
                 description: str,
                 tags: list[str],
                 training_run: Union[TrainingRun, None] = None,
                 ):
        """Class to hold information about a segmentation model.
        :param identifier_str: Identifier string. Must be unique.
        :param name: Human readable name of the model.
        :param type: Type of the model: "base" for untrained architectures. "trained" for models that already
            have trained weights.
        :param description: Description of the model.
        :param tags: List of tags for the model.
        :param training_run: An instance of the training run class. This can only be set for models of type "trained".
        :raises ValueError: If the identifier string is invalid.
        """
        self.identifier_str = identifier_str
        self.name = name
        self.type = type
        self.description = description
        self.tags = tags
        if training_run is not None and type == "base":
            raise ValueError("ModelInfo instantiated with TrainingRun object, but type is set to base.")
        self.training_run = training_run

    def to_json(self):
        """Convert the model information to a JSON-serializable dictionary."""
        return {
            "identifier_str": self.identifier_str,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "tags": self.tags,
        }

    def is_base_model(self) -> bool:
        return self.type == "base"
