import json
import os
from pathlib import Path
from typing import Union, Literal, List, override
from logging import getLogger

import nanoid
from pydantic import BaseModel, Field, ValidationError
from paths import TRAINED_MODEL_INFO_PATHS
from models.model_loader import ModelLoader, PathModelLoader
from models.model_info import ModelInfo

logger = getLogger(__name__)


class ModelRegistry:
    def __init__(self):
        """Registry to hold and manage multiple models."""
        self.model_infos: dict[str, ModelInfo] = {}
        self.model_loaders: dict[str, ModelLoader] = {}
        self.model_keys: set[str] = set()

    def register_model(self,
                       model_info: ModelInfo,
                       model_loader: ModelLoader):
        """Register a new model in the registry.
        :param model_info: ModelInfo object.
        :param model_loader: ModelLoader object.
        :raises ValueError: If the model identifier is already registered.
        """
        if model_info.identifier_str in self.model_infos:
            raise ValueError(f"Model with identifier {model_info.identifier_str} is already registered.")
        if model_info.identifier_str in self.model_loaders:
            raise ValueError(f"Model loader with identifier {model_info.identifier_str} is already registered.")
        if model_info.identifier_str in self.model_keys:
            raise KeyError(f"Model with identifier {model_info.identifier_str} is already registered.")
        self.model_infos[model_info.identifier_str] = model_info
        self.model_loaders[model_info.identifier_str] = model_loader
        self.model_keys.add(model_info.identifier_str)
        logger.info(f"Registered model {model_info.identifier_str}. Model is loadable: {model_loader.is_loadable()}")

    def register_model_from_path(self, model_info_path: str):
        model_info = ModelInfo.model_validate_json(Path(model_info_path).read_text())
        if os.path.exists(model_info.model_path) and os.path.isfile(model_info.model_path):
            model_loader = PathModelLoader(model_info.model_path)
            self.register_model(model_info, model_loader)
        else:
            raise Warning(f"Saved model path {model_info.model_path} does not exist or is not a file. Not registering: {model_info_path}")

    def get_new_key(self):
        finished = False
        while not finished:
            new_key = nanoid.generate(size=8)
            if new_key not in self.model_keys:
                finished = True
        return new_key

    def get_model_info(self, identifier_str: str) -> ModelInfo:
        """Get the model information for the given identifier."""
        if identifier_str not in self.model_infos:
            raise KeyError(f"Model with identifier {identifier_str} is not registered.")
        return self.model_infos[identifier_str]

    def get_model_loader(self, identifier_str: str) -> ModelLoader:
        """Get the model loader for the given identifier."""
        if identifier_str not in self.model_loaders:
            raise KeyError(f"Model loader with identifier {identifier_str} is not registered.")
        return self.model_loaders[identifier_str]

    def check_model_is_loadable(self, identifier_str: str) -> bool:
        """Check if the model with the given identifier is loadable."""
        model = self.get_model_loader(identifier_str)
        return model.is_loadable()

    def list_models(self, only_return_available: bool = True) -> list[ModelInfo]:
        """List all registered models.
        :param only_return_available: If True, only return models that are loadable. Default is True.
        :return: List of ModelInfo objects.
        """
        if only_return_available:
            # Only return loadable models
            return [model_info for model_info, model_loader in zip(self.model_infos.values(), self.model_loaders.values()) if model_loader.is_loadable()]
        return list(self.model_infos.values())

    def load_model(self, identifier_str: str):
        """Load the model with the given identifier."""
        return self.get_model_loader(identifier_str).load_model()
