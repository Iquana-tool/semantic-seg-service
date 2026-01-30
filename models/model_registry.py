import os
import warnings
from logging import getLogger
from pathlib import Path

import nanoid

from schemas.models import SemanticSegmentationModels as ModelInfo
from models.model_loader import ModelLoader, PathModelLoader

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
        key = model_info.registry_key
        if key in self.model_keys:
            raise ValueError(f"Model identifier '{key}' is already registered.")
        
        self.model_infos[model_info.registry_key] = model_info
        self.model_loaders[model_info.registry_key] = model_loader
        self.model_keys.add(model_info.registry_key)
        logger.info(f"Registered model {model_info.registry_key}. Model is loadable: {model_loader.is_loadable()}")

    def register_model_from_path(self, model_info_path: str):
        model_info = ModelInfo.model_validate_json(Path(model_info_path).read_text())
        if os.path.exists(model_info.model_path) and os.path.isfile(model_info.model_path):
            model_loader = PathModelLoader(model_info.model_path)
            self.register_model(model_info, model_loader)
        else:
            warnings.warn(f"Saved model path {model_info.model_path} does not exist or is not a file. "
                          f"Not registering: {model_info_path}")

    def get_new_key(self):
        finished = False
        while not finished:
            new_key = nanoid.generate(size=8)
            if new_key not in self.model_keys:
                finished = True
        return new_key

    def get_model_info(self, registry_key: str) -> ModelInfo:
        """Get the model information for the given identifier."""
        if registry_key not in self.model_infos:
            raise KeyError(f"Model with identifier {registry_key} is not registered.")
        return self.model_infos[registry_key]

    def get_model_loader(self, registry_key: str) -> ModelLoader:
        """Get the model loader for the given identifier."""
        if registry_key not in self.model_loaders:
            raise KeyError(f"Model loader with identifier {registry_key} is not registered.")
        return self.model_loaders[registry_key]

    def check_model_is_loadable(self, registry_key: str) -> bool:
        """Check if the model with the given identifier is loadable."""
        model = self.get_model_loader(registry_key)
        return model.is_loadable()

    def list_models(self, only_return_available: bool = True) -> list[ModelInfo]:
        """List all registered models.
        :param only_return_available: If True, only return models that are loadable. Default is True.
        :return: List of ModelInfo objects.
        """
        if only_return_available:
            # Only return loadable models
            return [self.model_infos[k] for k in self.model_keys if self.model_loaders[k].is_loadable()]
        return list(self.model_infos.values())

    def load_model(self, registry_key: str):
        """Load the model with the given identifier."""
        return self.get_model_loader(registry_key).load_model()
