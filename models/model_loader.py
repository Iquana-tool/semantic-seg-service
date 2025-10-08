import os

import torch
import torch.nn as nn
from typing import Any, Union, List
from collections.abc import Callable
from pydantic import BaseModel, Field


class ModelLoader:
    def is_available(self) -> bool:
        # Implement logic to check if the model can be loaded with the given kwargs
        pass

    def load_model(self, **kwargs) -> torch.nn.Module:
        pass

class BaseModelLoader(ModelLoader):
    """ Base Model loaders can not be serialized because they have specific functions. """
    def __init__(self, loader_function: Callable[Union[List[Any], None], nn.Module], **kwargs):
        """
        Class to handle loading of models.
        :param loader_function: Function that loads the model.
        :param kwargs: Parameters to be passed to the loader function.
        """
        self.loader_function = loader_function
        self.kwargs = kwargs

    def is_available(self):
        # Implement logic to check if the model can be loaded with the given kwargs
        pass

    def load_model(self, **kwargs):
        combined_kwargs = self.kwargs.copy()
        combined_kwargs.update(kwargs)
        return self.loader_function(combined_kwargs)


class PathModelLoader(BaseModel, ModelLoader):
    path_to_model: str = Field(..., description="Path to the model to be loaded. The model must be loadable via 'torch.save'.")

    def is_available(self):
        return os.path.exists(self.path_to_model)

    def load_model(self, **kwargs):
        return torch.load(self.path_to_model, **self.kwargs)
