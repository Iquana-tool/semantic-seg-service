import os

import torch
import torch.nn as nn
from typing import Any, Union, List
from collections.abc import Callable
from pydantic import BaseModel, Field


class ModelLoader(BaseModel):
    def is_available(self) -> bool:
        # Implement logic to check if the model can be loaded with the given kwargs
        pass

    def load_model(self, **kwargs) -> torch.nn.Module:
        pass

class BaseModelLoader(ModelLoader):
    """ Base Model loaders can not be serialized because they have specific functions. """
    loader_function: Callable
    kwargs: dict = Field(..., description="Keyword arguments to pass to the loader function")

    def is_available(self):
        # Implement logic to check if the model can be loaded with the given kwargs
        pass

    def load_model(self, **kwargs):
        combined_kwargs = self.kwargs.copy()
        combined_kwargs.update(kwargs)
        return self.loader_function(combined_kwargs)


class PathModelLoader(ModelLoader):
    path_to_model: str = Field(..., description="Path to the model to be loaded. The model must be loadable via 'torch.save'.")

    def is_available(self):
        return os.path.exists(self.path_to_model)

    def load_model(self, **kwargs):
        return torch.load(self.path_to_model, **self.kwargs)
