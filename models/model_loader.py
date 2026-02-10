import os

import torch


class ModelLoader:
    def is_loadable(self) -> bool:
        # Implement logic to check if the model can be loaded with the given kwargs
        pass

    def load_model(self, **kwargs) -> torch.nn.Module:
        pass

class BaseModelLoader(ModelLoader):
    """ Base Model loaders can not be serialized because they have specific functions. """
    def __init__(self, loader_function, **kwargs):
        """
        Class to handle loading of models.
        :param loader_function: Function that loads the model.
        :param kwargs: Parameters to be passed to the loader function.
        """
        self.loader_function = loader_function
        self.kwargs = kwargs

    def is_loadable(self):
        # Implement logic to check if the model can be loaded with the given kwargs
        pass

    def load_model(self, **kwargs):
        combined_kwargs = self.kwargs.copy()
        combined_kwargs.update(kwargs)
        return self.loader_function(**combined_kwargs)


class PathModelLoader(ModelLoader):
    def __init__(self, path_to_model, **kwargs):
        """
        Class to handle loading of models.
        :param loader_function: Function that loads the model.
        :param kwargs: Parameters to be passed to the loader function.
        """
        self.loader_function = None
        self.path_to_model = path_to_model
        self.kwargs = kwargs

    def is_loadable(self):
        return os.path.exists(self.path_to_model)

    def load_model(self, **kwargs):
        print(f"Loading model from  {self.path_to_model}")
        return torch.load(self.path_to_model, weights_only=False, **self.kwargs)
