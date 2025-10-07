from typing import Union, Literal, List

from models.model_loader import ModelLoader
from models.model_info import ModelInfo


class ModelRegistryEntry:
    def __init__(self, info: ModelInfo, loader: ModelLoader):
        self.info = info
        self.loader = loader


class ModelRegistry:
    """ Model Registry class to combine all available models."""
    def __init__(self):
        self.models: dict[str, ModelRegistryEntry] = {}

    def register(self, name, entry: ModelRegistryEntry):
        """ Register a new model in the registry by passing an instance of a ModelRegistryEntry class."""
        self.models[name] = entry

    def list_models(
            self,
            filter_type: Union[Literal["base", "trained"], None] = None,
            filter_availability: bool = False,
            return_as_json: bool = False,
    ) -> List[dict]:
        """ List models after applying some filters.
        :param filter_type: Filter after the type of the model. Available filters: 'base' for base, untrained
            architectures, 'trained' for trained models.
        :param filter_availability: Filter models based on availability. If set to true, only returns models that can
            be loaded in the current repo, eg. for example if their weights and configs are present. Each model loader
            class implements the availability logic by themselves.
        :param return_as_json: If set to true, return model info as json.
        :returns: A list of either the model info objects or a list of json serializable dicts.
        """
        filtered_models = self.models.values()
        if filter_type is not None:
            filtered_models = (
                model_entry
                for model_entry in filtered_models
                if model_entry.info.type == filter_type
            )

        if filter_availability:
            filtered_models = (
                model_entry
                for model_entry in filtered_models
                if model_entry.loader.is_available()
            )

        return [model_entry.info.to_json() if return_as_json else model_entry.info for model_entry in filtered_models]

    def __getitem__(self, item):
        return self.models[item]

    def get_available_new_key_for_base_model(self, base_model_key):
        """ Get a new key for a given base model key."""
        # Get all none base model keys
        existing = [key for key in self.models.keys() if base_model_key in key and key != base_model_key]
        if existing[-1].split("_")[-1] != str(len(existing) - 1):
            # If the last entry is not equal to the length of the list - 1, then we must have some mismatch.
            # Let's sort the array
            sorted_keys = sorted(existing, key=lambda x: int(x.split("_")[-1]))
            for i, key in enumerate(sorted_keys):
                if str(i) != key.split("_")[-1]:
                    # The index i is free, because the array is sorted but the index does not match
                    return base_model_key + "_" + str(i)
        return base_model_key + "_" + str(len(existing))

