import os
from typing import Union, Literal, List
from logging import getLogger
from pydantic import BaseModel, Field
from paths import MODEL_REGISTRY_ENTRY_PATHS
from models.model_loader import ModelLoader, PathModelLoader
from models.model_info import ModelInfo


logger = getLogger(__name__)


class ModelRegistryEntry(BaseModel):
    info: ModelInfo = Field(ModelInfo, alias="info", description="Model info holds all necessary info about the model.")
    loader: ModelLoader = Field(ModelLoader, alias="loader", description="Model loader class holds functionality about "
                                                                         "loading a model.")

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.model.json())


class ModelRegistry(BaseModel):
    """ Model Registry class to combine all available models."""
    models: dict[str, ModelRegistryEntry] = Field(default_factory=dict,
                                                  description="Dictionary mapping from registry keys to model entries.")

    def register(self, name, entry: ModelRegistryEntry):
        """ Register a new model in the registry by passing an instance of a ModelRegistryEntry class."""
        self.models[name] = entry

    def register_new_model_from_base_model(self, base_model_key):
        new_key = self.get_available_new_key_for_base_model(base_model_key)
        base_entry_copy = self.models[base_model_key].model_copy()
        base_entry_copy.info.identifier_str = new_key
        base_entry_copy.info.type = "trained"
        self.register(new_key, base_entry_copy)
        return self.models[new_key]

    def list_models(
            self,
            filter_type: Union[Literal["base", "trained"], None] = None,
            filter_availability: bool = False,
            filter_dataset: int = None,
            return_as_json: bool = False,
    ) -> List[dict]:
        """ List models after applying some filters.
        :param filter_type: Filter after the type of the model. Available filters: 'base' for base, untrained
            architectures, 'trained' for trained models.
        :param filter_availability: Filter models based on availability. If set to true, only returns models that can
            be loaded in the current repo, eg. for example if their weights and configs are present. Each model loader
            class implements the availability logic by themselves.
        :param filter_dataset: Filter models based on a specific dataset.
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

        if filter_dataset is not None:
            filtered_models = (
                model_entry
                for model_entry in filtered_models
                if model_entry.info.is_base_model() or model_entry.info.training_run.dataset_identifier == filter_dataset
            )

        return [model_entry.info.model_dump_json() if return_as_json else model_entry.info.model_dump() for model_entry in filtered_models]

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

    def get_task_id_of_model(self, key):
        return self.models[key].info.training_run.task_id

    def delete_model(self, key):
        entry = self.models[key]
        if entry.info.is_base_model():
            return {
                "success": False,
                "message": "Cannot delete base models!"
            }
        else:
            try:
                # Must be a pathmodel loader, because it is not base model
                loader: PathModelLoader = entry.loader
                os.remove(loader.path_to_model)
            except Exception as e:
                # Shouldnt happen, but okay, we move
                logger.error(e)
                pass
            finally:
                registry_entry_path = os.path.join(MODEL_REGISTRY_ENTRY_PATHS, key + ".json")
                os.remove(registry_entry_path)
                del self.models[key]
                return {
                    "success": True,
                    "message": f"Successfully deleted model {key}"
                }
