import os

import torch

from paths import MODEL_WEIGHTS_PATH
from models.model_registry import ModelRegistry, ModelRegistryEntry
from models.model_loader import BaseModelLoader as ModelLoader
from models.model_info import ModelInfo
import segmentation_models_pytorch as smp


def register_base_models(model_registry: ModelRegistry):
    model_registry.register("unet",
                            ModelRegistryEntry(
                                info=ModelInfo(
                                    identifier_str="unet",
                                    name="UNet",
                                    type="base",
                                    description="Placeholder.",
                                    tags=["Fast", "Legacy"]
                                ),
                                loader=ModelLoader(
                                    loader_function=smp.Unet,
                                    kwargs=dict()
                                ),
                            ))
    model_registry.register("unet++",
                            ModelRegistryEntry(
                                info=ModelInfo(
                                    identifier_str="unet++",
                                    name="UNet++",
                                    type="base",
                                    description="Placeholder.",
                                    tags=["Fast", "Legacy"]
                                ),
                                loader=ModelLoader(
                                    loader_function=smp.UnetPlusPlus,
                                    kwargs=dict()
                                ),
                            ))
    model_registry.register("deeplabv3",
                            ModelRegistryEntry(
                                info=ModelInfo(
                                    identifier_str="deeplabv3",
                                    name="DeepLabV3",
                                    type="base",
                                    description="Placeholder.",
                                    tags=["Slow", "Legacy"]
                                ),
                                loader=ModelLoader(
                                    loader_function=smp.DeepLabV3,
                                    kwargs=dict()
                                ),
                            ))
    model_registry.register("deeplabv3+",
                            ModelRegistryEntry(
                                info=ModelInfo(
                                    identifier_str="deeplabv3+",
                                    name="DeepLabV3+",
                                    type="base",
                                    description="Placeholder.",
                                    tags=["Slow", "Legacy"]
                                ),
                                loader=ModelLoader(
                                    loader_function=smp.DeepLabV3Plus,
                                    kwargs=dict()
                                ),
                            ))


def discover_trained_models(model_registry: ModelRegistry):
    """ Scans the saved models folder and tries to associate a base model with it. """
    saved_weights = os.listdir(MODEL_WEIGHTS_PATH)
    for weight_name in saved_weights:
        full_identifier, ext = weight_name.split(".")
        base_identifier, number = full_identifier.split("_")
        if base_identifier not in model_registry.models:
            # Base model isnt registered, so we skip
            continue
        else:
            base_entry = model_registry.models[base_identifier]

            def load_model_with_weights():
                # Load the base model
                base_model = base_entry.loader.loader_function(**base_entry.loader.kwargs)
                # Load the state dict
                base_model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS_PATH, weight_name)))
                return base_model

            model_registry.register(full_identifier,
                                    ModelRegistryEntry(
                                        info=ModelInfo(
                                            identifier_str=full_identifier,
                                            type="trained",
                                            name=base_entry.info.name,
                                            description=base_entry.info.description,
                                            tags=base_entry.info.tags,
                                        ),
                                        loader=ModelLoader(
                                            loader_function=load_model_with_weights,
                                        )
                                    ))
