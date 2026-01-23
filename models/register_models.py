import os

import torch

from paths import TRAINED_MODEL_WEIGHTS_PATH, TRAINED_MODEL_INFO_PATHS
from models.model_registry import ModelRegistry
from models.model_loader import BaseModelLoader as ModelLoader
from models.model_info import ModelInfo
import segmentation_models_pytorch as smp


def register_models(model_registry: ModelRegistry):
    model_registry.register_model(
        model_info=ModelInfo(
            identifier_str="unet",
            name="UNet",
            type="base",
            description="Placeholder.",
            tags=["Fast", "Legacy"]
        ),
        model_loader=ModelLoader(
            loader_function=smp.Unet,
            kwargs=dict()
        ),
    )
    model_registry.register_model(
        model_info=ModelInfo(
            identifier_str="unet++",
            name="UNet++",
            type="base",
            description="Placeholder.",
            tags=["Fast", "Legacy"]
        ),
        model_loader=ModelLoader(
            loader_function=smp.UnetPlusPlus,
            kwargs=dict()
        ),
    )
    model_registry.register_model(
        model_info=ModelInfo(
            identifier_str="deeplabv3",
            name="DeepLabV3",
            type="base",
            description="Placeholder.",
            tags=["Slow", "Legacy"]
        ),
        model_loader=ModelLoader(
            loader_function=smp.DeepLabV3,
            kwargs=dict()
        ),
    )
    model_registry.register_model(
        model_info=ModelInfo(
            identifier_str="deeplabv3+",
            name="DeepLabV3+",
            type="base",
            description="Placeholder.",
            tags=["Slow", "Legacy"]
        ),
        model_loader=ModelLoader(
            loader_function=smp.DeepLabV3Plus,
            kwargs=dict()
        ),
    )
    for trained_model in os.listdir(TRAINED_MODEL_INFO_PATHS):
        trained_model_path = os.path.join(TRAINED_MODEL_INFO_PATHS, trained_model)
        model_registry.register_model_from_path(trained_model_path)
