import os
import warnings
from logging import getLogger
from pathlib import Path

import nanoid
import segmentation_models_pytorch as smp
from iquana_toolbox.schemas.models import SemanticSegmentationModels as ModelInfo, SemanticSegmentationModels

from models.model_loader import ModelLoader, PathModelLoader, BaseModelLoader
from paths import TRAINED_MODEL_INFO_PATHS, TRAINED_MODEL_WEIGHTS_PATH

logger = getLogger(__name__)


class ModelRegistry:
    def __init__(self):
        """Registry to hold and manage multiple models."""
        self.model_infos: dict[str, ModelInfo] = {}
        self.model_loaders: dict[str, ModelLoader] = {}
        self.model_keys: set[str] = set()
        self.register_model(
            model_info=SemanticSegmentationModels(
                registry_key="unet",
                name="UNet",
                description="A symmetric encoder-decoder network using skip connections to retain spatial information. Excellent for medical imaging and small datasets.",
                tags=["Fast", "Skip-Connections", "Standard"],
                number_of_parameters=24400000,  # Approx for ResNet-34 backbone
                pretrained=False,
                finetunable=False,
                trainable=True,
                label_hierarchy=None,
                training_task_id=None
            ),
            model_loader=BaseModelLoader(
                loader_function=smp.Unet,
            ),
        )

        self.register_model(
            model_info=SemanticSegmentationModels(
                registry_key="unet++",
                name="UNet++",
                description="An evolution of UNet with nested and dense skip pathways designed to reduce the semantic gap between feature maps.",
                tags=["Intermediate", "Dense-Connections", "High-Precision"],
                number_of_parameters=26100000,
                pretrained=False,
                finetunable=False,
                trainable=True,
                label_hierarchy=None,
                training_task_id=None
            ),
            model_loader=BaseModelLoader(
                loader_function=smp.UnetPlusPlus,
            ),
        )

        self.register_model(
            model_info=SemanticSegmentationModels(
                registry_key="deeplabv3",
                name="DeepLabV3",
                description="Utilizes Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context by using filters at multiple sampling rates.",
                tags=["Slow", "Atrous-Convolution", "Context-Aware"],
                number_of_parameters=39600000,
                pretrained=False,
                finetunable=False,
                trainable=True,
                label_hierarchy=None,
                training_task_id=None
            ),
            model_loader=BaseModelLoader(
                loader_function=smp.DeepLabV3,
            ),
        )

        self.register_model(
            model_info=SemanticSegmentationModels(
                registry_key="deeplabv3plus",
                name="DeepLabV3+",
                description="Extends DeepLabV3 by adding a simple yet effective decoder module to refine the segmentation results along object boundaries.",
                tags=["Slow", "State-of-the-Art", "Boundary-Refinement"],
                number_of_parameters=40000000,
                pretrained=False,
                finetunable=False,
                trainable=True,
                label_hierarchy=None,
                training_task_id=None
            ),
            model_loader=BaseModelLoader(
                loader_function=smp.DeepLabV3Plus,
                kwargs=dict(encoder_name="resnet34", encoder_weights="imagenet")
            ),
        )

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
            return

        self.model_infos[model_info.registry_key] = model_info
        self.model_loaders[model_info.registry_key] = model_loader
        self.model_keys.add(model_info.registry_key)
        logger.info(f"Registered model {model_info.registry_key}. Model is loadable: {model_loader.is_loadable()}")

    def register_model_from_path(self, model_info_path: str):
        model_info = ModelInfo.model_validate_json(Path(model_info_path).read_text())
        if os.path.exists(model_info_path) and os.path.isfile(model_info_path):
            model_load_path = os.path.join(TRAINED_MODEL_WEIGHTS_PATH, f"{os.path.split(model_info_path)[-1].split(".")[0]}.pth")
            model_loader = PathModelLoader(model_load_path)
            self.register_model(model_info, model_loader)
        else:
            warnings.warn(f"Saved model path {model_info_path} does not exist or is not a file. "
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
            model_info_json_path = Path(os.path.join(TRAINED_MODEL_INFO_PATHS, f"{registry_key}.json"))
            if os.path.exists(model_info_json_path):
                return ModelInfo.model_validate_json(model_info_json_path.read_text())
            raise KeyError(f"Model info with identifier '{registry_key}' is not registered and no file was found "
                           f"on disk.")
        else:
            return self.model_infos[registry_key]

    def get_model_loader(self, registry_key: str) -> ModelLoader:
        """Get the model loader for the given identifier."""
        if registry_key not in self.model_loaders:
            model_weights_path = Path(os.path.join(TRAINED_MODEL_WEIGHTS_PATH, f"{registry_key}.pth"))
            if os.path.exists(model_weights_path) and os.path.isfile(model_weights_path):
                return PathModelLoader(model_weights_path)
            raise KeyError(f"Model loader with identifier {registry_key} is not registered and no file was found "
                           f"on disk.")
        else:
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
