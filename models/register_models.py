import os

import segmentation_models_pytorch as smp
from iquana_toolbox.schemas.models import SemanticSegmentationModels
from models.model_loader import BaseModelLoader as ModelLoader
from models.model_registry import ModelRegistry
from paths import TRAINED_MODEL_INFO_PATHS


def register_models(model_registry: ModelRegistry):
    model_registry.register_model(
        model_info=SemanticSegmentationModels(
            registry_key="unet",
            name="UNet",
            description="A symmetric encoder-decoder network using skip connections to retain spatial information. Excellent for medical imaging and small datasets.",
            tags=["Fast", "Skip-Connections", "Standard"],
            number_of_parameters=24400000,  # Approx for ResNet-34 backbone
            pretrained=False,
            finetunable=False,
            trainable=True,
            label_hierarchy=None
        ),
        model_loader=ModelLoader(
            loader_function=smp.Unet,
            kwargs=dict(encoder_name="resnet34", encoder_weights="imagenet")
        ),
    )

    model_registry.register_model(
        model_info=SemanticSegmentationModels(
            registry_key="unet++",
            name="UNet++",
            description="An evolution of UNet with nested and dense skip pathways designed to reduce the semantic gap between feature maps.",
            tags=["Intermediate", "Dense-Connections", "High-Precision"],
            number_of_parameters=26100000,
            pretrained=False,
            finetunable=False,
            trainable=True,
            label_hierarchy=None
        ),
        model_loader=ModelLoader(
            loader_function=smp.UnetPlusPlus,
            kwargs=dict(encoder_name="resnet34", encoder_weights="imagenet")
        ),
    )

    model_registry.register_model(
        model_info=SemanticSegmentationModels(
            registry_key="deeplabv3",
            name="DeepLabV3",
            description="Utilizes Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context by using filters at multiple sampling rates.",
            tags=["Slow", "Atrous-Convolution", "Context-Aware"],
            number_of_parameters=39600000,
            pretrained=False,
            finetunable=False,
            trainable=True,
            label_hierarchy=None
        ),
        model_loader=ModelLoader(
            loader_function=smp.DeepLabV3,
            kwargs=dict(encoder_name="resnet34", encoder_weights="imagenet")
        ),
    )

    model_registry.register_model(
        model_info=SemanticSegmentationModels(
            registry_key="deeplabv3plus",
            name="DeepLabV3+",
            description="Extends DeepLabV3 by adding a simple yet effective decoder module to refine the segmentation results along object boundaries.",
            tags=["Slow", "State-of-the-Art", "Boundary-Refinement"],
            number_of_parameters=40000000,
            pretrained=False,
            finetunable=False,
            trainable=True,
            label_hierarchy=None
        ),
        model_loader=ModelLoader(
            loader_function=smp.DeepLabV3Plus,
            kwargs=dict(encoder_name="resnet34", encoder_weights="imagenet")
        ),
    )
    for trained_model in os.listdir(TRAINED_MODEL_INFO_PATHS):
        trained_model_path = os.path.join(TRAINED_MODEL_INFO_PATHS, trained_model)
        model_registry.register_model_from_path(trained_model_path)
