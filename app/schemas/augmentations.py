import random
from typing import Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, Field
from torchvision import transforms as T
from torchvision.transforms import functional as TF


class Augmentations(BaseModel):
    """ Schema for augmentations. Augmentations are any image operations that should be applied before an image is fed to
        the model. This can include cropping etc, but also normalizing.
    """
    crop_relative_min: float = Field(
        default=0.25,
        description="Minimum relative size for random crop (e.g., 0.25 = up to 25% of image size)."
    )
    rotation_degrees: Optional[int] = Field(
        default=None,
        description="Degrees for random rotation. If None, no rotation is applied."
    )
    use_horizontal_flip: bool = Field(
        default=True,
        description="Whether to use horizontal flip of images."
    )
    use_vertical_flip: bool = Field(
        default=True,
        description="Whether to use vertical flip of images."
    )
    color_jitter: Optional[Tuple[float, float, float, float]] = Field(
        default=None,
        description="Color jitter parameters (Brightness, Contrast, Saturation, Hue). If None, not applied."
    )

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to the input tensor.
        Args:
            img: Input tensor of shape (C, H, W).
            mask: Input tensor of shape (C, H, W).
        Returns:
            Augmented tensor.
        """
        # Handle random crop with relative size
        if self.crop_relative_min > 1.:
            h, w = img.shape[-2], img.shape[-1]
            min_h = int(h * (1 - self.crop_relative_min))
            min_w = int(w * (1 - self.crop_relative_min))
            crop_h = np.random.randint(min_h, h)
            crop_w = np.random.randint(min_w, w)
            i, j, h, w = T.RandomCrop.get_params(img, (crop_h, crop_w))
            img = TF.crop(img, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
        if self.rotation_degrees is not None:
            angle = np.random.uniform(-self.rotation_degrees, self.rotation_degrees)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask.unsqueeze(0), angle, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
        if self.use_horizontal_flip and random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if self.use_vertical_flip and random.random() < 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        if self.color_jitter is not None:
            img =T.ColorJitter(
                    brightness=self.color_jitter[0],
                    contrast=self.color_jitter[1],
                    saturation=self.color_jitter[2],
                    hue=self.color_jitter[3]
                )(img)
        return img, mask
