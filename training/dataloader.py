import random
from logging import getLogger
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop, ColorJitter
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data import Dataset
from torchvision.io import read_image

from iquana_toolbox.schemas.training import Augmentations

logger = getLogger(__name__)


def augment(augmentations: Augmentations, img: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply augmentations to the input tensor.
    Args:
        img: Input tensor of shape (C, H, W).
        mask: Input tensor of shape (C, H, W).
    Returns:
        Augmented tensor.
    """
    # Handle random crop with relative size
    if augmentations.crop_relative_min > 1.:
        h, w = img.shape[-2], img.shape[-1]
        min_h = int(h * (1 - augmentations.crop_relative_min))
        min_w = int(w * (1 - augmentations.crop_relative_min))
        crop_h = np.random.randint(min_h, h)
        crop_w = np.random.randint(min_w, w)
        i, j, h, w = RandomCrop.get_params(img, (crop_h, crop_w))
        img = TF.crop(img, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
    if augmentations.rotation_degrees is not None:
        angle = np.random.uniform(-augmentations.rotation_degrees, augmentations.rotation_degrees)
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask.unsqueeze(0), angle, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
    if augmentations.use_horizontal_flip and random.random() < 0.5:
        img = TF.hflip(img)
        mask = TF.hflip(mask)
    if augmentations.use_vertical_flip and random.random() < 0.5:
        img = TF.vflip(img)
        mask = TF.vflip(mask)
    if augmentations.color_jitter is not None:
        img = ColorJitter(
            brightness=augmentations.color_jitter[0],
            contrast=augmentations.color_jitter[1],
            saturation=augmentations.color_jitter[2],
            hue=augmentations.color_jitter[3]
        )(img)
    return img, mask


class SegmentationTensorDataset(Dataset):
    """Dataset class."""

    def __init__(
            self,
            image_urls: list[str],
            mask_urls: list[str],
            augmentations: Augmentations,
            eval: bool = False,
            mean: list[float] = None,
            std: list[float] = None,
            image_size: Tuple[int, int] = (256, 256),
    ):
        """
        Args:
            image_urls (list[str]): list of image urls.
            mask_urls (list[str]): list of mask urls.
            augmentations (Augmentations): Apply the specified augmentations
            eval (bool): If True, this dataset is used for validation/ testing and augmentations should not be applied.
            mean (list[float]): Mean values to normalize image to. If None, no normalization is applied.
            std (list[float]): Std values to normalize image to. If None, no normalization is applied.
            image_size (tuple): Final image size (H, W)
        """
        self.image_urls = image_urls
        self.mask_urls = mask_urls
        self.eval = eval

        self.augmentations = augmentations
        self.mean = mean
        self.std = std
        self.image_size = list(image_size)

    def __len__(self):
        return len(self.image_urls)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image_url = self.image_urls[idx]
        mask_url = self.mask_urls[idx]

        image = read_image(image_url).float() / 255.0  # Normalize images to 0,1
        mask = read_image(mask_url).float()  # Masks stay in their classes.

        # Ensure image is CHW
        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.shape[0] not in [1, 3]:
            image = image.permute(2, 0, 1)

        # Ensure mask is HW
        if mask.ndim > 2:
            mask = mask.squeeze()

        # Apply augmentations if not eval
        if not self.eval:
            image, mask = augment(self.augmentations, image, mask)

        # Resize (after augmentations)
        image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask.unsqueeze(0), self.image_size, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

        # Normalize
        if self.mean is not None and self.std is not None:
            image = TF.normalize(image, self.mean, self.std)
        return image, mask


def get_dataloader(
        image_urls: list[str],
        mask_urls: list[str],
        augmentations: Augmentations,
        image_size: Tuple[int, int] = (256, 256),
        val_ratio: float = 0.1,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 4,
        min_samples_for_split: int = 10,
        seed: int = 42
) -> tuple:
    # 1. Create the base dataset (initially without augmentations)
    full_dataset = SegmentationTensorDataset(
        image_urls=image_urls,
        mask_urls=mask_urls,
        augmentations=None,  # Stay clean for now
        image_size=image_size,
    )
    dataset_size = len(image_urls)

    # 1. Handle tiny datasets
    if dataset_size < min_samples_for_split:
        train_ds = SegmentationTensorDataset(
            image_urls, mask_urls, augmentations, eval=False, image_size=image_size
        )
        return DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle), None

    # 2. Create split indices
    indices = list(range(dataset_size))
    val_size = max(1, int(dataset_size * val_ratio))
    train_size = dataset_size - val_size

    # Use torch.random for reproducibility
    train_indices, val_indices = random_split(
        indices, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # 3. Create two distinct dataset instances
    # This ensures train gets augmentations and val gets eval=True
    train_dataset = Subset(
        SegmentationTensorDataset(image_urls, mask_urls, augmentations, eval=False, image_size=image_size),
        train_indices
    )
    val_dataset = Subset(
        SegmentationTensorDataset(image_urls, mask_urls, augmentations, eval=True, image_size=image_size),
        val_indices
    )

    # 4. Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
