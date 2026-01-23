from logging import getLogger
from typing import Tuple

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from torchvision.io import read_image

from app.schemas.augmentations import Augmentations

logger = getLogger(__name__)


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
        return len(self.image_filenames)

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

        # Apply augmentations if not eval
        if not self.eval:
            image, mask = self.augmentations(image, mask)

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
    dataset = SegmentationTensorDataset(
        image_urls=image_urls,
        mask_urls=mask_urls,
        augmentations=augmentations,
        image_size=image_size,
    )

    # 1. Handle datasets too small to split
    if len(dataset) < min_samples_for_split:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        return loader, None

    # 2. Calculate lengths for split
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    # 3. Perform the split with a generator for reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    # 4. Create the dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Usually don't shuffle validation
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

