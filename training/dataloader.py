import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
from typing import Optional, Tuple


class SegmentationTensorDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        augment: bool = False,
        normalize: bool = True,
        image_size: Tuple[int, int] = (256, 256),
    ):
        """
        Args:
            dataset_root (str): Folder with 'images/' and 'masks/' subdirs
            augment (bool): Apply heavy data augmentation
            normalize (bool): Normalize image with mean/std
            image_size (tuple): Final image size (H, W)
        """
        self.image_dir = os.path.join(dataset_root, "images")
        self.mask_dir = os.path.join(dataset_root, "masks")

        self.image_filenames = sorted(
            [f for f in os.listdir(self.image_dir) if f.endswith(".pt")]
        )

        self.augment = augment
        self.normalize = normalize
        self.image_size = image_size

        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        # Color jitter transform (image only)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05,
        )

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = torch.load(image_path).float()
        mask = torch.load(mask_path).long()

        # Ensure image is CHW
        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.shape[0] not in [1, 3]:
            image = image.permute(2, 0, 1)

        if self.augment:
            image, mask = self.apply_augmentations(image, mask)

        # Resize (after augmentations)
        image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask.unsqueeze(0), self.image_size, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

        if self.normalize:
            image = TF.normalize(image, self.mean, self.std)

        return image, mask

    def apply_augmentations(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Random horizontal flip
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotation (±15 degrees)
        angle = random.uniform(-15, 15)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Apply color jitter only to images
        image = self.color_jitter(image)

        # Random crop
        crop_size = min(image.shape[1], image.shape[2], 224)
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(crop_size, crop_size))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        return image, mask


def get_dataloader(
    dataset_path: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    augment: bool = True,
    normalize: bool = True,
    image_size: Tuple[int, int] = (256, 256),
) -> DataLoader:
    dataset = SegmentationTensorDataset(
        dataset_root=dataset_path,
        augment=augment,
        normalize=normalize,
        image_size=image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader

