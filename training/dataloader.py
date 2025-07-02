import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Callable
import random


class SegmentationTensorDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ):
        """
        Args:
            dataset_root (str): Path to dataset folder containing 'images/' and 'masks/' subfolders with .pt files
            transform (Callable, optional): Transform to apply to the image
            mask_transform (Callable, optional): Transform to apply to the mask
        """
        self.image_dir = os.path.join(dataset_root, "images")
        self.mask_dir = os.path.join(dataset_root, "masks")

        self.image_filenames = sorted(
            [f for f in os.listdir(self.image_dir) if f.endswith(".pt")]
        )

        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = torch.load(image_path)
        mask = torch.load(mask_path)

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


def get_dataloader(
    dataset_path: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    mask_transform: Optional[Callable] = None,
) -> DataLoader:
    dataset = SegmentationTensorDataset(
        dataset_root=dataset_path,
        transform=transform,
        mask_transform=mask_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader
