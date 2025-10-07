import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from logging import getLogger
from typing import Tuple
from torch.utils.data import random_split
from app.schemas.augmentations import Augmentations
from app.schemas.data_profile import DataProfile
from app.schemas.preprocessing import Preprocessing

logger = getLogger(__name__)


class SegmentationTensorDataset(Dataset):
    """Dataset class."""
    def __init__(
        self,
        dataset_root: str,
        augmentations: Augmentations,
        preprocessing: Preprocessing,
        image_size: Tuple[int, int] = (256, 256),
    ):
        """
        Args:
            dataset_root (str): Folder with 'images/' and 'masks/' subdirs
            augmentations (Augmentations): Apply the specified augmentations
            preprocessing (Preprocessing): Apply the specified preprocessing
            image_size (tuple): Final image size (H, W)
        """
        self.image_dir = os.path.join(dataset_root, "images")
        self.mask_dir = os.path.join(dataset_root, "masks")

        self.image_filenames = sorted(
            [f for f in os.listdir(self.image_dir) if f.endswith(".pt")]
        )

        self.augmentations = augmentations
        self.preprocessing = preprocessing
        self.image_size = list(image_size)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = torch.load(image_path).float()
        mask = torch.load(mask_path)

        # Ensure image is CHW
        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.shape[0] not in [1, 3]:
            image = image.permute(2, 0, 1)

        # Apply augmentations
        image, mask = self.augmentations(image, mask)

        # Resize (after augmentations)
        image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask.unsqueeze(0), self.image_size, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

        # Preprocess
        image = self.preprocessing(image)

        return image, mask


def get_dataloader(
    dataset_path: str,
    augmentations: Augmentations,
    preprocessing: Preprocessing,
    data_profile: DataProfile,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    min_samples_for_split: int = 10,
    seed: int = 42
) -> tuple:
    """
    Returns (train_loader, val_loader, test_loader), or (single_loader, None, None) if not enough data.
    """
    dataset = SegmentationTensorDataset(
        dataset_root=dataset_path,
        augmentations=augmentations,
        preprocessing=preprocessing,
        image_size=data_profile.image_size,
    )
    n_total = len(dataset)
    if n_total < min_samples_for_split:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        return loader, None, None

    n_val = max(1, int(data_profile.val_ratio * n_total))
    n_test = max(1, int(data_profile.test_ratio * n_total))
    n_train = n_total - n_val - n_test

    if n_train < 3:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        return loader, None, None

    g = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=g)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

