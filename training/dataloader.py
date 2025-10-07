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
        eval: bool = False,
        mean: list[float] = None,
        std: list[float] = None,
        image_size: Tuple[int, int] = (256, 256),
    ):
        """
        Args:
            dataset_root (str): Folder with 'images/' and 'masks/' subdirs
            augmentations (Augmentations): Apply the specified augmentations
            eval (bool): If True, this dataset is used for validation/ testing and augmentations should not be applied.
            mean (list[float]): Mean values to normalize image to. If None, no normalization is applied.
            std (list[float]): Std values to normalize image to. If None, no normalization is applied.
            image_size (tuple): Final image size (H, W)
        """
        self.image_dir = os.path.join(dataset_root, "images")
        self.mask_dir = os.path.join(dataset_root, "masks")
        self.eval = eval
        self.image_filenames = [f for f in os.listdir(self.image_dir) if f.endswith(".pt")]

        self.augmentations = augmentations
        self.mean = mean
        self.std = std
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
    dataset_path: str,
    augmentations: Augmentations,
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
        image_size=data_profile.image_size,
    )

    if len(dataset) < min_samples_for_split:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        return loader, None, None

    # Create the datasets
    train_set = SegmentationTensorDataset(
        dataset_root=dataset_path,
        augmentations=augmentations,
        image_size=data_profile.image_size,
    )
    val_set = SegmentationTensorDataset(
        dataset_root=dataset_path,
        augmentations=augmentations,
        eval=True,
        image_size=data_profile.image_size,
    )
    test_set = SegmentationTensorDataset(
        dataset_root=dataset_path,
        augmentations=augmentations,
        eval=True,
        image_size=data_profile.image_size,
    )

    # Compute the filenames as sets for easy set subtraction
    filenames_set = set(dataset.image_filenames)
    train_filenames_set = set(data_profile.train_files)
    val_filenames_set = set(data_profile.val_files)
    test_filenames_set = set(data_profile.test_files)
    remaining_filenames_set = filenames_set - train_filenames_set - val_filenames_set - test_filenames_set

    # Update the dataset to only include the remaining filenames.
    dataset.image_filenames = list(remaining_filenames_set)
    n_val = max(1, int(data_profile.val_ratio * len(dataset)))
    n_test = max(1, int(data_profile.test_ratio * len(dataset)))
    n_train = len(dataset) - n_val - n_test
    # Split the dataset
    train_set_r, val_set_r, test_set_r = random_split(dataset,
                                                [n_train, n_val, n_test],
                                                generator=torch.Generator().manual_seed(seed))

    # Update the data profile with the new names
    data_profile.train_files += train_set_r.dataset.image_filenames
    data_profile.val_files += val_set_r.dataset.image_filenames
    data_profile.test_files += test_set_r.dataset.image_filenames
    # Update the datasets with the new names
    train_set.image_filenames = data_profile.train_files
    val_set.image_filenames = data_profile.val_files
    test_set.image_filenames = data_profile.test_files

    # Get the dataloaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

