from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from .pcam import PCAMDataset

import numpy as np

def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create Train and Validation DataLoaders
    using pre-split H5 files.
    """
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])
    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 2)

    # TODO: Define Transforms
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]
    )

    # TODO: Define Paths for X and Y (train and val)
    x_train = base_path / "camelyonpatch_level_2_split_train_x.h5"
    y_train = base_path / "camelyonpatch_level_2_split_train_y.h5"

    x_val = base_path / "camelyonpatch_level_2_split_valid_x.h5"
    y_val = base_path / "camelyonpatch_level_2_split_valid_y.h5"

    # TODO: Instantiate PCAMDataset for train and val
    train_dataset = PCAMDataset(x_train, y_train, transform=train_transform, filter_data=True)
    val_dataset = PCAMDataset(x_val, y_val, transform=val_transform, filter_data=False)
    labels = train_dataset.y_data[:].squeeze()
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    sample_weights = weights[labels]

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    # TODO: Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader
