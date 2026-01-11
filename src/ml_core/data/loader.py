from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import transforms

from .pcam import PCAMDataset


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create Train and Validation DataLoaders
    using pre-split H5 files.
    """
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])

    # TODO: Define Transforms
    # train_transform = ...
    # val_transform = ...

    # TODO: Define Paths for X and Y (train and val)
    
    # TODO: Instantiate PCAMDataset for train and val

    # TODO: Create DataLoaders
    # train_loader = ...
    # val_loader = ...
    
    raise NotImplementedError("Implement get_dataloaders")
