from pathlib import Path
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PCAMDataset(Dataset):
    """
    PatchCamelyon (PCAM) Dataset reader for H5 format.
    """

    def __init__(self, x_path: str, y_path: str, transform: Optional[Callable] = None, filter_data: bool = False):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.transform = transform
        self.filter_data = filter_data

        if not self.x_path.exists():
            raise FileNotFoundError(f"X file not found: {self.x_path}")
        if not self.y_path.exists():
            raise FileNotFoundError(f"Y file not found: {self.y_path}")

        self.x_file = h5py.File(self.x_path, "r")
        self.y_file = h5py.File(self.y_path, "r")
        self.x_data = self.x_file["x"]
        self.y_data = self.y_file["y"]

        if self.filter_data:
            means = self.x_data[:].mean(axis=(1, 2, 3))
            self.indices = np.where(
                (means > 5) & (means < 250)
            )[0]
        else:
            self.indices = np.arange(len(self.x_data))

    def __len__(self) -> int:
        # TODO: Return length of dataset
        # The dataloader will know hence how many batches to create
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement data retrieval
        # 1. Read data at idx
        # 2. Convert to uint8 (for PIL compatibility if using transforms)
        # 3. Apply transforms if they exist
        # 4. Return tensor image and label (as long)
        real_idx = self.indices[idx]

        x = self.x_data[real_idx]
        y = self.y_data[real_idx]

        x = np.clip(x, 0, 255).astype(np.uint8)

        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0

        y = torch.tensor(y).long().squeeze()

        return x, y
