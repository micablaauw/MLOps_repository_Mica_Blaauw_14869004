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

    def __init__(self, x_path: str, y_path: str, transform: Optional[Callable] = None):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.transform = transform

        if not self.x_path.exists():
            raise FileNotFoundError(f"X file not found: {self.x_path}")
        if not self.y_path.exists():
            raise FileNotFoundError(f"Y file not found: {self.y_path}")

        self.x_file = h5py.File(self.x_path, "r")
        self.y_file = h5py.File(self.y_path, "r")
        self.x_data = self.x_file["x"]
        self.y_data = self.y_file["y"]

    def __len__(self) -> int:
        # TODO: Return length of dataset
        # The dataloader will know hence how many batches to create
        return len(self.y_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement data retrieval
        # 1. Read data at idx
        # 2. Convert to uint8 (for PIL compatibility if using transforms)
        # 3. Apply transforms if they exist
        # 4. Return tensor image and label (as long)
        image = self.x_data[idx]
        label = self.y_data[idx]
        image = np.clip(image, 0, 255).astype(np.uint8)
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label = torch.tensor(label, dtype=torch.long)

        return image, label
