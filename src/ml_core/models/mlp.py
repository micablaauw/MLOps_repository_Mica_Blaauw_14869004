from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: List[int],
        hidden_units: List[int],
        num_classes: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        
        # TODO: Build the MLP architecture
        # If you are up to the task, explore other architectures or model types
        # Hint: Flatten -> [Linear -> ReLU -> Dropout] * N_layers -> Linear
        input_dim = 1
        for dim in input_shape:
            input_dim *= dim

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        x = x.view(x.size(0), -1)
        return self.model(x)
