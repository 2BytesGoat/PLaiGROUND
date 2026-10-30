"""Level-agnostic student MLP and the shared evaluator adapter."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class StudentMLP(nn.Module):
    """Flat ReLU MLP used by the campaign distillation candidates."""

    def __init__(self, input_dim: int, hidden: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = int(input_dim)
        for width in hidden:
            layers.extend([nn.Linear(last, int(width)), nn.ReLU()])
            last = int(width)
        layers.append(nn.Linear(last, 2))
        self.net = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.hidden = [int(value) for value in hidden]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class GridStudent(nn.Module):
    """Small 7x7 grid encoder followed by a flat player-feature head.

    The first 147 raw ``lean_stick`` values are the 7x7 tile ID, rotation, and
    phase grids. The remaining 44 values are the player, collision, flow, and
    hold/ceiling features. A shared convolutional trunk sees the grid; the
    player block is concatenated after the convolution.
    """

    def __init__(
        self,
        input_dim: int,
        hidden: list[int],
        *,
        tile_categories: int = 15,
        conv_channels: tuple[int, int] = (24, 32),
    ) -> None:
        super().__init__()
        if int(input_dim) != 191:
            raise ValueError(f"GridStudent expects 191 inputs, got {input_dim}")
        self.input_dim = int(input_dim)
        self.hidden = [int(value) for value in hidden]
        self.tile_categories = int(tile_categories)
        first, second = (int(value) for value in conv_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(self.tile_categories + 3, first, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(first, second, 3, padding=1),
            nn.ReLU(),
        )
        self.vector_norm = nn.LayerNorm(44)
        grid_size = 7 * 7 * second
        layers: list[nn.Module] = []
        last = grid_size + 44
        for width in self.hidden:
            layers.extend([nn.Linear(last, int(width)), nn.ReLU()])
            last = int(width)
        layers.append(nn.Linear(last, 2))
        self.head = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(
                f"GridStudent expects (batch, {self.input_dim}), "
                f"got {tuple(values.shape)}"
            )
        tile = values[:, :49].long().clamp(0, self.tile_categories - 1)
        tile_onehot = F.one_hot(tile, self.tile_categories)
        tile_onehot = tile_onehot.view(-1, 7, 7, self.tile_categories)
        tile_onehot = tile_onehot.permute(0, 3, 1, 2).float()
        rotation = values[:, 49:98].reshape(-1, 1, 7, 7)
        phase = values[:, 98:147].reshape(-1, 1, 7, 7)
        grid = torch.cat(
            [
                tile_onehot,
                torch.sin(rotation),
                torch.cos(rotation),
                phase,
            ],
            dim=1,
        )
        grid_features = self.conv(grid).flatten(1)
        vector = self.vector_norm(values[:, 147:])
        return self.head(torch.cat([grid_features, vector], dim=1))


class TorchPolicy:
    """Expose a PyTorch student through the SB3-style predict interface."""

    def __init__(self, model: nn.Module, expected_input_dim: int) -> None:
        self.model = model
        self.expected_input_dim = int(expected_input_dim)

    def predict(self, observations, deterministic: bool = True):
        values = np.asarray(observations, dtype=np.float32)
        single = values.ndim == 1
        batch = values.reshape(1, -1) if single else values
        if batch.shape[1] != self.expected_input_dim:
            raise ValueError(
                f"expected {self.expected_input_dim} inputs, got {batch.shape[1]}"
            )
        with torch.no_grad():
            logits = self.model(torch.as_tensor(batch, dtype=torch.float32))
            actions = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
        return (actions[0] if single else actions), None
