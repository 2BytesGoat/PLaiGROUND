"""Spatial encoder for the legacy 58-value Dragon Jump observation.

The first 49 values are a 7x7 grid of tile-symbol indices (0..15). The last 9
are direction/flow/velocity/state features. A tiny tile embedding + 2D CNN
keeps the grid structure that the flat MLP throws away.

This is a teacher-only experiment until a matching GDScript/ONNX runtime is
chosen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

GRID_SIZE = 7
GRID_VALUES = GRID_SIZE * GRID_SIZE
TAIL_VALUES = 9
TILE_VOCAB = 16


class LegacyGridExtractor(BaseFeaturesExtractor):
    """Embed 7x7 tile IDs, convolve, and append the 9 scalar tail values."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        embedding_dim: int = 8,
        conv_channels: tuple[int, int] = (16, 32),
    ) -> None:
        super().__init__(observation_space, features_dim)
        if observation_space.shape != (GRID_VALUES + TAIL_VALUES,):
            raise ValueError(
                "LegacyGridExtractor expects 58 observations, got "
                f"{observation_space.shape}"
            )

        self.grid_size = GRID_SIZE
        self.register_buffer(
            "_tile_vocab",
            torch.tensor(TILE_VOCAB, dtype=torch.long),
            persistent=False,
        )
        self.embedding = nn.Embedding(TILE_VOCAB, embedding_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(embedding_dim, conv_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_features = conv_channels[1] * GRID_SIZE * GRID_SIZE
        self.tail = nn.Sequential(
            nn.Linear(TAIL_VALUES, 32),
            nn.ReLU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(conv_features + 32, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        grid = observations[:, :GRID_VALUES].long()
        grid = torch.clamp(grid, 0, TILE_VOCAB - 1)
        grid = grid.view(-1, self.grid_size, self.grid_size)
        embedded = self.embedding(grid).permute(0, 3, 1, 2)
        conv = self.conv(embedded)
        tail = observations[:, GRID_VALUES:]
        return self.projection(torch.cat((conv, self.tail(tail)), dim=1))
