"""Semantic multi-hot autoencoder for the 754-value Dragon Jump observation.

The first 735 values are a 7x7x15 semantic cell grid; the remaining 19 are exact
player/route values. The autoencoder reconstructs only the grid. The 19 tail
values are concatenated unchanged after the latent vector.

A masked autoencoder (MAE) is used instead of a plain AE: random cells are
zeroed during training and the encoder gets an extra mask channel so it can
learn to infer context rather than only memorize the input.

This is teacher-only until a matching runtime/export exists.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import obs_spec

GRID_SIZE = obs_spec.GRID_SIZE
CELL_FEATURES = obs_spec.CELL_FEATURE_COUNT
GRID_VALUES = obs_spec.GRID_FEATURE_COUNT
TAIL_VALUES = obs_spec.PLAYER_FEATURE_COUNT
OBS_SIZE = obs_spec.MULTI_HOT_OBSERVATION_SIZE
TYPE_FEATURES = (
    obs_spec.CELL_POWERUP_KIND_OFFSET
)  # first 8 type bits, then powerup-kind bits are also binary through channel 11
BINARY_FEATURES = obs_spec.CELL_DISSOLVE_TIMER_OFFSET  # 12 binary channels
MASK_CHANNEL = CELL_FEATURES


class SemanticGridEncoder(nn.Module):
    """Convolutional encoder over 7x7 semantic cells plus an optional mask."""

    def __init__(
        self,
        latent_dim: int = 64,
        conv_channels: tuple[int, int] = (32, 64),
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(
                CELL_FEATURES + 1,
                conv_channels[0],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                conv_channels[0],
                conv_channels[1],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.projection = nn.Linear(
            conv_channels[1] * GRID_SIZE * GRID_SIZE,
            self.latent_dim,
        )
        self.latent_norm = nn.LayerNorm(self.latent_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        grid = observations[:, :GRID_VALUES].view(
            -1, GRID_SIZE, GRID_SIZE, CELL_FEATURES
        )
        mask = torch.zeros_like(grid[:, :, :, :1])
        return self._encode(grid, mask)

    def _encode(self, grid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_grid = grid * (1.0 - mask)
        model_input = torch.cat((masked_grid, mask), dim=3)
        model_input = model_input.permute(0, 3, 1, 2)
        latent = self.projection(self.conv(model_input))
        return self.latent_norm(latent)


class SemanticGridAutoencoder(nn.Module):
    """Masked autoencoder over the 7x7x15 semantic cell grid."""

    def __init__(
        self,
        latent_dim: int = 64,
        conv_channels: tuple[int, int] = (32, 64),
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.encoder = SemanticGridEncoder(
            latent_dim=latent_dim,
            conv_channels=conv_channels,
        )
        decoder_channels = conv_channels[1]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_channels * GRID_SIZE * GRID_SIZE),
            nn.ReLU(),
            nn.Unflatten(1, (decoder_channels, GRID_SIZE, GRID_SIZE)),
            nn.ConvTranspose2d(
                decoder_channels,
                conv_channels[0],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                conv_channels[0],
                CELL_FEATURES,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(
        self,
        observations: torch.Tensor,
        mask_prob: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid = observations[:, :GRID_VALUES].view(
            -1, GRID_SIZE, GRID_SIZE, CELL_FEATURES
        )
        mask = torch.zeros_like(grid[:, :, :, :1])
        if mask_prob > 0.0:
            mask = (
                torch.rand_like(mask[:, :, :, 0:1]) < mask_prob
            ).float()
        latent = self.encoder._encode(grid, mask)
        reconstruction = self.decoder(latent)
        return reconstruction, latent, mask


class FrozenSemanticAutoencoderExtractor(BaseFeaturesExtractor):
    """Frozen semantic MAE latent plus the exact 19-value tail."""

    def __init__(
        self,
        observation_space: spaces.Box,
        checkpoint_path: str | Path,
        features_dim: int | None = None,
    ) -> None:
        if observation_space.shape != (OBS_SIZE,):
            raise ValueError(
                "FrozenSemanticAutoencoderExtractor expects 754 observations, "
                f"got {observation_space.shape}"
            )
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        payload = torch.load(
            Path(self.checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )
        latent_dim = int(payload["latent_dim"])
        conv_channels = tuple(
            int(value) for value in payload.get("conv_channels", (32, 64))
        )
        encoder = SemanticGridEncoder(
            latent_dim=latent_dim,
            conv_channels=conv_channels,
        )
        encoder.load_state_dict(payload["encoder"])
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)

        resolved_features_dim = latent_dim + TAIL_VALUES
        if features_dim is not None and int(features_dim) != resolved_features_dim:
            raise ValueError(
                f"checkpoint latent={latent_dim} implies "
                f"features_dim={resolved_features_dim}, got {features_dim}"
            )
        super().__init__(observation_space, resolved_features_dim)
        self.encoder = encoder
        self._restore_encoder()

    def _restore_encoder(self) -> None:
        payload = torch.load(
            Path(self.checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )
        self.encoder.load_state_dict(payload["encoder"])
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def reload_encoder(self) -> "FrozenSemanticAutoencoderExtractor":
        self._restore_encoder()
        return self

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(observations)
        tail = observations[:, GRID_VALUES:]
        return torch.cat((latent, tail), dim=1)

    def train(self, mode: bool = True) -> "FrozenSemanticAutoencoderExtractor":
        super().train(False)
        self.encoder.eval()
        return self
