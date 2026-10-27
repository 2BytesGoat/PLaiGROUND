"""Frozen 7x7 grid autoencoder for the legacy 58-value observation.

The autoencoder is trained only on the 49-cell tile grid. The nine scalar tail
values (direction, flow, velocity, contact, jump phase, powerup) are left
untouched and concatenated after the latent vector. Freezing the encoder lets
PPO train a small MLP head instead of updating the spatial encoder jointly.

See docs/model-improvement-experiments.md in PLaiGROUND for the experiment.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

GRID_SIZE = 7
GRID_VALUES = GRID_SIZE * GRID_SIZE
TAIL_VALUES = 9
TILE_VOCAB = 16
VOID_INDEX = 16
ENCODER_VOCAB = VOID_INDEX + 1


def grid_tokens(observations: torch.Tensor) -> torch.Tensor:
    """Map the legacy 7x7 grid to 0..16.

    ``-1`` appears for stacked/unknown cells in the saved observations; it is
    represented as class 16 rather than clamped to empty.
    """
    grid = observations[:, :GRID_VALUES].round().long()
    return torch.where(
        grid < 0,
        torch.full_like(grid, VOID_INDEX),
        grid.clamp(0, VOID_INDEX),
    )


def grid_one_hot(observations: torch.Tensor) -> torch.Tensor:
    """Return a float ``[B, 17, 7, 7]`` one-hot cell tensor."""
    tokens = grid_tokens(observations).view(-1, GRID_SIZE, GRID_SIZE)
    return (
        torch.nn.functional.one_hot(tokens, ENCODER_VOCAB)
        .permute(0, 3, 1, 2)
        .float()
    )


class LegacyGridEncoder(nn.Module):
    """Small convolutional encoder over the 7x7 tile grid."""

    def __init__(
        self,
        latent_dim: int = 64,
        embedding_dim: int = 8,
        conv_channels: tuple[int, int] = (16, 32),
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.embedding = nn.Embedding(ENCODER_VOCAB, embedding_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(embedding_dim, conv_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_features = conv_channels[1] * GRID_VALUES
        self.projection = nn.Linear(conv_features, self.latent_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        grid = grid_tokens(observations)
        grid = grid.view(-1, GRID_SIZE, GRID_SIZE)
        embedded = self.embedding(grid).permute(0, 3, 1, 2)
        return self.projection(self.conv(embedded))


class OneHotLegacyGridEncoder(nn.Module):
    """Convolutional encoder over explicit 7x7x17 one-hot cell channels."""

    def __init__(
        self,
        latent_dim: int = 64,
        conv_channels: tuple[int, int] = (16, 32),
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(ENCODER_VOCAB, conv_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_features = conv_channels[1] * GRID_VALUES
        self.projection = nn.Linear(conv_features, self.latent_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.projection(self.conv(grid_one_hot(observations)))


def build_grid_encoder(
    input_encoding: str,
    latent_dim: int,
    embedding_dim: int,
    conv_channels: tuple[int, int],
) -> nn.Module:
    if input_encoding == "one_hot":
        return OneHotLegacyGridEncoder(
            latent_dim=latent_dim,
            conv_channels=conv_channels,
        )
    return LegacyGridEncoder(
        latent_dim=latent_dim,
        embedding_dim=embedding_dim,
        conv_channels=conv_channels,
    )



class LegacyGridAutoencoder(nn.Module):
    """Reconstruct the 49 grid cells from a compact latent vector."""

    def __init__(
        self,
        latent_dim: int = 64,
        embedding_dim: int = 8,
        conv_channels: tuple[int, int] = (16, 32),
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.encoder = LegacyGridEncoder(
            latent_dim=latent_dim,
            embedding_dim=embedding_dim,
            conv_channels=conv_channels,
        )
        decoder_channels = conv_channels[1]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_channels * GRID_VALUES),
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
                ENCODER_VOCAB,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(observations)
        return self.decoder(latent), latent


class OneHotLegacyGridAutoencoder(LegacyGridAutoencoder):
    """Same decoder, but the encoder consumes explicit 7x7x17 cell channels."""

    def __init__(
        self,
        latent_dim: int = 64,
        conv_channels: tuple[int, int] = (16, 32),
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            embedding_dim=1,
            conv_channels=conv_channels,
        )
        self.encoder = OneHotLegacyGridEncoder(
            latent_dim=latent_dim,
            conv_channels=conv_channels,
        )


class FrozenLegacyAutoencoderExtractor(BaseFeaturesExtractor):
    """Use a pretrained, frozen grid encoder plus the exact scalar tail.

    The output is ``[latent(64), tail(9)]``. Only the PPO MLP head is trained.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        checkpoint_path: str | Path,
        features_dim: int | None = None,
    ) -> None:
        if observation_space.shape != (GRID_VALUES + TAIL_VALUES,):
            raise ValueError(
                "FrozenLegacyAutoencoderExtractor expects 58 observations, got "
                f"{observation_space.shape}"
            )

        payload = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location="cpu",
            weights_only=False,
        )
        latent_dim = int(payload["latent_dim"])
        embedding_dim = int(payload.get("embedding_dim", 8))
        conv_channels = tuple(int(v) for v in payload.get("conv_channels", (16, 32)))
        input_encoding = str(payload.get("input_encoding", "embedding"))
        encoder = build_grid_encoder(
            input_encoding=input_encoding,
            latent_dim=latent_dim,
            embedding_dim=embedding_dim,
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
        self.input_encoding = input_encoding
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self._restore_encoder()

    def _restore_encoder(self) -> None:
        """Reload the trained encoder after SB3's policy initialization pass."""
        payload = torch.load(
            Path(self.checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )
        self.encoder.load_state_dict(payload["encoder"])
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def reload_encoder(self) -> "FrozenLegacyAutoencoderExtractor":
        """Restore frozen weights lost to a parent module's init_weights pass."""
        self._restore_encoder()
        return self

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(observations)
        tail = observations[:, GRID_VALUES:]
        return torch.cat((latent, tail), dim=1)

    def train(self, mode: bool = True) -> "FrozenLegacyAutoencoderExtractor":
        # The spatial encoder must stay in eval mode and frozen. The PPO head
        # lives outside this module and still trains normally.
        super().train(False)
        self.encoder.eval()
        return self
