"""Lean v1 MLP architecture.

The lean policy is intentionally simple:

    model input -> 512 -> 512 -> action value

The input width is vocabulary-dependent because the Python preprocessing
one-hots tile IDs, state_id, powerup slots, and collision types.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

LEAN_HIDDEN_SIZES = (512, 512)
LEAN_ACTIVATION = nn.Tanh
LEAN_OUTPUT_DIM = 2  # idle / jump


class LeanMLP(nn.Module):
    """Basic 512x512 MLP for the lean observation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = LEAN_OUTPUT_DIM,
        hidden_sizes: Iterable[int] = LEAN_HIDDEN_SIZES,
        activation_fn: type[nn.Module] = LEAN_ACTIVATION,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        if int(output_dim) <= 0:
            raise ValueError("output_dim must be positive")
        sizes = tuple(int(value) for value in hidden_sizes)
        if not sizes or any(value <= 0 for value in sizes):
            raise ValueError("hidden_sizes must contain positive widths")

        layers: list[nn.Module] = []
        last = int(input_dim)
        for width in sizes:
            layers.extend([nn.Linear(last, width), activation_fn()])
            last = width
        layers.append(nn.Linear(last, int(output_dim)))
        self.net = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_sizes = sizes

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def lean_policy_kwargs() -> dict:
    """SB3 policy kwargs for the basic lean 512×512 baseline."""
    return {
        "net_arch": {
            "pi": list(LEAN_HIDDEN_SIZES),
            "vf": list(LEAN_HIDDEN_SIZES),
        },
        "activation_fn": LEAN_ACTIVATION,
    }


def build_lean_mlp(input_dim: int) -> LeanMLP:
    return LeanMLP(input_dim=input_dim)
