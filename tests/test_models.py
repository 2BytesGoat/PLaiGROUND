"""Tests for the declared lean model architecture."""

from __future__ import annotations

import unittest
import torch

from plaiground.models import LeanMLP, build_lean_mlp, lean_policy_kwargs


class LeanMLPTest(unittest.TestCase):
    def test_declared_512_512_architecture_and_forward_shape(self) -> None:
        model = LeanMLP(input_dim=513, output_dim=2)
        self.assertEqual(model.input_dim, 513)
        self.assertEqual(model.hidden_sizes, (512, 512))
        first = model.net[0]
        second = model.net[2]
        self.assertEqual(first.in_features, 513)
        self.assertEqual(first.out_features, 512)
        self.assertEqual(second.in_features, 512)
        self.assertEqual(second.out_features, 512)
        output = model(torch.zeros((4, 513), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (4, 2))

    def test_policy_kwargs_and_builder(self) -> None:
        kwargs = lean_policy_kwargs()
        self.assertEqual(kwargs["net_arch"]["pi"], [512, 512])
        self.assertEqual(kwargs["net_arch"]["vf"], [512, 512])
        model = build_lean_mlp(954)
        self.assertIsInstance(model, LeanMLP)
        self.assertEqual(model.hidden_sizes, (512, 512))

    def test_rejects_invalid_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            LeanMLP(input_dim=0)
        with self.assertRaises(ValueError):
            LeanMLP(input_dim=10, hidden_sizes=(512, 0))


if __name__ == "__main__":
    unittest.main()
