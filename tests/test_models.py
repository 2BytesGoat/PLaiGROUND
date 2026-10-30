"""Tests for the declared lean model architecture."""

from __future__ import annotations

import unittest
import torch

from plaiground.models import LeanMLP, build_lean_mlp, lean_policy_kwargs
from plaiground.models.student import GridStudent


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


class GridStudentTest(unittest.TestCase):
    def test_forward_shape_and_grid_input_contract(self) -> None:
        model = GridStudent(191, [64, 64], conv_channels=(8, 16))
        values = torch.zeros((3, 191), dtype=torch.float32)
        values[:, :49] = 2.0
        values[:, 49:98] = 1.5707964
        output = model(values)
        self.assertEqual(tuple(output.shape), (3, 2))

    def test_rejects_non_lean_stick_input(self) -> None:
        with self.assertRaises(ValueError):
            GridStudent(957, [64, 64])


if __name__ == "__main__":
    unittest.main()
