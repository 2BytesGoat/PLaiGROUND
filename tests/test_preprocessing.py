"""Tests for the model observation preprocessing pipeline."""

from __future__ import annotations

import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import obs_spec  # noqa: E402
from preprocessing import (  # noqa: E402
    Clip,
    CollisionFeatures,
    DecodeRawState,
    FeatureUnion,
    FlowFeatures,
    PlayerFeatures,
    Standardize,
    StatePipeline,
    TileFeatures,
    build_raw_state_pipeline,
)


def _raw(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=obs_spec.RAW_OBSERVATION_SIZE).astype(np.float32)
    # Make the tile IDs look like the actual game's non-negative IDs.
    raw[obs_spec.RAW_TILE_ID_OFFSET:obs_spec.RAW_TILE_STATE_OFFSET] = (
        rng.integers(0, 6, size=obs_spec.RAW_TILE_ID_COUNT)
    )
    return raw


class ObservationPreprocessingTest(unittest.TestCase):
    def test_decode_raw_state_shapes_and_fields(self) -> None:
        raw = _raw()
        decoded = DecodeRawState().transform(raw[None, :])
        self.assertEqual(decoded.flat.shape, (1, obs_spec.RAW_OBSERVATION_SIZE))
        self.assertEqual(decoded.tile_ids.shape, (1, 7, 7))
        self.assertEqual(decoded.tile_states.shape, (1, 49, 5))
        self.assertEqual(decoded.collisions.shape, (1, 8, 3))
        self.assertEqual(decoded.flow.shape, (1, 8))
        self.assertEqual(set(decoded.player), set(obs_spec.RAW_PLAYER_FIELDS))
        self.assertAlmostEqual(
            float(decoded.player["cell_x"][0]),
            float(raw[obs_spec.RAW_PLAYER_OFFSET]),
        )

        single = decoded.first()
        self.assertEqual(single.tile_ids.shape, (7, 7))
        self.assertEqual(single.player["cell_y"], float(decoded.player["cell_y"][0]))

    def test_raw_identity_pipeline_is_single_batch_consistent(self) -> None:
        raw = _raw()
        batch = np.stack([raw, raw + 1.0])
        pre = build_raw_state_pipeline("raw")
        out = pre.fit_transform(batch)
        self.assertEqual(out.shape, batch.shape)
        self.assertEqual(pre.output_size, obs_spec.RAW_OBSERVATION_SIZE)
        np.testing.assert_allclose(pre.transform(raw), raw)
        np.testing.assert_allclose(pre.transform(batch), batch)

    def test_feature_pipeline_rebuilds_flat_raw_contract(self) -> None:
        raw = _raw()
        pre = build_raw_state_pipeline("features")
        out = pre.fit_transform(raw[None, :])
        self.assertEqual(out.shape, (1, obs_spec.RAW_OBSERVATION_SIZE))
        self.assertEqual(pre.output_size, obs_spec.RAW_OBSERVATION_SIZE)
        np.testing.assert_allclose(pre.transform(raw), raw)
        np.testing.assert_allclose(out[0], raw)

    def test_one_hot_tiles_grow_the_output(self) -> None:
        raw = _raw()
        pre = build_raw_state_pipeline(
            "features",
            tile_names=["Wall", "Floor", "Hazard", "Powerup", "Secret", "Pickup"],
            tile_one_hot=True,
        )
        out = pre.fit_transform(raw[None, :])
        expected = (
            49 * 6
            + 49 * obs_spec.RAW_TILE_STATE_FLOATS
            + len(obs_spec.RAW_PLAYER_FIELDS)
            + obs_spec.RAW_DIRECTIONS * obs_spec.RAW_COLLISION_FLOATS
            + obs_spec.RAW_FLOW_COUNT
        )
        self.assertEqual(out.shape, (1, expected))
        self.assertEqual(pre.output_size, expected)
        self.assertEqual(pre.transform(raw).shape, (expected,))

    def test_union_and_standardize_chain(self) -> None:
        raw_batch = np.stack([_raw(0), _raw(1), _raw(2)])
        pre = StatePipeline([
            ("decode", DecodeRawState()),
            ("features", FeatureUnion([
                ("tiles", TileFeatures()),
                ("player", PlayerFeatures()),
                ("collisions", CollisionFeatures()),
                ("flow", FlowFeatures()),
            ])),
            ("standardize", Standardize()),
        ])
        out = pre.fit_transform(raw_batch)
        self.assertEqual(out.shape, raw_batch.shape)
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-4)
        single = pre.transform(raw_batch[0])
        self.assertEqual(single.shape, (obs_spec.RAW_OBSERVATION_SIZE,))
        np.testing.assert_allclose(single, out[0])

    def test_clip(self) -> None:
        values = np.asarray([[-10.0, 0.0, 10.0]], dtype=np.float32)
        step = Clip(-1.0, 1.0)
        step.fit(values)
        np.testing.assert_allclose(step.transform(values), [[-1.0, 0.0, 1.0]])

    def test_named_params_and_set_params(self) -> None:
        pre = build_raw_state_pipeline("features")
        params = pre.get_params(deep=True)
        self.assertIn("steps", params)
        self.assertIn("features__flow__scale", params)
        pre.set_params(features__flow__scale=2.0)
        self.assertEqual(pre.named_steps["features"].named_transformers["flow"].scale, 2.0)

    def test_sklearn_clone(self) -> None:
        from sklearn.base import clone

        pre = build_raw_state_pipeline("features")
        cloned = clone(pre)
        self.assertIsInstance(cloned, StatePipeline)
        self.assertEqual(cloned.named_steps["decode"].__class__, DecodeRawState)
        self.assertEqual(
            cloned.named_steps["features"].named_transformers["flow"].scale,
            1.0,
        )

    def test_pickle_round_trip(self) -> None:
        raw = _raw()
        pre = build_raw_state_pipeline("features")
        expected = pre.fit_transform(raw[None, :])
        restored = pickle.loads(pickle.dumps(pre))
        np.testing.assert_allclose(restored.transform(raw[None, :]), expected)

    def test_lean_layout_is_explicitly_unavailable(self) -> None:
        with self.assertRaises(NotImplementedError):
            build_raw_state_pipeline("lean")


if __name__ == "__main__":
    unittest.main()
