"""Unit checks for the raw observation/reward contract constants."""

from __future__ import annotations

import unittest
import numpy as np

from plaiground.observations import spec as obs_spec


class RawRewardContractTest(unittest.TestCase):
    def test_raw_layout_and_gold_ratio(self) -> None:
        self.assertEqual(obs_spec.RAW_OBSERVATION_SIZE, 342)
        self.assertEqual(obs_spec.RAW_PLAYER_FLOATS, 16)
        self.assertEqual(len(obs_spec.RAW_PLAYER_FIELDS), 16)
        self.assertAlmostEqual(obs_spec.RAW_GOLD_RATIO_CAP, 3.0)
        self.assertIn("gold_ratio", obs_spec.RAW_PLAYER_FIELDS)

        flat = np.zeros(obs_spec.RAW_OBSERVATION_SIZE, dtype=np.float32)
        index = obs_spec.RAW_PLAYER_FIELDS.index("gold_ratio")
        flat[obs_spec.RAW_PLAYER_OFFSET + index] = 2.5
        decoded = obs_spec.decode_raw(flat)
        self.assertAlmostEqual(decoded["player"]["gold_ratio"], 2.5)
        self.assertEqual(decoded["tile_ids"].shape, (7, 7))
        self.assertEqual(decoded["tile_states"].shape, (49, 5))
        self.assertEqual(decoded["collisions"].shape, (8, 3))
        self.assertEqual(decoded["flow"].shape, (8,))


if __name__ == "__main__":
    unittest.main()
