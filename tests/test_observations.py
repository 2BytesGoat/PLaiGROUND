"""Observation-contract tests, including the versioned legacy hold encoding."""

from __future__ import annotations

import unittest

import numpy as np

from plaiground.observations import spec


class ObservationSpecTest(unittest.TestCase):
    def test_legacy_contract_is_frozen(self) -> None:
        self.assertEqual(spec.expected_obs_size(spec.ENCODING_LEGACY), 58)
        self.assertEqual(spec.spec_version(spec.ENCODING_LEGACY), 1)

    def test_legacy_stick_appends_hold_flags_without_moving_legacy_values(self) -> None:
        self.assertEqual(
            spec.expected_obs_size(spec.ENCODING_LEGACY_STICK), 61
        )
        self.assertEqual(spec.spec_version(spec.ENCODING_LEGACY_STICK), 6)
        self.assertEqual(spec.LEGACY_ON_CEILING_OFFSET, 58)
        self.assertEqual(spec.LEGACY_JUMP_HELD_OFFSET, 59)
        self.assertEqual(spec.LEGACY_STICK_ACTIVE_OFFSET, 60)

        values = np.arange(61, dtype=np.float32)
        decoded = spec.decode_legacy_stick(values)
        np.testing.assert_array_equal(
            decoded["legacy"], np.arange(58, dtype=np.float32)
        )
        self.assertEqual(decoded["on_ceiling"], 58.0)
        self.assertEqual(decoded["jump_held"], 59.0)
        self.assertEqual(decoded["stick_active"], 60.0)

    def test_legacy_stick_validation_rejects_old_width(self) -> None:
        with self.assertRaises(ValueError):
            spec.validate_obs(np.zeros(58, dtype=np.float32), spec.ENCODING_LEGACY_STICK)

    def test_lean_stick_appends_flags_without_moving_lean_values(self) -> None:
        self.assertEqual(spec.expected_obs_size(spec.ENCODING_LEAN_STICK), 191)
        self.assertEqual(spec.spec_version(spec.ENCODING_LEAN_STICK), 7)
        self.assertEqual(spec.LEAN_STICK_ON_CEILING_OFFSET, 188)
        self.assertEqual(spec.LEAN_STICK_JUMP_HELD_OFFSET, 189)
        self.assertEqual(spec.LEAN_STICK_STICK_ACTIVE_OFFSET, 190)

        values = np.arange(191, dtype=np.float32)
        decoded = spec.decode_lean_stick(values)
        self.assertEqual(decoded["on_ceiling"], 188.0)
        self.assertEqual(decoded["jump_held"], 189.0)
        self.assertEqual(decoded["stick_active"], 190.0)
        np.testing.assert_array_equal(
            decoded["flow"],
            np.arange(spec.LEAN_FLOW_OFFSET, spec.LEAN_OBSERVATION_SIZE, dtype=np.float32),
        )

    def test_lean_stick_validation_rejects_lean_width(self) -> None:
        with self.assertRaises(ValueError):
            spec.validate_obs(
                np.zeros(188, dtype=np.float32), spec.ENCODING_LEAN_STICK
            )


if __name__ == "__main__":
    unittest.main()
