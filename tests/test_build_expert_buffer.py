"""Tests for route-level filtering in the expert replay builder."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.build_expert_buffer import _success_mask


class ExpertBufferMaskTest(unittest.TestCase):
    def _data(self, root: Path):
        signatures = np.asarray(
            ["a", "a", "a", "b", "b", "b", "c", "c", "d", "d"],
            dtype="U8",
        )
        path = root / "expert.npz"
        np.savez(
            path,
            actions=np.zeros(len(signatures), dtype=np.int64),
            route_success=np.asarray(
                [True, True, True, True, True, True, True, True,
                 False, False],
            ),
            route_signature=signatures,
            step=np.arange(len(signatures), dtype=np.int32),
            route_run_time=np.asarray(
                [4.5, 4.5, 4.5, 5.0, 5.0, 5.0, 6.0, 6.0, 4.5, 4.5],
                dtype=np.float32,
            ),
            replayed_run_time=np.asarray(
                [0.2, 1.0, 4.5, 0.2, 1.0, 5.0, 0.2, 4.0, 0.2, 1.0],
                dtype=np.float32,
            ),
        )
        return path, np.load(path)

    def test_replay_gain_keeps_or_drops_whole_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path, data = self._data(Path(tmp))
            mask = _success_mask(
                path, data, max_replay_gain=0.25
            )
            # a and b replay at source speed; c is an anomalous 2.0 s
            # speedup and is dropped with all of its early transitions.
            self.assertEqual(
                mask.tolist(),
                [True, True, True, True, True, True, False, False,
                 False, False],
            )

    def test_max_replayed_time_keeps_whole_fast_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path, data = self._data(Path(tmp))
            mask = _success_mask(
                path, data, max_replayed_run_time=4.6
            )
            # a and c finish inside the threshold; b is not, so none of b's
            # early transitions leak into the expert buffer.
            self.assertEqual(
                mask.tolist(),
                [True, True, True, False, False, False, True, True,
                 False, False],
            )


if __name__ == "__main__":
    unittest.main()
