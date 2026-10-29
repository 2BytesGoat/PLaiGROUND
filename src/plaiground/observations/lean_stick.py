"""Preprocessing for the lean + hold/ceiling observation."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import spec


class LeanStickPreprocessor:
    """Apply the fitted lean 188 pipeline, then append three raw flags.

    The model input order is:

        lean-one-hot(954) + on_ceiling + jump_held + stick_active

    This keeps the existing lean model features and their weights in the same
    columns, making a 188 -> 191 warm start a simple first-layer expansion.
    """

    def __init__(self, lean_pipeline: Any) -> None:
        self.lean_pipeline = lean_pipeline

    @property
    def output_size(self) -> int:
        return int(self.lean_pipeline.output_size) + 3

    @property
    def n_features_in_(self) -> int:
        return spec.LEAN_STICK_OBSERVATION_SIZE

    def _split(self, X: Any) -> tuple[np.ndarray, np.ndarray, bool]:
        array = np.asarray(X, dtype=np.float32)
        single = array.ndim == 1
        batch = array.reshape(1, -1) if single else array
        if batch.ndim != 2 or batch.shape[1] != spec.LEAN_STICK_OBSERVATION_SIZE:
            raise ValueError(
                "lean_stick preprocessing expects "
                f"({spec.LEAN_STICK_OBSERVATION_SIZE},) or "
                f"(N, {spec.LEAN_STICK_OBSERVATION_SIZE}); got {array.shape}"
            )
        return batch[:, : spec.LEAN_OBSERVATION_SIZE], batch[:, spec.LEAN_OBSERVATION_SIZE :], single

    def fit(self, X: Any, y: Any = None) -> "LeanStickPreprocessor":
        lean, _flags, _single = self._split(X)
        self.lean_pipeline.fit(lean, y)
        return self

    def transform(self, X: Any) -> np.ndarray:
        lean, flags, single = self._split(X)
        base = np.asarray(self.lean_pipeline.transform(lean), dtype=np.float32)
        if base.ndim == 1:
            base = base.reshape(1, -1)
        result = np.concatenate([base, flags], axis=1)
        return result[0] if single else result

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        lean, flags, single = self._split(X)
        base = np.asarray(self.lean_pipeline.fit_transform(lean, y), dtype=np.float32)
        if base.ndim == 1:
            base = base.reshape(1, -1)
        result = np.concatenate([base, flags], axis=1)
        return result[0] if single else result

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"lean_pipeline": self.lean_pipeline}

    def set_params(self, **params: Any) -> "LeanStickPreprocessor":
        if "lean_pipeline" in params:
            self.lean_pipeline = params["lean_pipeline"]
        return self
