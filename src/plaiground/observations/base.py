"""Base interfaces for the lean observation pipeline.

The pipeline is intentionally sklearn-shaped (``fit``, ``transform``,
``fit_transform``) but operates on one flat observation or a batch. Steps may
return a structured intermediate such as ``LeanStateBatch``; the final step
must return a float array.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator


class StateTransformer(BaseEstimator):
    """Base class for one observation-transform step."""

    def fit(self, X: Any, y: Any = None) -> "StateTransformer":
        return self

    def transform(self, X: Any) -> Any:
        raise NotImplementedError

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        return self.fit(X, y).transform(X)


def is_single_array(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == 1


def restore_single(value: Any) -> Any:
    """Remove the batch axis added for a single observation."""
    if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == 1:
        return value[0]
    if hasattr(value, "first"):
        return value.first()
    return value


def as_batch(value: Any) -> Any:
    """Add a batch axis to a single NumPy observation, if needed."""
    if isinstance(value, np.ndarray) and value.ndim == 1:
        return value[None, :]
    return value
