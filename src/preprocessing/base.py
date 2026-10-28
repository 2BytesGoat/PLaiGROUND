"""Base interfaces for observation preprocessing.

The pipeline is intentionally sklearn-shaped (``fit``, ``transform``,
``fit_transform``) but operates on either one flat observation or a batch of
flat observations. Steps may return a structured intermediate such as
``RawStateBatch``; the final step in a model pipeline must return a float
array.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator


class StateTransformer(BaseEstimator):
    """Base class for one observation-transform step.

    ``fit`` is stateless by default. A stateful step (for example a scaler)
    should override it and store fitted parameters with a trailing underscore.
    """

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


class Identity(StateTransformer):
    """Pass observations through, optionally casting them to float32."""

    def __init__(self, dtype: str | np.dtype = np.float32) -> None:
        self.dtype = dtype

    def fit(self, X: Any, y: Any = None) -> "Identity":
        array = np.asarray(X)
        if array.ndim not in (1, 2):
            raise ValueError(f"Identity expects a 1D/2D array, got {array.shape}")
        self.n_features_in_ = int(array.shape[-1])
        return self

    def transform(self, X: Any) -> np.ndarray:
        return np.asarray(X, dtype=self.dtype)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"dtype": self.dtype}

    def set_params(self, **params: Any) -> "Identity":
        if "dtype" in params:
            self.dtype = params["dtype"]
        return self


class Clip(StateTransformer):
    """Clip a numeric feature array to a fixed range."""

    def __init__(self, min_value: float = -5.0, max_value: float = 5.0) -> None:
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def fit(self, X: Any, y: Any = None) -> "Clip":
        array = np.asarray(X)
        if array.ndim not in (1, 2):
            raise ValueError(f"Clip expects a 1D/2D array, got {array.shape}")
        if self.max_value < self.min_value:
            raise ValueError("max_value must be >= min_value")
        self.n_features_in_ = int(array.shape[-1])
        return self

    def transform(self, X: Any) -> np.ndarray:
        return np.clip(np.asarray(X), self.min_value, self.max_value)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"min_value": self.min_value, "max_value": self.max_value}

    def set_params(self, **params: Any) -> "Clip":
        if "min_value" in params:
            self.min_value = float(params["min_value"])
        if "max_value" in params:
            self.max_value = float(params["max_value"])
        return self


class Standardize(StateTransformer):
    """Zero-mean/unit-variance scaling fitted on training samples."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = float(epsilon)

    def fit(self, X: Any, y: Any = None) -> "Standardize":
        array = np.asarray(X, dtype=np.float64)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise ValueError(f"Standardize expects a 1D/2D array, got {array.shape}")
        self.mean_ = array.mean(axis=0)
        self.scale_ = array.std(axis=0)
        self.scale_ = np.maximum(self.scale_, self.epsilon)
        self.n_features_in_ = int(array.shape[1])
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise RuntimeError("Standardize must be fitted before transform")
        array = np.asarray(X, dtype=np.float64)
        single = array.ndim == 1
        if single:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"Expected observations with {self.mean_.shape[0]} features, "
                f"got {array.shape}"
            )
        result = ((array - self.mean_) / self.scale_).astype(np.float32)
        return result[0] if single else result

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"epsilon": self.epsilon}

    def set_params(self, **params: Any) -> "Standardize":
        if "epsilon" in params:
            self.epsilon = float(params["epsilon"])
        return self
