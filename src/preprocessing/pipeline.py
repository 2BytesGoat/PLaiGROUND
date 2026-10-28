"""Composable observation preprocessing pipelines."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.base import BaseEstimator

from .base import (
    StateTransformer,
    as_batch,
    is_single_array,
    restore_single,
)


def _validate_steps(steps: Iterable[tuple[str, Any]]) -> list[tuple[str, Any]]:
    result = list(steps)
    if not result:
        raise ValueError("pipeline needs at least one step")
    names: set[str] = set()
    for name, step in result:
        if not isinstance(name, str) or not name:
            raise ValueError(f"step name must be a non-empty string: {name!r}")
        if name in names:
            raise ValueError(f"duplicate step name: {name!r}")
        names.add(name)
        if not (hasattr(step, "fit") and hasattr(step, "transform")):
            raise TypeError(f"step {name!r} needs fit() and transform()")
    return result


def _fit_transform(step: Any, data: Any, y: Any = None) -> Any:
    if hasattr(step, "fit_transform"):
        return step.fit_transform(data, y)
    return step.fit(data, y).transform(data)


def _as_matrix(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    if array.ndim != 2:
        raise ValueError(
            "a model-facing preprocessing step must return a 1D/2D array; "
            f"got {array.shape}"
        )
    return array


class StatePipeline(BaseEstimator):
    """Sequential transform with sklearn-like names and parameters.

    Each step receives the previous step's output. The input may be one flat
    observation or a batch; the output keeps that rank. Intermediate values
    may be structured (for example ``RawStateBatch``), so this is a
    purpose-built pipeline rather than ``sklearn.pipeline.Pipeline``.
    """

    def __init__(self, steps: Iterable[tuple[str, Any]]) -> None:
        # Keep the constructor parameter object unchanged for sklearn.clone();
        # validate it into a separate list used by the methods.
        self.steps = steps
        self._steps = _validate_steps(steps)

    @property
    def named_steps(self) -> dict[str, Any]:
        return dict(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self._steps[key][1]
        return self.named_steps[key]

    def _apply(self, X: Any, method: str, y: Any = None) -> Any:
        data = X
        for _name, step in self._steps:
            if method == "fit_transform":
                data = _fit_transform(step, data, y)
            else:
                data = getattr(step, method)(data)
        return data

    def fit(self, X: Any, y: Any = None) -> "StatePipeline":
        data = X
        for _name, step in self._steps[:-1]:
            data = _fit_transform(step, data, y)
        self._steps[-1][1].fit(data, y)
        self.n_features_in_ = int(np.asarray(X).shape[-1])
        output = self.transform(X)
        array = np.asarray(output)
        if array.ndim not in (1, 2):
            raise ValueError(
                "the final pipeline step must return a 1D/2D model array; "
                f"got {array.shape}"
            )
        self.n_features_out_ = int(array.shape[-1])
        return self

    def transform(self, X: Any) -> Any:
        single = is_single_array(X)
        data = as_batch(X) if single else X
        result = self._apply(data, "transform")
        return restore_single(result) if single else result

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        single = is_single_array(X)
        data = as_batch(X) if single else X
        result = self._apply(data, "fit_transform", y)
        self.n_features_in_ = int(np.asarray(X).shape[-1])
        array = np.asarray(result)
        if array.ndim not in (1, 2):
            raise ValueError(
                "the final pipeline step must return a 1D/2D model array; "
                f"got {array.shape}"
            )
        self.n_features_out_ = int(array.shape[-1])
        return restore_single(result) if single else result

    @property
    def output_size(self) -> int:
        if hasattr(self, "n_features_out_"):
            return int(self.n_features_out_)
        raise RuntimeError(
            "pipeline output size is unknown until fit()/fit_transform() has run"
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = {"steps": self.steps}
        if deep:
            for name, step in self._steps:
                getter = getattr(step, "get_params", None)
                if getter is None:
                    continue
                for key, value in getter(deep=True).items():
                    params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params: Any) -> "StatePipeline":
        if "steps" in params:
            self.steps = params.pop("steps")
            self._steps = _validate_steps(self.steps)
        named = self.named_steps
        for key, value in params.items():
            if "__" not in key:
                raise ValueError(f"unknown pipeline parameter: {key!r}")
            name, parameter = key.split("__", 1)
            if name not in named:
                raise ValueError(f"unknown pipeline step: {name!r}")
            step = named[name]
            setter = getattr(step, "set_params", None)
            if setter is None:
                raise TypeError(f"step {name!r} does not support set_params")
            setter(**{parameter: value})
        return self


class FeatureUnion(StateTransformer):
    """Run several branches on the same value and concatenate the outputs."""

    def __init__(self, transformers: Iterable[tuple[str, Any]]) -> None:
        self.transformers = transformers
        self._transformers = _validate_steps(transformers)

    @property
    def named_transformers(self) -> dict[str, Any]:
        return dict(self._transformers)

    def __len__(self) -> int:
        return len(self._transformers)

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self._transformers[key][1]
        return self.named_transformers[key]

    def _apply(self, X: Any, method: str, y: Any = None) -> np.ndarray:
        outputs = []
        for _name, step in self._transformers:
            if method == "fit_transform":
                value = _fit_transform(step, X, y)
            else:
                value = step.transform(X)
            outputs.append(_as_matrix(value))
        if not outputs:
            raise ValueError("FeatureUnion needs at least one branch")
        count = outputs[0].shape[0]
        for output in outputs[1:]:
            if output.shape[0] != count:
                raise ValueError(
                    "FeatureUnion branches returned different batch sizes: "
                    f"{[item.shape[0] for item in outputs]}"
                )
        return np.concatenate(outputs, axis=1).astype(np.float32, copy=False)

    def fit(self, X: Any, y: Any = None) -> "FeatureUnion":
        self._apply(X, "fit_transform", y)
        self.n_features_out_ = int(self._apply(X, "transform").shape[1])
        return self

    def transform(self, X: Any) -> np.ndarray:
        return self._apply(X, "transform")

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        result = self._apply(X, "fit_transform", y)
        self.n_features_out_ = int(result.shape[1])
        return result

    @property
    def output_size(self) -> int:
        if hasattr(self, "n_features_out_"):
            return int(self.n_features_out_)
        raise RuntimeError(
            "FeatureUnion output size is unknown until fit()/fit_transform() has run"
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = {"transformers": self.transformers}
        if deep:
            for name, step in self._transformers:
                getter = getattr(step, "get_params", None)
                if getter is None:
                    continue
                for key, value in getter(deep=True).items():
                    params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params: Any) -> "FeatureUnion":
        if "transformers" in params:
            self.transformers = params.pop("transformers")
            self._transformers = _validate_steps(self.transformers)
        named = self.named_transformers
        for key, value in params.items():
            if "__" not in key:
                raise ValueError(f"unknown FeatureUnion parameter: {key!r}")
            name, parameter = key.split("__", 1)
            if name not in named:
                raise ValueError(f"unknown FeatureUnion branch: {name!r}")
            step = named[name]
            setter = getattr(step, "set_params", None)
            if setter is None:
                raise TypeError(f"branch {name!r} does not support set_params")
            setter(**{parameter: value})
        return self


# A short alias for scripts that prefer the sklearn spelling.
Pipeline = StatePipeline
