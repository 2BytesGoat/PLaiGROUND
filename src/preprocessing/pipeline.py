"""Sequential lean observation preprocessing."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.base import BaseEstimator

from .base import as_batch, is_single_array, restore_single


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


class StatePipeline(BaseEstimator):
    """Sequential transform with sklearn-like names and parameters.

    Input may be one flat observation or a batch; the output keeps that rank.
    Intermediate values may be structured (``LeanStateBatch``).
    """

    def __init__(self, steps: Iterable[tuple[str, Any]]) -> None:
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
