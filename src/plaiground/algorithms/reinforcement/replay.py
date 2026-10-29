"""Replay buffers for protected multi-level DDQN training.

The ordinary SB3 ``ReplayBuffer`` is a single FIFO ring. When a new level is
introduced, old-level transitions are gradually evicted and the policy can
forget them even if the level is still in the training mix.

``ProtectedReplayBuffer`` keeps two independent buffers:

* an ordinary online buffer for current mixed-level transitions;
* a fixed anchor buffer collected from the prior selected model.

Every sample is a mix of both. ``RareReplayBuffer`` stores a current-state
bucket, return-to-go, and old-policy Q value for each transition. Anchor slots
are weighted by inverse state frequency and positive (return-to-go - Q) gaps,
so valuable or rare mid-route states are not drowned out by common states.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


def pad_anchor_observations(
    anchor: ReplayBuffer, observation_space
) -> ReplayBuffer:
    """Append zero-valued features to an older, narrower anchor buffer.

    This is used when a lean 188 model is expanded to lean_stick 191. The old
    anchor's first 188 lean features stay in place; the three new hold flags
    are zero because old 1-1/1-2 transitions have no stick state.
    """
    old_shape = tuple(anchor.obs_shape)
    new_shape = tuple(observation_space.shape)
    if old_shape == new_shape:
        return anchor
    if len(old_shape) != 1 or len(new_shape) != 1 or new_shape[0] < old_shape[0]:
        raise ValueError(
            f"cannot pad anchor observation shape {old_shape} to {new_shape}"
        )
    for name in ("observations", "next_observations"):
        values = getattr(anchor, name)
        padded = np.zeros(values.shape[:-1] + new_shape, dtype=values.dtype)
        padded[..., : old_shape[0]] = values
        setattr(anchor, name, padded)
    anchor.obs_shape = new_shape
    anchor.observation_space = observation_space
    return anchor


def state_bucket_key(info: dict[str, Any]) -> str:
    """Build a state bucket for rare/high-profit replay protection.

    This is the *current* state, not the episode start. The bucket is deliberately
    coarse: cell/facing/time-bin/powerup state, so the reservoir covers valuable
    route states without exploding into per-frame uniqueness.
    """
    level = info.get("level", "unknown")
    cell = info.get("cell")
    if isinstance(cell, (list, tuple)) and len(cell) == 2:
        cell_key = f"{int(cell[0])},{int(cell[1])}"
    else:
        position = info.get("global_position")
        if isinstance(position, (list, tuple)) and len(position) == 2:
            cell_key = f"{int(position[0]) // 16},{int(position[1]) // 16}"
        else:
            cell_key = "unknown"
    facing = info.get("facing_direction", info.get("start_facing", "?"))
    run_time = info.get("run_time")
    try:
        run_bin = int(float(run_time) / 0.5)
    except (TypeError, ValueError):
        run_bin = -1
    powerups = info.get("powerup_stack")
    if isinstance(powerups, (list, tuple)):
        power_key = ",".join(str(int(value)) for value in powerups)
    else:
        power_key = "?"
    return f"{level}|{cell_key}|{facing}|{run_bin}|{power_key}"


class RareReplayBuffer(ReplayBuffer):
    """SB3 replay buffer with a bucket ID per stored transition."""

    def __init__(
        self,
        *args,
        rare_weight_cap: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rare_weight_cap = float(rare_weight_cap)
        self.bucket_ids = np.full(
            (self.buffer_size, self.n_envs), -1, dtype=np.int64
        )
        self.bucket_names: dict[str, int] = {}
        # Return-to-go and old-policy Q values; filled by the collector.
        self.returns = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.q_values = np.full(
            (self.buffer_size, self.n_envs), np.nan, dtype=np.float32
        )
        self.profit_gaps = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        # Sampling priority, filled from positive return/Q gaps.
        self.priorities = np.ones(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.gap_threshold: float | None = None
        # The anchor is fixed during training, so its sampling distribution can
        # be cached across minibatches. Invalidate whenever stored data changes.
        self._probability_cache: tuple[int, np.ndarray] | None = None

    def _bucket_id(self, info: dict[str, Any]) -> int:
        key = state_bucket_key(info)
        value = self.bucket_names.get(key)
        if value is None:
            value = len(self.bucket_names)
            self.bucket_names[key] = value
        return value

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        index = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        for env_index, info in enumerate(infos[: self.n_envs]):
            self.bucket_ids[index, env_index] = self._bucket_id(info)
            self.returns[index, env_index] = 0.0
            self.q_values[index, env_index] = np.nan
            self.profit_gaps[index, env_index] = 0.0
            self.priorities[index, env_index] = 1.0
        self._probability_cache = None

    def set_return(
        self, slot: int, env_index: int, return_to_go: float
    ) -> None:
        if 0 <= slot < self.buffer_size and 0 <= env_index < self.n_envs:
            self.returns[slot, env_index] = float(return_to_go)
            self._probability_cache = None

    def set_priority(self, slot: int, env_index: int, value: float) -> None:
        """Legacy hook: assign a manual sampling priority."""
        if 0 <= slot < self.buffer_size and 0 <= env_index < self.n_envs:
            self.priorities[slot, env_index] = max(0.0, float(value))
            self._probability_cache = None

    def reset(self) -> None:
        super().reset()
        self.bucket_ids.fill(-1)
        self.bucket_names.clear()
        self.returns.fill(0.0)
        self.q_values.fill(np.nan)
        self.profit_gaps.fill(0.0)
        self.priorities.fill(1.0)
        self.gap_threshold = None
        self._probability_cache = None

    def compute_q_values(
        self, q_net, batch_size: int = 4096
    ) -> int:
        """Fill old-policy Q(s,a) for all currently stored transitions."""
        size = self.size()
        if size == 0:
            return 0
        observations = self.observations[:size].reshape(
            (-1,) + self.obs_shape
        )
        actions = self.actions[:size].reshape(-1, 1).astype(np.int64)
        device = next(q_net.parameters()).device
        values = np.empty(observations.shape[0], dtype=np.float32)
        with torch.no_grad():
            for start in range(0, observations.shape[0], batch_size):
                end = min(start + batch_size, observations.shape[0])
                batch = torch.as_tensor(
                    observations[start:end], device=device
                )
                q_values = q_net(batch)
                selected = q_values.gather(
                    1,
                    torch.as_tensor(
                        actions[start:end], device=device, dtype=torch.long
                    ),
                )
                values[start:end] = selected.squeeze(1).cpu().numpy()
        self.q_values[:size] = values.reshape(size, self.n_envs)
        return size

    def compute_gap_priorities(
        self, quantile: float = 0.9
    ) -> dict[str, Any]:
        """Set priorities from positive (return_to_go - Q_old) gaps.

        The threshold is a quantile of positive relative gaps. States below the
        threshold still receive rarity weight, but no extra profit weight.
        """
        size = self.size()
        if size == 0:
            return {"threshold": None, "positive": 0, "stored": 0}
        returns = self.returns[:size].astype(np.float64)
        q_values = self.q_values[:size].astype(np.float64)
        valid = np.isfinite(q_values)
        gaps = np.zeros_like(returns)
        relative = np.zeros_like(returns)
        gaps[valid] = returns[valid] - q_values[valid]
        relative[valid] = np.maximum(0.0, gaps[valid]) / (
            np.abs(returns[valid]) + 1.0
        )
        positive = relative[relative > 0.0]
        if positive.size:
            threshold = float(
                np.quantile(positive, min(max(quantile, 0.0), 1.0))
            )
        else:
            threshold = 0.0
        priorities = np.where(relative >= threshold, relative, 0.0)
        self.profit_gaps[:size] = gaps.astype(np.float32)
        self.priorities[:size] = priorities.astype(np.float32)
        self.gap_threshold = threshold
        self._probability_cache = None
        return {
            "threshold": threshold,
            "positive": int(positive.size),
            "stored": int(valid.sum()),
            "max_gap": float(gaps[valid].max()) if valid.any() else 0.0,
        }

    def bucket_counts(self) -> dict[str, int]:
        if self.size() == 0:
            return {}
        ids = self.bucket_ids[: self.size()].reshape(-1)
        ids = ids[ids >= 0]
        counts = np.bincount(ids, minlength=len(self.bucket_names))
        names = {value: key for key, value in self.bucket_names.items()}
        return {
            names[index]: int(count)
            for index, count in enumerate(counts)
            if count
        }

    def sample(
        self, batch_size: int, env=None
    ) -> ReplayBufferSamples:
        size = self.size()
        if size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        ids = self.bucket_ids[:size]
        valid = ids >= 0
        if not valid.any():
            return super().sample(batch_size=batch_size, env=env)

        cache = getattr(self, "_probability_cache", None)
        if cache is not None and cache[0] == size:
            probabilities = cache[1]
        else:
            counts = np.bincount(
                ids[valid], minlength=len(self.bucket_names)
            )
            weights = np.ones(size, dtype=np.float64)
            for slot in range(size):
                row = ids[slot][valid[slot]]
                if row.size:
                    weights[slot] = 1.0 / max(
                        1, int(min(counts[b] for b in row))
                    )
            # Profit-gap weighting: priorities are positive
            # (return_to_go - Q_old) gaps above the configured quantile.
            # Normalize per snapshot so the signal cannot overwhelm rarity
            # protection.
            priority = self.priorities[:size].max(axis=1).astype(np.float64)
            max_priority = float(priority.max()) if priority.size else 0.0
            if max_priority > 0.0:
                weights *= 1.0 + priority / max_priority
            weights = np.minimum(weights, self.rare_weight_cap)
            probabilities = weights / weights.sum()
            self._probability_cache = (size, probabilities)
        indices = np.random.choice(
            size, size=batch_size, replace=True, p=probabilities
        )
        return self._get_samples(indices, env=env)


class ProtectedReplayBuffer:
    """Online FIFO buffer plus a fixed, rare-weighted anchor partition."""

    def __init__(
        self,
        online_size: int,
        observation_space,
        action_space,
        *,
        device: str = "auto",
        n_envs: int = 1,
        anchor_buffer: ReplayBuffer | None = None,
        anchor_fraction: float = 0.25,
    ) -> None:
        if not 0.0 <= float(anchor_fraction) <= 1.0:
            raise ValueError("anchor_fraction must be between 0 and 1")
        if anchor_buffer is not None and anchor_buffer.n_envs != n_envs:
            raise ValueError(
                "anchor buffer n_envs "
                f"({anchor_buffer.n_envs}) must match training n_envs ({n_envs})"
            )
        self.n_envs = int(n_envs)
        self.observation_space = observation_space
        self.action_space = action_space
        self.anchor_fraction = float(anchor_fraction)
        self.online = ReplayBuffer(
            online_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
        )
        self.anchor = anchor_buffer
        anchor_slots = anchor_buffer.buffer_size if anchor_buffer is not None else 0
        self.buffer_size = self.online.buffer_size + anchor_slots
        self.optimize_memory_usage = False

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        self.online.add(obs, next_obs, action, reward, done, infos)

    def reset(self) -> None:
        self.online.reset()

    def size(self) -> int:
        anchor_size = self.anchor.size() if self.anchor is not None else 0
        return self.online.size() + anchor_size

    def sample(
        self, batch_size: int, env=None
    ) -> ReplayBufferSamples:
        online_size = self.online.size()
        anchor_size = self.anchor.size() if self.anchor is not None else 0
        if online_size == 0 and anchor_size == 0:
            raise ValueError("cannot sample from empty online and anchor buffers")
        if online_size == 0:
            return self.anchor.sample(batch_size=batch_size, env=env)
        if anchor_size == 0:
            return self.online.sample(batch_size=batch_size, env=env)

        anchor_batch = int(round(batch_size * self.anchor_fraction))
        anchor_batch = min(max(anchor_batch, 1), batch_size - 1)
        online_batch = batch_size - anchor_batch
        anchor_samples = self.anchor.sample(anchor_batch, env=env)
        online_samples = self.online.sample(online_batch, env=env)
        parts = []
        for anchor_part, online_part in zip(anchor_samples, online_samples):
            if anchor_part is None and online_part is None:
                parts.append(None)
            elif anchor_part is None:
                parts.append(online_part)
            elif online_part is None:
                parts.append(anchor_part)
            else:
                parts.append(torch.cat((anchor_part, online_part), dim=0))
        return ReplayBufferSamples(*parts)

    def stats(self) -> dict[str, Any]:
        return {
            "online_slots": self.online.size(),
            "anchor_slots": self.anchor.size() if self.anchor is not None else 0,
            "anchor_fraction": self.anchor_fraction,
            "anchor_buckets": (
                self.anchor.bucket_counts()
                if isinstance(self.anchor, RareReplayBuffer)
                else {}
            ),
        }
