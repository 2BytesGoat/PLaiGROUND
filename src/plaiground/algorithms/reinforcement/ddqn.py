"""Train a Double DQN policy from an experiment manifest.

Double DQN differs from vanilla DQN only in the TD target: the online network
selects the next action and the target network evaluates it. The policy and
checkpoint format remain DQN-compatible.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from stable_baselines3 import DQN

from plaiground.experiments.runner import parse_train_args, run_from_args

from .dqn import run_training


class DoubleDQN(DQN):
    """DQN with Double-DQN targets."""

    @staticmethod
    def _margin_loss(q_values, actions, margin: float):
        """Rank the stored action above every alternative."""
        action_index = actions.long().reshape(-1, 1)
        q_action = q_values.gather(1, action_index)
        mask = torch.zeros_like(q_values, dtype=torch.bool)
        mask.scatter_(1, action_index, True)
        q_other = q_values.masked_fill(mask, -float("inf")).max(
            dim=1, keepdim=True
        ).values
        return F.relu(float(margin) - (q_action - q_other)).mean()

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        expert_td_values = []
        expert_margin_values = []
        old_margin_values = []
        expert_buffer = getattr(self, "expert_buffer", None)
        expert_batch_size = int(
            getattr(self, "expert_batch_size", 64)
        )
        behavior_margin = float(
            getattr(self, "behavior_margin", 0.5)
        )
        behavior_margin_weight = float(
            getattr(self, "behavior_margin_weight", 0.0)
        )
        expert_td_weight = float(
            getattr(self, "expert_td_weight", 1.0)
        )
        old_margin_weight = float(
            getattr(self, "old_margin_weight", 0.0)
        )
        old_margin_batch_size = int(
            getattr(self, "old_margin_batch_size", 64)
        )
        old_anchor = getattr(
            self.replay_buffer, "anchor", None
        )
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            with torch.no_grad():
                next_online_q = self.q_net(replay_data.next_observations)
                next_actions = next_online_q.argmax(dim=1, keepdim=True)
                next_target_q = self.q_net_target(replay_data.next_observations)
                next_q_values = next_target_q.gather(1, next_actions)
                targets = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(
                current_q_values,
                dim=1,
                index=replay_data.actions.long(),
            )
            loss = F.smooth_l1_loss(current_q_values, targets)
            expert_td = torch.zeros((), device=loss.device)
            expert_margin = torch.zeros((), device=loss.device)
            old_margin = torch.zeros((), device=loss.device)

            if expert_buffer is not None and expert_batch_size > 0:
                expert_data = expert_buffer.sample(
                    expert_batch_size, env=self._vec_normalize_env
                )
                with torch.no_grad():
                    expert_next_online = self.q_net(
                        expert_data.next_observations
                    )
                    expert_next_actions = expert_next_online.argmax(
                        dim=1, keepdim=True
                    )
                    expert_next_target = self.q_net_target(
                        expert_data.next_observations
                    )
                    expert_next_q = expert_next_target.gather(
                        1, expert_next_actions
                    )
                    expert_targets = (
                        expert_data.rewards
                        + (1 - expert_data.dones)
                        * self.gamma
                        * expert_next_q
                    )
                expert_q = self.q_net(expert_data.observations)
                expert_current_q = expert_q.gather(
                    1, expert_data.actions.long()
                )
                expert_td = F.smooth_l1_loss(
                    expert_current_q, expert_targets
                )
                expert_margin = self._margin_loss(
                    expert_q, expert_data.actions, behavior_margin
                )
                loss = (
                    loss
                    + expert_td_weight * expert_td
                    + behavior_margin_weight * expert_margin
                )

            if (
                old_anchor is not None
                and old_margin_weight > 0.0
                and old_margin_batch_size > 0
            ):
                old_data = old_anchor.sample(
                    old_margin_batch_size, env=self._vec_normalize_env
                )
                old_q = self.q_net(old_data.observations)
                old_margin = self._margin_loss(
                    old_q, old_data.actions, behavior_margin
                )
                loss = loss + old_margin_weight * old_margin

            losses.append(loss.item())
            expert_td_values.append(float(expert_td.item()))
            expert_margin_values.append(float(expert_margin.item()))
            old_margin_values.append(float(old_margin.item()))

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            )
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record(
            "train/n_updates", self._n_updates, exclude="tensorboard"
        )
        self.logger.record("train/loss", float(sum(losses) / len(losses)))
        self.logger.record(
            "train/expert_td",
            float(sum(expert_td_values) / len(expert_td_values)),
        )
        self.logger.record(
            "train/expert_margin",
            float(
                sum(expert_margin_values) / len(expert_margin_values)
            ),
        )
        self.logger.record(
            "train/old_margin",
            float(sum(old_margin_values) / len(old_margin_values)),
        )


def main() -> None:
    run = run_from_args(parse_train_args(__doc__ or "Train Double DQN."))
    run_training(run, model_cls=DoubleDQN, checkpoint_prefix="ddqn")


if __name__ == "__main__":
    main()
