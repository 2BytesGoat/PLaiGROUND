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

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
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
            losses.append(loss.item())

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


def main() -> None:
    run = run_from_args(parse_train_args(__doc__ or "Train Double DQN."))
    run_training(run, model_cls=DoubleDQN, checkpoint_prefix="ddqn")


if __name__ == "__main__":
    main()
