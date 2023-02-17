from typing import Union

import torch
from habitat_baselines.rl.ppo import Net
from habitat_baselines.utils.common import CategoricalNet, GaussianNet
from torch import nn


class DAggerPolicyMixin:
    """Avoids computing value or action_log_probs, which are RL-only, and .act() returns
    only the action. Also endows the parent class with a self-updating
    self.recurrent_hidden_states"""

    action_distribution: Union[CategoricalNet, GaussianNet]
    critic: nn.Module
    net: nn.Module
    action_distribution_type: str
    prev_distribution: None
    rnn_hidden_states: torch.Tensor
    num_recurrent_layers: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_hidden_states(self, num_envs, device):
        output_size = self.net.output_size if hasattr(self, "net") else 0
        self.rnn_hidden_states = torch.zeros(
            num_envs,
            self.num_recurrent_layers,
            output_size,
            device=device,
        )

    def act(self, observations, prev_actions, masks, deterministic=False):
        if not hasattr(self, "net"):  # if we're using a non-network policy
            return super().act(observations, prev_actions, masks, deterministic)[1]

        features, self.rnn_hidden_states, _ = self.net(
            observations, self.rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)

        with torch.no_grad():
            if deterministic:
                if self.action_distribution_type == "categorical":
                    action = distribution.mode()
                elif self.action_distribution_type == "gaussian":
                    action = distribution.mean
                else:
                    raise NotImplementedError(
                        "Distribution type {} is not supported".format(
                            self.action_distribution_type
                        )
                    )
            else:
                action = distribution.sample()

        return action

    def evaluate_log_probs(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info,
    ):
        """Given a state and action, computes the policy's action distribution for that
        state and then returns the log probability of the given action under this
        distribution."""
        features, _, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks, rnn_build_seq_info
        )
        distribution = self.action_distribution(features)
        log_probs = distribution.log_probs(action)
        return log_probs

    def get_logits(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info,
    ):
        """Assumes a Categorical action distribution. Given a state, computes the
        policy's action distribution for that state and returns its logits."""
        features, _, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks, rnn_build_seq_info
        )
        distribution = self.action_distribution(features)
        logits = distribution.logits
        return logits
