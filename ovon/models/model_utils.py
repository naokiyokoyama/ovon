from typing import Union

import torch
from habitat_baselines.utils.common import CategoricalNet, GaussianNet
from torch import nn


class DAggerPolicyMixin:
    """Avoids computing value or action_log_probs, which are RL-only, and
    .evaluate_actions() will be overridden to produce the correct gradients."""

    action_distribution: Union[CategoricalNet, GaussianNet]
    net: nn.Module
    action_distribution_type: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def act(
        self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False
    ):
        features, rnn_hidden_states, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
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
        n = action.shape[0]
        value = torch.zeros(n, 1, device=action.device)
        action_log_probs = torch.zeros(n, 1, device=action.device)
        return value, action, action_log_probs, rnn_hidden_states

    def evaluate_actions(
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
