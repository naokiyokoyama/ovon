from typing import Union

import torch
from habitat_baselines.rl.ppo import Net
from habitat_baselines.utils.common import CategoricalNet, GaussianNet
from torch import nn


class ComputeLogProbsMixin:
    action_distribution: Union[CategoricalNet, GaussianNet]
    critic: nn.Module
    net: nn.Module
    action_distribution_type: str
    prev_distribution: None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        """Copied from policy.py, but self.prev_distribution is saved for later use, and
        don't waste time computing value or action_log_probs which are RL-only"""
        features, rnn_hidden_states, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        self.prev_distribution = distribution = self.action_distribution(features)

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

        value, action_log_probs = None, None
        return value, action, action_log_probs, rnn_hidden_states

    def compute_log_probs(self, actions):
        a = actions.clone() if torch.is_inference(actions) else actions
        log_probs = self.prev_distribution.log_probs(a)
        return log_probs


class RecurrentPolicyMixin:
    """Endows the parent class with a self-updating self.recurrent_hidden_states"""

    recurrent_hidden_states: torch.Tensor
    num_recurrent_layers: int
    net: Net

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_hidden_states(self, num_envs, device):
        output_size = self.net.output_size if hasattr(self, "net") else 0
        self.recurrent_hidden_states = torch.zeros(
            num_envs,
            self.num_recurrent_layers,
            output_size,
            device=device,
        )

    def act(
        self,
        observations,
        prev_actions,
        masks,
        deterministic=False,
    ):
        value, action, action_log_probs, self.recurrent_hidden_states = super().act(
            observations,
            self.recurrent_hidden_states.detach(),
            prev_actions,
            masks,
            deterministic,
        )
        return value, action, action_log_probs
