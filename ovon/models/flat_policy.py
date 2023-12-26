from itertools import chain

import einops
import gym.spaces as spaces
import torch.nn as nn
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import CriticHead, Policy
from habitat_baselines.utils.common import (CategoricalNet,
                                            CustomFixedCategorical,
                                            GaussianNet, get_num_actions)

from ovon.models.transformer_wrappers import TransformerWrapper


@baseline_registry.register_policy
class FlatPolicy(nn.Module, Policy):
    def __init__(self, config, obs_space, action_space, **kwargs):
        super().__init__()

        policy_cfg = config.habitat_baselines.rl.policy
        self.policy_core = TransformerWrapper(
            policy_cfg.model_id, policy_cfg.use_b16, policy_cfg.debug_mode
        )

        hidden_size = policy_cfg.ac_hidden_size
        assert isinstance(
            action_space, spaces.Discrete
        ), "Only discrete action spaces are supported right now (but this is easy to fix)"
        self.distrib_head = CategoricalNet(hidden_size, action_space.n)
        self.critic = CriticHead(hidden_size)

    @property
    def visual_encoder(self):
        return self.policy_core.visual_encoder

    def get_policy_components(self):
        return [self]

    @property
    def recurrent_hidden_size(self):
        return self.policy_core.rnn_hidden_dim

    @property
    def num_recurrent_layers(self):
        return self.policy_core.get_num_rnn_layers()

    def parameters(self):
        return chain(
            self.policy_core.get_trainable_params(),
            self.distrib_head.parameters(),
            self.critic.parameters(),
        )

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.policy_core.forward(observations, rnn_hidden_states, masks)
        return self.critic(features)

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info,
    ):
        distrib, features, _ = self.get_distrib(
            observations, rnn_hidden_states, masks, rnn_build_seq_info
        )
        value = self.critic(features)

        if len(action.shape) == 3:
            # [batch_size, seq_len, data_dim]
            log_probs = distrib.log_prob(einops.rearrange(action, "b t 1 -> b t"))
            log_probs = einops.rearrange(log_probs, "b t -> b t 1")
        else:
            log_probs = distrib.log_probs(action)

        distribution_entropy = distrib.entropy()

        return (
            value,
            log_probs,
            distribution_entropy,
            rnn_hidden_states,
            {},
        )

    def get_distrib(
        self,
        observations,
        rnn_hidden_states,
        masks,
        rnn_build_seq_info=None,
        flatten=False,
    ):
        features, rnn_hidden_states = self.policy_core.forward(
            observations, rnn_hidden_states, masks, rnn_build_seq_info
        )
        distrib = self.distrib_head(features)

        return distrib, features, rnn_hidden_states

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):

        distrib, features, rnn_hidden_states = self.get_distrib(
            observations, rnn_hidden_states, masks
        )
        values = self.critic(features)

        if deterministic:
            action = distrib.mode()
        else:
            action = distrib.sample()
        action_log_probs = distrib.log_probs(action)

        return (
            values,
            action,
            action_log_probs,
            rnn_hidden_states
        )

    @classmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        return cls(config, observation_space, action_space, **kwargs)
