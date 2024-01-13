from typing import Dict, Optional, Tuple

import torch
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import NetPolicy

from ovon.models.clip_policy import OVONNet, PointNavResNetCLIPPolicy
from ovon.models.transformer_encoder import TransformerEncoder


@baseline_registry.register_policy
class OVONTransformerPolicy(PointNavResNetCLIPPolicy):
    is_transformer = True

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        transformer_config,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        backbone: str = "clip_avgpool",
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        depth_ckpt: str = "",
        fusion_type: str = "concat",
        attn_heads: int = 3,
        use_vis_query: bool = False,
        use_residual: bool = True,
        residual_vision: bool = False,
        unfreeze_xattn: bool = False,
        **kwargs,
    ):
        self.unfreeze_xattn = unfreeze_xattn
        if policy_config is not None:
            discrete_actions = policy_config.action_distribution_type == "categorical"
            self.action_distribution_type = policy_config.action_distribution_type
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"

        NetPolicy.__init__(
            self,
            OVONTransformerNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                discrete_actions=discrete_actions,
                depth_ckpt=depth_ckpt,
                fusion_type=fusion_type,
                attn_heads=attn_heads,
                use_vis_query=use_vis_query,
                use_residual=use_residual,
                residual_vision=residual_vision,
                transformer_config=transformer_config,
            ),
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

    @classmethod
    def from_config(cls, config: "DictConfig", *args, **kwargs):
        tf_cfg = config.habitat_baselines.rl.policy.transformer_config
        return super().from_config(config, transformer_config=tf_cfg, *args, **kwargs)

    @property
    def num_recurrent_layers(self):
        return self.net.state_encoder.n_layers

    @property
    def num_heads(self):
        return self.net.state_encoder.n_head

    @property
    def context_len(self):
        return self.net.state_encoder.max_position_embeddings

    @property
    def recurrent_hidden_size(self):
        return self.net.state_encoder.n_embed

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        import os

        if os.environ.get("OVON_IL_DONT_CHEAT", "0") == "1":
            return NetPolicy.act(
                self,
                observations,
                rnn_hidden_states,
                prev_actions,
                masks,
                deterministic=deterministic,
            )

        # TODO: Currently returning dummy values for imitation learning
        num_envs = observations["rgb"].shape[0]
        device = rnn_hidden_states.device
        action_log_probs = torch.zeros(num_envs, 1).to(device)
        value = torch.zeros(num_envs, 1).to(device)

        action = observations["teacher_label"].to(device).long()

        return value, action, action_log_probs, rnn_hidden_states


class OVONTransformerNet(OVONNet):
    """Same as OVONNet but uses transformer instead of LSTM."""

    def __init__(self, transformer_config, *args, **kwargs):
        self.transformer_config = transformer_config
        super().__init__(*args, **kwargs)

    @property
    def output_size(self):
        return self.transformer_config.n_hidden

    def build_state_encoder(self):
        state_encoder = TransformerEncoder(
            self.rnn_input_size, config=self.transformer_config
        )
        return state_encoder

    def encode_state(
        self,
        out: torch.Tensor,
        rnn_hidden_states: torch.Tensor,
        masks: torch.Tensor,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        full_rnn_state = False

        n_envs = masks.shape[0]
        seq_len = masks.shape[1]

        rnn_build_seq_info = {
            "dims": torch.tensor([n_envs, seq_len]),
            "is_first": torch.tensor(True),
            "old_context_length": torch.tensor(0),
        }

        out, rnn_hidden_states, *output = self.state_encoder(
            out,
            rnn_hidden_states,
            masks,
            rnn_build_seq_info,
            full_rnn_state=full_rnn_state,
        )

        return out, rnn_hidden_states
