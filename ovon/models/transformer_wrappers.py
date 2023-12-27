from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import transformers
from habitat import logger
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder

from ovon.models.llamarl.configuration_llamarl import LlamaRLConfig
from ovon.models.llamarl.modeling_llamarl import LlamaRLModel


class TransformerWrapper(nn.Module):
    def __init__(
        self,
        model_id,
        use_b16,
        debug_mode=False,
        observation_space=None,
        inter_episodes_attention=False,
        reset_position_index=True,
        force_blind_policy=False,
        fuse_keys=None,
    ):
        super().__init__()
        self._debug_mode = debug_mode
        self.inter_episodes_attention = inter_episodes_attention
        self.reset_position_index = reset_position_index
        if model_id == "gpt":
            self.hf_config = transformers.GPT2Config(
                n_embd=512,
                n_layer=2,
                n_head=8,
            )
            self.n_layers = self.hf_config.n_layer
            self.n_embed = self.hf_config.n_embd
            self.n_head = self.hf_config.n_head

            # self.feats_proj = nn.Linear(14, self.hf_config.n_embd)
            self.model = transformers.GPT2Model(self.hf_config)
            self.model.wte.weight.requires_grad_(False)
            self.context_len = 32
        elif model_id == "llama":
            self.hf_config = transformers.LlamaConfig(
                hidden_size=512,
                intermediate_size=512 * 4,
                num_hidden_layers=2,
                num_attention_heads=8,
            )
            self.n_layers = self.hf_config.num_hidden_layers
            self.n_embed = self.hf_config.hidden_size
            self.n_head = self.hf_config.num_attention_heads
            # self.feats_proj = nn.Linear(14, self.hf_config.n_embd)
            self.model = LlamaRLModel(self.hf_config)
            self.model.embed_tokens.weight.requires_grad_(False)
            self.context_len = 32

        else:
            raise ValueError(f"Unrecognized {model_id}")

        if use_b16:
            logger.info("Setting model data type to bfloat16.")
            create_type = torch.bfloat16
            self.model = self.model.to(create_type)
        else:
            create_type = torch.float32

        self._create_type = create_type
        self._non_causal_mask = None
        self.tmp_skip_debug = False

        from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalSensor
        from habitat.tasks.nav.nav import (
            EpisodicCompassSensor,
            EpisodicGPSSensor,
            HeadingSensor,
            ImageGoalSensor,
            IntegratedPointGoalGPSAndCompassSensor,
            PointGoalSensor,
            ProximitySensor,
        )
        from habitat.tasks.nav.object_nav_task import ObjectGoalSensor

        if fuse_keys is None:
            fuse_keys = observation_space.spaces.keys()
            # removing keys that correspond to goal sensors
            goal_sensor_keys = {
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid,
                ObjectGoalSensor.cls_uuid,
                EpisodicGPSSensor.cls_uuid,
                PointGoalSensor.cls_uuid,
                HeadingSensor.cls_uuid,
                ProximitySensor.cls_uuid,
                EpisodicCompassSensor.cls_uuid,
                ImageGoalSensor.cls_uuid,
                InstanceImageGoalSensor.cls_uuid,
            }
            fuse_keys = [k for k in fuse_keys if k not in goal_sensor_keys]

        from gym import spaces

        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        else:
            use_obs_space = spaces.Dict({
                k: observation_space.spaces[k]
                for k in fuse_keys
                if len(observation_space.spaces[k].shape) == 3
            })
        resnet_baseplanes = 32
        normalize_visual_inputs = False
        backbone = "resnet18"
        self.visual_encoder = ResNetEncoder(
            use_obs_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )
        hidden_size = self.n_embed - 14
        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), hidden_size),
                nn.ReLU(True),
            )

        logger.info("Done loading llama")

    # @property
    # def n_layers(self):
    #     return self.hf_config.n_layer

    # @property
    # def n_head(self):
    #     return self.hf_config.n_head

    # @property
    # def n_embed(self):
    #     return self.hf_config.n_embd

    @property
    def d_model(self):
        return self.model.config.hidden_size

    @property
    def hidden_window_dim(self):
        if self._debug_mode:
            return self.d_model + 2
        else:
            return self.d_model

    def _verify_input(self, non_causal_tokens, causal_embeds, att_masks):
        if self.tmp_skip_debug:
            return
        debug_info = causal_embeds[..., -2:]
        # Check each env individually.
        for env_prefix_tokens, env_debug_info, env_att_masks in zip(
            non_causal_tokens, debug_info, att_masks
        ):
            env_att_masks = env_att_masks.view(-1)
            valid_debug_info = env_debug_info[env_att_masks].int()
            step_idx = valid_debug_info[:, 0]
            ep_idx = valid_debug_info[:, 1]

            if (ep_idx[0] != ep_idx).all():
                raise ValueError(f"The episode indices don't match {ep_idx}")

            n_attend = env_att_masks.sum()
            if step_idx.max() + 1 != n_attend:
                raise ValueError(
                    f"Attention mask is wrong {env_att_masks}, compare to steps"
                    f" {step_idx}"
                )
            step_diff = step_idx[1:] - step_idx[:-1]
            if not (len(step_diff) == 0 or (step_diff == 1).all()):
                raise ValueError(f"Steps are inconsistent {step_diff}")

    def decode(self, non_causal_embeds, causal_embeds, att_masks):
        if self._debug_mode:
            self._verify_input(non_causal_embeds, causal_embeds, att_masks)
            # Remove the debug info.
            causal_embeds = causal_embeds[..., :-2]

        causal_embeds = causal_embeds.to(self._create_type)
        all_embeds = torch.cat([non_causal_embeds, causal_embeds], dim=1)

        non_causal_mask = self._get_non_causal_mask(non_causal_embeds)
        non_causal_mask = non_causal_mask.expand(non_causal_embeds.shape[0], -1)
        att_masks = att_masks.view(*att_masks.shape[:2])
        att_masks = torch.cat([non_causal_mask, att_masks], dim=-1)
        assert len(att_masks.shape) == 2

        seq_out = self.model(inputs_embeds=all_embeds, attention_mask=att_masks.int())
        context_len = causal_embeds.shape[1]
        return seq_out.last_hidden_state[:, -context_len:].to(torch.float32)

    def postprocess_past_key_value(self, past_key_values):
        # past_key_values.shape -> [nL, 2(k and v), bs(nE), nH, nS, nE/nH]
        past_key_values = torch.stack([torch.stack(x) for x in past_key_values])
        return past_key_values.permute(2, 0, 1, 3, 4, 5)[..., -1, :].flatten(2, 4)

    def stack_past_key_values(self, past_key_values):
        past_key_values = torch.stack([torch.stack(x) for x in past_key_values])
        return past_key_values

    def preprocess_past_key_value(self, past_key_values):
        # past_key_values.shape -> [nS, bs, nL, 2*nH*nE/nH]
        nS, bs, nL, _ = past_key_values.shape
        nH = self.n_head
        nE = self.n_embed
        return past_key_values.reshape(nS, bs, nL, 2, nH, nE // nH).permute(
            2, 3, 1, 4, 0, 5
        )

    def forward(self, observations, rnn_hidden_states, masks, rnn_build_seq_info):
        if len(rnn_hidden_states.shape) == 4:
            IS_UPDATE = False
        else:
            IS_UPDATE = True

        if rnn_hidden_states.shape[0] > 0 and not IS_UPDATE:
            past_key_values = self.preprocess_past_key_value(rnn_hidden_states)
        else:
            past_key_values = None

        feats_keys = [
            "is_holding",
            "joint",
            "obj_start_sensor",
            "relative_resting_position",
        ]
        feats = torch.cat([observations[k] for k in feats_keys], dim=-1)
        if len(feats.shape) == 2 and not IS_UPDATE:
            feats = feats[:, None]
            flat_obs = observations
            reshape = False
        elif len(feats.shape) == 2 and IS_UPDATE:
            feats = feats[None]
            flat_obs = observations
            reshape = False
        else:
            n_env, n_steps, *_ = observations["head_depth"].shape
            flat_obs = {
                k: observations[k].flatten(0, 1)
                for k in self.visual_encoder.visual_keys
            }
            reshape = True

        visual_feats = self.visual_encoder(flat_obs)

        visual_feats = self.visual_fc(visual_feats)

        if reshape:
            visual_feats = visual_feats.reshape(n_env, n_steps, *visual_feats.shape[1:])
        elif not IS_UPDATE:
            visual_feats = visual_feats[:, None]
        elif IS_UPDATE:
            visual_feats = visual_feats[None]
        else:
            assert False, "Don't know what to do!"

        feats = torch.cat([visual_feats, feats], dim=-1)
        if len(masks.shape) != 3:
            masks = masks.T.float()
        else:
            masks = masks[..., 0].T.float()

        if self.reset_position_index:
            position_ids = torch.tile(
                torch.arange(masks.shape[1]), (masks.shape[0], 1)
            ).to(masks.device)
            position_ids = (
                position_ids - torch.cummax(position_ids * (1 - masks), dim=-1)[0]
            )
            position_ids = position_ids[:, -feats.shape[1] :]
        else:
            position_ids = None

        if self.inter_episodes_attention:
            masks = None

        output = self.model(
            inputs_embeds=feats,
            past_key_values=past_key_values,
            attention_mask=masks,
            position_ids=position_ids,
        )

        feats = output.last_hidden_state
        feats = feats.squeeze(1)

        if len(feats.shape) == 3 and not IS_UPDATE:
            feats = feats.squeeze(1)
        elif len(feats.shape) == 3 and IS_UPDATE:
            feats = feats.squeeze(0)

        return feats, self.postprocess_past_key_value(output.past_key_values), output

    def get_trainable_params(self):
        return chain(
            [p for name, p in self.model.named_parameters() if "wte" not in name],
            self.visual_encoder.parameters(),
            self.visual_fc.parameters(),
        )

    def _get_non_causal_mask(self, non_causal_inputs):
        if self._non_causal_mask is None:
            # Cache this mask so we don't have to reallocate.
            self._non_causal_mask = torch.ones(
                # Only take the token length, we expand to fit the batch sizes
                # later.
                (1, non_causal_inputs.shape[-1]),
                device=non_causal_inputs.device,
                dtype=torch.bool,
            )
        return self._non_causal_mask


class MinimalTransformerWrapper(nn.Module):
    def __init__(self, input_size: int, config):
        super().__init__()
        self.model_name = config.model_name
        self.inter_episodes_attention = config.inter_episodes_attention
        self.reset_position_index = config.reset_position_index
        self.add_sequence_idx_embed = config.add_sequence_idx_embed
        self.n_layers = config.n_layers
        self.n_embed = config.n_hidden
        self.n_mlp_hidden = config.n_mlp_hidden
        self.n_head = config.n_heads
        self.activation = config.activation
        self.position_embed_type = config.position_embed_type
        self.depth_dropout_p = config.depth_dropout_p
        self.gated_residual = config.gated_residual

        self.context_len = config.context_len
        self.banded_attention = config.banded_attention
        self.orphan_steps_attention = config.orphan_steps_attention
        self.add_context_loss = config.add_context_loss
        self.max_position_embeddings = config.max_position_embeddings
        self.feats_proj = nn.Linear(input_size, self.n_embed)
        self.feats_out = nn.Linear(self.n_embed, self.n_embed)

        if self.model_name == "gpt":
            self.hf_config = transformers.GPT2Config(
                vocab_size=0,
                n_embd=self.n_embed,
                n_layer=self.n_layers,
                n_head=self.n_head,
            )

            self.model = transformers.GPT2Model(self.hf_config)
            self.model.wte.weight.requires_grad_(False)
        elif self.model_name == "llamarl":
            self.hf_config = LlamaRLConfig(
                hidden_size=self.n_embed,
                intermediate_size=self.n_mlp_hidden,
                num_hidden_layers=self.n_layers,
                num_attention_heads=self.n_head,
                hidden_act=self.activation,
                inter_episodes_attention=self.inter_episodes_attention,
                reset_position_index=self.reset_position_index,
                add_sequence_idx_embed=self.add_sequence_idx_embed,
                position_embed_type=self.position_embed_type,
                gated_residual=self.gated_residual,
                context_len=self.context_len,
                banded_attention=self.banded_attention,
                orphan_steps_attention=self.orphan_steps_attention,
                depth_dropout_p=self.depth_dropout_p,
                max_position_embeddings=self.max_position_embeddings,
            )

            self.model = LlamaRLModel(self.hf_config)

        else:
            raise ValueError(f"Unrecognized {self.model_name}")

        logger.info("Done loading llama")

    def postprocess_past_key_value(self, past_key_values, full_rnn_state=False):
        # past_key_values.shape -> [nL, 2(k and v), bs(nE), nH, nS, nE/nH]
        past_key_values = torch.stack([torch.stack(x) for x in past_key_values])
        if not full_rnn_state:
            return past_key_values.permute(2, 0, 1, 3, 4, 5)[..., -1, :].flatten(2, 4)
        else:
            return past_key_values.permute(4, 2, 0, 1, 3, 5).flatten(3, 5)

    def stack_past_key_values(self, past_key_values, last_step=False):
        if last_step:
            past_key_values = torch.stack(
                [torch.stack([y[..., -1, :] for y in x]) for x in past_key_values]
            )
        else:
            past_key_values = torch.stack([torch.stack(x) for x in past_key_values])
        return past_key_values

    def preprocess_past_key_value(self, past_key_values):
        # past_key_values.shape -> [nS, bs, nL, 2*nH*nE/nH]
        bs, nS, nL, _ = past_key_values.shape
        nH = self.n_head
        nE = self.n_embed
        return past_key_values.reshape(bs, nS, nL, 2, nH, nE // nH).permute(
            2, 3, 0, 4, 1, 5
        )

    def forward(
        self,
        feats,
        rnn_hidden_states,
        masks,
        rnn_build_seq_info,
        full_rnn_state=False,
        **kwargs,
    ):
        if rnn_build_seq_info is None:
            past_key_values = (
                rnn_hidden_states if np.prod(rnn_hidden_states.shape) > 0 else None
            )
            n_envs = rnn_hidden_states.shape[2]
            seq_len = 1
            masks = masks.squeeze(-1).float()
            stop_grad_steps = 0
        else:
            n_envs, seq_len = rnn_build_seq_info["dims"]
            past_key_values = None
            masks = masks.squeeze(-1).unflatten(0, (n_envs, seq_len)).float()
            if "stop_grad_steps" in rnn_build_seq_info:
                stop_grad_steps = rnn_build_seq_info["stop_grad_steps"]
            else:
                stop_grad_steps = 0

        feats = feats.unflatten(0, (n_envs, seq_len))

        if rnn_build_seq_info is not None:
            old_context_length = rnn_build_seq_info["old_context_length"]
        else:
            old_context_length = 0

        feats = torch.concat(
            [
                feats[:, :old_context_length].detach(),
                feats[:, old_context_length:],
            ],
            dim=1,
        )

        if (
            rnn_build_seq_info is not None
            and not rnn_build_seq_info["is_first"]
            and not self.add_context_loss
        ):
            feats = torch.concat(
                [
                    feats[:, : rnn_build_seq_info["old_context_length"]].detach(),
                    feats[:, rnn_build_seq_info["old_context_length"] :],
                ],
                dim=1,
            )

        if stop_grad_steps:
            feats_ = feats[:, :stop_grad_steps].detach()
            masks_ = masks[:, :stop_grad_steps].detach()
            feats = feats[:, stop_grad_steps:]

            # TODO check why torch.no_grad doesn't work.
            feats_ = self.feats_proj(feats_)
            output_ = self.model(
                inputs_embeds=feats_,
                past_key_values=None,
                attention_mask=masks_,
            )
            feats_ = output_.last_hidden_state
            feats_ = self.feats_out(feats_)

            past_key_values = output_.past_key_values

            feats_ = feats_.detach()
            past_key_values = self.stack_past_key_values(past_key_values).detach()

        feats = self.feats_proj(feats)
        output = self.model(
            inputs_embeds=feats,
            past_key_values=past_key_values,
            attention_mask=masks,
            **kwargs,
        )

        feats = output.last_hidden_state
        feats = self.feats_out(feats)

        if (
            rnn_build_seq_info is not None
            and not rnn_build_seq_info["is_first"]
            and not self.add_context_loss
        ):
            feats = feats[:, rnn_build_seq_info["old_context_length"] :]

        if stop_grad_steps:
            feats = torch.concat([feats_, feats], dim=1)
        feats = feats.flatten(0, 1)
        if kwargs:
            return (
                feats,
                self.stack_past_key_values(
                    output.past_key_values, last_step=not full_rnn_state
                ),
                output,
            )
        else:
            return feats, self.stack_past_key_values(
                output.past_key_values, last_step=not full_rnn_state
            )

    def get_trainable_params(self):
        return chain(
            self.feats_proj.parameters(),
            self.model.named_parameters(),
            self.feats_out.parameters(),
        )

    @property
    def num_recurrent_layers(self):
        return self.n_layers

    @property
    def recurrent_hidden_size(self):
        return self.n_embed
