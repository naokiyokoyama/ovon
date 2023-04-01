import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import clip
import torch
from gym import spaces
from habitat.tasks.nav.nav import EpisodicCompassSensor, EpisodicGPSSensor
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.utils.common import get_num_actions
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from ovon.models.encoders.clip_cross_attn import (
    CLIPCrossAttentionEncoder,
    forward_attn_avg_pool,
)
from ovon.task.sensors import ClipObjectGoalSensor


@baseline_registry.register_policy
class PointNavResNetCLIPPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        backbone: str = "resnet50_clip_avgpool",
        force_blind_policy: bool = False,
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        fuse_keys: Optional[List[str]] = None,
        add_clip_linear_projection: bool = False,
        **kwargs,
    ):
        if policy_config is not None:
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"
            )
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"

        super().__init__(
            PointNavResNetCLIPNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                fuse_keys=fuse_keys,
                force_blind_policy=force_blind_policy,
                discrete_actions=discrete_actions,
                add_clip_linear_projection=add_clip_linear_projection,
            ),
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        # Exclude cameras for rendering from the observation space.
        ignore_names: List[str] = []
        for agent_config in config.habitat.simulator.agents.values():
            ignore_names.extend(
                agent_config.sim_sensors[k].uuid
                for k in config.habitat_baselines.video_render_views
                if k in agent_config.sim_sensors
            )
        filtered_obs = spaces.Dict(
            OrderedDict(
                (
                    (k, v)
                    for k, v in observation_space.items()
                    if k not in ignore_names
                )
            )
        )
        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            rnn_type=config.habitat_baselines.rl.ddppo.rnn_type,
            num_recurrent_layers=config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            backbone=config.habitat_baselines.rl.ddppo.backbone,
            force_blind_policy=config.habitat_baselines.force_blind_policy,
            policy_config=config.habitat_baselines.rl.policy,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            fuse_keys=None,
            add_clip_linear_projection=config.habitat_baselines.rl.policy.add_clip_linear_projection,
        )


class PointNavResNetCLIPNet(Net):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        fuse_keys: Optional[List[str]],
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        clip_model: str = "RN50",
        add_clip_linear_projection: bool = False,
    ):
        super().__init__()
        self.prev_action_embedding: nn.Module
        self.discrete_actions = discrete_actions
        self.add_clip_linear_projection = add_clip_linear_projection
        self._n_prev_action = 32
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(
                action_space.n + 1, self._n_prev_action
            )
        else:
            num_actions = get_num_actions(action_space)
            self.prev_action_embedding = nn.Linear(
                num_actions, self._n_prev_action
            )
        rnn_input_size = self._n_prev_action  # test
        rnn_input_size_info = {"prev_action": self._n_prev_action}

        self.visual_encoder = ResNetCLIPEncoder(
            observation_space,
            backbone_type=backbone,
            clip_model=clip_model,
        )
        if not self.visual_encoder.is_blind:
            visual_fc_in_size = self.visual_encoder.output_shape[0]
            # If both attention and avgpool are used, only pass the avgpool through the
            # visual_fc linear layer; attnpool output will be compared with CLIP text or
            # passed directly to the state encoder.
            if "attn_avg_pool" in backbone:
                visual_fc_in_size -= 1024  # CLIP embedding size is 1024
            self.visual_fc = nn.Sequential(
                nn.Linear(visual_fc_in_size, hidden_size),
                nn.ReLU(True),
            )
        print("Obs space: {}".format(observation_space.spaces))

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32
            rnn_input_size_info["object_goal"] = 32

        if ClipObjectGoalSensor.cls_uuid in observation_space.spaces:
            if self.visual_encoder.using_cross_mlp:
                clip_object_goal_size = 32
            else:
                clip_embedding = 1024 if clip_model == "RN50" else 768
                print(
                    f"Clip embedding: {clip_embedding}, "
                    f"Add CLIP linear: {add_clip_linear_projection}"
                )
                if self.add_clip_linear_projection:
                    if os.environ.get("LINPROJ_DEBUG", "0") == "1":
                        self.obj_categories_embedding = nn.Sequential(
                            nn.Linear(clip_embedding, 512),
                            nn.ReLU(True),
                            nn.Linear(512, 32),
                            nn.ReLU(True),
                        )
                        clip_object_goal_size = 32
                    else:
                        self.obj_categories_embedding = nn.Linear(
                            clip_embedding, 256
                        )
                        clip_object_goal_size = 256
                else:
                    clip_object_goal_size = clip_embedding
            rnn_input_size += clip_object_goal_size
            rnn_input_size_info["clip_object_goal"] = clip_object_goal_size

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
            rnn_input_size_info["gps_embedding"] = 32

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32
            rnn_input_size_info["compass_embedding"] = 32

        # State encoder is directly fed CLIP rgb attn if both rgb attn and avgpool are
        # used AND cross attention is NOT used to inject the rgb attn into the text
        # embedding instead.
        if (
            not (
                self.visual_encoder.using_cross_attn
                or self.visual_encoder.using_cross_mlp
            )
            and self.visual_encoder.using_both_clip_attn_avg_pool
        ):
            if self.add_clip_linear_projection:
                clip_attnpool_size = 256
                self.clip_rgb_embedding = nn.Linear(1024, clip_attnpool_size)
            else:
                clip_attnpool_size = 1024
            rnn_input_size += clip_attnpool_size
            rnn_input_size_info["clip_attnpool"] = clip_attnpool_size

        self._hidden_size = hidden_size

        rnn_input_size_info["visual_hidden_size"] = (
            0 if self.is_blind else self._hidden_size
        )

        print("RNN input size info: ")
        total = 0
        for k, v in rnn_input_size_info.items():
            print(f"  {k}: {v}")
            total += v
        total_str = f"  Total RNN input size: {total}"
        print("  " + "-" * (len(total_str) - 2))
        print(total_str)

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = []
        aux_loss_state = {}
        clip_rgb = None
        if not self.is_blind:
            # We CANNOT use observations.get() here because
            # self.visual_encoder(observations) is an expensive operation. Therefore,
            # we need `# noqa: SIM401`
            if (  # noqa: SIM401
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                in observations
            ):
                raw_visual_feats = observations[
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                ]
            else:
                raw_visual_feats = self.visual_encoder(observations)

            visual_fc_in = raw_visual_feats
            if self.visual_encoder.using_both_clip_attn_avg_pool:
                # Remove CLIP rgb attn from visual_fc_in
                clip_rgb, visual_fc_in = (
                    raw_visual_feats[:, :1024],
                    raw_visual_feats[:, 1024:],
                )
            elif self.visual_encoder.using_only_clip_attnpool:
                clip_rgb = raw_visual_feats

            visual_feats = self.visual_fc(visual_fc_in)
            aux_loss_state["perception_embed"] = visual_feats
            x.append(visual_feats)

        if (
            not (
                self.visual_encoder.using_cross_attn
                or self.visual_encoder.using_cross_mlp
            )
            and self.visual_encoder.using_both_clip_attn_avg_pool
        ):
            assert clip_rgb is not None
            if self.add_clip_linear_projection:
                clip_rgb = self.clip_rgb_embedding(clip_rgb)
            x.append(clip_rgb)

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if ClipObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ClipObjectGoalSensor.cls_uuid].type(
                torch.float32
            )
            if (
                self.visual_encoder.using_cross_attn
                or self.visual_encoder.using_cross_mlp
            ):
                assert clip_rgb is not None
                if self.visual_encoder.using_cross_attn:
                    object_goal = self.visual_encoder.clip_cross_attn(
                        clip_rgb, object_goal
                    )
                else:
                    object_goal = self.visual_encoder.clip_cross_mlp(
                        torch.cat([clip_rgb, object_goal], dim=1)
                    )

            if self.add_clip_linear_projection:
                object_goal = self.obj_categories_embedding(object_goal)
            x.append(object_goal)

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(
                self.compass_embedding(compass_observations.squeeze(dim=1))
            )

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(
                self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid])
            )

        prev_actions = prev_actions.squeeze(-1)
        start_token = torch.zeros_like(prev_actions)
        # The mask means the previous action will be zero, an extra dummy action
        prev_actions = self.prev_action_embedding(
            torch.where(masks.view(-1), prev_actions + 1, start_token)
        )

        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = out

        return out, rnn_hidden_states, aux_loss_state


class ResNetCLIPEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        backbone_type: str,
        clip_model="RN50",
    ):
        super().__init__()

        self.backbone_type = backbone_type
        self.rgb = "rgb" in observation_space.spaces
        self.depth = "depth" in observation_space.spaces

        if not self.is_blind:
            model, preprocess = clip.load(clip_model)

            # expected input: C x H x W (np.uint8 in [0-255])
            self.preprocess = T.Compose(
                [
                    # resize and center crop to 224
                    preprocess.transforms[0],
                    preprocess.transforms[1],
                    # already tensor, but want float
                    T.ConvertImageDtype(torch.float),
                    # normalize with CLIP mean, std
                    preprocess.transforms[4],
                ]
            )
            # expected output: C x H x W (np.float32)

            self.backbone = model.visual

            if self.rgb and self.depth:
                self.backbone.attnpool = nn.Identity()
                self.output_shape = (2048,)
            elif self.using_only_clip_attnpool:
                # Retains the final attention layer of CLIP visual model
                self.output_shape = (1024,)
            elif "none" in backbone_type:
                # Removes the final attention layer of CLIP visual model
                self.backbone.attnpool = nn.Identity()
                self.output_shape = (2048, 7, 7)
            elif self.using_only_clip_avgpool:
                # Replaces the final attention layer of CLIP visual model w/ avg pooling
                self.backbone.attnpool = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten()
                )
                self.output_shape = (2048,)
            elif self.using_both_clip_attn_avg_pool:
                # Adds an avg pooling head in parallel to final attention layer
                self.backbone.adaptive_avgpool = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten()
                )
                self.output_shape = (1024 + 2048,)

                # Overwrite forward method to return both attnpool and avgpool
                # concatenated together (attnpool + avgpool).
                bound_method = forward_attn_avg_pool.__get__(
                    self.backbone, self.backbone.__class__
                )
                setattr(self.backbone, "forward", bound_method)
            else:
                raise NotImplementedError(
                    f"Backbone type {backbone_type} not implemented"
                )

            if self.using_cross_attn:
                # Adds cross attention layer to compare curr/goal CLIP embeddings.
                # Output size is also 1024. If used, its output is sent to the state
                # encoder INSTEAD of the CLIP text (goal) embedding.
                self.clip_cross_attn = CLIPCrossAttentionEncoder()
            elif self.using_cross_mlp:
                # Adds two-layer mlp with sizes 512 and 32 that compares the
                # curr/goal CLIP embeddings. Output size is 32.
                self.clip_cross_mlp = nn.Sequential(
                    nn.Linear(1024 * 2, 512),
                    nn.ReLU(),
                    nn.Linear(512, 32),
                    nn.ReLU(),
                )

            for param in self.backbone.parameters():
                param.requires_grad = False
            for module in self.backbone.modules():
                if "BatchNorm" in type(module).__name__:
                    module.momentum = 0.0
            self.backbone.eval()

    @property
    def is_blind(self):
        return self.rgb is False and self.depth is False

    @property
    def using_cross_attn(self):
        return "crossattn" in self.backbone_type

    @property
    def using_cross_mlp(self):
        return "crossmlp" in self.backbone_type

    @property
    def using_only_clip_attnpool(self):
        return "attnpool" in self.backbone_type

    @property
    def using_only_clip_avgpool(self):
        return "avgpool" in self.backbone_type

    @property
    def using_both_clip_attn_avg_pool(self):
        return "attn_avg_pool" in self.backbone_type

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.is_blind:
            return None

        cnn_input = []
        if self.rgb:
            rgb_observations = observations["rgb"]
            rgb_observations = rgb_observations.permute(
                0, 3, 1, 2
            )  # BATCH x CHANNEL x HEIGHT X WIDTH
            rgb_observations = torch.stack(
                [self.preprocess(rgb_image) for rgb_image in rgb_observations]
            )  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32
            rgb_x = self.backbone(rgb_observations).type(torch.float32)
            cnn_input.append(rgb_x)

        if self.depth:
            depth_observations = observations["depth"][
                ..., 0
            ]  # [BATCH x HEIGHT X WIDTH]
            ddd = torch.stack(
                [depth_observations] * 3, dim=1
            )  # [BATCH x 3 x HEIGHT X WIDTH]
            ddd = torch.stack(
                [
                    self.preprocess(
                        TF.convert_image_dtype(depth_map, torch.uint8)
                    )
                    for depth_map in ddd
                ]
            )  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32
            depth_x = self.backbone(ddd).float()
            cnn_input.append(depth_x)

        if self.rgb and self.depth:
            x = F.adaptive_avg_pool2d(cnn_input[0] + cnn_input[1], 1)
            x = x.flatten(1)
        else:
            x = torch.cat(cnn_input, dim=1)

        return x
