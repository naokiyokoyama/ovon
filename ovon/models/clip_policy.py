from collections import OrderedDict
from typing import Dict, List, Optional

import clip
import torch
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.ppo import NetPolicy
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF


@baseline_registry.register_policy
class PointNavResNetClipPolicy(NetPolicy):
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
        **kwargs,
    ):
        if policy_config is not None:
            discrete_actions = policy_config.action_distribution_type == "categorical"
            self.action_distribution_type = policy_config.action_distribution_type
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
                ((k, v) for k, v in observation_space.items() if k not in ignore_names)
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
        )


class PointNavResNetCLIPNet(PointNavResNetNet):
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
    ):
        super().__init__(
            observation_space,
            action_space,
            hidden_size,
            num_recurrent_layers,
            rnn_type,
            "resnet18",  # dummy value; instantiates a temporary encoder
            32,  # dummy value; instantiates a temporary encoder
            fuse_keys,
            force_blind_policy,
            discrete_actions,
        )

        # Re-define (overwrite) the visual_encoder attribute from ResNet to CLIP
        self.visual_encoder = ResNetCLIPEncoder(
            observation_space,
            pooling="avgpool" if "avgpool" in backbone else "attnpool",
        )
        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                nn.Linear(self.visual_encoder.output_shape[0], hidden_size),
                nn.ReLU(True),
            )


class ResNetCLIPEncoder(nn.Module):
    def __init__(self, observation_space: spaces.Dict, pooling="attnpool"):
        super().__init__()

        self.rgb = "rgb" in observation_space.spaces
        self.depth = "depth" in observation_space.spaces

        if not self.is_blind:
            model, preprocess = clip.load("RN50")

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
            elif pooling == "none":
                self.backbone.attnpool = nn.Identity()
                self.output_shape = (2048, 7, 7)
            elif pooling == "avgpool":
                self.backbone.attnpool = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten()
                )
                self.output_shape = (2048,)
            else:
                self.output_shape = (1024,)

            for param in self.backbone.parameters():
                param.requires_grad = False
            for module in self.backbone.modules():
                if "BatchNorm" in type(module).__name__:
                    module.momentum = 0.0
            self.backbone.eval()

    @property
    def is_blind(self):
        return self.rgb is False and self.depth is False

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
            rgb_x = self.backbone(rgb_observations).float()
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
                    self.preprocess(TF.convert_image_dtype(depth_map, torch.uint8))
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
