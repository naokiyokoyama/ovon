from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import clip
import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from habitat.tasks.nav.nav import EpisodicCompassSensor, EpisodicGPSSensor
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.ddppo.policy.resnet import resnet18
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.models.rnn_state_encoder import \
    build_rnn_state_encoder
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.utils.common import get_num_actions
from torch import nn as nn
from torchvision import transforms as T

from ovon.task.sensors import (ClipGoalSelectorSensor, ClipImageGoalSensor,
                               ClipObjectGoalSensor)


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
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        add_clip_linear_projection: bool = False,
        depth_ckpt: str = "",
        late_fusion: bool = False,
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
                discrete_actions=discrete_actions,
                add_clip_linear_projection=add_clip_linear_projection,
                depth_ckpt=depth_ckpt,
                late_fusion=late_fusion,
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
        try:
            """TODO: Eventually eliminate this check. Safeguard for loading
            configs from checkpoints created before commit where depth_ckpt was
            added to OVONPolicyConfig
            """
            depth_ckpt = config.habitat_baselines.rl.policy.depth_ckpt
        except:
            depth_ckpt = ""
        try:
            late_fusion = config.habitat_baselines.rl.policy.late_fusion
        except:
            late_fusion = False
        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            rnn_type=config.habitat_baselines.rl.ddppo.rnn_type,
            num_recurrent_layers=config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            backbone=config.habitat_baselines.rl.ddppo.backbone,
            policy_config=config.habitat_baselines.rl.policy,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            add_clip_linear_projection=config.habitat_baselines.rl.policy.add_clip_linear_projection,
            depth_ckpt=depth_ckpt,
            late_fusion=late_fusion,
        )

    def freeze_visual_encoders(self):
        for param in self.net.visual_encoder.parameters():
            param.requires_grad_(False)
        for param in self.net.visual_fc.parameters():
            param.requires_grad_(False)

    def unfreeze_visual_encoders(self):
        for param in self.net.visual_encoder.parameters():
            param.requires_grad_(True)
        for param in self.net.visual_fc.parameters():
            param.requires_grad_(True)

    def freeze_state_encoder(self):
        for param in self.net.state_encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_state_encoder(self):
        for param in self.net.state_encoder.parameters():
            param.requires_grad_(True)

    def freeze_actor(self):
        for param in self.action_distribution.parameters():
            param.requires_grad_(False)

    def unfreeze_actor(self):
        for param in self.action_distribution.parameters():
            param.requires_grad_(True)


class PointNavResNetCLIPNet(Net):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        discrete_actions: bool = True,
        clip_model: str = "RN50",
        depth_ckpt: str = "",
        add_clip_linear_projection: bool = False,
        late_fusion: bool = False,
    ):
        super().__init__()
        self.prev_action_embedding: nn.Module
        self.discrete_actions = discrete_actions
        self.add_clip_linear_projection = add_clip_linear_projection
        self.late_fusion = late_fusion
        self._n_prev_action = 32
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(
                action_space.n + 1, self._n_prev_action
            )
        else:
            num_actions = get_num_actions(action_space)
            self.prev_action_embedding = nn.Linear(num_actions, self._n_prev_action)
        rnn_input_size = self._n_prev_action  # test
        rnn_input_size_info = {"prev_action": self._n_prev_action}

        self.visual_encoder = ResNetCLIPEncoder(
            observation_space,
            backbone_type=backbone,
            clip_model=clip_model,
            depth_ckpt=depth_ckpt,
        )
        visual_fc_input = self.visual_encoder.output_shape[0]
        if ClipImageGoalSensor.cls_uuid in observation_space.spaces:
            # Goal image is a goal embedding, gets processed as a separate
            # input rather than along with the visual features
            visual_fc_input -= 1024
        self.visual_fc = nn.Sequential(
            nn.Linear(visual_fc_input, hidden_size),
            nn.ReLU(True),
        )
        print("Observation space info:")
        for k, v in observation_space.spaces.items():
            print(f"  {k}: {v}")

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]) + 1
            )
            self.obj_categories_embedding = nn.Embedding(self._n_object_categories, 32)
            rnn_input_size += 32
            rnn_input_size_info["object_goal"] = 32

        if (
            ClipObjectGoalSensor.cls_uuid in observation_space.spaces
            or ClipImageGoalSensor.cls_uuid in observation_space.spaces
        ):
            clip_embedding = 1024 if clip_model == "RN50" else 768
            print(
                f"CLIP embedding: {clip_embedding}, "
                f"Add CLIP linear: {add_clip_linear_projection}"
            )
            if self.add_clip_linear_projection:
                self.obj_categories_embedding = nn.Linear(clip_embedding, 256)
                object_goal_size = 256
            else:
                object_goal_size = clip_embedding

            if not late_fusion:
                rnn_input_size += object_goal_size
                rnn_input_size_info["clip_goal"] = object_goal_size

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[EpisodicGPSSensor.cls_uuid].shape[
                0
            ]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
            rnn_input_size_info["gps_embedding"] = 32

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[0] == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32
            rnn_input_size_info["compass_embedding"] = 32

        if not self.is_blind:
            rnn_input_size += hidden_size
            rnn_input_size_info["visual_feats"] = hidden_size

        self._hidden_size = hidden_size

        print("RNN input size info: ")
        total = 0
        for k, v in rnn_input_size_info.items():
            print(f"  {k}: {v}")
            total += v
        if total - rnn_input_size != 0:
            print(f"  UNACCOUNTED: {total - rnn_input_size}")
        total_str = f"  Total RNN input size: {total}"
        print("  " + "-" * (len(total_str) - 2))
        print(total_str)

        self.state_encoder = build_rnn_state_encoder(
            rnn_input_size,
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
        clip_image_goal = None
        object_goal = None
        if not self.is_blind:
            # We CANNOT use observations.get() here because
            # self.visual_encoder(observations) is an expensive operation. Therefore,
            # we need `# noqa: SIM401`
            if (  # noqa: SIM401
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY in observations
            ):
                visual_feats = observations[
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                ]
            else:
                visual_feats = self.visual_encoder(observations)

            if ClipImageGoalSensor.cls_uuid in observations:
                clip_image_goal = visual_feats[:, :1024]
                visual_feats = self.visual_fc(visual_feats[:, 1024:])
            else:
                visual_feats = self.visual_fc(visual_feats)

            aux_loss_state["perception_embed"] = visual_feats
            x.append(visual_feats)

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if ClipObjectGoalSensor.cls_uuid in observations and not self.late_fusion:
            object_goal = observations[ClipObjectGoalSensor.cls_uuid]
            if self.add_clip_linear_projection:
                object_goal = self.obj_categories_embedding(object_goal)
            x.append(object_goal)

        if ClipImageGoalSensor.cls_uuid in observations and not self.late_fusion:
            assert clip_image_goal is not None
            if self.add_clip_linear_projection:
                clip_image_goal = self.obj_categories_embedding(clip_image_goal)
            x.append(clip_image_goal)

        if ClipGoalSelectorSensor.cls_uuid in observations:
            assert (
                ClipObjectGoalSensor.cls_uuid in observations
                and ClipImageGoalSensor.cls_uuid in observations
            ), "Must have both object and image goals to use goal selector."
            image_goal_embedding = x.pop()
            object_goal = x.pop()
            assert image_goal_embedding.shape == object_goal.shape
            clip_goal = torch.where(
                observations[ClipGoalSelectorSensor.cls_uuid].bool(),
                image_goal_embedding,
                object_goal,
            )
            x.append(clip_goal)

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(self.compass_embedding(compass_observations.squeeze(dim=1)))

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid]))

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

        if self.late_fusion:
            object_goal = observations[ClipObjectGoalSensor.cls_uuid]
            out = (out + visual_feats) * object_goal

        return out, rnn_hidden_states, aux_loss_state


class ResNetCLIPEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        backbone_type="attnpool",
        clip_model="RN50",
        depth_ckpt: str = "",
    ):
        super().__init__()

        self.backbone_type = backbone_type
        self.rgb = "rgb" in observation_space.spaces
        self.depth = "depth" in observation_space.spaces

        if not self.is_blind:
            model, preprocess = clip.load(clip_model)

            # expected input: H x W x C (np.uint8 in [0-255])
            if (
                observation_space.spaces["rgb"].shape[0] != 224
                or observation_space.spaces["rgb"].shape[1] != 224
            ):
                print("Using CLIP preprocess for resizing+cropping to 224x224")
                preprocess_transforms = [
                    # resize and center crop to 224
                    preprocess.transforms[0],
                    preprocess.transforms[1],
                ]
            else:
                preprocess_transforms = []
            preprocess_transforms.extend(
                [
                    # already tensor, but want float
                    T.ConvertImageDtype(torch.float),
                    # normalize with CLIP mean, std
                    preprocess.transforms[4],
                ]
            )
            self.preprocess = T.Compose(preprocess_transforms)
            # expected output: H x W x C (np.float32)

            self.backbone = model.visual

            assert self.rgb

            if self.depth:
                assert depth_ckpt != ""
                self.depth_backbone = copy_depth_encoder(depth_ckpt)
                depth_size = 512
            else:
                self.depth_backbone = None
                depth_size = 0
            if "none" in backbone_type:
                self.backbone.attnpool = nn.Identity()
                self.output_shape = (2048, 7, 7)
            elif self.using_both_clip_avg_attn_pool:
                # Adds an avg pooling head in parallel to final attention layer
                self.backbone.adaptive_avgpool = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten()
                )
                self.output_shape = (1024 + 2048,)  # attnpool + avgpool

                # Overwrite forward method to return both attnpool and avgpool
                # concatenated together (attnpool + avgpool).
                bound_method = forward_avg_attn_pool.__get__(
                    self.backbone, self.backbone.__class__
                )
                setattr(self.backbone, "forward", bound_method)
            elif self.using_only_clip_avgpool:
                self.backbone.attnpool = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten()
                )
                self.output_shape = (2048 + depth_size,)
            elif self.using_only_clip_attnpool:
                self.output_shape = (1024 + depth_size,)
                if ClipImageGoalSensor.cls_uuid in observation_space.spaces:
                    self.output_shape = (self.output_shape[0] + 1024,)

            for param in self.backbone.parameters():
                param.requires_grad = False
            for module in self.backbone.modules():
                if "BatchNorm" in type(module).__name__:
                    module.momentum = 0.0
            self.backbone.eval()

            if self.depth:
                for param in self.depth_backbone.parameters():
                    param.requires_grad = False
                self.depth_backbone.eval()

    @property
    def is_blind(self):
        return self.rgb is False and self.depth is False

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        cnn_input = []
        if ClipImageGoalSensor.cls_uuid in observations:
            # Stack them into the same batch
            rgb_observations = torch.cat(
                [
                    observations["rgb"],
                    observations[ClipImageGoalSensor.cls_uuid],
                ],
                dim=0,
            )
        else:
            rgb_observations = observations["rgb"]

        rgb_observations = rgb_observations.permute(
            0, 3, 1, 2
        )  # BATCH x CHANNEL x HEIGHT X WIDTH
        rgb_observations = torch.stack(
            [self.preprocess(rgb_image) for rgb_image in rgb_observations]
        )  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32
        rgb_x = self.backbone(rgb_observations)

        if ClipImageGoalSensor.cls_uuid in observations:
            # Split them back out
            if self.using_both_clip_avg_attn_pool:
                rgb_x, goal_x = rgb_x
            else:
                rgb_x, goal_x = rgb_x[:, :-1024], rgb_x[:, -1024:]
            cnn_input.append(goal_x.type(torch.float32))

        cnn_input.append(rgb_x.type(torch.float32))

        if self.depth and "depth" in observations:
            depth_feats = self.depth_backbone({"depth": observations["depth"]})
            cnn_input.append(depth_feats)

        x = torch.cat(cnn_input, dim=1)
        print(
            "CNN ouptut: {}, input: {}, keys: {}".format(
                [i.shape for i in cnn_input],
                [i.shape for i in rgb_observations],
                observations.keys(),
            )
        )

        return x

    @property
    def using_only_clip_attnpool(self):
        return "attnpool" in self.backbone_type

    @property
    def using_only_clip_avgpool(self):
        return "avgpool" in self.backbone_type

    @property
    def using_both_clip_avg_attn_pool(self):
        return "avgattnpool" in self.backbone_type


class ResNet18DepthEncoder(nn.Module):
    def __init__(self, depth_encoder, visual_fc):
        super().__init__()
        self.encoder = depth_encoder
        self.visual_fc = visual_fc

    def forward(self, x):
        x = self.encoder(x)
        x = self.visual_fc(x)
        return x

    def load_state_dict(self, state_dict, strict: bool = True):
        # TODO: allow dicts trained with both attn and avg pool to be loaded
        ignore_attnpool = False
        if ignore_attnpool:
            pass
        return super().load_state_dict(state_dict, strict=strict)


def copy_depth_encoder(depth_ckpt):
    """
    Returns an encoder that stacks the encoder and visual_fc of the provided
    depth checkpoint
    :param depth_ckpt: path to a resnet18 depth pointnav policy
    :return: nn.Module representing the backbone of the depth policy
    """
    # Initialize encoder and fc layers
    base_planes = 32
    ngroups = 32
    spatial_size = 128

    observation_space = SpaceDict(
        {
            "depth": spaces.Box(
                low=0.0, high=1.0, shape=(256, 256, 1), dtype=np.float32
            ),
        }
    )
    depth_encoder = ResNetEncoder(
        observation_space,
        base_planes,
        ngroups,
        spatial_size,
        make_backbone=resnet18,
    )

    flat_output_shape = 2048
    hidden_size = 512
    visual_fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(flat_output_shape, hidden_size),
        nn.ReLU(True),
    )

    pretrained_state = torch.load(depth_ckpt, map_location="cpu")

    # Load weights into depth encoder
    depth_encoder_state_dict = {
        k[len("actor_critic.net.visual_encoder.") :]: v
        for k, v in pretrained_state["state_dict"].items()
        if k.startswith("actor_critic.net.visual_encoder.")
    }
    depth_encoder.load_state_dict(depth_encoder_state_dict)

    # Load weights in fc layers
    visual_fc_state_dict = {
        k[len("actor_critic.net.visual_fc.") :]: v
        for k, v in pretrained_state["state_dict"].items()
        if k.startswith("actor_critic.net.visual_fc.")
    }
    visual_fc.load_state_dict(visual_fc_state_dict)

    modified_depth_encoder = ResNet18DepthEncoder(depth_encoder, visual_fc)

    return modified_depth_encoder


def forward_avg_attn_pool(self, x):
    """
    Adapted from https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L138
    Expects a batch of images where the batch number is even. The whole batch
    is passed through all layers except the last layer; the first half of the
    batch will be passed through avgpool and the second half will be passed
    through attnpool. The outputs of both pools are concatenated returned.
    """

    assert hasattr(self, "adaptive_avgpool")
    assert x.shape[0] % 2 == 0, "Batch size must be even"

    def stem(x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    x = x.type(self.conv1.weight.dtype)
    x = stem(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x_avgpool, x_attnpool = x.chunk(2, dim=0)
    x_avgpool = self.adaptive_avgpool(x_avgpool)
    x_attnpool = self.attnpool(x_attnpool)

    return x_avgpool, x_attnpool


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    depth_encoder = copy_depth_encoder(args.ckpt)
    print(depth_encoder)
    output = depth_encoder({"depth": torch.rand(1, 256, 256, 1)})
    print("success. output shape: ", output.shape)
