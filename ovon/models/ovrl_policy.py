from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from gym import spaces
from habitat import logger
from habitat.tasks.nav.nav import EpisodicCompassSensor, EpisodicGPSSensor
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.utils.common import get_num_actions

from ovon.models.encoders.visual_encoder import VisualEncoder
from ovon.models.transforms import get_transform
from ovon.task.sensors import ClipObjectGoalSensor
from ovon.utils.utils import load_encoder


class OVRLPolicyNet(Net):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

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
        use_augmentations: bool = True,
        augmentations_name: str = "resize",
        use_augmentations_test_time: bool = True,
        randomize_augmentations_over_envs: bool = False,
        rgb_image_size: int = 224,
        resnet_baseplanes: int = 64,
        avgpooled_image: bool = False,
        drop_path_rate: float = 0.0,
        pretrained_encoder: str = None,
        freeze_backbone: bool = True,
        run_type: str = "train",
    ):
        super().__init__()

        self.prev_action_embedding: nn.Module
        self.discrete_actions = discrete_actions
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
        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action  # test

        name = "resize"
        if use_augmentations and run_type == "train":
            name = augmentations_name
        if use_augmentations_test_time and run_type == "eval":
            name = augmentations_name
        self.visual_transform = get_transform(name, size=rgb_image_size)
        self.visual_transform.randomize_environments = (
            randomize_augmentations_over_envs
        )

        self.visual_encoder = VisualEncoder(
            image_size=rgb_image_size,
            backbone=backbone,
            input_channels=3,
            resnet_baseplanes=resnet_baseplanes,
            resnet_ngroups=resnet_baseplanes // 2,
            avgpooled_image=avgpooled_image,
            drop_path_rate=drop_path_rate,
        )

        self.visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.visual_encoder.output_size,
                hidden_size,
            ),
            nn.ReLU(True),
        )

        # pretrained weights
        if pretrained_encoder is not None:
            msg = load_encoder(self.visual_encoder, pretrained_encoder)
            logger.info(
                "Using weights from {}: {}".format(pretrained_encoder, msg)
            )

        # freeze backbone
        if freeze_backbone:
            for p in self.visual_encoder.backbone.parameters():
                p.requires_grad = False
        logger.info("RGB encoder is {}".format(backbone))

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

        if ClipObjectGoalSensor.cls_uuid in observation_space.spaces:
            rnn_input_size += 1024 if clip_model == "RN50" else 768

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32

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

        self._hidden_size = hidden_size

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

        N = rnn_hidden_states.size(0)
        if not self.is_blind:
            # We CANNOT use observations.get() here because self.visual_encoder(observations)
            # is an expensive operation. Therefore, we need `# noqa: SIM401`
            if (  # noqa: SIM401
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                in observations
            ):
                visual_feats = observations[
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                ]
            else:
                # visual encoder
                rgb = observations["rgb"]

                rgb = self.visual_transform(rgb, N)
                visual_feats = self.visual_encoder(rgb)

            visual_feats = self.visual_fc(visual_feats)
            aux_loss_state["perception_embed"] = visual_feats
            x.append(visual_feats)

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if ClipObjectGoalSensor.cls_uuid in observations:
            object_goal = (
                observations[ClipObjectGoalSensor.cls_uuid].float().cuda()
            )
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


@baseline_registry.register_policy
class OVRLPolicy(NetPolicy):
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
        clip_model: str = "RN50",
        use_augmentations: bool = True,
        augmentations_name: str = "resize",
        use_augmentations_test_time: bool = True,
        randomize_augmentations_over_envs: bool = False,
        rgb_image_size: int = 224,
        resnet_baseplanes: int = 64,
        avgpooled_image: bool = False,
        drop_path_rate: float = 0.0,
        pretrained_encoder: str = None,
        freeze_backbone: bool = False,
        run_type: str = "train",
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
            OVRLPolicyNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                fuse_keys=fuse_keys,
                force_blind_policy=force_blind_policy,
                discrete_actions=discrete_actions,
                clip_model=clip_model,
                use_augmentations=use_augmentations,
                augmentations_name=augmentations_name,
                use_augmentations_test_time=use_augmentations_test_time,
                randomize_augmentations_over_envs=randomize_augmentations_over_envs,
                rgb_image_size=rgb_image_size,
                resnet_baseplanes=resnet_baseplanes,
                avgpooled_image=avgpooled_image,
                drop_path_rate=drop_path_rate,
                pretrained_encoder=pretrained_encoder,
                freeze_backbone=freeze_backbone,
                run_type=run_type,
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
            force_blind_policy=config.habitat_baselines.force_blind_policy,
            policy_config=config.habitat_baselines.rl.policy,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            fuse_keys=None,
            backbone=config.habitat_baselines.rl.policy.backbone,
            clip_model=config.habitat_baselines.rl.policy.clip_model,
            use_augmentations=config.habitat_baselines.rl.policy.use_augmentations,
            augmentations_name=config.habitat_baselines.rl.policy.augmentations_name,
            use_augmentations_test_time=config.habitat_baselines.rl.policy.use_augmentations_test_time,
            randomize_augmentations_over_envs=config.habitat_baselines.rl.policy.randomize_augmentations_over_envs,
            rgb_image_size=config.habitat_baselines.rl.policy.rgb_image_size,
            resnet_baseplanes=config.habitat_baselines.rl.policy.resnet_baseplanes,
            avgpooled_image=config.habitat_baselines.rl.policy.avgpooled_image,
            drop_path_rate=config.habitat_baselines.rl.policy.drop_path_rate,
            pretrained_encoder=config.habitat_baselines.rl.policy.pretrained_encoder,
            freeze_backbone=config.habitat_baselines.rl.policy.freeze_backbone,
        )
