from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.config.default_structured_configs import ObsTransformConfig
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.utils.common import inference_mode
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from ovon.models.clip_policy import ResNetCLIPEncoder


@baseline_registry.register_obs_transformer()
class CLIPEncoder(ObservationTransformer):
    """Renames the entry corresponding to the given key string within the observations
    dict to 'teacher_actions'"""

    def __init__(self, backbone: str, clip_model: str = "RN50"):
        super().__init__()
        dummy_obs_space = spaces.Dict(
            {
                "rgb": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(1,),
                    dtype=np.float32,
                )
            }
        )
        self._encoder = ResNetCLIPEncoder(
            dummy_obs_space,
            pooling="avgpool" if "avgpool" in backbone else "attnpool",
            clip_model=clip_model,
        )

    def transform_observation_space(self, observation_space: spaces.Dict, **kwargs):
        assert "rgb" in observation_space.spaces, f"CLIP needs rgb observation!"
        new_obs_space = spaces.Dict(
            {
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY: spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=self._encoder.output_shape,
                    dtype=np.float32,
                ),
                **observation_space.spaces,
            }
        )
        new_obs_space.spaces.pop("rgb")
        return new_obs_space

    @classmethod
    def from_config(cls, config: DictConfig):
        return cls(backbone=config.backbone)

    def forward(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with inference_mode():
            observations[
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
            ] = self._encoder(observations)
        observations.pop("rgb")
        return observations


@dataclass
class CLIPEncoderConfig(ObsTransformConfig):
    type: str = CLIPEncoder.__name__
    backbone: str = "resnet50_clip_avgpool"


cs = ConfigStore.instance()

cs.store(
    package="habitat_baselines.rl.policy.obs_transforms.clip_encoder",
    group="habitat_baselines/rl/policy/obs_transforms",
    name="clip_encoder",
    node=CLIPEncoderConfig,
)
