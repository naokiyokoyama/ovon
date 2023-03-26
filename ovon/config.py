from dataclasses import dataclass

from habitat.config.default_structured_configs import LabSensorConfig
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfig,
    HabitatBaselinesRLConfig,
    PolicyConfig,
    RLConfig,
)
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.config_store import ConfigStore
from hydra.plugins.search_path_plugin import SearchPathPlugin

cs = ConfigStore.instance()


@dataclass
class ClipObjectGoalSensorConfig(LabSensorConfig):
    type: str = "ClipObjectGoalSensor"
    prompt: str = "Find and go to {category}"
    cache: str = "data/ovon_cache.pickle"


@dataclass
class OVONPolicyConfig(PolicyConfig):
    name: str = "OVONPolicy"
    backbone: str = "resnet50"
    use_augmentations: bool = True
    augmentations_name: str = "jitter+shift"
    use_augmentations_test_time: bool = True
    randomize_augmentations_over_envs: bool = False
    rgb_image_size: int = 256
    resnet_baseplanes: int = 32
    avgpooled_image: bool = False
    drop_path_rate: float = 0.0
    freeze_backbone: bool = True
    pretrained_encoder: str = "data/visual_encoders/omnidata_DINO_02.pth"

    clip_model: str = "RN50"
    add_clip_linear_projection: bool = False


@dataclass
class OVONRLConfig(RLConfig):
    policy: OVONPolicyConfig = OVONPolicyConfig()


@dataclass
class OVONBaselinesRLConfig(HabitatBaselinesRLConfig):
    rl: OVONRLConfig = OVONRLConfig()


# -----------------------------------------------------------------------------
# Register configs in the Hydra ConfigStore
# -----------------------------------------------------------------------------


cs.store(
    package=f"habitat.task.lab_sensors.clip_objectgoal_sensor",
    group="habitat/task/lab_sensors",
    name="clip_objectgoal_sensor",
    node=ClipObjectGoalSensorConfig,
)

cs.store(
    group="habitat_baselines",
    name="habitat_baselines_rl_config_base",
    node=OVONBaselinesRLConfig,
)


class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="habitat",
            path="pkg://config/tasks/",
        )
