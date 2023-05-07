from dataclasses import dataclass

from habitat.config.default_structured_configs import (EnvironmentConfig,
                                                       HabitatConfig,
                                                       LabSensorConfig,
                                                       MeasurementConfig,
                                                       SimulatorConfig)
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesRLConfig, PolicyConfig, RLConfig)
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
class ClipImageGoalSensorConfig(LabSensorConfig):
    type: str = "ClipImageGoalSensor"


@dataclass
class ClipGoalSelectorSensorConfig(LabSensorConfig):
    type: str = "ClipGoalSelectorSensor"
    image_sampling_probability: float = 0.8


@dataclass
class ImageGoalRotationSensorConfig(LabSensorConfig):
    type: str = "ImageGoalRotationSensor"
    sample_angle: bool = True


@dataclass
class AngleSuccessMeasurementConfig(MeasurementConfig):
    type: str = "AngleSuccess"
    success_angle: float = 25.0


@dataclass
class AngleToGoalMeasurementConfig(MeasurementConfig):
    type: str = "AngleToGoal"


@dataclass
class ImageNavRewardMeasurementConfig(MeasurementConfig):
    type: str = "ImageNavReward"
    success_reward: float = 2.5
    angle_success_reward: float = 2.5
    slack_penalty: float = -0.01
    use_atg_reward: bool = True
    use_dtg_reward: bool = True
    use_atg_fix: bool = True
    atg_reward_distance: float = 1.0


@dataclass
class OVONObjectGoalIDMeasurementConfig(MeasurementConfig):
    type: str = "OVONObjectGoalID"
    cache: str = "data/clip_embeddings/ovon_stretch_final_cache.pkl"


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
    depth_ckpt: str = ""
    late_fusion: bool = False


@dataclass
class OVONRLConfig(RLConfig):
    policy: OVONPolicyConfig = OVONPolicyConfig()


@dataclass
class OVONBaselinesRLConfig(HabitatBaselinesRLConfig):
    rl: OVONRLConfig = OVONRLConfig()


@dataclass
class NavmeshSettings:
    agent_max_climb: float = 0.20
    cell_height: float = 0.20


@dataclass
class OVONSimulatorConfig(SimulatorConfig):
    type: str = "OVONSim-v0"
    navmesh_settings: NavmeshSettings = NavmeshSettings()


@dataclass
class OVONHabitatConfig(HabitatConfig):
    simulator: SimulatorConfig = OVONSimulatorConfig()


# -----------------------------------------------------------------------------
# Register configs in the Hydra ConfigStore
# -----------------------------------------------------------------------------

cs.store(
    group="habitat",
    name="habitat_config_base",
    node=OVONHabitatConfig,
)

cs.store(
    package=f"habitat.task.lab_sensors.clip_objectgoal_sensor",
    group="habitat/task/lab_sensors",
    name="clip_objectgoal_sensor",
    node=ClipObjectGoalSensorConfig,
)

cs.store(
    package=f"habitat.task.lab_sensors.clip_imagegoal_sensor",
    group="habitat/task/lab_sensors",
    name="clip_imagegoal_sensor",
    node=ClipImageGoalSensorConfig,
)

cs.store(
    package=f"habitat.task.lab_sensors.clip_goal_selector_sensor",
    group="habitat/task/lab_sensors",
    name="clip_goal_selector_sensor",
    node=ClipGoalSelectorSensorConfig,
)

cs.store(
    package=f"habitat.task.lab_sensors.image_goal_rotation_sensor",
    group="habitat/task/lab_sensors",
    name="image_goal_rotation_sensor",
    node=ImageGoalRotationSensorConfig,
)

cs.store(
    package="habitat.task.measurements.angle_success",
    group="habitat/task/measurements",
    name="angle_success",
    node=AngleSuccessMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.angle_to_goal",
    group="habitat/task/measurements",
    name="angle_to_goal",
    node=AngleToGoalMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.imagenav_reward",
    group="habitat/task/measurements",
    name="imagenav_reward",
    node=ImageNavRewardMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.ovon_object_goal_id",
    group="habitat/task/measurements",
    name="ovon_object_goal_id",
    node=OVONObjectGoalIDMeasurementConfig,
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
