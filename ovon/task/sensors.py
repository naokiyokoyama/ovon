import random
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from gym import spaces
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.core.utils import try_cv2_import

cv2 = try_cv2_import()


from ovon.utils.utils import load_pickle

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_sensor
class ClipObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id, and we will generate the prompt corresponding to it
    so that it's usable by CLIP's text encoder.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalPromptSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "clip_objectgoal"

    def __init__(
        self,
        *args: Any,
        config: "DictConfig",
        **kwargs: Any,
    ):
        self.cache = load_pickle(config.cache)
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0, high=np.inf, shape=(1024,), dtype=np.int64)

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: Any,
        **kwargs: Any,
    ) -> Optional[int]:
        category = (
            episode.object_category
            if hasattr(episode, "object_category")
            else ""
        )
        if category not in self.cache:
            print("Missing category: {}".format(category))
        return self.cache[category]


@registry.register_sensor
class ClipImageGoalSensor(Sensor):

    cls_uuid: str = "clip_imagegoal"

    def __init__(
        self,
        sim: "HabitatSim",
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        # TODO: actually calculate the correct height and width
        self.height = 256
        self.width = 256
        self._sim = sim
        super().__init__(config=config)
        self._curr_ep_id = None
        self.image_goal = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype=np.uint8,
        )

    def _reset(self, episode):
        self._curr_ep_id = episode.episode_id
        sampled_goal = random.choice(episode.goals)
        sampled_viewpoint = random.choice(sampled_goal.view_points)
        observations = self._sim.get_observations_at(
            position=sampled_viewpoint.agent_state.position,
            rotation=sampled_viewpoint.agent_state.rotation,
            keep_agent_at_new_pose=False,
        )
        assert observations is not None
        self.image_goal = observations["rgb"]
        # Mutate the episode
        episode.goals = [sampled_goal]

    def get_observation(
        self,
        observations,
        episode: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        if self.image_goal is None or self._curr_ep_id != episode.episode_id:
            self._reset(episode)
        assert self.image_goal is not None
        return self.image_goal


@registry.register_sensor
class ClipGoalSelectorSensor(Sensor):
    cls_uuid: str = "clip_goal_selector"

    def __init__(
        self,
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(config=config)
        self._image_sampling_prob = config.image_sampling_probability
        self._curr_ep_id = None
        self._use_image_goal = True

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.bool,
        )

    def _reset(self, episode):
        self._curr_ep_id = episode.episode_id
        self._use_image_goal = random.random() < self._image_sampling_prob

    def get_observation(
        self,
        observations,
        episode: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        if self._curr_ep_id != episode.episode_id:
            self._reset(episode)
        return np.array([self._use_image_goal], dtype=np.bool)
