from collections import defaultdict
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
from habitat.core.embodied_task import Measure
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import NavigationEpisode, NavigationTask, Success

from ovon.dataset.semantic_utils import ObjectCategoryMapping
from ovon.utils.utils import load_json, load_pickle

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_measure
class OVONDistanceToGoal(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "distance_to_goal"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._metric = None
        if self._config.distance_to == "VIEW_POINTS":
            goals = task._dataset.goals_by_category[episode.goals_key]
            self._episode_view_points = [
                view_point.agent_state.position
                for goal in goals
                for view_point in goal.view_points
            ]

            if episode.children_object_categories is not None:
                for (
                    children_category
                ) in episode.children_object_categories:
                    scene_id = episode.scene_id.split("/")[-1]
                    goal_key = f"{scene_id}_{children_category}"

                    # Ignore if there are no valid viewpoints for goal
                    if goal_key not in task._dataset.goals_by_category:
                        continue
                    self._episode_view_points.extend(
                        [
                            vp.agent_state.position
                            for goal in task._dataset.goals_by_category[
                                goal_key
                            ]
                            for vp in goal.view_points
                        ]
                    )

        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self,
        episode: NavigationEpisode,
        task: NavigationTask,
        *args: Any,
        **kwargs: Any,
    ):
        current_position = self._sim.get_agent_state().position

        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            if self._config.distance_to == "POINT":
                goals = task._dataset.goals_by_category[episode.goals_key]
                distance_to_target = self._sim.geodesic_distance(
                    current_position,
                    [goal.position for goal in goals],
                    episode,
                )
            elif self._config.distance_to == "VIEW_POINTS":
                distance_to_target = self._sim.geodesic_distance(
                    current_position, self._episode_view_points, episode
                )
            else:
                logger.error(
                    "Non valid distance_to parameter was provided"
                    f"{self._config.distance_to}"
                )

            self._previous_position = (
                current_position[0],
                current_position[1],
                current_position[2],
            )
            self._metric = distance_to_target


@registry.register_measure
class OVONObjectGoalID(Measure):
    cls_uuid: str = "object_goal_id"

    def __init__(self, config: "DictConfig", *args: Any, **kwargs: Any):
        cache = load_pickle(config.cache)
        self.vocab = sorted(list(cache.keys()))
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = self.vocab.index(episode.object_category)

    def update_metric(
        self,
        episode: NavigationEpisode,
        task: NavigationTask,
        *args: Any,
        **kwargs: Any,
    ):
        pass


@registry.register_measure
class FailureModeMeasure(Measure):
    """
    Last Mile Navigation failure measures.
    """

    cls_uuid: str = "failure_modes"

    def __init__(
        self, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._config = config
        self._goal_seen = False
        self._elapsed_steps = 0
        self._ovon_categories = load_json(config.categories_file)
        self.cat_map = ObjectCategoryMapping(
            config.mapping_file,
            coverage_meta_file="data/coverage_meta/train.pkl",
            frame_coverage_threshold=0.05
        )
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, observations, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [OVONDistanceToGoal.cls_uuid, Success.cls_uuid]
        )
        self._goal_seen = False
        self._max_area = 0
        self._elapsed_steps = 0
        self._reached_within_success_area = False
        self.set_goal_positions(task)
        self.update_metric(episode=episode, task=task, observations=observations, *args, **kwargs)  # type: ignore
    
    def set_goal_positions(self, task):
        sim = task._sim
        categories = self._ovon_categories["train"]

        semantic_scene = sim.semantic_annotations()
        category_to_positions = defaultdict(list)
        for obj in semantic_scene.objects:
            relabelled_category = self.cat_map[obj.category.name()]
            if relabelled_category is not None and relabelled_category in categories:
                category_to_positions[obj.category.name()].append(obj.aabb.center)
        print("Total categories: {} - {}".format(len(category_to_positions), len(semantic_scene.objects)))
        
        self.category_to_positions = category_to_positions

    def visible_goal_area(self, observations, episode, task):
        scene_id = episode.scene_id.split("/")[-1]
        object_category = episode.object_category
        goal_key = f"{scene_id}_{object_category}"
        goals = task._dataset.goals_by_category[goal_key]
        object_ids = [g.object_id for g in goals]
        semantic_scene = task._sim.semantic_annotations()
        objs = [o for o in semantic_scene.objects if o.id in object_ids]

        semantic_observation = observations["semantic"]
        mask = np.zeros_like(semantic_observation)
        for obj in objs:
            mask += (semantic_observation == obj.semantic_id).astype(np.int32)
        area = np.sum(mask) /  np.prod(semantic_observation.shape)
        return area

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, episode, task: NavigationTask, observations, *args: Any, **kwargs: Any
    ):
        area = self.visible_goal_area(observations, episode, task)
        self._max_area = max(self._max_area, area)
        if area >= 0.01:
            self._goal_seen = True

        distance_to_target = task.measurements.measures[
            OVONDistanceToGoal.cls_uuid
        ].get_metric()
        is_success = task.measurements.measures[Success.cls_uuid].get_metric()

        if distance_to_target < 0.25 and not is_success:
            self._reached_within_success_area = True

        metrics = {
            "exploration": 0.0,
            "last_mile_nav": 0.0,
            "recognition": 0.0,
            "stop_failure": 0.0,
            "nearest_object": "",
            "nearest_object_distance": 0,
            "objects_within_2m": [],
        }

        scene_id = episode.scene_id.split("/")[-1]
        object_keys = [k for k in task._dataset.goals_by_category.keys() if k.startswith(scene_id)]
        current_position = task._sim.get_agent_state().position

        nearest_object = ""
        nearest_object_distance = 100
        objects_within_2m = []
        for object_key, positions in self.category_to_positions.items():
            # goals = task._dataset.goals_by_category[object_key]
            # distance = min([self._euclidean_distance(current_position, goal.position) for goal in goals])
            distance = min([self._euclidean_distance(current_position, position) for position in positions])
            if distance < 2.0 and nearest_object_distance > distance:
                nearest_object = object_key.split("_")[-1]
                nearest_object_distance = distance
            if distance < 2.0:
                objects_within_2m.append(object_key.split("_")[-1])
        metrics["nearest_object"] = nearest_object
        metrics["nearest_object_distance"] = float(nearest_object_distance)
        metrics["objects_within_2m"] = ",".join(objects_within_2m)
        metrics["position"] = ",".join([str(i) for i in current_position.tolist()])

        if not is_success:
            if not self._goal_seen and distance_to_target > 2.0:
                metrics["exploration"] = 1.0
            if self._goal_seen and distance_to_target <= 2.0:
                metrics["last_mile_nav"] = 1.0
            if self._goal_seen and distance_to_target > 2.0:
                metrics["recognition"] = 1.0
        metrics["area_seen"] = self._max_area
        metrics["reached_within_success_area"] = self._reached_within_success_area

        self._metric = metrics
