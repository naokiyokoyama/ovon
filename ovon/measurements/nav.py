import os.path as osp
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationEpisode, NavigationTask

from ovon.dataset.ovon_dataset import extract_goals_from_gz
from ovon.task.simulator import OVONSim

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_measure
class OVONDistanceToGoal(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "distance_to_goal"

    def __init__(
        self,
        sim: OVONSim,
        config: "DictConfig",
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any,
    ):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self.task = task
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._metric = None
        self._update_scene_goals()
        if self._config.distance_to == "VIEW_POINTS":
            goals = self._sim.curr_scene_goals[episode.goals_key]
            self._episode_view_points = [
                view_point.agent_state.position
                for goal in goals
                for view_point in goal.view_points
            ]

            if episode.children_object_categories is not None:
                for children_category in episode.children_object_categories:
                    scene_id = episode.scene_id.split("/")[-1]
                    goal_key = f"{scene_id}_{children_category}"

                    # Ignore if there are no valid viewpoints for goal
                    if goal_key not in self._sim.curr_scene_goals:
                        continue
                    self._episode_view_points.extend(
                        [
                            vp.agent_state.position
                            for goal in self._sim.curr_scene_goals[goal_key]
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
        self._update_scene_goals()

        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            if self._config.distance_to == "POINT":
                goals = self._sim.curr_scene_goals[episode.goals_key]
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

    def _update_scene_goals(self):
        if len(self._sim.curr_scene_goals) == 0:
            scene_id = osp.basename(self._sim._current_scene)
            gz_path = self.task._dataset.scene_id_to_path[scene_id]
            self._sim.curr_scene_goals = extract_goals_from_gz(gz_path)
