#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import List

import habitat_sim
import numpy as np
from habitat.core.simulator import AgentState
from habitat.datasets.pointnav.pointnav_generator import (ISLAND_RADIUS_LIMIT,
                                                          _ratio_sample_rate)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.object_nav_task import (ObjectGoal,
                                               ObjectGoalNavEpisode,
                                               ObjectViewLocation)
from habitat.tasks.utils import compute_pixel_coverage
from habitat.utils.geometry_utils import quaternion_from_two_vectors
from habitat_sim.utils.common import quat_to_coeffs


def _direction_to_quaternion(direction_vector: np.array):
    origin_vector = np.array([0, 0, -1])
    output = quaternion_from_two_vectors(origin_vector, direction_vector)
    output = output.normalized()
    return output


def _get_multipath(sim: HabitatSim, start, ends):
    multi_goal = habitat_sim.MultiGoalShortestPath()
    multi_goal.requested_start = start
    multi_goal.requested_ends = ends
    sim.pathfinder.find_path(multi_goal)
    return multi_goal


def _get_action_shortest_path(
    sim: HabitatSim, start_pos, start_rot, goal_pos, goal_radius=0.05
):
    sim.set_agent_state(start_pos, start_rot, reset_sensors=True)
    greedy_follower = sim.make_greedy_follower()
    return greedy_follower.find_path(goal_pos)


def is_compatible_episode(
    s,
    t,
    sim: HabitatSim,
    goals: List[ObjectGoal],
    near_dist,
    far_dist,
    geodesic_to_euclid_ratio,
    same_floor_flag=False,
):
    FAIL_TUPLE = False, 0, 0, [], [], [], []
    if sim.island_radius(s) < ISLAND_RADIUS_LIMIT:
        # print('=====> Island radius failure')
        return FAIL_TUPLE
    s = np.array(s)

    # assert(len(goals_repeated) == len(t))
    # TWO TARGETS MAY BE BETWEEN TWO GOALS
    goal_targets = [
        [vp.agent_state.position for vp in goal.view_points] for goal in goals
    ]

    # Check if atleast one goal is on the same floor as the agent
    if same_floor_flag:
        valid = False
        for gt in goal_targets:
            gt = np.array(gt)
            valid = np.any(np.abs(gt[:, 1] - s[1]) < 0.5)
            if valid:
                break
        if not valid:
            return FAIL_TUPLE

    closest_goal_targets = (sim.geodesic_distance(s, vps) for vps in goal_targets)
    closest_goal_targets, goals_sorted = zip(
        *sorted(zip(closest_goal_targets, goals), key=lambda x: x[0])
    )
    d_separation = closest_goal_targets[0]

    if (
        np.isinf(d_separation)
        # np.inf in closest_goal_targets
        or not near_dist <= d_separation <= far_dist
    ):
        # print('=====> Distance threshold failure: {}, {}, {}, {}'.format(
        #     near_dist, d_separation, far_dist, np.inf in closest_goal_targets
        # ))
        return FAIL_TUPLE

    # shortest_path = sim.get_straight_shortest_path_points(s, closest_target)
    shortest_path = None
    # shortest_path = closest_goal_targets[0].points
    shortest_path_pos = shortest_path
    euclid_dist = np.linalg.norm(s - goals_sorted[0].position)
    distances_ratio = d_separation / euclid_dist
    if distances_ratio < geodesic_to_euclid_ratio and (
        np.random.rand() > _ratio_sample_rate(distances_ratio, geodesic_to_euclid_ratio)
    ):
        # print(f'=====> Distance ratios: {distances_ratio} {geodesic_to_euclid_ratio}')
        return FAIL_TUPLE

    geodesic_distances = closest_goal_targets

    angle = np.random.uniform(0, 2 * np.pi)
    source_rotation = [
        0,
        np.sin(angle / 2),
        0,
        np.cos(angle / 2),
    ]  # Pick random starting rotation

    ending_state = None

    return (
        True,
        d_separation,
        euclid_dist,
        source_rotation,
        geodesic_distances,
        goals_sorted,
        shortest_path,
        ending_state,
    )


def build_goal(
    sim: HabitatSim,
    object_id: int,
    object_name_id: int,
    object_category_name: str,
    object_category_id: int,
    object_position,
    object_aabb: habitat_sim.geo.BBox,
    object_obb: habitat_sim.geo.OBB,
    object_gmobb: habitat_sim.geo.OBB,
    cell_size: float = 1.0,
    grid_radius: float = 10.0,
    turn_radians: float = np.pi / 9,
    max_distance: float = 1.0,
    debug_mode: bool = False,
):
    object_position = object_aabb.center
    eps = 1e-5

    x_len, _, z_len = object_aabb.sizes / 2.0 + max_distance
    x_bxp = np.arange(-x_len, x_len + eps, step=cell_size) + object_position[0]
    z_bxp = np.arange(-z_len, z_len + eps, step=cell_size) + object_position[2]
    candidate_poses = [
        np.array([x, object_position[1], z]) for x, z in itertools.product(x_bxp, z_bxp)
    ]

    def down_is_navigable(pt, search_dist=2.0):
        pf = sim.pathfinder
        delta_y = 0.05
        max_steps = int(search_dist / delta_y)
        step = 0
        is_navigable = pf.is_navigable(pt, 2)
        while not is_navigable:
            pt[1] -= delta_y
            is_navigable = pf.is_navigable(pt)
            step += 1
            if step == max_steps:
                return False
        return True

    def _get_iou(x, y, z):
        pt = np.array([x, y, z])

        # TODO: What is this?
        if not (
            object_obb.distance(pt) <= max_distance
            and habitat_sim.geo.OBB(object_aabb).distance(pt) <= max_distance
        ):
            return -0.5, pt, None, "Unknown error"

        if not down_is_navigable(pt):
            return -1.0, pt, None, "Down is not navigable"
        pf = sim.pathfinder
        pt = np.array(pf.snap_point(pt))

        goal_direction = object_position - pt
        goal_direction[1] = 0

        q = _direction_to_quaternion(goal_direction)

        cov = 0
        agent = sim.get_agent(0)
        for act in [
            HabitatSimActions.look_down,
            HabitatSimActions.look_up,
            HabitatSimActions.look_up,
        ]:
            agent.act(act)
            for v in agent._sensors.values():
                v.set_transformation_from_spec()
            obs = sim.get_observations_at(pt, q, keep_agent_at_new_pose=True)
            cov += compute_pixel_coverage(obs["semantic"], object_id)

        return cov, pt, q, "Success"

    def _visualize_rejected_viewpoints(x, y, z):
        pt = np.array([x, y, z])

        pt[1] -= 0.5
        pf = sim.pathfinder

        goal_direction = object_position - pt
        goal_direction[1] = 0

        q = _direction_to_quaternion(goal_direction)

        import os

        import imageio
        from habitat.utils.visualizations.utils import observations_to_image

        obs = sim.get_observations_at(pt, q, keep_agent_at_new_pose=True)
        imageio.imsave(
            os.path.join(
                "data/images/objnav_dataset_gen",
                f"rejected_{object_category_name}_{object_name_id}_{object_id}_{x}_{z}_.png",
            ),
            observations_to_image(obs, info={}),
        )

    candidate_poses_ious_orig = list(_get_iou(*pos) for pos in candidate_poses)
    n_orig_poses = len(candidate_poses_ious_orig)
    n_unknown_rejected = 0
    n_down_not_navigable_rejected = 0
    for p in candidate_poses_ious_orig:
        if p[-1] == "Unknown error":
            n_unknown_rejected += 1
        elif p[-1] == "Down is not navigable":
            n_down_not_navigable_rejected += 1
    candidate_poses_ious_orig_2 = [p for p in candidate_poses_ious_orig if p[0] > 0]

    # Reject candidate_poses that do not satisfy island radius constraints
    candidate_poses_ious = [
        p
        for p in candidate_poses_ious_orig_2
        if sim.island_radius(p[1]) >= ISLAND_RADIUS_LIMIT
    ]
    n_island_candidates_rejected = len(candidate_poses_ious_orig_2) - len(
        candidate_poses_ious
    )

    best_iou = (
        max(v[0] for v in candidate_poses_ious) if len(candidate_poses_ious) != 0 else 0
    )

    keep_thresh = 0

    view_locations = [
        ObjectViewLocation(AgentState(pt.tolist(), quat_to_coeffs(q).tolist()), iou)
        for iou, pt, q, _ in candidate_poses_ious
        if iou is not None and iou > keep_thresh
    ]

    if debug_mode:
        import ovon.dataset.debug_utils as debug_utils

        debug_utils.plot_area(
            candidate_poses_ious,
            [v.agent_state.position for v in view_locations],
            [object_position],
            object_category_name + object_name_id,
        )

    view_locations = sorted(view_locations, reverse=True, key=lambda v: v.iou)
    if len(view_locations) == 0:
        print(f"No valid views found for {object_name_id}_{object_id}: {best_iou}")
        return None

    if debug_mode:
        for view in view_locations:
            obs = sim.get_observations_at(
                view.agent_state.position, view.agent_state.rotation
            )

            import os

            import imageio
            from habitat.utils.visualizations.utils import (
                get_image_with_obj_overlay, observations_to_image)

            obs["rgb"] = get_image_with_obj_overlay(obs, [object_id])
            imageio.imsave(
                os.path.join(
                    "data/images/objnav_dataset_gen/",
                    f"{object_name_id}_{object_id}_{view.iou}_{view.agent_state.position}.png",
                ),
                observations_to_image(obs, info={}).astype(np.uint8),
            )

    goal = ObjectGoal(
        position=object_position.tolist(),
        view_points=view_locations,
        object_id=object_id,
        object_category=object_category_name,
        object_name=object_name_id,
    )

    return goal


def _create_episode(
    episode_id,
    scene_id,
    start_position,
    start_rotation,
    goals,
    shortest_paths=None,
    scene_state=None,
    info=None,
    scene_dataset_config="default",
):
    return ObjectGoalNavEpisode(
        episode_id=str(episode_id),
        goals=goals,
        scene_id=scene_id,
        object_category=goals[0].object_category,
        start_position=start_position,
        start_rotation=start_rotation,
        shortest_paths=shortest_paths,
        info=info,
        scene_dataset_config=scene_dataset_config,
    )


def generate_objectnav_episode_v2(
    sim: HabitatSim,
    goals: List[ObjectGoal],
    cluster_centers: List[List[float]],
    distance_to_clusters: np.float32,
    scene_state=None,
    num_episodes: int = -1,
    closest_dist_limit: float = 0.2,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.05,
    number_retries_per_cluster: int = 1000,
    scene_dataset_config: str = "default",
    same_floor_flag: bool = False,
):
    r"""Generator function that generates PointGoal navigation episodes.
    An episode is trivial if there is an obstacle-free, straight line between
    the start and goal positions. A good measure of the navigation
    complexity of an episode is the ratio of
    geodesic shortest path position to Euclidean distance between start and
    goal positions to the corresponding Euclidean distance.
    If the ratio is nearly 1, it indicates there are few obstacles, and the
    episode is easy; if the ratio is larger than 1, the
    episode is difficult because strategic navigation is required.
    To keep the navigation complexity of the precomputed episodes reasonably
    high, we perform aggressive rejection sampling for episodes with the above
    ratio falling in the range [1, 1.1].
    Following this, there is a significant decrease in the number of
    straight-line episodes.
    :param sim: simulator with loaded scene for generation.
    :param num_episodes: number of episodes needed to generate
    :param is_gen_shortest_path: option to generate shortest paths
    :param shortest_path_success_distance: success distance when agent should
    stop during shortest path generation
    :param shortest_path_max_steps maximum number of steps shortest path
    expected to be
    :param closest_dist_limit episode geodesic distance lowest limit
    :param furthest_dist_limit episode geodesic distance highest limit
    :param geodesic_to_euclid_min_ratio geodesic shortest path to Euclid
    :param same_floor_flag should object exist on same floor as agent's start?
    :return: navigation episode that satisfy specified distribution for
    currently loaded into simulator scene.
    """
    assert num_episodes > 0
    # cache this transformation
    target_positions = np.array(
        list(
            itertools.chain(
                *(
                    (view_point.agent_state.position for view_point in g.view_points)
                    for g in goals
                )
            )
        )
    )
    ############################################################################
    # Filter out invalid clusters
    ############################################################################
    valid_mask = (distance_to_clusters >= closest_dist_limit) & (
        distance_to_clusters <= furthest_dist_limit
    )
    if same_floor_flag:
        # Ensure that cluster is on same floor as atleast 1 object viewpoint
        for i, cluster_info in enumerate(cluster_centers):
            valid_mask[i] = valid_mask[i] & np.any(
                np.abs(cluster_info["center"][1] - target_positions[:, 1]) < 0.5
            )

    valid_clusters = []
    for i in range(len(cluster_centers)):
        if valid_mask[i].item():
            valid_clusters.append(cluster_centers[i])

    if len(valid_clusters) == 0:
        raise RuntimeError(
            f"No valid clusters: {len(valid_clusters)}/{len(cluster_centers)}"
        )
    cluster_centers = valid_clusters
    NC = len(cluster_centers)
    ############################################################################
    # Divide episodes across clusters
    ############################################################################
    episodes_per_cluster = np.zeros((len(cluster_centers),), dtype=np.int32)
    if NC <= num_episodes:
        # Case 1: There are more episodes than clusters
        ## Divide episodes equally across clusters
        episodes_per_cluster[:] = num_episodes // NC
        ## Place the residual episodes into random clusters
        residual_episodes = num_episodes - NC * (num_episodes // NC)
        if residual_episodes > 0:
            random_order = np.random.permutation(NC)
            for i in random_order[:residual_episodes]:
                episodes_per_cluster[i] += 1
    else:
        # Case 2: There are fewer episodes than clusters
        ## Sample one episode per cluster for a random subset of clusters.
        random_order = np.random.permutation(NC)
        for i in random_order[:num_episodes]:
            episodes_per_cluster[i] = 1

    ############################################################################
    # Generate episodes for each cluster
    ############################################################################
    pathfinder = sim.pathfinder
    for i, num_cluster_episodes in enumerate(episodes_per_cluster):
        episode_count = 0
        cluster_center = cluster_centers[i]["center"]
        cluster_radius = max(3 * cluster_centers[i]["stddev"], 2.0)
        while episode_count < num_cluster_episodes and num_cluster_episodes > 0:
            for _ in range(number_retries_per_cluster):
                source_position = pathfinder.get_random_navigable_point_near(
                    cluster_center, cluster_radius
                )
                if (
                    source_position is None
                    or np.any(np.isnan(source_position))
                    or not sim.is_navigable(source_position)
                ):
                    print(f"Skipping cluster {cluster_center}")
                    num_cluster_episodes = 0
                    break
                if sim.island_radius(source_position) < ISLAND_RADIUS_LIMIT:
                    continue
                compat_outputs = is_compatible_episode(
                    source_position,
                    target_positions,
                    sim,
                    goals,
                    near_dist=closest_dist_limit,
                    far_dist=furthest_dist_limit,
                    geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
                    same_floor_flag=same_floor_flag,
                )
                is_compatible = compat_outputs[0]

                if is_compatible:
                    (
                        is_compatible,
                        dist,
                        euclid_dist,
                        source_rotation,
                        geodesic_distances,
                        goals_sorted,
                        shortest_path,
                        ending_state,
                    ) = compat_outputs
                    if shortest_path is None:
                        shortest_paths = None
                    else:
                        shortest_paths = [shortest_path]

                    episode = _create_episode(
                        episode_id=episode_count,
                        scene_id=sim.habitat_config.scene,
                        start_position=source_position,
                        start_rotation=source_rotation,
                        shortest_paths=shortest_paths,
                        scene_state=scene_state,
                        info={
                            "geodesic_distance": dist,
                            "euclidean_distance": euclid_dist,
                            "closest_goal_object_id": goals_sorted[0].object_id,
                        },
                        goals=goals_sorted,
                        scene_dataset_config=scene_dataset_config,
                    )

                    episode_count += 1
                    yield episode
                    break
