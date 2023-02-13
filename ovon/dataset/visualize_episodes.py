import argparse
import os

import cv2
import habitat
import habitat_sim
from habitat.config import read_write
from habitat.config.default import get_config
from habitat.config.default_structured_configs import \
    HabitatSimSemanticSensorConfig
from habitat.utils.visualizations import maps
from ovon.utils.utils import (draw_bounding_box, draw_point, is_on_same_floor,
                              load_dataset)

SCENES_ROOT = "data/scene_datasets/hm3d"
MAP_RESOLUTION = 512


def get_objnav_config(scene):
    TASK_CFG = "config/tasks/objectnav_stretch_hm3d.yaml"
    SCENE_DATASET_CFG = os.path.join(
        SCENES_ROOT, "hm3d_annotated_basis.scene_dataset_config.json"
    )
    objnav_config = get_config(TASK_CFG)

    deviceId = 0

    with read_write(objnav_config):
        # TODO: find a better way to do it.
        objnav_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor = (
            HabitatSimSemanticSensorConfig()
        )
        objnav_config.habitat.simulator.habitat_sim_v0.gpu_device_id = deviceId
        objnav_config.habitat.simulator.scene = scene
        objnav_config.habitat.simulator.scene_dataset = SCENE_DATASET_CFG
        objnav_config.habitat.simulator.habitat_sim_v0.enable_physics = True

        objnav_config.habitat.task.measurements = {}

    return objnav_config


def get_sim(objnav_config):
    sim = habitat.sims.make_sim("Sim-v0", config=objnav_config.habitat.simulator)

    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = (
        objnav_config.habitat.simulator.agents.main_agent.radius
    )
    navmesh_settings.agent_height = (
        objnav_config.habitat.simulator.agents.main_agent.height
    )
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings, include_static_objects=True)

    return sim


def setup(scene):
    ovon_config = get_objnav_config(scene)
    sim = get_sim(ovon_config)
    return sim


def visualize_episodes(sim, dataset, object_category):
    top_down_map = None
    episodes = dataset["episodes"]
    goals_by_category = dataset["goals_by_category"]
    ref_floor_height = None

    for episode in episodes:
        if not ref_floor_height is None and not is_on_same_floor(
            episode["start_position"][1], ref_floor_height
        ):
            continue
        if episode["object_category"] != object_category:
            continue

        if top_down_map is None:
            top_down_map = maps.get_topdown_map(
                sim.pathfinder,
                height=episode["start_position"][1],
                map_resolution=MAP_RESOLUTION,
                draw_border=True,
            )
            ref_floor_height = episode["start_position"][1]

        draw_point(
            sim,
            top_down_map,
            episode["start_position"],
            maps.MAP_SOURCE_POINT_INDICATOR,
        )

    for goal_category, goals in goals_by_category.items():
        if object_category in goal_category:
            for goal in goals:
                if not is_on_same_floor(goal["position"][1], ref_floor_height):
                    continue
                top_down_map = draw_point(
                    sim,
                    top_down_map,
                    goal["position"],
                    maps.MAP_TARGET_POINT_INDICATOR,
                    point_padding=6,
                )
                for view_point in goal["view_points"]:
                    top_down_map = draw_point(
                        sim,
                        top_down_map,
                        view_point["agent_state"]["position"],
                        maps.MAP_VIEW_POINT_INDICATOR,
                    )

                draw_bounding_box(
                    sim, top_down_map, goal["object_id"], ref_floor_height
                )

    top_down_map = maps.colorize_topdown_map(top_down_map)

    return top_down_map


def save_visual(img, path):
    cv2.imwrite(path, img)


def visualize(episodes_path, output_path):
    dataset = load_dataset(episodes_path)
    sim = setup(dataset["episodes"][0]["scene_id"])
    top_down_map = visualize_episodes(sim, dataset, object_category="bed")
    save_visual(top_down_map, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=str,
        required=True,
        help="Path to episode dataset",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="output path of visualization",
    )
    args = parser.parse_args()
    visualize(args.episodes, args.output_path)
