import argparse
import copy
import gzip
import itertools
import math
import multiprocessing
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple, Union

import GPUtil
import habitat
import habitat_sim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import trimesh
from habitat.config.default_structured_configs import \
    HabitatSimSemanticSensorConfig
from habitat_sim import bindings as hsim
from habitat_sim._ext.habitat_sim_bindings import SemanticObject
from habitat_sim.agent.agent import AgentConfiguration, AgentState
from habitat_sim.simulator import Simulator
from habitat_sim.utils.common import quat_from_coeffs
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from ovon.dataset.objectnav_generator import ObjectGoalGenerator
from ovon.dataset.ovon_dataset import OVONEpisode
from ovon.dataset.pose_sampler import PoseSampler
from ovon.dataset.semantic_utils import (ObjectCategoryMapping, WordnetMapping,
                                         get_hm3d_semantic_scenes)
from ovon.dataset.visualization import (get_bounding_box, get_color, get_depth,
                                        objects_in_view)
from ovon.utils.utils import save_image


class LanguageGoalGenerator(ObjectGoalGenerator):
    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.visuals_dir = "data/visualizations/language_goals_debug/"
        os.makedirs(self.visuals_dir, exist_ok=True)
    
    @staticmethod
    def max_coverage_viewpoint(observations, frame_coverages):
        max_iou = 0
        for obs, frame_coverage in zip(observations, frame_coverages):
            if frame_coverage > max_iou:
                max_iou = frame_coverage
                max_cov_viewpoint = obs
        return max_cov_viewpoint

    def get_observations_at(self, sim, viewpoint):
        position = viewpoint["agent_state"]["position"]
        rotation = quat_from_coeffs(viewpoint["agent_state"]["rotation"])
        sim.agents[0].set_state(AgentState(position=position, rotation=rotation))
        obs = sim.get_sensor_observations()
        return obs

    def _make_prompt(self, obj, observation, sim):
        object_ids_in_view = objects_in_view(observation["semantic_sensor"], obj.semantic_id)
        objs_in_view = list(
            filter(
                lambda obj: obj is not None
                and (
                    self.cat_map[obj.category.name()] is not None
                    or "wall" in obj.category.name().lower()
                ),
                [*map(self.semantic_id_to_obj.get, object_ids_in_view)],
            )
        )

        drawn_img, bbs, area_covered = get_bounding_box(
            observation, [obj] + objs_in_view, depths=None
        )
        return f"Go to the {obj.category.name()}.", observation, np.asarray(ToPILImage()(drawn_img))

    def _make_goal(
        self,
        sim: Simulator,
        pose_sampler: PoseSampler,
        obj: SemanticObject,
        with_viewpoints: bool,
        with_start_poses: bool,
        scene: str,
    ):
        if with_start_poses:
            assert with_viewpoints

        states = pose_sampler.sample_agent_poses_radially(obj.aabb.center, obj)
        observations = self._render_poses(sim, states)
        observations, states = self._can_see_object(observations, states, obj)

        if len(observations) == 0:
            return None

        frame_coverages = self._compute_frame_coverage(observations, obj.semantic_id)

        keep_goal = self._threshold_object_goals(frame_coverages)

        if sum(keep_goal) == 0:
            return None

        result = {
            "object_category": self.cat_map[obj.category.name()],
            "object_id": obj.id,
            "position": obj.aabb.center.tolist(),
            "children_object_categories": [],
        }

        if not with_viewpoints:
            return result

        if self.sample_dense_viewpoints:
            goal_viewpoints = self._make_object_viewpoints(sim, obj)
            if len(goal_viewpoints) == 0:
                return None
            result["view_points"] = goal_viewpoints
        else:
            goal_viewpoints = self._states_to_viewpoints(states)
            result["view_points"] = goal_viewpoints

        max_cov_vp = self.max_coverage_viewpoint(observations, frame_coverages)
        _, observation, img_bb = self._make_prompt(obj, max_cov_vp, sim)
        
        cat_name = "{}_{}".format(obj.category.name().replace("/", "_"), obj.semantic_id)

        output_path = "{}/raw/{}/{}.png".format(self.visuals_dir, scene, cat_name)
        save_image(observation["color_sensor"], output_path)
        output_path = "{}/annotated/{}/{}.png".format(self.visuals_dir, scene, cat_name)
        save_image(img_bb, output_path)

        if not with_start_poses:
            return result
        return result

    def make_language_goals(
        self,
        scene: str,
        with_viewpoints: bool,
        with_start_poses: bool,
    ) -> List[Dict[str, Any]]:
        sim = self._config_sim(scene)
        pose_sampler = PoseSampler(sim=sim, **self.pose_sampler_args)
        scene_id = scene.split("/")[-1].split(".")[0]

        output_path = "{}/raw/{}/".format(self.visuals_dir, scene_id)
        os.makedirs(output_path, exist_ok=True)
        output_path = "{}/annotated/{}/".format(self.visuals_dir, scene_id)
        os.makedirs(output_path, exist_ok=True)

        objects = [
            o
            for o in sim.semantic_scene.objects
            if self.cat_map[o.category.name()] is not None
        ]
        self.semantic_id_to_obj = {o.semantic_id: o for o in objects}

        language_goals = {}
        results = []
        for obj in tqdm(objects, total=len(objects), dynamic_ncols=True):
            goal = self._make_goal(
                sim, pose_sampler, obj, with_viewpoints, with_start_poses, scene_id
            )
            if goal is not None and len(goal["view_points"]) > 0:
                if goal["object_category"] not in language_goals:
                    language_goals[goal["object_category"]] = []

                results.append((obj.id, obj.category.name(), len(goal["view_points"])))

        all_goals = []
        # for object_category, goals in tqdm(language_goals.items()):
        #     obj_goals = copy.deepcopy(goals)

        #     start_positions, start_rotations = self._sample_start_poses(
        #         sim,
        #         obj_goals,
        #     )

        #     if len(start_positions) == 0:
        #         print("Start poses none for: {}".format(object_category))
        #         continue

        #     all_goals.append(
        #         {
        #             "object_goals": goals,
        #             "start_positions": start_positions,
        #             "start_rotations": start_rotations,
        #         }
        #     )
        sim.close()
        return all_goals

    @staticmethod
    def _create_episode(
        episode_id,
        scene_id,
        start_position,
        start_rotation,
        object_category,
        shortest_paths=None,
        info=None,
        scene_dataset_config="default",
        children_object_categories=None,
    ):
        return OVONEpisode(
            episode_id=str(episode_id),
            goals=[],
            scene_id=scene_id,
            object_category=object_category,
            start_position=start_position,
            start_rotation=start_rotation,
            shortest_paths=shortest_paths,
            info=info,
            scene_dataset_config=scene_dataset_config,
            children_object_categories=children_object_categories,
        )

    def make_episodes(
        self,
        object_goals: Dict,
        scene: str,
        episodes_per_object: int = -1,
        split: str = "train",
    ):
        dataset = habitat.datasets.make_dataset("ObjectNav-v1")
        dataset.category_to_task_category_id = {}
        dataset.category_to_scene_annotation_category_id = {}

        goals_by_category = defaultdict(list)
        episode_count = 0
        print("Total number of object goals: {}".format(len(object_goals)))
        for goal in object_goals:
            object_goal = goal["object_goals"][0]
            scene_id = scene.split("/")[-1]
            goals_category_id = "{}_{}".format(scene_id, object_goal["object_category"])
            print(
                "Goal category: {} - viewpoints: {}, episodes: {}".format(
                    goals_category_id,
                    sum([len(gg["view_points"]) for gg in goal["object_goals"]]),
                    len(goal["start_positions"]),
                )
            )

            goals_by_category[goals_category_id].extend(goal["object_goals"])

            start_positions = goal["start_positions"]
            start_rotations = goal["start_rotations"]

            episodes_for_object = []
            for start_position, start_rotation in zip(start_positions, start_rotations):
                episode = self._create_episode(
                    episode_id=episode_count,
                    scene_id=scene.replace("data/scene_datasets/", ""),
                    scene_dataset_config="./data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
                    start_position=start_position,
                    start_rotation=start_rotation,
                    info={
                        "geodesic_distance": 0,
                        "euclidean_distance": 0,
                    },
                    object_category=object_goal["object_category"],
                )
                episodes_for_object.append(episode)
                episode_count += 1

            if split != "train" and episodes_per_object > 0:
                episodes_for_object = random.sample(
                    episodes_for_object,
                    min(episodes_per_object, len(episodes_for_object)),
                )

            dataset.episodes.extend(episodes_for_object)

            # Clean up children object categories
            for o_g in goal["object_goals"]:
                del o_g["children_object_categories"]

        dataset.goals_by_category = goals_by_category
        return dataset


def make_episodes_for_scene(args):
    (
        scene,
        outpath,
        device_id,
        split,
        start_poses_per_object,
        episodes_per_object,
        disable_euc_to_geo_ratio_check,
    ) = args
    if isinstance(scene, tuple) and outpath is None:
        scene, outpath = scene

    scene_name = os.path.basename(scene).split(".")[0]
    print(
        "Processing scene: {}, output_path: {}".format(
            scene, os.path.join(outpath, "{}.json.gz".format(scene_name))
        )
    )
    if os.path.exists(os.path.join(outpath, "{}.json.gz".format(scene_name))):
        print("Skipping scene: {}".format(scene))
        return

    language_goal_maker = LanguageGoalGenerator(
        semantic_spec_filepath="data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
        img_size=(512, 512),
        hfov=90,
        agent_height=1.41,
        agent_radius=0.17,
        sensor_height=1.31,
        pose_sampler_args={
            "r_min": 0.5,
            "r_max": 2.0,
            "r_step": 0.5,
            "rot_deg_delta": 10.0,
            "h_min": 0.8,
            "h_max": 1.4,
            "sample_lookat_deg_delta": 5.0,
        },
        mapping_file="ovon/dataset/source_data/Mp3d_category_mapping.tsv",
        categories=None,
        coverage_meta_file="data/coverage_meta/{}.pkl".format(split),
        frame_cov_thresh=0.05,
        goal_vp_cell_size=0.25,
        goal_vp_max_dist=1.0,
        start_poses_per_obj=start_poses_per_object,
        start_poses_tilt_angle=30.0,
        start_distance_limits=(1.0, 30.0),
        min_geo_to_euc_ratio=1.05,
        start_retries=2000,
        max_viewpoint_radius=1.0,
        wordnet_mapping_file="data/wordnet/wordnet_mapping.json",
        device_id=device_id,
        sample_dense_viewpoints=True,
        disable_euc_to_geo_ratio_check=disable_euc_to_geo_ratio_check,
    )

    object_goals = language_goal_maker.make_language_goals(
        scene=scene, with_viewpoints=True, with_start_poses=True
    )
    print("Scene: {}".format(scene))
    episode_dataset = language_goal_maker.make_episodes(
        object_goals,
        scene,
        episodes_per_object=episodes_per_object,
        split=split,
    )

    scene_name = os.path.basename(scene).split(".")[0]
    save_to = os.path.join(outpath, f"{scene_name}.json.gz")
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    print("Total episodes: {}".format(len(episode_dataset.episodes)))
    language_goal_maker.save_to_disk(episode_dataset, save_to)


def make_episodes_for_split(
    scenes: List[str],
    split: str,
    outpath: str,
    tasks_per_gpu: int = 1,
    enable_multiprocessing: bool = False,
    start_poses_per_object: int = 2000,
    episodes_per_object: int = -1,
    disable_euc_to_geo_ratio_check: bool = False,
):
    dataset = habitat.datasets.make_dataset("OVON-v1")

    os.makedirs(outpath.format(split), exist_ok=True)
    save_to = os.path.join(
        outpath.format(split).replace("content/", ""), f"{split}.json.gz"
    )
    LanguageGoalGenerator.save_to_disk(dataset, save_to)

    deviceIds = GPUtil.getAvailable(order="memory", limit=1, maxLoad=1.0, maxMemory=1.0)

    if enable_multiprocessing:
        gpus = len(GPUtil.getAvailable(limit=256))
        cpu_threads = gpus * 16
        print("In multiprocessing setup - cpu {}, GPU: {}".format(cpu_threads, gpus))

        items = []
        for i, s in enumerate(scenes):
            deviceId = deviceIds[0]
            if i < gpus * tasks_per_gpu or len(deviceIds) == 0:
                deviceId = i % gpus
            items.append(
                (
                    s,
                    outpath.format(split),
                    deviceId,
                    split,
                    start_poses_per_object,
                    episodes_per_object,
                    disable_euc_to_geo_ratio_check,
                )
            )

        mp_ctx = multiprocessing.get_context("forkserver")
        with mp_ctx.Pool(cpu_threads) as pool, tqdm(
            total=len(scenes), position=0
        ) as pbar:
            for _ in pool.imap_unordered(make_episodes_for_scene, items):
                pbar.update()
    else:
        for scene in tqdm(scenes, total=len(scenes), dynamic_ncols=True):
            make_episodes_for_scene(
                (
                    scene,
                    outpath.format(split),
                    deviceIds[0],
                    split,
                    start_poses_per_object,
                    episodes_per_object,
                    disable_euc_to_geo_ratio_check,
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/datasets/langaugenav/hm3d/v1_stretch/",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--tasks-per-gpu",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--multiprocessing",
        dest="enable_multiprocessing",
        action="store_true",
    )
    parser.add_argument(
        "--start-poses-per-object",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--episodes-per-object",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--disable-euc-geo-ratio-check",
        action="store_true",
        dest="disable_euc_to_geo_ratio_check",
    )

    args = parser.parse_args()
    scenes = None
    if args.scene is not None:
        scene_id = args.scene.split(".")[0] + ".basis.glb"
        scenes = [scene_id]
    else:
        scenes = list(
            get_hm3d_semantic_scenes("data/scene_datasets/hm3d", [args.split])[
                args.split
            ]
        )
        scenes = sorted(scenes)

    if args.num_scenes > 0:
        scenes = scenes[: args.num_scenes]
    print(scenes)
    print(
        "Start poses per object: {}, Episodes per object: {}, Split: {}".format(
            args.start_poses_per_object, args.episodes_per_object, args.split
        )
    )

    outpath = os.path.join(args.output_path, "{}/content/".format(args.split))
    make_episodes_for_split(
        scenes,
        args.split,
        outpath,
        args.tasks_per_gpu,
        args.enable_multiprocessing,
        args.start_poses_per_object,
        args.episodes_per_object,
        args.disable_euc_to_geo_ratio_check,
    )
