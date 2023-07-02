import argparse
import copy
import multiprocessing
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

import GPUtil
import habitat
import numpy as np
from habitat_sim._ext.habitat_sim_bindings import SemanticObject
from habitat_sim.agent.agent import AgentState
from habitat_sim.simulator import Simulator
from habitat_sim.utils.common import quat_from_coeffs
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from ovon.dataset.languagenav_dataset import LanguageNavEpisode
from ovon.dataset.objectnav_generator import ObjectGoalGenerator
from ovon.dataset.pose_sampler import PoseSampler
from ovon.dataset.semantic_utils import get_hm3d_semantic_scenes
from ovon.dataset.visualization import (draw_bbox_on_img, get_bounding_box,
                                        objects_in_view)
from ovon.utils.utils import (load_json, load_pickle, save_image, save_pickle,
                              write_json)


class LanguageGoalGenerator(ObjectGoalGenerator):
    def __init__(
        self,
        outpath: str,
        visuals_dir: str = "data/visualizations/language_goals_debug/",
        caption_annotation_file: Optional[str] = None,
        blacklist_categories: List[str] = [],
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.visuals_dir = visuals_dir
        self.outpath = outpath
        self.caption_annotations = None

        for category in blacklist_categories:
            if self.cat_map.get(category) is not None:
                print("Removing {} from category map".format(category))
                self.cat_map._mapping.pop(category)

        print("Caption annotation file: {}".format(caption_annotation_file))
        if caption_annotation_file is not None:
            self.caption_annotations = load_json(caption_annotation_file)

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

    def dedup_bboxes(self, bbox_metadata):
        filtered_bbox_metadata = []
        category_to_instance_map = defaultdict(list)
        for metadata in bbox_metadata:
            category_to_instance_map[metadata["category"]].append(metadata)
            if len(category_to_instance_map[metadata["category"]]) == 1:
                filtered_bbox_metadata.append(metadata)
        return filtered_bbox_metadata, category_to_instance_map

    def _make_prompt(
        self,
        pose_sampler: PoseSampler,
        target_obj: SemanticObject,
        sim: Simulator,
    ):
        states = pose_sampler.sample_agent_poses_radially(
            target_obj.aabb.center,
            target_obj,
            radius_min=1.0,
            radius_max=2.0,
        )
        observations = self._render_poses(sim, states)
        observations, states = self._can_see_object(observations, states, target_obj)

        if len(observations) == 0:
            return None

        frame_coverages = self._compute_frame_coverage(observations, target_obj.semantic_id)

        keep_goal = self._threshold_object_goals(frame_coverages)

        if sum(keep_goal) == 0:
            return None

        observation = self.max_coverage_viewpoint(observations, frame_coverages)

        object_ids_in_view, avd, obj_ids, fl_ids = objects_in_view(
            observation, target_obj.semantic_id
        )
        objs_in_view = list(
            filter(
                lambda obj: obj is not None
                and (
                    self.cat_map[obj.category.name()] is not None
                ),
                [*map(self.semantic_id_to_obj.get, object_ids_in_view)],
            )
        )
        obj_filtered = list(
            filter(
                lambda obj: obj is not None
                and (
                    self.cat_map[obj.category.name()] is not None
                ),
                [*map(self.semantic_id_to_obj.get, fl_ids)],
            )
        )
        ob_cat_filter = {}
        fl_ids_to_obj = [*map(self.semantic_id_to_obj.get, obj_ids)]
        for ob, depth in zip(fl_ids_to_obj, avd):
            if ob is not None and self.cat_map[ob.category.name()] is not None:
                ob_cat_filter[self.cat_map[ob.category.name()]] = depth

        drawn_img, bbox_metadata, _ = get_bounding_box(
            observation, [target_obj] + objs_in_view, target=target_obj, depths=None
        )

        bbox_metadata, category_to_instance_map = self.dedup_bboxes(bbox_metadata)

        if len(category_to_instance_map) < 3:
            return None

        bboxes = [bbox["bbox"] for bbox in bbox_metadata]
        labels = ["{}_{}".format(bbox["category"], bbox["semantic_id"]) for bbox in bbox_metadata]

        drawn_img = draw_bbox_on_img(observation, bboxes, labels)

        obs_aug = {
            "color_sensor": drawn_img.permute(1, 2, 0).detach().cpu().numpy(),
            "semantic_sensor": observation["semantic_sensor"],
        }
        drawn_img, _, _ = get_bounding_box(
            obs_aug, obj_filtered, target=target_obj, depths=None, bbox_color="green"
        )
        result = {
            "dummy_prompt": f"Go to the {target_obj.category.name()}.",
            "observation": observation,
            "annotated_observation": np.asarray(ToPILImage()(drawn_img)),
            "bbox_metadata": bbox_metadata,
            "category_to_instance_map": category_to_instance_map,
        }
        return result

    def _make_goal(
        self,
        sim: Simulator,
        pose_sampler: PoseSampler,
        obj: SemanticObject,
        with_viewpoints: bool,
        scene: str,
    ):
        assert with_viewpoints

        states = pose_sampler.sample_agent_poses_radially(obj.aabb.center, obj)
        observations = self._render_poses(sim, states)
        observations, states = self._can_see_object(observations, states, obj)

        if len(observations) == 0:
            return None, None

        frame_coverages = self._compute_frame_coverage(observations, obj.semantic_id)

        keep_goal = self._threshold_object_goals(frame_coverages)

        if sum(keep_goal) == 0:
            return None, None

        result = {
            "object_category": self.cat_map[obj.category.name()],
            "object_id": obj.id,
            "position": obj.aabb.center.tolist(),
        }

        if not with_viewpoints:
            return result, None

        if self.sample_dense_viewpoints:
            goal_viewpoints = self._make_object_viewpoints(sim, obj)
            if len(goal_viewpoints) == 0:
                return None, None
            result["view_points"] = goal_viewpoints
        else:
            goal_viewpoints = self._states_to_viewpoints(states)
            result["view_points"] = goal_viewpoints

        prompt_result = self._make_prompt(pose_sampler, obj, sim)
        # Discard object if we can't generate prompt
        if prompt_result is None:
            return None, None

        observation = prompt_result["observation"]
        img_bb = prompt_result["annotated_observation"]

        cat_name = "{}_{}".format(
            obj.category.name().replace("/", "_"), obj.semantic_id
        )
        prompt_meta = {
            "metadata": prompt_result["bbox_metadata"],
        }

        raw_output_path = "{}/raw/{}/{}.png".format(self.outpath, scene, cat_name)
        save_image(observation["color_sensor"], raw_output_path)
        prompt_meta["observation"] = raw_output_path

        annotated_output_path = "{}/annotated/{}/{}.png".format(
            self.outpath, scene, cat_name
        )
        save_image(img_bb, annotated_output_path)
        prompt_meta["annotate_observation"] = annotated_output_path

        return result, prompt_meta

    def make_language_goals(
        self,
        scene: str,
        with_viewpoints: bool,
        with_start_poses: bool,
    ) -> List[Dict[str, Any]]:
        sim = self._config_sim(scene)
        pose_sampler = PoseSampler(sim=sim, **self.pose_sampler_args)
        scene_id = scene.split("/")[-1].split(".")[0]

        output_path = "{}/annotated/{}".format(self.outpath, scene_id)
        os.makedirs(output_path, exist_ok=True)
        output_path = "{}/raw/{}".format(self.outpath, scene_id)
        os.makedirs(output_path, exist_ok=True)

        objects = [
            o
            for o in sim.semantic_scene.objects
            if self.cat_map[o.category.name()] is not None
        ]
        self.semantic_id_to_obj = {o.semantic_id: o for o in sim.semantic_scene.objects}

        language_goals = {}
        results = []
        promp_meta = {}
        if not os.path.exists("{}/{}.pkl".format(self.outpath, scene_id)):
            print("Computing language goals for scene: {}".format(scene_id))
            for obj in tqdm(objects, total=len(objects), dynamic_ncols=True):
                goal, prompt = self._make_goal(
                    sim, pose_sampler, obj, with_viewpoints, scene_id
                )
                if goal is not None and len(goal["view_points"]) > 0 and len(prompt["metadata"]) > 2:
                    goal_uuid = "{}_{}".format(goal["object_category"], obj.semantic_id)
                    if goal_uuid not in language_goals:
                        language_goals[goal_uuid] = []

                    results.append((obj.id, obj.category.name(), len(goal["view_points"])))
                    promp_meta[goal_uuid] = prompt
                    language_goals[goal_uuid].append(goal)
            save_pickle(language_goals, "{}/{}.pkl".format(self.outpath, scene_id))
            write_json(promp_meta, "{}/{}_prompt_meta.json".format(self.outpath, scene_id))
        else:
            print("Loading cached language goals for scene: {}".format(scene_id))
            language_goals = load_pickle("{}/{}.pkl".format(self.outpath, scene_id))

        all_goals = []
        if with_start_poses:
            for goal_uuid, goals in tqdm(language_goals.items()):
                obj_goals = copy.deepcopy(goals)
                prompt = self.caption_annotations.get(goal_uuid)

                if prompt is None:
                    continue

                (
                    start_positions,
                    start_rotations,
                    geodesic_distances,
                    euclidean_distances,
                ) = self._sample_start_poses(
                    sim,
                    obj_goals,
                )

                if len(start_positions) == 0:
                    continue

                all_goals.append(
                    {
                        "language_goals": goals,
                        "start_positions": start_positions,
                        "start_rotations": start_rotations,
                        "geodesic_distances": geodesic_distances,
                        "euclidean_distances": euclidean_distances,
                        "instructions": prompt["instructions"],
                        "goal_uuid": goal_uuid,
                    }
                )

        sim.close()
        return all_goals

    @staticmethod
    def _create_episode(
        episode_id,
        scene_id,
        start_position,
        start_rotation,
        object_category,
        instructions,
        object_instance_id,
        shortest_paths=None,
        info=None,
        scene_dataset_config="default",
    ):
        return LanguageNavEpisode(
            episode_id=str(episode_id),
            goals=[],
            scene_id=scene_id,
            object_category=object_category,
            object_instance_id=object_instance_id,
            instructions=instructions,
            start_position=start_position,
            start_rotation=start_rotation,
            shortest_paths=shortest_paths,
            info=info,
            scene_dataset_config=scene_dataset_config,
        )

    def make_episodes(
        self,
        language_goals: Dict,
        scene: str,
        episodes_per_object: int = -1,
        split: str = "train",
    ):
        dataset = habitat.datasets.make_dataset("LanguageNav-v1")
        dataset.category_to_task_category_id = {}
        dataset.category_to_scene_annotation_category_id = {}

        goals_by_instance = defaultdict(list)
        episode_count = 0
        print("Total number of object goals: {}".format(len(language_goals)))
        for goal in language_goals:
            language_goal = goal["language_goals"][0]
            scene_id = scene.split("/")[-1]
            goals_category_id = "{}_{}".format(scene_id, goal["goal_uuid"])
            print(
                "Goal category: {} - viewpoints: {}, episodes: {}".format(
                    goals_category_id,
                    sum([len(gg["view_points"]) for gg in goal["language_goals"]]),
                    len(goal["start_positions"]),
                )
            )

            goals_by_instance[goals_category_id].extend(goal["language_goals"])

            start_positions = goal["start_positions"]
            start_rotations = goal["start_rotations"]
            euclidean_distances = goal["euclidean_distances"]
            geodesic_distances = goal["geodesic_distances"]

            episodes_for_object = []
            for start_position, start_rotation, euc_dist, geo_dist in zip(start_positions, start_rotations, euclidean_distances, geodesic_distances):
                episode = self._create_episode(
                    episode_id=episode_count,
                    object_category=language_goal["object_category"],
                    object_instance_id=goal["goal_uuid"],
                    scene_id=scene.replace("data/scene_datasets/", ""),
                    scene_dataset_config="./data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
                    start_position=start_position,
                    start_rotation=start_rotation,
                    info={
                        "geodesic_distance": euc_dist,
                        "euclidean_distance": geo_dist,
                        "instructions": goal["instructions"],
                    },
                    instructions=list(goal["instructions"].values()),
                )
                episodes_for_object.append(episode)
                episode_count += 1

            if split != "train" and episodes_per_object > 0:
                episodes_for_object = random.sample(
                    episodes_for_object,
                    min(episodes_per_object, len(episodes_for_object)),
                )
            
            # Clean up children object categories
            for l_g in goal["language_goals"]:
                del l_g["children_object_categories"]

            dataset.episodes.extend(episodes_for_object)

        dataset.goals_by_instance = goals_by_instance
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
        visuals_dir,
        with_start_poses,
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
    
    caption_annotation_file = os.path.join(outpath, "{}_prompt_meta_annotated.json".format(scene_name))
    if not os.path.exists(caption_annotation_file):
        caption_annotation_file = None

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
        frame_cov_thresh=0.10,
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
        visuals_dir=visuals_dir,
        outpath=outpath,
        caption_annotation_file=caption_annotation_file,
        blacklist_categories=["window", "window frame"],
    )

    language_goals = language_goal_maker.make_language_goals(
        scene=scene, with_viewpoints=True, with_start_poses=with_start_poses
    )
    print("Scene: {}".format(scene))
    episode_dataset = language_goal_maker.make_episodes(
        language_goals,
        scene,
        episodes_per_object=episodes_per_object,
        split=split,
    )

    scene_name = os.path.basename(scene).split(".")[0]
    save_to = os.path.join(outpath, f"{scene_name}.json.gz")
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    print("Total episodes: {}".format(len(episode_dataset.episodes)))
    if len(episode_dataset.episodes) != 0:
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
    visuals_dir: str = "data/visualizations/language_goals_debug/",
    with_start_poses: bool = True,
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
                    visuals_dir,
                    with_start_poses,
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
                    visuals_dir,
                    with_start_poses,
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
        default="data/datasets/langaugenav/hm3d/v2_stretch/",
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
    parser.add_argument(
        "--visuals-dir", type=str, default="data/visualizations/language_goals_debug/"
    )
    parser.add_argument(
        "--with-start-poses",
        action="store_true",
        dest="with_start_poses",
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
    print("With start poses: {}".format(args.with_start_poses))

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
        args.visuals_dir,
        args.with_start_poses,
    )
