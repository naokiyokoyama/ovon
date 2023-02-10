import argparse
import copy
import csv
import glob
import gzip
import itertools
import json
import lzma
import math
import multiprocessing
import os
import os.path as osp
import pickle
import sys
import traceback
from collections import defaultdict

import GPUtil
import habitat
import habitat_sim
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import trimesh
from habitat.config.default import get_agent_config, get_config
from habitat.config.default_structured_configs import \
    HabitatSimSemanticSensorConfig
from habitat.config.read_write import read_write
from ovon.dataset.episode_generator import (build_goal,
                                            generate_objectnav_episode_v2)
from ovon.dataset.hm3d_constants import GOAL_CATEGORIES, HM3D_SCENES
from sklearn.cluster import AgglomerativeClustering

font = {"size": 22}
matplotlib.rc("font", **font)

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


SCENES_ROOT = "./data/scene_datasets/hm3d"
COMPRESSION = ".gz"
VERSION_ID = "v1"
OBJECT_ON_SAME_FLOOR = False
NUM_VAL_OBJS_JSON = 28
NUM_TEST_OBJS_JSON = 59
MIN_OBJECT_DISTANCE = 1.0
MAX_OBJECT_DISTANCE = 30.0


OUTPUT_OBJ_FOLDER = f"./data/datasets/objectnav/hm3d_semantic_v0.2/{VERSION_ID}"
PLOT_FOLDER = "data/hm3d_semantic_v0.2_objectnav_plots"
OUTPUT_JSON_FOLDER = f"./data/datasets/objectnav/hm3d_semantic_v0.2/{VERSION_ID}"
NUM_GPUS = len(GPUtil.getAvailable(limit=256))
TASKS_PER_GPU = 12

wordlist = GOAL_CATEGORIES

category_to_task_category_id = {k: int(v) for v, k in enumerate(wordlist)}
# NOTE: Just a dummy set of values. This is not used anywhere afaik.
category_to_scene_annotation_category_id = {k: int(v) for v, k in enumerate(wordlist)}


ANNOTATION_CORRECTIONS = {  # Values -  (original name, fixed name)
    ############################################################################
    # v2 scenes
    ("00020-XYyR54sxe6b", 153): ("table", "chair"),
    ("00020-XYyR54sxe6b", 154): ("table", "chair"),
    ("00020-XYyR54sxe6b", 155): ("table", "chair"),
    ("00245-741Fdj7NLF9", 102): ("chair", "stool"),
    ("00245-741Fdj7NLF9", 103): ("chair", "stool"),
    ("00245-741Fdj7NLF9", 104): ("chair", "stool"),
    ("00245-741Fdj7NLF9", 105): ("chair", "stool"),
    ("00256-92vYG1q49FY", 851): ("unknown", "tv"),
    ("00612-GsQBY83r3hb", 31): ("bench", "sofa"),
    ("00940-oHcTqmvveM7", 508): ("bed comforter", "bed"),
    ############################################################################
    # v1 scenes
    ("00109-GTV2Y73Sn5t", 725): ("bed", "sofa"),
    ("00109-GTV2Y73Sn5t", 733): ("bed", "sofa"),
    ("00263-GGBvSFddQgs", 49): ("kitchen table", "chair"),
    ("00263-GGBvSFddQgs", 50): ("kitchen table", "chair"),
    ("00263-GGBvSFddQgs", 51): ("kitchen table", "chair"),
    ("00263-GGBvSFddQgs", 52): ("kitchen table", "chair"),
    ("00263-GGBvSFddQgs", 429): ("flowerpot", "lamp"),
    ("00386-b3WpMbPFB6q", 562): ("bed", "sofa"),
    ("00404-QN2dRqwd84J", 384): ("tv stand", "tv"),
    ("00404-QN2dRqwd84J", 507): ("box", "tv"),
    ("00904-S9CUp5RsFY9", 608): ("bathroom utensil", "flower"),
    ("00904-S9CUp5RsFY9", 488): ("lamp", "flower"),
    ("00926-egVUByjgpHU", 92): ("kitchen countertop item", "tv"),
}


def get_objnav_config(i, scene):

    CFG = "habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"
    SCENE_CFG = f"{SCENES_ROOT}/hm3d_annotated_basis.scene_dataset_config.json"
    objnav_config = get_config(CFG)

    with read_write(objnav_config):
        agent_config = get_agent_config(objnav_config.habitat.simulator)

        del agent_config.sim_sensors["depth_sensor"]
        agent_config.sim_sensors.update(
            {"semantic_sensor": HabitatSimSemanticSensorConfig()}
        )
        FOV = 90

        for sensor, sensor_config in agent_config.sim_sensors.items():
            agent_config.sim_sensors[sensor].hfov = FOV
            agent_config.sim_sensors[sensor].width //= 2
            agent_config.sim_sensors[sensor].height //= 2

        objnav_config.habitat.task.measurements = {}

        deviceIds = GPUtil.getAvailable(
            order="memory", limit=1, maxLoad=1.0, maxMemory=1.0
        )
        if i < NUM_GPUS * TASKS_PER_GPU or len(deviceIds) == 0:
            deviceId = i % NUM_GPUS
        else:
            deviceId = deviceIds[0]
        objnav_config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
            deviceId  # i % NUM_GPUS
        )
        objnav_config.habitat.dataset.data_path = (
            "./data/datasets/pointnav/hm3d/v1/val/val.json.gz"
        )
        objnav_config.habitat.dataset.scenes_dir = "./data/scene_datasets/"
        objnav_config.habitat.dataset.split = "train"
        objnav_config.habitat.simulator.scene = scene
        objnav_config.habitat.simulator.scene_dataset = SCENE_CFG
    return objnav_config


def get_simulator(objnav_config):
    sim = habitat.sims.make_sim("Sim-v0", config=objnav_config.habitat.simulator)
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = (
        objnav_config.habitat.simulator.agents.main_agent.radius
    )
    navmesh_settings.agent_height = (
        objnav_config.habitat.simulator.agents.main_agent.height
    )
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    return sim


def dense_sampling_trimesh(triangles, density=25.0, max_points=200000):
    # Create trimesh mesh from triangles
    t_vertices = triangles.reshape(-1, 3)
    t_faces = np.arange(0, t_vertices.shape[0]).reshape(-1, 3)
    t_mesh = trimesh.Trimesh(vertices=t_vertices, faces=t_faces)
    surface_area = t_mesh.area
    n_points = min(int(surface_area * density), max_points)
    t_pts, _ = trimesh.sample.sample_surface_even(t_mesh, n_points)
    return t_pts


def get_gravity_mobb(object_obb: habitat_sim.geo.OBB):
    bounding_area = [
        (object_obb.local_to_world @ np.array([x, y, z, 1]))[:-1]
        for x, y, z in itertools.product(*([[-1, 1]] * 3))
    ]
    bounding_area = np.array(bounding_area, dtype=np.float32)
    # print('Bounding Area: %s' % bounding_area)
    # TODO Maybe Cache this
    return habitat_sim.geo.compute_gravity_aligned_MOBB(
        habitat_sim.geo.GRAVITY, bounding_area
    )


def get_scene_key(glb_path):
    # <ROOT_DIRECTORY>/xxx-<SCENE_KEY>/<SCENE_KEY>.basis.glb
    return osp.basename(glb_path).split(".")[0]


def get_file_opener(fname):
    ext = os.path.splitext(fname)[-1]

    if ext == ".gz":
        file_opener = gzip.open
    elif ext == ".xz":
        file_opener = lzma.open
    else:
        print(ext)
        assert False
        # ile_opener = open
    return file_opener


def save_dataset(dset: habitat.Dataset, fname: str):
    file_opener = get_file_opener(fname)
    # compression = gzip if format == 'gzip' else lzma
    # dset = dset.from_json(dset.to_json())
    # ddset = dset.from_json(json.loads(json.dumps(dset.to_json())))
    if (
        os.path.basename(os.path.dirname(fname)) == "content"
        and len(dset.episodes) == 0
    ):
        print("WARNING UNEXPECTED EMPTY EPISODES: %s" % fname)
        return
    with file_opener(fname, "wt") as f:
        if len(dset.episodes) == 0:
            print("WARNING EMPTY EPISODES: %s" % fname)
            f.write(
                json.dumps(
                    {
                        "episodes": [],
                        "category_to_task_category_id": dset.category_to_task_category_id,
                        "category_to_scene_annotation_category_id": dset.category_to_scene_annotation_category_id,
                    }
                )
            )
        else:
            dset_str = dset.to_json()
            f.write(dset_str)


def generate_scene(args):
    i, scene, split = args
    objnav_config = get_objnav_config(i, scene)
    print(scene)

    ############################################################################
    # Get mapping from raw names to categories                                 #
    ############################################################################
    HM3D_RAW_TO_CAT_MAPPING = {}

    with open(
        "ovon/dataset/source_data/Mp3d_category_mapping_fixed_taxonomy.tsv", "r"
    ) as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter="\t")
        is_first_row = True
        for row in tsv_reader:
            if is_first_row:
                is_first_row = False
                continue
            orig_raw_name = row[0].strip().lower()
            fixed_raw_name = row[1].strip().lower()
            cat_name = row[2]
            # Override the category name for plant
            if "plant" in fixed_raw_name or "flower" in fixed_raw_name:
                cat_name = "plant"
            HM3D_RAW_TO_CAT_MAPPING[orig_raw_name] = cat_name
            HM3D_RAW_TO_CAT_MAPPING[fixed_raw_name] = cat_name

    sim = get_simulator(objnav_config)
    print(
        "Number of semantic objects: {}".format(len(sim.semantic_annotations().objects))
    )

    rgm = sim.get_rigid_object_manager()
    obj_ids = [int(x.split(",")[5]) for x in rgm.get_objects_info()[1:]]
    print("Num rigid object: {}".format(obj_ids))
    total_objects = len(sim.semantic_annotations().objects)
    # Check there exists a navigable point
    test_point = sim.sample_navigable_point()
    if total_objects == 0 or not sim.is_navigable(np.array(test_point)):
        print("Scene is not navigable/ no objects are available in scene: %s" % scene)
        sim.close()
        return scene, total_objects, defaultdict(list), None

    objects = []

    for source_id, source_obj in enumerate(
        tqdm.tqdm(
            sim.semantic_annotations().objects,
            desc="Generating object data",
        )
    ):
        if source_obj is None:
            print("=====> Source object is None. Skipping.")
            continue

        raw_name = copy.deepcopy(source_obj.category.name(""))
        ########################################################################
        # Apply annotation correction if available
        obj_id = int(source_obj.id.split("_")[-1])
        scene_name = scene.split("/")[-2]
        if (scene_name, obj_id) in ANNOTATION_CORRECTIONS:
            expec_raw_name = ANNOTATION_CORRECTIONS[(scene_name, obj_id)][1]
            if raw_name != expec_raw_name:
                print(
                    f"Raw name: {raw_name}, Annot corr: {ANNOTATION_CORRECTIONS[(scene_name, obj_id)]}"
                )
            assert raw_name == expec_raw_name
            raw_name = ANNOTATION_CORRECTIONS[(scene_name, obj_id)][1]
        ########################################################################
        raw_name = raw_name.strip().lower()
        category_name = HM3D_RAW_TO_CAT_MAPPING.get(raw_name)
        if raw_name not in HM3D_RAW_TO_CAT_MAPPING:
            continue
        category_name = HM3D_RAW_TO_CAT_MAPPING[raw_name]
        category_id = source_obj.category.index("")

        if category_name not in wordlist:
            continue
        if np.all(source_obj.obb.sizes == 0):
            continue
        if category_name == None:
            print("ERROR NONE CATEGORY NAME: %s %d" % (scene, source_id))
            continue

        obj = {
            "center": source_obj.aabb.center,
            "id": int(source_obj.id.split("_")[-1]),
            "object_name": source_obj.id,
            "obb": source_obj.obb,
            "aabb": source_obj.aabb,
            "gravity_mobb": get_gravity_mobb(source_obj.obb),
            "category_id": category_id,
            "category_name": category_name,
        }
        objects.append(obj)

    print("Scene loaded.")
    print("Total objects post filtering: {}".format(len(objects)))
    scene_key = get_scene_key(scene)
    fname_obj = f"{OUTPUT_OBJ_FOLDER}/{split}/content/{scene_key}_objs.pkl"
    fname = f"{OUTPUT_JSON_FOLDER}/{split}/content/{scene_key}.json{COMPRESSION}"
    print("Write objects pkl at: {}".format(fname_obj))

    ############################################################################
    # Pre-compute goals
    ############################################################################
    if os.path.exists(fname_obj):
        with open(fname_obj, "rb") as f:
            goals_by_category = pickle.load(f)
        total_objects_by_cat = {k: len(v) for k, v in goals_by_category.items()}
    else:
        goals_by_category = defaultdict(list)
        cell_size = objnav_config.habitat.simulator.agents.main_agent.radius / 2.0
        categories_to_counts = {}
        for obj in tqdm.tqdm(objects, desc="Objects for %s:" % scene):
            # print("Object id: %d" % obj["id"])
            # print(obj["category_name"])
            if obj["category_name"] not in categories_to_counts:
                categories_to_counts[obj["category_name"]] = [0, 0]
            categories_to_counts[obj["category_name"]][1] += 1

            goal = build_goal(
                sim,
                object_id=obj["id"],
                object_name_id=obj["object_name"],
                object_category_name=obj["category_name"],
                object_category_id=obj["category_id"],
                object_position=obj["center"],
                object_aabb=obj["aabb"],
                object_obb=obj["obb"],
                object_gmobb=obj["gravity_mobb"],
                cell_size=cell_size,
                grid_radius=3.0,
            )
            if goal == None:
                continue
            categories_to_counts[obj["category_name"]][0] += 1
            goals_by_category[osp.basename(scene) + "_" + obj["category_name"]].append(
                goal
            )
        for obj_cat in sorted(list(categories_to_counts.keys())):
            nvalid, ntotal = categories_to_counts[obj_cat]
            print(f"Category: {obj_cat:<15s} | {nvalid:03d}/{ntotal:03d} instances")
        os.makedirs(osp.dirname(fname_obj), exist_ok=True)
        total_objects_by_cat = {k: len(v) for k, v in goals_by_category.items()}
        with open(fname_obj, "wb") as f:
            pickle.dump(goals_by_category, f)

    ############################################################################
    # Cluster points on the navmesh
    ############################################################################
    scene_name = scene.split("/")[-1].split(".")[0]
    scene_name_wno = scene.split("/")[-2]
    obj_save_path = os.path.join(
        OUTPUT_OBJ_FOLDER, split, "content", f"{scene_name}_clusters.pkl"
    )
    if os.path.isfile(obj_save_path):
        with open(obj_save_path, "rb") as fp:
            obj_data = pickle.load(fp)
            cluster_infos = obj_data["cluster_infos"]
            goal_category_to_cluster_distances = obj_data[
                "goal_category_to_cluster_distances"
            ]
    else:
        # Discover navmesh clusters
        navmesh_triangles = np.array(sim.pathfinder.build_navmesh_vertices())
        navmesh_pc = dense_sampling_trimesh(navmesh_triangles)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            affinity="euclidean",
            distance_threshold=1.0,
        ).fit(navmesh_pc)
        labels = clustering.labels_
        n_clusters = clustering.n_clusters_
        cluster_infos = []
        for i in range(n_clusters):
            center = navmesh_pc[labels == i, :].mean(axis=0)
            if sim.pathfinder.is_navigable(center):
                center = np.array(sim.pathfinder.snap_point(center)).tolist()
                locs = navmesh_pc[labels == i, :].tolist()
                stddev = np.linalg.norm(np.std(locs, axis=0)).item()
                cluster_infos.append({"center": center, "locs": locs, "stddev": stddev})
        print(f"====> Calculated cluster infos. # clusters: {n_clusters}")
        os.makedirs(osp.join(PLOT_FOLDER, "split", scene_name_wno), exist_ok=True)
        # Calculate distances from goals to cluster centers
        goal_category_to_cluster_distances = {}
        for cat, data in goals_by_category.items():
            object_vps = []
            for inst_data in data:
                for view_point in inst_data.view_points:
                    object_vps.append(view_point.agent_state.position)
            goal_distances = []
            for i, cluster_info in enumerate(cluster_infos):
                dist = sim.geodesic_distance(cluster_info["center"], object_vps)
                goal_distances.append(dist)
            goal_category_to_cluster_distances[cat] = goal_distances

            # Plot distances for visualization
            plt.figure(figsize=(8, 8))
            hist_data = list(filter(math.isfinite, goal_distances))
            hist_data = pd.DataFrame.from_dict({"Geodesic distance": hist_data})
            sns.histplot(data=hist_data, x="Geodesic distance")
            plt.title(cat)
            plt.tight_layout()
            plt.savefig(osp.join(PLOT_FOLDER, "split", scene_name_wno, f"{cat}.png"))
        with open(obj_save_path, "wb") as fp:
            pickle.dump(
                {
                    "cluster_infos": cluster_infos,
                    "goal_category_to_cluster_distances": goal_category_to_cluster_distances,
                },
                fp,
            )

    ############################################################################
    # Compute ObjectNav episodes
    ############################################################################
    if os.path.exists(fname):
        print("Scene already generated. Skipping")
        sim.close()
        return scene, total_objects, total_objects_by_cat, None

    if True:
        if split == "val_v0.2":
            total_objs_json = NUM_VAL_OBJS_JSON
        elif split == "test_v0.2":
            total_objs_json = NUM_TEST_OBJS_JSON
        else:
            total_objs_json = 50000
        total_valid_cats = len(total_objects_by_cat)
        dset = habitat.datasets.make_dataset("ObjectNav-v1")
        dset.category_to_task_category_id = category_to_task_category_id
        dset.category_to_scene_annotation_category_id = (
            category_to_scene_annotation_category_id
        )
        dset.goals_by_category = goals_by_category
        scene_dataset_config = (
            f"{SCENES_ROOT}/hm3d_annotated_basis.scene_dataset_config.json"
        )
        with tqdm.tqdm(total=total_objs_json, desc=scene) as pbar:

            for goal_cat, goals in goals_by_category.items():
                eps_per_obj = int(total_objs_json / total_valid_cats + 0.5)
                eps_generated = 0
                try:
                    for ep in generate_objectnav_episode_v2(
                        sim,
                        goals,
                        cluster_infos,
                        np.array(goal_category_to_cluster_distances[goal_cat]),
                        num_episodes=eps_per_obj,
                        closest_dist_limit=MIN_OBJECT_DISTANCE,
                        furthest_dist_limit=MAX_OBJECT_DISTANCE,
                        scene_dataset_config=scene_dataset_config,
                        same_floor_flag=OBJECT_ON_SAME_FLOOR,
                    ):
                        dset.episodes.append(ep)
                        pbar.update()
                        eps_generated += 1
                except RuntimeError:
                    traceback.print_exc()
                    obj_cat = goals[0].object_name.split("_")[0]
                    print(f"Skipping category {obj_cat}")
                    pbar.update(eps_per_obj - eps_generated)

        for ep in dset.episodes:
            # STRIP OUT PATH
            ep.scene_id = ep.scene_id[len("./data/scene_datasets/") :]

        os.makedirs(osp.dirname(fname), exist_ok=True)
        save_dataset(dset, fname)
    sim.close()
    return scene, total_objects, total_objects_by_cat, fname


def read_dset(json_fname):
    dset2 = habitat.datasets.make_dataset("ObjectNav-v1")
    file_opener = get_file_opener(json_fname)

    # compression = gzip if os.path.splitext(json_fname)[-1] == 'gz' else lzma
    with file_opener(json_fname, "rt") as f:
        # print(json_fname)
        dset2.from_json(f.read())
    return dset2


def prepare_inputs(split):
    scenes = [f"{SCENES_ROOT}/{scene}" for scene in HM3D_SCENES[split]]
    return [(i, scene, split) for i, scene in enumerate(scenes)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["train_v0.2", "val_v0.2", "test_v0.2", "train", "*"],
        required=True,
        type=str,
    )
    args = parser.parse_args()

    mp_ctx = multiprocessing.get_context("forkserver")

    np.random.seed(1234)
    if args.split == "*":
        inputs = []
        for split in ["train_v0.2", "val_v0.2", "test_v0.2"]:
            inputs += prepare_inputs(split)
    else:
        inputs = prepare_inputs(args.split)

    GPU_THREADS = NUM_GPUS * TASKS_PER_GPU
    print(GPU_THREADS)
    print("*" * 1000)
    CPU_THREADS = multiprocessing.cpu_count()

    # Generate episodes for all scenes
    with mp_ctx.Pool(CPU_THREADS, maxtasksperchild=1) as pool, tqdm.tqdm(
        total=len(inputs)
    ) as pbar, open("data/train_subtotals.json", "w") as f:
        total_all = 0
        subtotals = []
        for scene, subtotal, subtotal_by_cat, fname in pool.imap_unordered(
            generate_scene, inputs
        ):
            pbar.update()
            total_all += subtotal
            subtotals.append(subtotal_by_cat)
        print(total_all)
        print(subtotals)

        json.dump({"total_objects:": total_all, "subtotal": subtotals}, f)

    if args.split == "*":
        splits = ["train_v0.2", "val_v0.2", "test_v0.2"]
    else:
        splits = [args.split]

    # Create minival split and outer files
    for split in splits:
        dset = habitat.datasets.make_dataset("ObjectNav-v1")
        dset.category_to_task_category_id = category_to_task_category_id
        dset.category_to_scene_annotation_category_id = (
            category_to_scene_annotation_category_id
        )
        global_dset = f"{OUTPUT_JSON_FOLDER}/{split}/{split}.json{COMPRESSION}"
        if os.path.exists(global_dset):
            os.remove(global_dset)
        if not os.path.exists(os.path.dirname(global_dset)):
            os.makedirs(os.path.dirname(global_dset), exist_ok=True)
        jsons_gz = glob.glob(
            f"{OUTPUT_JSON_FOLDER}/{split}/content/*.json{COMPRESSION}"
        )

        if split == "val":
            # Create a minival split
            MINI_SAMPLE = 30
            print("Sampling %d episodes for minival" % MINI_SAMPLE)
            n_minival_scenes = len(HM3D_SCENES["minival"])
            MINI_SAMPLE_PER_SCENE = MINI_SAMPLE // n_minival_scenes
            min_dset_fname = (
                f"{OUTPUT_JSON_FOLDER}/{split}_mini/{split}_mini.json{COMPRESSION}"
            )
            # Create directories corresponding to minival
            os.makedirs(os.path.dirname(min_dset_fname), exist_ok=True)
            os.makedirs(
                os.path.join(os.path.dirname(min_dset_fname), "content"), exist_ok=True
            )
            # Sample episodes for each minival scene
            for scene_name in HM3D_SCENES["minival"]:
                scene_hash = scene_name.split("/")[-1].split(".")[0]
                scene_json_gz = [
                    json_gz for json_gz in jsons_gz if scene_hash in json_gz
                ]
                assert len(scene_json_gz) == 1
                scene_json_gz = scene_json_gz[0]
                dset2 = read_dset(scene_json_gz)
                scene_episodes = list(
                    dset2.get_episode_iterator(
                        num_episode_sample=min(
                            MINI_SAMPLE_PER_SCENE, dset2.num_episodes
                        ),
                        cycle=False,
                    )
                )
                print(
                    "Sampling {} episodes for scene {} in minival".format(
                        len(scene_episodes), scene_name
                    )
                )
                min_scn_dset_fname = f"{OUTPUT_JSON_FOLDER}/{split}_mini/content/{scene_hash}.json{COMPRESSION}"
                min_scn_dset = habitat.datasets.make_dataset("ObjectNav-v1")
                min_scn_dset.category_to_task_category_id = category_to_task_category_id
                min_scn_dset.category_to_scene_annotation_category_id = (
                    category_to_scene_annotation_category_id
                )
                scene_key = get_scene_key(dset2.episodes[0].scene_id)
                fname_obj = f"{OUTPUT_OBJ_FOLDER}/{split}/content/{scene_key}_objs.pkl"
                with open(fname_obj, "rb") as f:
                    goals_by_category = pickle.load(f)
                min_scn_dset.goals_by_category = goals_by_category

                min_scn_dset.episodes = scene_episodes
                save_dataset(min_scn_dset, min_scn_dset_fname)
            # Save global mini dataset
            global_min_dset = (
                f"{OUTPUT_JSON_FOLDER}/{split}_mini/{split}_mini.json{COMPRESSION}"
            )
            save_dataset(dset, global_min_dset)

        save_dataset(dset, global_dset)
