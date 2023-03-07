import argparse
import json
import os
import os.path as osp
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np
from habitat_sim._ext.habitat_sim_bindings import BBox, SemanticObject
from habitat_sim.agent.agent import AgentState
from ovon.dataset.pose_sampler import PoseSampler
from ovon.dataset.semantic_utils import get_hm3d_semantic_scenes
from ovon.dataset.visualise_objects import get_objnav_config, get_simulator
from ovon.dataset.visualization import (
    get_best_viewpoint_with_posesampler,
    get_bounding_box,
)
from scipy.spatial import distance
from torchvision.transforms import ToPILImage
from tqdm import tqdm


def create_html(file_name, relationships, visualised=True, threshold=0):
    html_head = """
    <html>
    <head>
        <meta charset="utf-8">
        <title>Objects Spatial Relationships</title>
    </head>"""
    html_style = """
    <style>
        /* Three image containers (use 25% for four, and 50% for two, etc) */
        .column {
        float: left;
        width: 20.00%;
        padding: 5px;
        }

        /* Clear floats after image containers */
        {
        box-sizing: border-box;
        }

        .row {
        display: flex;
        }
    </style>
    """
    cnt = 0
    html_body = ""
    for scene in relationships.keys():
        for obj_id in relationships[scene].keys():
            info_dict = relationships[scene][obj_id]
            if visualised and info_dict["area"] >= threshold:
                cnt += 1
                name = info_dict["img_ref"]
                cov_sum = np.sum(info_dict["cov"])
                html_body += f"""<input type="checkbox" id="{scene}/{name}" name="{scene}/{name}">"""
                if cnt % 5 == 1:
                    html_body += """<div class="row">"""
                html_body += f"""
                            <div class="column">
                                <img src="../images/relationships_{dim}d/{scene}/{info_dict['img_ref']}.png" alt="{info_dict['img_ref']}" style="width:100%">
                                <h5>{info_dict['img_ref']} cov = {cov_sum:.3f}, frac = {info_dict['area']:.3f}, dist = {info_dict['distance']:.3f}</h5>
                            </div>
                            """
                if cnt % 5 == 0:
                    html_body += "</div>"
            # Filtered Objects
            elif not visualised and info_dict["area"] < threshold:
                cnt += 1
                if cnt % 5 == 1:
                    html_body += """<div class="row">"""
                html_body += f"""
                            <div class="column">
                                <input type="checkbox" id="{name}" name="{name}">
                                <img src="../images/relationships_{dim}d/{scene}/{info_dict['img_ref']}.png" alt="{name}" style="width:100%">
                                <h5>cov = {sum(info_dict['cov']):.3f}, frac = {info_dict['area']:.3f}</h5>
                            </div>
                            """
                if cnt % 5 == 0:
                    html_body += "</div>"
    html_body = (
        f"""
                <body>
                <h2> Visualising {cnt} Relationships </h2>
                """
        + html_body
    )
    html_body += """</body>
                    </html>"""
    f = open(file_name, "w")
    f.write(html_head + html_style + html_body)
    f.close()


def is_above(b, a):
    b_center = b.aabb.center
    a_center = a.aabb.center
    _, a_y, _ = a.aabb.sizes / 2
    _, b_y, _ = b.aabb.sizes / 2

    # Lower plane of b is above upper plane of a
    # The displacement in other directions is not greater than the displacement in y direection
    disp = b_center - a_center
    disp_y = disp[1]
    disp[1] = 0
    if b_center[1] - b_y > a_center[1] + a_y and np.linalg.norm(disp) < disp_y:
        return True
    return False


def find_relation_above(pt1, pt2):
    disp = pt1 - pt2
    disp_y = disp[1]
    disp[1] = 0
    if np.linalg.norm(disp) < disp_y:
        return True


def get_relation_3d(sim, agent, pose_sampler, a, b, closest_points=None):
    """Finds spatial relationship [above,below,near] from 3D bounding boxes of objects and returns the image"""
    name_b = b.category.name()
    name_a = a.category.name()

    if closest_points is not None:
        pta, ptb = closest_points
        if find_relation_above(pta, ptb):
            if a.aabb.center[1] - b.aabb.center[1] < 0:
                rel = f"{name_b} above {name_a}"
            else:
                rel = f"{name_b} below {name_a}"
        else:
            rel = f"{name_b} near {name_a}"

    else:
        if is_above(b, a):
            rel = f"{name_b} above {name_a}"
        elif is_above(a, b):
            rel = f"{name_b} below {name_a}"
        else:
            rel = f"{name_b} near {name_a}"
    check, view = get_best_viewpoint_with_posesampler(sim, pose_sampler, [a, b])
    if check:
        cov, pose, _ = view
        agent.set_state(pose)
        obs = sim.get_sensor_observations()
        drawn_img, _, area = get_bounding_box(obs, [a, b])
        return True, rel, drawn_img, cov, area
    return False, None, None, 0, 0


def get_relation_2d(sim, agent, pose_sampler, a, b):
    """Finds spatial relationship [above, below, near] from the 2D viewpoint with BB for objects and returns image"""
    check, view = get_best_viewpoint_with_posesampler(sim, pose_sampler, [a, b])
    name_b = b.category.name()
    name_a = a.category.name()
    if check:
        cov, pose, _ = view
        agent.set_state(pose)
        obs = sim.get_sensor_observations()
        drawn_img, bb, area = get_bounding_box(obs, a, b)
        if area > 0:
            center1 = np.array([(bb[0][0] + bb[0][2]) / 2, (bb[0][1] + bb[0][3]) / 2])
            center2 = np.array([(bb[1][0] + bb[1][2]) / 2, (bb[1][1] + bb[1][3]) / 2])
            x_disp, y_disp = center2 - center1
            if abs(x_disp) > abs(y_disp):  # left/right/near relationship
                if x_disp > 0:
                    rel = f"{name_b} near {name_a}"
                else:
                    rel = f"{name_b} near {name_a}"
            else:  # above/below relationship
                if y_disp > 0:
                    rel = f"{name_b} below {name_a}"
                else:
                    rel = f"{name_b} above {name_a}"
            return True, rel, drawn_img, np.sum(cov), area
    return False, None, None, None, None


def get_surface_points(obj_a):
    xmin, ymin, zmin = obj_a.aabb.center - obj_a.aabb.sizes / 2
    xmax, ymax, zmax = obj_a.aabb.center + obj_a.aabb.sizes / 2

    x_coords = np.linspace(xmin, xmax, num=100)
    y_coords = np.linspace(ymin, ymax, num=100)
    z_coords = np.linspace(zmin, zmax, num=100)

    points_minx = np.array([(xmin, y, z) for y, z in zip(y_coords, z_coords)])
    points_maxx = np.array([(xmax, y, z) for y, z in zip(y_coords, z_coords)])
    points_miny = np.array([(x, ymin, z) for x, z in zip(x_coords, z_coords)])
    points_maxy = np.array([(x, ymax, z) for x, z in zip(x_coords, z_coords)])
    points_minz = np.array([(x, y, zmin) for x, y in zip(x_coords, y_coords)])
    points_maxz = np.array([(x, y, zmax) for x, y in zip(x_coords, y_coords)])

    points = np.concatenate(
        (points_minx, points_maxx, points_miny, points_maxy, points_minz, points_maxz)
    )
    return points


def is_close_to(obj_a, obj_b, delta):
    pts_a = get_surface_points(obj_a)
    pts_b = get_surface_points(obj_b)
    dist = distance.cdist(pts_a, pts_b, "euclidean")
    pta, ptb = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
    if dist[pta, ptb] < delta:
        return True, dist[pta, ptb], pts_a[pta], pts_b[ptb]

    else:
        return False, None, None, None


def get_spatial_relationships(
    sim,
    objects: List[SemanticObject],
    scene_key: str,
    pose_sampler: PoseSampler,
    relationships: List[Tuple],
    dim: int,
    delta: float = 0.5,
):
    agent = sim.get_agent(0)
    filtered_objects = [
        obj for obj in objects if obj.category.name() in FILTERED_CATEGORIES
    ]
    if not osp.isdir(
        f"/nethome/akutumbaka3/files/ovonproject/data/images/relationships_{dim}d/{scene_key}"
    ):
        os.mkdir(
            f"/nethome/akutumbaka3/files/ovonproject/data/images/relationships_{dim}d/{scene_key}"
        )

    relationships[scene_key] = {}

    for a in filtered_objects:
        for b in filtered_objects:
            name_a = a.category.name()
            name_b = b.category.name()

            if np.linalg.norm(a.aabb.center - b.aabb.center) > 2:
                continue

            close, min_distance, pta, ptb = is_close_to(a, b, delta)
            closest_points = (pta, ptb)

            if name_a != name_b and close:
                if dim == 2:
                    check, rel, img, cov, frac = get_relation_2d(
                        sim, agent, pose_sampler, a, b
                    )
                if dim == 3:
                    check, rel, img, cov, frac = get_relation_3d(
                        sim, agent, pose_sampler, a, b, closest_points
                    )
                if check and all(cov_obj >= 0.05 for cov_obj in cov):
                    preposition = rel.split(" ")[1]
                    name = rel.replace("/", " ")
                    # relationship_map[scene_key][preposition]
                    print(f"Found relationship: {name}")
                    relationships[scene_key][b.semantic_id] = {
                        "scene": scene_key,
                        "relation": preposition,
                        "a": name_a,
                        "b": name_b,
                        "a_id": a.semantic_id,
                        "b_id": b.semantic_id,
                        "distance": min_distance,
                        "cov": cov,
                        "area": frac,
                        "img_ref": name,
                    }
                    (ToPILImage()(img)).convert("RGB").save(
                        f"/nethome/akutumbaka3/files/ovonproject/data/images/relationships_{dim}d/{scene_key}/{name}.png"
                    )


def main():
    if (force) or (
        not (
            os.path.isfile(
                f"/nethome/akutumbaka3/files/ovonproject/data/relationships{dim}d_{split}.pickle"
            )
        )
    ):
        relationships = {}
        HM3D_SCENES = get_hm3d_semantic_scenes("data/scene_datasets/hm3d")
        print("Total number of scenes: ", len(list(HM3D_SCENES[split])))
        for i, scene in enumerate(tqdm(list(HM3D_SCENES[split]))):
            scene_key = os.path.basename(scene).split(".")[0]
            cfg = get_objnav_config(i, scene_key)
            sim = get_simulator(cfg)
            objects_info = sim.semantic_scene.objects
            pose_sampler = PoseSampler(
                sim=sim,
                r_min=0.1,
                r_max=2.0,
                r_step=0.25,
                rot_deg_delta=10.0,
                h_min=0.8,
                h_max=1.4,
                sample_lookat_deg_delta=5.0,
            )
            print(f"Starting scene: {scene_key}")
            get_spatial_relationships(
                sim, objects_info, scene_key, pose_sampler, relationships, dim, delta
            )
            sim.close()
        with open(
            f"/nethome/akutumbaka3/files/ovonproject/data/relationships{dim}d_{split}.pickle",
            "wb",
        ) as handle:
            pickle.dump(relationships, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(
            f"/nethome/akutumbaka3/files/ovonproject/data/relationships{dim}d_{split}.pickle",
            "rb",
        ) as handle:
            relationships = pickle.load(handle)

    create_html(
        f"/nethome/akutumbaka3/files/ovonproject/data/webpage/relationships_visualized_{split}.html",
        relationships,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--split",
        help="split of data to be used",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-f",
        "--filtered_data",
        help="path of json which contains the filtered categories",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dim",
        help="Dimension of BB to generate spatial relationships",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--delta",
        help="The distance threshold to identify whether two objects are next to each other",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--force",
        help="Get all relationships from scenes and redo relationship calculation even if data is already available",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    split = args.split
    dim = args.dim
    force = args.force
    delta = args.delta
    if dim not in [2, 3]:
        print("Invalid Dimension. Please select dim = 2 or 3!")
        sys.exit(1)
    f = open(args.filtered_data)
    FILTERED_CATEGORIES = json.load(f)
    main()
