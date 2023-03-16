import argparse
import json
import os
import os.path as osp
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np
from habitat_sim._ext.habitat_sim_bindings import BBox, SemanticObject
from habitat_sim.agent.agent import Agent, AgentState
from habitat_sim.simulator import Simulator
from ovon.dataset.pose_sampler import PoseSampler
from ovon.dataset.semantic_utils import ObjectCategoryMapping, get_hm3d_semantic_scenes
from ovon.dataset.visualise_objects import get_objnav_config, get_simulator
from ovon.dataset.visualization import (
    get_best_viewpoint_with_posesampler,
    get_bounding_box,
)
from scipy.spatial import distance
from torchvision.transforms import ToPILImage
from tqdm import tqdm


def create_html(
    file_name: str,
    relationships: Dict,
    visualised: bool = True,
    threshold: float = 0.05,
) -> None:
    html_head = """
    <html>
    <head>
        <meta charset="utf-8">
        <title>Objects Spatial Relationships</title>
    </head>
    """
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
    html_script = """
    <script>
        var li_relationships = []
        function addRelationships(cb) {
        if (cb.checked) {
            li_relationships.push(cb.id);
        }
        else {
            var index = li_relationships.indexOf(cb.id);
            if (index > -1) {
                li_relationships.splice(index, 1);
            }
        }
        localStorage.setItem("relationships",li_relationships)
        }
    </script>
    """
    vis_cnt = 0
    html_body = ""
    for scene in relationships.keys():
        for info_dict in relationships[scene]:
            if visualised and np.sum(info_dict["area"]) >= threshold:
                vis_cnt += 1

                name = info_dict["name"]
                img_ref = info_dict["img_ref"]
                cov_sum = np.sum(info_dict["cov"])
                area = np.sum(info_dict["area"])

                if vis_cnt % 5 == 1:
                    html_body += """<div class="row">"""
                html_body += f"""
                            <input type="checkbox" id="{scene}_{img_ref}" name="{scene}_{name} onclick="addRelationships(this);"">
                            <div class="column">
                                <img src="../images/relationships_{dim}d/{scene}/{img_ref}.png" alt="{img_ref}" style="width:100%">
                                <h5>{img_ref} cov = {cov_sum:.3f}, frac = {area:.3f}, dist = {info_dict['distance']:.3f}</h5>
                            </div>
                            """
                if vis_cnt % 5 == 0:
                    html_body += "</div>"
    html_body = (
        f"""
    <body>
        <h2> Visualising {vis_cnt} Relationships </h2>
        """
        + html_body
        + """</body>
        </html>"""
    )
    f = open(file_name, "w")
    f.write(html_head + html_style + html_script + html_body)
    f.close()


def get_relation_3d(
    sim: Simulator,
    agent: Agent,
    pose_sampler: PoseSampler,
    a: SemanticObject,
    b: SemanticObject,
    closest_points: Tuple = None,
):
    """Finds spatial relationship [above,below,near] from 3D bounding boxes of objects and returns the image"""
    name_b = b.category.name()
    name_a = a.category.name()

    pt_1, pt_2 = closest_points
    disp = pt_1 - pt_2
    y_disp_greater = disp[1] >= np.linalg.norm(disp[0:3:2])

    def is_above(b: SemanticObject, a: SemanticObject, eps=0.05) -> bool:
        b_center = b.aabb.center
        a_center = a.aabb.center
        _, a_y, _ = a.aabb.sizes / 2
        _, b_y, _ = b.aabb.sizes / 2

        if b_center[1] - b_y + eps > a_center[1] + a_y:
            return True
        return False

    if is_above(b, a) and y_disp_greater:
        rel = f"{name_b} above {name_a}"
    elif is_above(a, b) and y_disp_greater:
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


def get_relation_2d(
    sim: Simulator,
    agent: Agent,
    pose_sampler: PoseSampler,
    a: SemanticObject,
    b: SemanticObject,
    eps: float = 10,
):
    """Finds spatial relationship [above, below, near] from the 2D viewpoint with BB for objects and returns image"""
    check, view = get_best_viewpoint_with_posesampler(sim, pose_sampler, [a, b])
    name_b = b.category.name()
    name_a = a.category.name()
    if check:
        cov, pose, _ = view
        agent.set_state(pose)
        obs = sim.get_sensor_observations()
        drawn_img, bb, area = get_bounding_box(obs, [a, b])

        def get_intersection_area(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            # compute the area of intersection rectangle
            interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))

            # find relative orientation of the rectanges (check if displacement in y is less than that of x)
            y_disp = max((boxA[0] - boxB[2]), (boxB[0] - boxA[2])) < max(
                (boxA[1] - boxB[3]), (boxB[1] - boxA[3])
            )
            if interArea == 0:
                return 0, y_disp

            return interArea, y_disp

        if np.sum(area) > 0:
            intersection, y_disp = get_intersection_area(bb[0], bb[1])
            xmin1, ymin1, xmax1, ymax1 = bb[0]
            xmin2, ymin2, xmax2, ymax2 = bb[1]

            if (intersection) == (np.min(area)):
                if area[0] > area[1]:
                    rel = f"{name_b} on {name_a}"
                else:
                    return False, None, None, None, None
            elif ymin1 + eps > ymax2 and y_disp:
                rel = f"{name_b} above {name_a}"
            elif ymin2 + eps > ymax1 and y_disp:
                rel = f"{name_b} below {name_a}"
            else:
                rel = f"{name_b} near {name_a}"
            return True, rel, drawn_img, (cov), area
    return False, None, None, None, None


def is_close_to(obj_a: SemanticObject, obj_b: SemanticObject, delta: float) -> Tuple:
    def get_surface_points(obj_a: SemanticObject) -> np.ndarray:
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
            (
                points_minx,
                points_maxx,
                points_miny,
                points_maxy,
                points_minz,
                points_maxz,
            )
        )
        return points

    pts_a = get_surface_points(obj_a)
    pts_b = get_surface_points(obj_b)
    dist = distance.cdist(pts_a, pts_b, "euclidean")
    pta, ptb = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
    if dist[pta, ptb] < delta:
        return True, dist[pta, ptb], pts_a[pta], pts_b[ptb]

    return (False, "", None, None)


def get_spatial_relationships(
    sim: Simulator,
    objects: List[SemanticObject],
    scene_key: str,
    pose_sampler: PoseSampler,
    relationships: List[Tuple],
    dim: int,
    delta: float = 0.5,
) -> None:
    agent = sim.get_agent(0)
    filtered_objects = [
        obj for obj in objects if obj.category.name() in FILTERED_CATEGORIES
    ]
    if not osp.isdir(outpath + f"images/relationships_{dim}d/{scene_key}"):
        os.mkdir(outpath + f"images/relationships_{dim}d/{scene_key}")

    cat_map = ObjectCategoryMapping(
        mapping_file="ovon/dataset/source_data/Mp3d_category_mapping.tsv",
        allowed_categories=None,
        coverage_meta_file=f"data/coverage_meta/{split}.pkl",
        frame_coverage_threshold=0.05,
    )

    relationships[scene_key] = []

    for a in filtered_objects:
        for b in filtered_objects:
            name_a = cat_map[a.category.name()]
            name_b = cat_map[b.category.name()]
            if name_a is None or name_b is None:
                continue
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
                    name = rel.replace("/", "_")
                    img_ref = (
                        name.replace(" ", "_") + f"_{b.semantic_id}_{a.semantic_id}"
                    )

                    print(f"Found relationship: {name}")
                    relationships[scene_key].append(
                        {
                            "scene": scene_key,
                            "ref_object": name_a,
                            "target_object": name_b,
                            "ref_obj_semantic_id": a.semantic_id,
                            "target_obj_semantic_id": b.semantic_id,
                            "distance": min_distance,
                            "cov": cov,
                            "area": frac,
                            "name": name,
                            "img_ref": img_ref,
                        }
                    )
                    (ToPILImage()(img)).convert("RGB").save(
                        outpath
                        + f"images/relationships_{dim}d/{scene_key}/{img_ref}.png"
                    )


def main():
    if (force) or (not (os.path.isfile(relationships_outpath + filename + ".pkl"))):
        relationships = {}
        HM3D_SCENES = get_hm3d_semantic_scenes("data/scene_datasets/hm3d")
        print("Total number of scenes: ", len(list(HM3D_SCENES[split])))
        for i, scene in enumerate(
            tqdm(sorted(list(HM3D_SCENES[split]), reverse=True)[:num_scenes])
        ):
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
            relationships_outpath + filename + ".pkl",
            "wb",
        ) as handle:
            pickle.dump(relationships, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(
            relationships_outpath + filename + ".pkl",
            "rb",
        ) as handle:
            relationships = pickle.load(handle)

    if html:
        create_html(
            html_outpath + filename + ".html",
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
        default="data/obj/filtered_raw_categories.json",
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
    parser.add_argument(
        "--num_scenes",
        help="Number of scenes to generate prompts for",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--outpath",
        help="Output path for relationships dictionary and HTML webpage",
        type=str,
        default="data/",
    )
    parser.add_argument(
        "--html",
        help="Create HTML webpage or not",
        default=True,
        action="store_true",
    )
    args = parser.parse_args()
    split = args.split
    dim = args.dim
    force = args.force
    delta = args.delta
    num_scenes = args.num_scenes

    # Output Path Information
    outpath = args.outpath
    relationships_outpath = outpath + "relationships/"
    html_outpath = outpath + "webpage/"
    filename = f"relationships_{dim}d_{split}_{num_scenes}"
    html = args.html

    if dim not in [2, 3]:
        print("Invalid Dimension. Please select dim = 2 or 3!")
        sys.exit(1)
    f = open(args.filtered_data)
    FILTERED_CATEGORIES = json.load(f)
    main()
