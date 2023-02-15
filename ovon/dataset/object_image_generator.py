
import copy
import os
import os.path as osp
import pickle
import json
import numpy as np
import GPUtil
import habitat
import habitat_sim
import matplotlib.pyplot as plt
from habitat.utils.geometry_utils import quaternion_from_two_vectors
from habitat.config.default import get_agent_config, get_config
from habitat.tasks.utils import compute_pixel_coverage
from habitat.config.read_write import read_write
from habitat.config.default_structured_configs import HabitatSimSemanticSensorConfig
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from ovon.dataset.hm3d_constants import HM3D_SCENES
from ovon.dataset.generate_objectnav_dataset import get_scene_key
from ovon.dataset.pose_sampler import PoseSampler
from PIL import Image, ImageDraw
import itertools
import math
pi = math.pi


SCENES_ROOT = "data/scene_datasets/hm3d"
NUM_GPUS = len(GPUtil.getAvailable(limit=256))
TASKS_PER_GPU = 12

def create_html(objects, file_name, scene_key):
    html_text = """
    <html>
    <head>
        <meta charset="utf-8">
        <title>Objects Spatial Relationships</title>
    </head>
    <style>
        .image-list-small {
        font-family: Arial, Helvetica, sans-serif;
        margin: 0 auto;
        text-align: center;
        max-width: 640px;
        padding: 0;
        }

        .image-list-small li {
        display: inline-block;
        width: 250px;
        margin: 0 12px 30px;
        }


        /* Photo */

        .image-list-small li > a {
        display: block;
        text-decoration: none;
        background-size: cover;
        background-repeat: no-repeat;
        height: 250px;
        width: 250 px;
        margin: 0;
        padding: 0;
        border: 4px solid #ffffff;
        outline: 1px solid #d0d0d0;
        box-shadow: 0 2px 1px #DDD;
        }

        .image-list-small .details {
        margin-top: 13px;
        }


        /* Title */

        .image-list-small .details h3 {
        display: block;
        font-size: 12px;
        margin: 0 0 3px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        }

        .image-list-small .details h3 a {
        color: #303030;
        text-decoration: none;
        }

        .image-list-small .details .image-author {
        display: block;
        color: #717171;
        font-size: 11px;
        font-weight: normal;
        margin: 0;
        }
    </style>
    <body>
    <ul class="image-list-small">"""
    for obj in set(objects):
        html_text += f"""
            <li>
            <a href="#" style="background-image: url('../../images/objects/{scene_key}/{obj}.png');"></a>
            <div class="details">
                <h3><a href="#"> {obj} </a></h3>
            </div>
            </li>
            """
    html_text += """</ul>
                    </body>
                    </html>"""
    f = open(file_name,"w")
    f.write(html_text)
    f.close()


def _get_iou_pose(sim, pose, object):
    agent = sim.get_agent(0)
    obs = agent.set_state(pose)
    cov = compute_pixel_coverage(obs["semantic"], object.semantic_id)
    if(cov >= maxcov):
        maxcov = cov
    
    return maxcov, pose, "Success"


def get_best_viewpoint_with_posesampler(sim, object, pose_sampler):
    candidate_states = pose_sampler.sample_agent_poses_radially(object)
    candidate_poses_ious = list(_get_iou_pose(sim, pos,object) for pos in candidate_states)
    candidate_poses_ious_filtered = [p for p in candidate_poses_ious if p[0] > 0]
    candidate_poses_sorted = sorted(candidate_poses_ious_filtered, key=lambda x: x[0], reverse=True)
    if candidate_poses_sorted:  
        return True, candidate_poses_sorted[0]
    else :
        return False, None

def get_bounding_box(obs, object):
    a_args = np.argwhere(obs["semantic"] == (object.semantic_id))
    try : 
        bb_a = (np.min(a_args[:,0]), np.max(a_args[:,0]),np.min(a_args[:,1]), np.max(a_args[:,1]))
        area = (bb_a[1]-bb_a[0])*(bb_a[3] - bb_a[2])
        return True, bb_a, area/(obs["semantic"].shape[0]*obs["semantic"].shape[1])
    except ValueError :
        return False, None, None

def get_objects(sim, objects_info, scene_key, obj_mapping): 
    f = open('data/obj/filtered_raw_categories.json')
    FILTERED_CATEGORIES = json.load(f)
    filtered_objects = [obj for obj in objects_info if obj.category.name() in FILTERED_CATEGORIES]
    threshold_fractions = [0.05,0.1,0.15,0.20]
    size_filtered = {}
    objects_visualized = []
    cnt = 0

    if not osp.isdir(f"data/images/objects/{scene_key}"):
        os.makedir(f"data/images/objects/{scene_key}")

    pose_sampler = PoseSampler(
        sim=sim,
        r_min=0.5,
        r_max=2.0,
        r_step=0.5,
        rot_deg_delta=10.0,
        h_min=0.8,
        h_max=1.4,
        sample_lookat_deg_delta=5.0,
    )

    for object in filtered_objects :
        name = object.category.name()        
        check, view = get_best_viewpoint_with_posesampler(sim, object, pose_sampler)
        if check: 
            cov, pt, q, angle, _ = view
            obs = set_agent_state(sim,pt,q, angle)
            bb_check, bb, fraction = get_bounding_box(obs,object)
            if bb_check and (name not in obj_mapping or scene_key not in obj_mapping[name] or cov > obj_mapping[name][scene_key]):

                for f in threshold_fractions:
                    if(fraction < f):
                        size_filtered[f]+=1



                #Add object visualization to mapping
                obj_mapping[name][scene_key] = cov

                #Draw bounding box and save image
                i=Image.fromarray(obs["rgb"][:,:,:3], 'RGB')
                draw=ImageDraw.Draw(i)
                draw.rectangle([(bb[3],bb[1]),(bb[2],bb[0])],outline="red", width = 3)
                i.save(f"data/images/objects/{scene_key}/{name}.png")

                objects_visualized.append(object.category.name().strip())
                cnt+=1

        else :
            print("No viewpoint for object :", name)
    print(f"Visualised {cnt} number of objects for scene {scene_key} out of total {len(filtered_objects)}!")
    #create_html(objects_visualized, f"data/webpage/{scene_key}/objects.html", scene_key)

def get_objnav_config(i, scene):
    CFG = "habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"
    SCENE_CFG = f"{SCENES_ROOT}/hm3d_annotated_basis.scene_dataset_config.json"
    objnav_config = get_config(CFG)

    with read_write(objnav_config):
        agent_config = get_agent_config(objnav_config.habitat.simulator)

        #Stretch agent
        agent_config.height = 1.41
        agent_config.radius = 0.17

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


def main():
    object_map = {} #map between object instance -> (coverage, scene)
                    #sort for each object and generate 4-5 visualisation
    for i,glb_path in enumerate(HM3D_SCENES['train'][:5]):
        scene_key  = get_scene_key(glb_path)
        print(f"Starting scene: {scene_key}")
        cfg = get_objnav_config(i,scene_key)
        sim = get_simulator(cfg)
        objects_info = sim.semantic_scene.objects
        get_objects(sim, objects_info, scene_key, object_map)
        print(object_map.keys())
        print(object_map['chair'].keys())
        sim.close()

if __name__ == "__main__":
    main()