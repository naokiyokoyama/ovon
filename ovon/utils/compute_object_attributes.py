import argparse

import clip
import numpy as np
import requests
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

from ovon.dataset.languagenav_generator import LanguageGoalGenerator
from ovon.dataset.pose_sampler import PoseSampler
from ovon.dataset.semantic_utils import get_hm3d_semantic_scenes
from ovon.utils.utils import save_image


def init_clip(model_type="RN50"):
    model, preprocessor = clip.load(model_type)
    model = model.eval()
    model = model.cuda()
    return model, preprocessor


def init_blip2():
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=torch.device("cuda")
    )
    return model, vis_processors


def get_attribute(model, preprocessor, object_category, observation, attributes):
    augmented_object_categories = ["{} {}".format(a, object_category) for a in attributes]
    cosine_similarity = torch.nn.CosineSimilarity()
    with torch.no_grad():
        text = clip.tokenize(augmented_object_categories, context_length=77).cuda()
        obs_preprocessed = preprocessor(Image.fromarray(observation)).unsqueeze(0).cuda()
        image_features = model.encode_image(obs_preprocessed)
        text_features = model.encode_text(text)
        image_features = image_features.repeat(text_features.shape[0], 1)
        print(image_features.shape, text_features.shape)
        probs = cosine_similarity(image_features, text_features).detach().cpu().numpy()

        return attributes[np.argmax(probs)]

def blip2_caption(model, preprocessor, object_category, observation, attrs):
    with torch.no_grad():
        obs = Image.fromarray(observation).convert("RGB")
        print("img: {}".format(np.array(obs).shape))
        obs_preprocessed = preprocessor["eval"](obs).unsqueeze(0).cuda()
        attribute = model.generate({"image": obs_preprocessed, "prompt": "Question: describe the {}? Answer:".format(object_category)})
        return attribute


def viewpoint_per_object(scene, language_goal_maker):
    object_to_viewpoints = {}
    # model, preprocessor = init_clip()
    model, preprocessor = init_blip2()

    attributes = [a.strip() for a in open("data/hm3d_meta/colors.txt", "r").readlines()]

    sim = language_goal_maker._config_sim(scene)
    pose_sampler = PoseSampler(sim=sim, **language_goal_maker.pose_sampler_args)

    objects = [
        o
        for o in sim.semantic_scene.objects
        if language_goal_maker.cat_map[o.category.name()] is not None
    ]
    semantic_id_to_obj = {o.semantic_id: o for o in sim.semantic_scene.objects}
    language_goal_maker.semantic_id_to_obj = semantic_id_to_obj

    for obj in objects:
        prompt_result = language_goal_maker._make_prompt(pose_sampler, obj, sim)
        if prompt_result is None:
            continue
        print(obj.id, prompt_result["dummy_prompt"])
        attribute = blip2_caption(model, preprocessor, obj.category.name(), prompt_result["observation"]["color_sensor"], attributes)
        print("Attribute: {} - {}".format(attribute, obj.category.name()))
        save_image(prompt_result["observation"]["color_sensor"], "data/visualizations/language_goals_debug/{}_{}.png".format(attribute, obj.category.name()))

    print("Total objects: {}".format(len(objects)))


def main(args):
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
        coverage_meta_file="data/coverage_meta/{}.pkl".format(args.split),
        frame_cov_thresh=0.10,
        goal_vp_cell_size=0.25,
        goal_vp_max_dist=1.0,
        start_poses_per_obj=100,
        start_poses_tilt_angle=30.0,
        start_distance_limits=(1.0, 30.0),
        min_geo_to_euc_ratio=1.05,
        start_retries=2000,
        max_viewpoint_radius=1.0,
        wordnet_mapping_file="data/wordnet/wordnet_mapping.json",
        device_id=0,
        sample_dense_viewpoints=True,
        disable_euc_to_geo_ratio_check=True,
        visuals_dir="data/visualizations/language_goals_debug/",
        outpath="",
        caption_annotation_file=None,
        blacklist_categories=["window", "window frame"],
    )
    scenes = list(
        get_hm3d_semantic_scenes("data/scene_datasets/hm3d", [args.split])[
            args.split
        ]
    )
    scenes = sorted(scenes)

    viewpoint_per_object(scenes[0], language_goal_maker)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    main(args)
