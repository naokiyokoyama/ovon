import argparse
import json
import os
import os.path as osp
import time

import cv2
import tqdm
from add_text import add_text_to_image
from blip_model import BLIP2
from label_image import BLIPLabeller
from llava_descriptor import LLaVA, LLaVALabeller


def parse_json(json_path: str, data_parent_dir: str):
    imgs_list, bboxes_list, classes_list, target_indices = [], [], [], []
    with open(json_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        to_iter = data
    else:
        to_iter = list(data.values())
    for sample in tqdm.tqdm(to_iter):
        image_path = osp.join(data_parent_dir, sample["observation"])
        imgs_list.append(image_path)
        bboxes = [
            (
                (int(i["bbox"][0]), int(i["bbox"][1])),
                (int(i["bbox"][2]), int(i["bbox"][3])),
            )
            for i in sample["metadata"]
        ]
        classes = [i["category"] for i in sample["metadata"]]
        bboxes_list.append(bboxes)
        classes_list.append(classes)
        for idx, i in enumerate(sample["metadata"]):
            if i["is_target"]:
                target_indices.append(idx)

    return imgs_list, bboxes_list, classes_list, target_indices


def main(json_path, data_parent_dir):
    model = LLaVA("data/pretrained_models/LLaVA-7B-v0")
    for j in json_path:
        process_json(j, data_parent_dir, model)


def process_json(json_path, data_parent_dir, model):
    print("Parsing jsons...")
    imgs, bboxes, classes, target_indices = parse_json(
        json_path, data_parent_dir
    )

    # print("Loading blip...")
    # blip_model = BLIP2()
    # print("Done loading.")
    # model = None

    for i_path, b, c, t in zip(imgs, bboxes, classes, target_indices):
        annotated_path = i_path.replace("raw", "annotated")
        print(f"Processing {i_path}...")
        if not osp.isfile(i_path) or not osp.isfile(annotated_path):
            print(
                f"SKIPPING! Existence of (img, annot): "
                f"({osp.isfile(i_path)}, {osp.isfile(annotated_path)})"
            )
            continue
        i = cv2.cvtColor(cv2.imread(i_path), cv2.COLOR_BGR2RGB)
        # labeller = BLIPLabeller(
        #     blip_model=None,
        #     img_rgb=i,
        #     bboxes=b,
        #     classes=c,
        #     target_idx=t,
        #     llava_model=model,

        # )
        labeller = LLaVALabeller(
            model=model,
            img_rgb=i,
            bboxes=b,
            classes=c,
            target_idx=t,
        )
        arrangements = labeller.label_image()
        if len(arrangements) == 0:
            target = c[t]
            arrangements = [f"COULD NOT GROUND {target}."]
            success = "failure"
        else:
            success = "success"

        suffix = osp.basename(json_path).split(".")[0].split("_")[0]
        filename = f"arrangements_{suffix}/{success}_{time.time()}.jpg"

        if not osp.isdir(f"arrangements_{suffix}"):
            os.mkdir(f"arrangements_{suffix}")

        arrangements = list(set(arrangements))

        img = cv2.imread(annotated_path)
        for a in arrangements:
            img = add_text_to_image(img, a)

        # Identify objects that were verified
        for idx, (top_left_pt, bottom_right_pt) in enumerate(b):
            #if idx not in labeller.bad_indices:
            cv2.rectangle(
                img, top_left_pt, bottom_right_pt, (0, 255, 0), 2
            )

        cv2.imwrite(filename, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze dataset with BLIP-2."
    )
    parser.add_argument("data_parent_dir", help="Path to dir containing data/")
    parser.add_argument(
        "json_path", nargs="+", help="Path(s) to the JSON file(s)"
    )
    args = parser.parse_args()

    main(args.json_path, args.data_parent_dir)
