import argparse
import glob
import os
import random

import numpy as np
from tqdm import tqdm

from ovon.utils.utils import load_dataset, load_json, write_dataset, write_json


def split_val_unseen(path, output_path, n_bins=50):
    categories = []
    category_count = {}

    for split in ["train", "val"]:
        files = glob.glob(
            os.path.join(path, "{}/content/".format(split), "*.json.gz")
        )

        for file in tqdm(files):
            dataset = load_dataset(file)

            goal_categories = [
                key.split("_")[1]
                for key in dataset["goals_by_category"].keys()
            ]
            categories.extend(goal_categories)
            for cat in goal_categories:
                if cat not in category_count:
                    category_count[cat] = 0
                category_count[cat] += 1

    ordered_categories = [
        k
        for k, v in sorted(
            category_count.items(), key=lambda item: item[1], reverse=True
        )
    ]
    category_to_category_id = {
        cat: i for i, cat in enumerate(ordered_categories)
    }

    category_ids = [category_to_category_id[cat] for cat in categories]
    _, bins = np.histogram(category_ids, bins=n_bins)

    category_to_bins = np.digitize(category_ids, bins)
    print(category_to_bins)
    print("Bins: {}".format(bins))
    print("Total categorues: {}".format(len(categories)))
    bin_to_categories = {i: [] for i in range(int(bins.max() + 1))}

    for category, bin in zip(categories, category_to_bins):
        bin_to_categories[bin].append(category)

    val_categories = []
    for bin in bin_to_categories:
        if len(bin_to_categories[bin]) == 0:
            continue
        val_categories.append(np.random.choice(bin_to_categories[bin]))

    train_categories = list(set(categories) - set(val_categories))

    print(train_categories)
    print(val_categories)
    print("Total train categories: {}".format(len(train_categories)))
    print("Total val categories: {}".format(len(val_categories)))

    write_json(
        train_categories,
        os.path.join(output_path, "ovon_train_categories.json"),
    )
    write_json(
        val_categories, os.path.join(output_path, "ovon_val_categories.json")
    )

    write_json(
        {k: v for k, v in category_count.items() if k in train_categories},
        os.path.join(output_path, "ovon_train_category_count.json"),
    )
    write_json(
        {k: v for k, v in category_count.items() if k in val_categories},
        os.path.join(output_path, "ovon_val_category_count.json"),
    )


def filter_episodes(dataset, categories):
    filtered_episodes = []
    for episode in dataset["episodes"]:
        if episode["object_category"] not in categories:
            continue
        filtered_episodes.append(episode)

    return filtered_episodes, dataset["goals_by_category"]


def filter_and_save_dataset(
    path, output_path, categories, max_episodes=2000, split="train"
):
    files = glob.glob(os.path.join(path, "*.json.gz"))

    num_added = 0
    num_gz_files = len(files)

    for idx, file in tqdm(enumerate(files)):
        dataset = load_dataset(file)
        filtered_episodes, filtered_goals = filter_episodes(
            dataset, categories
        )

        if split == "val":
            num_left = max_episodes - num_added
            num_gz_remaining = num_gz_files - idx
            num_needed = min(
                num_left / num_gz_remaining, len(dataset["episodes"])
            )

            filtered_episodes = random.sample(
                filtered_episodes, int(num_needed)
            )
            num_added += len(filtered_episodes)

        dataset["goals_by_category"] = filtered_goals
        dataset["episodes"] = filtered_episodes

        write_dataset(
            dataset,
            os.path.join(output_path, os.path.basename(file)),
        )
    print("Total episodes: {}".format(num_added))


def split_dataset(path, output_path, max_episodes=2000):
    train_categories = load_json("data/hm3d_meta/ovon_train_categories.json")
    val_categories = load_json("data/hm3d_meta/ovon_val_categories.json")

    os.makedirs(os.path.join(output_path, "train/content"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "val_seen/content"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "val_unseen/content"), exist_ok=True)

    # filter_and_save_dataset(
    #     os.path.join(path, "train/content"),
    #     os.path.join(output_path, "train/content"),
    #     train_categories,
    # )
    filter_and_save_dataset(
        os.path.join(path, "val/content"),
        os.path.join(output_path, "val_seen/content"),
        train_categories,
        max_episodes,
        split="val",
    )
    filter_and_save_dataset(
        os.path.join(path, "val/content"),
        os.path.join(output_path, "val_unseen/content"),
        val_categories,
        max_episodes,
        split="val",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument(
        "--split-dataset", action="store_true", dest="split_dataset"
    )
    args = parser.parse_args()

    if args.split_dataset:
        split_dataset(args.path, args.output_path)
    else:
        split_val_unseen(args.path, args.output_path)
