import glob
import gzip
import json
import os.path as osp
from collections import defaultdict

import matplotlib.pyplot as plt

PLOT = True


def main(val_split_dir: str):
    # Get basename of val_split_dir, remove trailing slash if present first though
    val_split_dir = val_split_dir.rstrip("/")
    val_split_dir_basename = osp.basename(val_split_dir)

    json_gz_files = glob.glob(f"{val_split_dir}/content/*.json.gz")

    category_to_scene_count = defaultdict(int)

    for gz in json_gz_files:
        with gzip.open(gz) as f:
            data = json.load(f)
        categories = [k.split("glb_")[-1] for k in data["goals_by_category"].keys()]
        for category in categories:
            category_to_scene_count[category] += 1

    print("Num categories:", len(category_to_scene_count))

    # Count area under curve of bar graph
    area_under_curve = sum(category_to_scene_count.values())

    print("Area under curve:", area_under_curve)
    category_to_total_count = scale_dict_values(category_to_scene_count, 3000)
    category_to_scene_counts = distribute_values(
        category_to_total_count, category_to_scene_count
    )

    print("Sum of values:", sum(category_to_total_count.values()))

    # Get sum of values in category_to_scene_counts
    sum_of_values = sum([sum(x) for x in category_to_scene_counts.values()])
    print("Sum of values:", sum_of_values)

    scene_id_to_category_to_count = {}
    for gz in json_gz_files:
        with gzip.open(gz) as f:
            data = json.load(f)

        category_to_count = defaultdict(int)
        categories = [k.split("glb_")[-1] for k in data["goals_by_category"].keys()]
        for category in categories:
            category_to_count[category] += category_to_scene_counts[category].pop()

        scene_id = osp.basename(gz).replace(".json.gz", "")
        scene_id_to_category_to_count[scene_id] = category_to_count

    # Assert all values of category_to_scene_counts are now []
    for category, counts in category_to_scene_counts.items():
        assert len(counts) == 0, f"category: {category}, counts: {counts}"

    # Save scene_id_to_category_to_count to a json file
    with open(f"{val_split_dir_basename}_scene_id_to_category_to_count.json", "w") as f:
        json.dump(scene_id_to_category_to_count, f, indent=4)

    if PLOT:
        # Plot a bar graph of the number of scenes per category
        # with category names as x-axis labels
        scale = 2.0
        plt.figure(figsize=(8 * scale, 6 * scale))
        plt.xlabel("Category name")
        plt.ylabel("Number of scenes")
        # Sort bars by height
        sorted_category_to_scene_count = sorted(
            category_to_scene_count.items(), key=lambda x: -x[1]
        )
        plt.bar(
            [x[0] for x in sorted_category_to_scene_count],
            [category_to_total_count[x[0]] for x in sorted_category_to_scene_count],
        )
        plt.bar(
            [x[0] for x in sorted_category_to_scene_count],
            [x[1] for x in sorted_category_to_scene_count],
        )
        plt.xticks(rotation=90)

        # Save plot to a file
        plt.savefig(f"{val_split_dir_basename}_hist.png")


def scale_dict_values(dictionary, target_sum):
    original_sum = sum(dictionary.values())
    scaling_factor = target_sum / original_sum

    scaled_dict = {}
    scaled_sum = 0

    for key, value in dictionary.items():
        scaled_value = round(value * scaling_factor)
        scaled_dict[key] = scaled_value
        scaled_sum += scaled_value

    # Adjust for rounding errors
    if scaled_sum != target_sum:
        diff = target_sum - scaled_sum
        sorted_values = sorted(
            scaled_dict.items(),
            key=lambda x: abs(x[1] - (scaled_sum / len(scaled_dict))),
        )
        for i in range(abs(diff)):
            key = sorted_values[i][0]
            scaled_dict[key] += 1 if diff > 0 else -1

    return scaled_dict


def distribute_values(sum_dict, length_dict):
    result_dict = {}

    for key in sum_dict:
        value_sum = sum_dict[key]
        value_length = length_dict[key]

        quotient = value_sum // value_length
        remainder = value_sum % value_length

        values = [quotient] * value_length

        for i in range(remainder):
            values[i] += 1

        result_dict[key] = values

    return result_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("val_split_dir", type=str)
    args = parser.parse_args()

    main(args.val_split_dir)
