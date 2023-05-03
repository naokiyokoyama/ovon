import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ovon.utils.utils import count_episodes


def plot_statistics(path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _, categories = count_episodes(path)

    categories = {
        "objects": list(categories.keys()),
        "frequency": list(categories.values()),
    }

    df = pd.DataFrame.from_dict(categories)
    df.sort_values(by="frequency", inplace=True, ascending=False)
    print(df.columns)

    fig, axs = plt.subplots(1, 1, figsize=(8, 50))

    plot = sns.barplot(data=df, x="frequency", y="objects", ax=axs)

    fig.savefig(output_path, dpi=100, bbox_inches="tight", pad_inches=0.1, transparent=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/datasets/ovon/hm3d/v4_stretch/val_seen/content/")
    parser.add_argument("--output-path", type=str, default="val_unseen.png")
    args = parser.parse_args()

    plot_statistics(args.path, args.output_path)
