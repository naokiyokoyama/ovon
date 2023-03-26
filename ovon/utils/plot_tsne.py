import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

from ovon.utils.utils import load_pickle


def plot_tsne(path, output_path):
    clip_embeddings = load_pickle(path)

    keys = list(clip_embeddings.keys())
    # for key in keys:
    #     if key not in [
    #         "chair",
    #         "bed",
    #         "plant",
    #         "toilet",
    #         "tv_monitor",
    #         "sofa",
    #     ]:
    #         del clip_embeddings[key]

    features = list(clip_embeddings.values())
    categories = list(clip_embeddings.keys())

    tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=500)
    tsne_results = tsne.fit_transform(features)

    df = pd.DataFrame(
        {
            "tsne-2d-one": tsne_results[:, 0],
            "tsne-2d-two": tsne_results[:, 1],
            "labels": categories,
        }
    )

    pallete_size = np.unique(categories).shape[0]

    # colors = ['#4f7ac9', '#e68752', '#6fc769', '#d06565', '#9470b0', '#886140', '#d783be', '#797979', '#d0b86d', '#87c4dd', '#4f7ac9', '#e68752', '#6fc769', '#d06565', '#9470b0']
    colors = sns.color_palette("hls", pallete_size)
    color_map = {}
    for i in range(len(categories)):
        color_map[categories[i]] = colors[i]

    print(color_map)

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        data=df,
        x="tsne-2d-one",
        y="tsne-2d-two",
        hue="labels",
        palette=color_map,
        legend="full",
        s=120,
    )

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    return tsne_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="path to the dataset"
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="path to the dataset"
    )

    args = parser.parse_args()

    plot_tsne(args.path, args.output_path)


if __name__ == "__main__":
    main()
