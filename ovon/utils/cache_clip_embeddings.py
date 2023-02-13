import argparse

import clip
import numpy as np
import torch
from ovon.utils.utils import load_json, save_pickle, write_json

PROMPT = "Find and goto {category}"


def tokenize_and_batch(clip, goal_categories):
    tokens = []
    for category in goal_categories:
        prompt = PROMPT.format(category=category)
        tokens.append(clip.tokenize(prompt, context_length=77).numpy())
    return torch.tensor(np.array(tokens)).cuda()


def save_to_disk(text_embedding, goal_categories, output_path):
    output = {}
    for goal_category, embedding in zip(goal_categories, text_embedding):
        output[goal_category] = embedding.detach().cpu().numpy()
    save_pickle(output, output_path)


def cache_embeddings(categories_file_path, output_path, clip_model="RN50"):
    model, _ = clip.load(clip_model)
    goal_categories = load_json(categories_file_path)

    batch = tokenize_and_batch(clip, goal_categories)

    with torch.no_grad():
        print(batch.shape)
        text_embedding = model.encode_text(batch.flatten(0, 1)).float()
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    save_to_disk(text_embedding, goal_categories, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories-file-path",
        type=str,
        required=True,
        help="file path of OVON categories",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="output path of clip features",
    )
    args = parser.parse_args()
    cache_embeddings(args.categories_file_path, args.output_path)
