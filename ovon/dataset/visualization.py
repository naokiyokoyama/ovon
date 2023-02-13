import os
from typing import Dict, List

import cv2
import numpy as np
from habitat_sim.utils.common import d3_40_colors_rgb
from numpy import ndarray
from PIL import Image

IMAGE_DIR = "data/images/ovon_dataset_gen/debug"
MAX_DIST = [0, 0, 200]  # Blue
NON_NAVIGABLE = [150, 150, 150]  # Grey
POINT_COLOR = [150, 150, 150]  # Grey
VIEW_POINT_COLOR = [0, 200, 0]  # Green
CENTER_POINT_COLOR = [200, 0, 0]  # Red


def obs_to_frame(obs):
    rgb = cv2.cvtColor(obs["color_sensor"], cv2.COLOR_BGRA2RGB)
    dmap = (obs["depth_sensor"] / 10 * 255).astype(np.uint8)
    dmap_colored = cv2.applyColorMap(dmap, cv2.COLORMAP_VIRIDIS)

    semantic_obs = obs["semantic_sensor"]
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    return np.concatenate([rgb, dmap_colored, semantic_img], axis=1)


def save_candidate_imgs(
    obs: List[Dict[str, ndarray]],
    frame_covs: List[float],
    save_to: str,
) -> None:
    """Write coverage stats on candidate images and save all to disk"""
    os.makedirs(save_to, exist_ok=True)

    font = cv2.FONT_HERSHEY_SIMPLEX
    loc = (50, 50)
    lt = cv2.LINE_AA
    w = (255, 255, 255)
    b = (0, 0, 0)

    for i, (o, fc) in enumerate(zip(obs, frame_covs)):
        txt = f"Frame coverage: {round(fc, 2)}"
        img = obs_to_frame(o)
        img = cv2.putText(img, txt, loc, font, 1, b, 4, lt)
        img = cv2.putText(img, txt, loc, font, 1, w, 2, lt)
        cv2.imwrite(os.path.join(save_to, f"candidate_{i}.png"), img)
