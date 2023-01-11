import os

import imageio
import numpy as np

IMAGE_DIR = "data/images/objnav_dataset_gen/debug"
MAX_DIST = [0, 0, 200]  # Blue
NON_NAVIGABLE = [150, 150, 150]  # Grey
POINT_COLOR = [150, 150, 150]  # Grey
VIEW_POINT_COLOR = [0, 200, 0]  # Green
CENTER_POINT_COLOR = [200, 0, 0]  # Red


def plot_area(points, points_iou, object_center, object_name_id):
    max_coord = 1000  # int(np.max(points) * 100)
    image = np.zeros((max_coord, max_coord, 3), dtype=np.uint8)

    def mark_points(points, color):
        int_points = [
            (int(p[0] * 10 + max_coord / 2), int(p[2] * 10 + max_coord / 2))
            for p in points
        ]
        for p in int_points:
            image[p[0], p[1]] = color

    def iou_points(points, color):
        int_points = [
            (
                int(p[0] * 10 + max_coord / 2),
                int(p[2] * 10 + max_coord / 2),
                iou,
            )
            for iou, p, _ in points
        ]
        for p in int_points:
            if p[2] == -1:
                image[p[0], p[1]] = NON_NAVIGABLE
            elif p[2] == -0.5:
                image[p[0], p[1]] = MAX_DIST
            else:
                color = int((p[2] + 1.0) * 255)
                image[p[0], p[1]] = [color, color, color]

    iou_points(points, POINT_COLOR)
    mark_points(points_iou, VIEW_POINT_COLOR)
    mark_points(object_center, CENTER_POINT_COLOR)

    imageio.imsave(os.path.join(IMAGE_DIR, f"objnav_image_{object_name_id}.png"), image)
