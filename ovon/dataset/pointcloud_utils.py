from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d
import quaternion
from habitat_sim._ext.habitat_sim_bindings import BBox
from numpy import ndarray
from scipy import stats


def project_semantics_to_world(
    obs: List[Dict[str, ndarray]],
    camera_poses: List[Dict[str, ndarray]],
    hfovs: List[float],  # in radians
    object_id: int,
) -> List[ndarray]:
    """Projects depth values into a world point cloud. Pixels are only
    projected that match the provided object_id.
    """
    points = []
    if len(obs) == 0:
        return points

    for o, pose, hfov in zip(obs, camera_poses, hfovs):
        assert "depth_sensor" in o
        assert "color_sensor" in o
        assert hfov <= (2 * np.pi)

        H, W = o["depth_sensor"].shape
        assert W == H  # need a small refactor if not

        K = np.array(
            [
                [1 / np.tan(hfov / 2.0), 0.0, 0.0, 0.0],
                [0.0, 1 / np.tan(hfov / 2.0), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, W))
        depth = o["depth_sensor"][None, :, :]
        xs = xs.reshape(1, W, W)
        ys = ys.reshape(1, W, W)
        xys = np.vstack((xs * depth, ys * depth, -depth, np.ones(depth.shape)))
        xys = xys.reshape(4, -1)
        xy_c0 = np.matmul(np.linalg.inv(K), xys)

        rotation = quaternion.as_rotation_matrix(pose.rotation)
        T_world_camera = np.eye(4)
        T_world_camera[0:3, 0:3] = rotation
        T_world_camera[0:3, 3] = pose.position

        xyz = np.matmul(T_world_camera, xy_c0)
        xyz = xyz / xyz[3, :]

        # filter on matching semantic labels.
        sem_px_match = o["semantic_sensor"].reshape(-1) == object_id
        points.append(xyz[:, sem_px_match])

    return points


def prune_to_largest_cluster(
    pcd: open3d.geometry.PointCloud, eps: float, min_points: int = 4
) -> open3d.geometry.PointCloud:
    """Extract the largest DBSCAN cluster from the point cloud. This removes
    label splatter. If no clusters are found, it returns None.
    """
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    mode = stats.mode(labels).mode

    if mode.shape[0] == 0:
        return None

    points = np.array(pcd.points)[labels == mode.item()]

    if points.shape[0] < min_points:
        return None

    return open3d.geometry.PointCloud(
        points=open3d.utility.Vector3dVector(points)
    )


def points_to_surface_area(
    points: ndarray,
    aabb: BBox,
    voxel_size: float,
    dbscan_slack: float,
    export_pcd_name: Optional[str] = None,
) -> Tuple[float, open3d.t.geometry.TriangleMesh]:
    """Computes the convex hull of the set of 4D points (4, n). Points are cropped
    to a bounding box, voxelized, and clustered with DBSCAN.
    """
    assert points.shape[0] == 4

    # must have at least 4 points
    if points.shape[1] < 4:
        return 0.0, None

    # bounding boxes could be undefined
    if aabb.sizes.sum() == 0.0:
        return 0.0, None

    x = aabb.sizes / 2.0
    aabb_o3d = open3d.geometry.AxisAlignedBoundingBox.create_from_points(
        open3d.utility.Vector3dVector(
            np.stack([aabb.center - x, aabb.center + x], axis=0)
        )
    )

    pcd = (
        open3d.geometry.PointCloud(
            points=open3d.utility.Vector3dVector((points[:3] / points[-1]).T)
        )
        .crop(aabb_o3d)
        .voxel_down_sample(voxel_size=voxel_size)
    )

    pcd = prune_to_largest_cluster(pcd, eps=voxel_size + dbscan_slack)
    if pcd is None:
        return 0.0, None

    if export_pcd_name is not None:
        open3d.io.write_point_cloud(f"{export_pcd_name}.ply", pcd)

    mesh = pcd.compute_convex_hull()[0]
    return float(mesh.get_surface_area()), mesh


def points_to_surface_area_in_scene(
    pts: ndarray,
    o3d_obj_scene: open3d.t.geometry.RaycastingScene,
    voxel_size: float,
    dbscan_slack: float,
) -> float:
    """Computes the convex hull of the set of 4D points. Points are voxelized
    and pruned to lie inside the provided Open3D scene. They are then
    clustered with DBSCAN.
    """
    pcd = open3d.geometry.PointCloud(
        points=open3d.utility.Vector3dVector((pts[:3] / pts[-1]).T)
    ).voxel_down_sample(voxel_size=voxel_size)

    pts_3d = np.asarray(pcd.points).astype(np.float32)
    d = np.asarray(o3d_obj_scene.compute_signed_distance(pts_3d))
    pcd.points = open3d.utility.Vector3dVector(pts_3d[d <= 0.0])
    if np.array(pcd.points).shape[0] == 0:
        return 0.0

    pcd = prune_to_largest_cluster(pcd, eps=voxel_size + dbscan_slack)
    if pcd is None:
        return 0.0

    return pcd.compute_convex_hull()[0].get_surface_area()


def mesh_to_scene(
    mesh: open3d.t.geometry.TriangleMesh,
) -> open3d.t.geometry.RaycastingScene:
    scene = open3d.t.geometry.RaycastingScene()
    scene.add_triangles(open3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene
