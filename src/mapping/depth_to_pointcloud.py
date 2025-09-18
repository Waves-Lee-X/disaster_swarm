import numpy as np
import open3d as o3d

def depth_to_pointcloud(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    return pcd

