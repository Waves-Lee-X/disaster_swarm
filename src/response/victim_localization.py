import numpy as np

def pixel_depth_to_world(u, v, depth, fx, fy, cx, cy, T_wc):
    """像素 + 深度 → 世界坐标（简单针孔+外参）"""
    z = float(depth[v, u])
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pt_c = np.array([x, y, z, 1.0], dtype=float)
    pt_w = T_wc @ pt_c
    return pt_w[:3]

# 多帧观测融合（示例平均）
def fuse_observations(pts_world: list):
    if not pts_world: return None
    arr = np.array(pts_world, dtype=float)
    return arr.mean(axis=0)
