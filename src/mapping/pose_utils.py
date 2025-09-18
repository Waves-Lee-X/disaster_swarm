import numpy as np
import airsim

def get_cam_intrinsics_from_fov(width=640, height=480, fov_deg=90.0):
    fx = fy = (width/2) / np.tan(np.deg2rad(fov_deg)/2)
    cx, cy = width/2, height/2
    return fx, fy, cx, cy

def pose_to_matrix(pose: airsim.Pose):
    """AirSim Pose -> 4x4 世界坐标变换矩阵（简化示例）"""
    q = pose.orientation
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    R = quat_to_rotm(np.array([w,x,y,z]))
    t = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def quat_to_rotm(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ], dtype=float)
