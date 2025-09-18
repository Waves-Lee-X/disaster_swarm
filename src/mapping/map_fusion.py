from pathlib import Path
import glob, numpy as np, open3d as o3d
from src.mapping.depth_to_pointcloud import depth_to_pointcloud
from src.mapping.pose_utils import get_cam_intrinsics_from_fov

DEPTH_DIR = Path("data/collected/depth")
OUT_PLY = Path("output/global_map.ply")
OUT_PLY.parent.mkdir(parents=True, exist_ok=True)

def main():
    fx, fy, cx, cy = get_cam_intrinsics_from_fov(640, 480, 90.0)
    npy_files = sorted(glob.glob(str(DEPTH_DIR / "*.npy")))[:400]  # 防止过大
    if not npy_files:
        print("No depth frames in data/collected/depth")
        return

    pc_all = o3d.geometry.PointCloud()
    for i, f in enumerate(npy_files):
        d = np.load(f)
        pcd = depth_to_pointcloud(d, fx, fy, cx, cy)
        # 简化演示：未使用外参，对应帧直接堆叠
        pc_all += pcd
        if (i+1) % 50 == 0:
            print(f"Accumulated {i+1} frames")

    pc_all = pc_all.voxel_down_sample(voxel_size=0.1)
    o3d.io.write_point_cloud(str(OUT_PLY), pc_all)
    print(f"Saved: {OUT_PLY}")

if __name__ == "__main__":
    main()

