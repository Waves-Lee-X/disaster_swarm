import airsim
import numpy as np
import cv2
import time
import math
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import List, Tuple

# ===============================
# 基本配置
# ===============================
DRONES = ["Drone1", "Drone2", "Drone3", "Drone4", "Drone5"]

MAP_AREA = [-20.0, 20.0, -20.0, 20.0, 0, 20.0]

Z_LAYERS_LOW  = [0, 5.0]
Z_LAYERS_MID  = [5.0, 15.0]
Z_LAYERS_HIGH = [15.0, 20.0]
Z_LAYERS: List[float] = Z_LAYERS_LOW + Z_LAYERS_MID + Z_LAYERS_HIGH

XY_STEP = 12.0
CRUISE_VELOCITY = 5.0
SAFE_TAKEOFF_Z = 2.0
CAPTURE_DELAY = 0.35
SAFE_FORWARD_DIST = 3.0
WAYPOINT_TIMEOUT = 15.0

# ===== 新角度（避免机翼） =====
ANGLES = [
    (-25,   0), (-25,  90), (-25, -90), (-25, 180),
    (-45,  30), (-45, -30),
    ( 10,   0), ( 10,  90), ( 10, -90)
]

# stall detection
MAX_STALL_TIME = 5.0
STALL_VEL_THRESH = 0.3

# ===============================
# 输出目录
# ===============================
ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "data/collected"
RGB_DIR   = OUT_DIR / "rgb"
DEPTH_DIR = OUT_DIR / "depth"
LABEL_DIR = OUT_DIR / "labels"
META_DIR  = OUT_DIR / "meta"
for d in [RGB_DIR, DEPTH_DIR, LABEL_DIR, META_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 日志
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s")
log = logging.getLogger("dataset_gen")

# ===============================
# 工具函数
# ===============================
def ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def boustrophedon_points_for_area(area: List[float],
                                  z_layers: List[float],
                                  step: float) -> List[Tuple[float,float,float]]:
    xmin, xmax, ymin, ymax, _, _ = area
    xs = np.arange(xmin, xmax + step, step, dtype=float)
    ys = np.arange(ymin, ymax + step, step, dtype=float)

    points = []
    for z in z_layers:
        reverse = False
        for yi in ys:
            if not reverse:
                for xi in xs:
                    points.append((float(xi), float(yi), float(z)))
            else:
                for xi in xs[::-1]:
                    points.append((float(xi), float(yi), float(z)))
            reverse = not reverse
    return points

def distribute_points(points: List[Tuple[float,float,float]], num_drones: int) -> List[List[Tuple[float,float,float]]]:
    buckets = [[] for _ in range(num_drones)]
    for i, p in enumerate(points):
        buckets[i % num_drones].append(p)
    return buckets

def set_camera_pose(client: airsim.MultirotorClient, drone: str, pitch_deg: float, yaw_deg: float):
    pose = airsim.Pose(
        airsim.Vector3r(0, 0, 0),
        airsim.to_quaternion(math.radians(pitch_deg), 0.0, math.radians(yaw_deg))
    )
    client.simSetCameraPose("0", pose, vehicle_name=drone)

def get_rgb_depth(client: airsim.MultirotorClient, drone: str):
    resp = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)
    ], vehicle_name=drone)
    if len(resp) < 2 or resp[0].width == 0:
        return None, None
    img = np.frombuffer(resp[0].image_data_uint8, dtype=np.uint8)\
            .reshape(resp[0].height, resp[0].width, 3)
    depth = np.array(resp[1].image_data_float, dtype=np.float32)\
            .reshape(resp[1].height, resp[1].width)
    return img, depth

def save_sample(drone: str, tag: str, img: np.ndarray, depth: np.ndarray,
                pitch: float, yaw: float, front_min_d: float = None):
    name = f"{drone}_{ts()}_{tag}"
    cv2.imwrite(str((RGB_DIR / f"{name}.png")), img[:, :, ::-1])
    np.save(DEPTH_DIR / f"{name}.npy", depth)
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(DEPTH_DIR / f"{name}.png"), depth_vis)
    (LABEL_DIR / f"{name}.txt").write_text("")
    meta = {
        "drone": drone,
        "pitch_deg": pitch,
        "yaw_deg": yaw,
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "front_min_d": None if front_min_d is None else float(front_min_d)
    }
    (META_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2))

# ===============================
# 多角度拍摄
# ===============================
def capture_at_waypoint(client: airsim.MultirotorClient, drone: str, tag: str, front_min_d: float = None):
    for (pitch, yaw) in ANGLES:
        set_camera_pose(client, drone, pitch, yaw)
        img, depth = get_rgb_depth(client, drone)
        if img is not None and depth is not None:
            save_sample(drone, tag + f"_p{int(pitch)}_y{int(yaw)}", img, depth, pitch, yaw, front_min_d)

# ===============================
# 无人机线程
# ===============================
def drone_worker(drone: str, points: List[Tuple[float,float,float]]):
    client = airsim.MultirotorClient()
    client.confirmConnection()
    log.info(f"[{drone}] connected.")
    client.enableApiControl(True, drone)
    client.armDisarm(True, drone)

    client.takeoffAsync(vehicle_name=drone).join()
    client.moveToZAsync(float(SAFE_TAKEOFF_Z), CRUISE_VELOCITY, vehicle_name=drone).join()

    captured_sets = 0
    skipped_points = 0

    try:
        for i, (x, y, z) in enumerate(points):
            t0 = time.time()
            stall_t0 = None

            log.info(f"[{drone}] wp#{i} -> target=({x:.1f},{y:.1f},{z:.1f})")

            client.moveToPositionAsync(float(x), float(y), float(z),
                                       CRUISE_VELOCITY, vehicle_name=drone)

            while True:
                time.sleep(0.5)
                state = client.getMultirotorState(vehicle_name=drone)
                pos = state.kinematics_estimated.position
                vel = state.kinematics_estimated.linear_velocity
                speed = np.linalg.norm([vel.x_val, vel.y_val, vel.z_val])

                if abs(pos.x_val - x) < 1.0 and abs(pos.y_val - y) < 1.0 and abs(pos.z_val - z) < 1.0:
                    log.info(f"[{drone}] wp#{i} reached.")
                    break

                if speed < STALL_VEL_THRESH:
                    if stall_t0 is None:
                        stall_t0 = time.time()
                    elif time.time() - stall_t0 > MAX_STALL_TIME:
                        log.warning(f"[{drone}] wp#{i} stalled, capturing and skipping...")
                        capture_at_waypoint(client, drone, f"{i:05d}_stalled")
                        skipped_points += 1
                        break
                else:
                    stall_t0 = None

                if time.time() - t0 > WAYPOINT_TIMEOUT:
                    log.warning(f"[{drone}] wp#{i} timeout, capturing and skipping...")
                    capture_at_waypoint(client, drone, f"{i:05d}_timeout")
                    skipped_points += 1
                    break

            time.sleep(CAPTURE_DELAY)
            tag = f"{i:05d}_x{int(x)}_y{int(y)}_z{int(z)}"
            capture_at_waypoint(client, drone, tag)
            captured_sets += 1

            if i % 5 == 0:
                progress = (i+1)/len(points)*100
                log.info(f"[{drone}] heartbeat: wp#{i}/{len(points)}, {progress:.1f}%, captured={captured_sets}, skipped={skipped_points}")

    finally:
        try:
            client.landAsync(vehicle_name=drone).join()
        except Exception:
            pass
        client.armDisarm(False, drone)
        client.enableApiControl(False, drone)
        log.info(f"[{drone}] finished. total={captured_sets}, skipped={skipped_points}")

# ===============================
# 主程序
# ===============================
def main():
    start = time.time()
    all_points = boustrophedon_points_for_area(MAP_AREA, Z_LAYERS, XY_STEP)
    log.info(f"[assign] total waypoints={len(all_points)}")

    buckets = distribute_points(all_points, len(DRONES))
    for dn, pts in zip(DRONES, buckets):
        log.info(f"[assign] {dn} -> {len(pts)} pts")

    with ThreadPoolExecutor(max_workers=len(DRONES)) as ex:
        futs = [ex.submit(drone_worker, dn, pts) for dn, pts in zip(DRONES, buckets)]
        for f in as_completed(futs):
            f.result()

    elapsed = time.time() - start
    log.info(f"Mission finished in {elapsed:.1f} seconds.")

if __name__ == "__main__":
    main()


