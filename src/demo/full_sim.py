import time
from src.control.swarm_control import SwarmController
from src.perception.detector import SARDetection
from src.allocation.auction import auction_task
import airsim
import numpy as np

def main():
    # 初始化控制与感知
    swarm = SwarmController(num_drones=5, vehicle_prefix="Drone", default_alt=-6)
    detectors = {v: SARDetection(model_path="yolov8n.pt", vehicle=v) for v in swarm.vehicles}
    client = airsim.MultirotorClient()
    client.confirmConnection()
    try:
        # 主循环：boids + 简单 detection -> 任务分配
        for t in range(200):
            # boids step
            swarm.step_boids(dt=0.5)
            # collect detections and generate tasks
            tasks = []
            task_id = 0
            for v, det in detectors.items():
                ds = det.detect()
                for d in ds:
                    # convert detection center + depth to world coords (approx)
                    cx, cy = d['center']
                    depth = d['depth'] if d['depth'] is not None else 10.0
                    # approximate back-projection: this is placeholder; for accurate transform use camera intrinsics and pose
                    # we create a task at estimated (x,y) offset from vehicle
                    s = client.getMultirotorState(vehicle_name=v)
                    pos = s.kinematics_estimated.position
                    # randomize small offset
                    tx = pos.x_val + np.random.uniform(-2,2)
                    ty = pos.y_val + np.random.uniform(-2,2)
                    tz = pos.z_val
                    tasks.append({'id': f"{t}_{task_id}", 'x': tx, 'y': ty, 'z': tz, 'type': 'search'})
                    task_id += 1
            if len(tasks) > 0:
                assignments = auction_task(client, tasks, swarm.vehicles)
                print(f"[demo] assignments: {assignments}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        swarm.shutdown()
        print("Demo finished.")

if __name__ == "__main__":
    main()
