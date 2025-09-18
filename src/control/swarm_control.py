import airsim
import numpy as np
from typing import Dict, Tuple

class SwarmControl:
    def __init__(self, drones):
        self.drones = drones
        self.clients = {d: airsim.MultirotorClient() for d in drones}
        for c in self.clients.values():
            c.confirmConnection()

    def takeoff_all(self, timeout=15.0):
        for d, c in self.clients.items():
            c.enableApiControl(True, d)
            c.armDisarm(True, d)
            c.takeoffAsync(timeout_sec=float(timeout), vehicle_name=d).join()

    def land_all(self, timeout=15.0):
        for d, c in self.clients.items():
            try:
                c.landAsync(timeout_sec=float(timeout), vehicle_name=d).join()
            except Exception:
                pass
            c.armDisarm(False, d)
            c.enableApiControl(False, d)

    def move_to(self, drone: str, x: float, y: float, z: float, vel: float = 5.0, timeout: float = 10.0):
        c = self.clients[drone]
        c.moveToPositionAsync(float(x), float(y), float(z), float(vel),
                              vehicle_name=drone, timeout_sec=float(timeout)).join()

    def get_states(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        positions, velocities = {}, {}
        for d, c in self.clients.items():
            s = c.getMultirotorState(d)
            p = s.kinematics_estimated.position
            v = s.kinematics_estimated.linear_velocity
            positions[d] = np.array([p.x_val, p.y_val, p.z_val], dtype=float)
            velocities[d] = np.array([v.x_val, v.y_val, v.z_val], dtype=float)
        return positions, velocities
