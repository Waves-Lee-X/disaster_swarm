"""
SAREnv: Gym 环境（single-agent or vectorized multi-agent simplified wrapper）
设计：每 step 返回 obs, reward, done, info
Obs: 对每架 drone -> pos(3) + vel(3) + battery (1) + nearest victim rel pos (2) => shape (N,9)
Action: 每架 drone 的目标速度 (vx, vy, vz) clipped [-3,3]
说明：本文件提供可在 AirSim 中运行的真实仿真接口；训练时建议使用并行环境和简化模拟速度（time acceleration or skipping frames）。
"""

import gym
import numpy as np
import airsim
from gym import spaces
import time

class SAREnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, num_drones=5, num_victims=10, vehicle_prefix="Drone", max_episode_steps=500):
        super().__init__()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.num_drones = num_drones
        self.vehicles = [f"{vehicle_prefix}{i+1}" for i in range(num_drones)]
        self.num_victims = num_victims
        # state spaces
        # obs per agent: x,y,z, vx,vy,vz, battery, nearest_vx, nearest_vy
        self.obs_dim = 9
        self.action_dim = 3
        obs_low = np.tile(np.array([-1e3, -1e3, -200, -20, -20, -20, 0.0, -1e3, -1e3]), (num_drones,1))
        obs_high = np.tile(np.array([1e3, 1e3, 0, 20, 20, 20, 1.0, 1e3, 1e3]), (num_drones,1))
        self.observation_space = spaces.Box(low=obs_low.flatten(), high=obs_high.flatten(), dtype=np.float32)
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(num_drones, self.action_dim), dtype=np.float32)
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        self.reset()

    def reset(self):
        # spawn/position drones and victims
        self.victims = np.random.uniform(-40, 40, (self.num_victims, 2))
        # takeoff and position drones at origin scattered
        for i, v in enumerate(self.vehicles):
            try:
                self.client.enableApiControl(True, v)
                self.client.armDisarm(True, v)
                self.client.takeoffAsync(vehicle_name=v).join()
                x = np.random.uniform(-5, 5)
                y = np.random.uniform(-5, 5)
                self.client.moveToPositionAsync(x, y, -6, 3, vehicle_name=v).join()
            except Exception:
                pass
        # battery initial
        self.battery = {v: 1.0 for v in self.vehicles}
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for v in self.vehicles:
            s = self.client.getMultirotorState(vehicle_name=v)
            pos = s.kinematics_estimated.position
            vel = s.kinematics_estimated.linear_velocity
            pos_arr = np.array([pos.x_val, pos.y_val, pos.z_val])
            vel_arr = np.array([vel.x_val, vel.y_val, vel.z_val])
            # nearest victim
            dists = np.linalg.norm(self.victims - pos_arr[:2], axis=1)
            if len(dists) == 0:
                rel = np.array([0.0, 0.0])
            else:
                idx = np.argmin(dists)
                rel = self.victims[idx] - pos_arr[:2]
            obs_agent = np.concatenate([pos_arr, vel_arr, [self.battery[v]], rel])
            obs.append(obs_agent)
        obs = np.stack(obs)  # (N,9)
        return obs.flatten().astype(np.float32)

    def step(self, actions):
        # actions shape (N,3)
        for i, v in enumerate(self.vehicles):
            a = actions[i]
            # moveByVelocityAsync
            try:
                self.client.moveByVelocityAsync(float(a[0]), float(a[1]), float(a[2]), 0.5, vehicle_name=v)
            except Exception:
                pass
            # battery cost
            self.battery[v] = max(0.0, self.battery[v] - 0.001 * np.linalg.norm(a)**2)
        self.step_count += 1
        obs = self._get_obs()
        rewards = self._compute_rewards()
        done = self.step_count >= self.max_episode_steps
        info = {}
        # Gym expects (obs, reward, done, info). Since we flattened multi-agent, reward is per-agent; sum as scalar
        return obs, float(np.mean(rewards)), done, info

    def _compute_rewards(self):
        rewards = np.zeros(self.num_drones)
        # if any drone within 2m of a victim -> reward and remove victim
        positions = [self.client.getMultirotorState(vehicle_name=v).kinematics_estimated.position for v in self.vehicles]
        pos2 = np.array([[p.x_val, p.y_val] for p in positions])
        if self.victims.shape[0] == 0:
            return rewards
        for i in range(self.num_drones):
            dists = np.linalg.norm(self.victims - pos2[i], axis=1)
            if len(dists) > 0:
                minidx = np.argmin(dists)
                if dists[minidx] < 2.0:
                    rewards[i] += 20.0
                    # remove victim
                    self.victims = np.delete(self.victims, minidx, axis=0)
            # penalize high velocity (energy)
            vel = self.client.getMultirotorState(vehicle_name=self.vehicles[i]).kinematics_estimated.linear_velocity
            vnorm = np.linalg.norm([vel.x_val, vel.y_val, vel.z_val])
            rewards[i] -= 0.01 * (vnorm**2)
        # coverage bonus (mean distance to victims)
        if self.victims.shape[0] > 0:
            meand = np.mean([np.min(np.linalg.norm(self.victims - p2, axis=1)) for p2 in pos2])
            rewards += -0.001 * meand
        return rewards

    def render(self, mode='human'):
        pass

    def close(self):
        pass
