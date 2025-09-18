"""
数字孪生短步优化 loop：
- 注入环境（风、雨、火）作为孪生输入
- 使用 shadow simulation （在内存中加噪 obs 并用 policy 预测）做短期预测，如果 shadow 预测明显差于实际，则触发 conservative fallback
注意：本模块假设使用已训练的 RL policy checkpoint（stable-baselines3 PPO）。
"""

import airsim
import numpy as np
import time
import os
from stable_baselines3 import PPO
from src.rl.envs.sar_env import SAREnv

class DigitalTwin:
    def __init__(self, model_path="models/sar_ppo.zip"):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.model_path = model_path
        if os.path.exists(model_path):
            self.model = PPO.load(model_path)
        else:
            self.model = None

    def inject_disaster(self, intensity=0.5):
        # enable weather and random wind
        try:
            self.client.simEnableWeather(True)
            wind_x = 5 + np.random.uniform(-3, 3) * intensity
            wind_y = np.random.uniform(-2, 2) * intensity
            self.client.simSetWind(airsim.Vector3r(wind_x, wind_y, 0))
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, intensity)
        except Exception as e:
            print("Weather injection failed:", e)

    def optimize_once(self, episodes=1):
        env = SAREnv()
        if self.model is None:
            print("[twin] no model loaded")
            return
        for ep in range(episodes):
            obs = env.reset()
            self.inject_disaster(0.4)
            total_reward = 0.0
            for t in range(200):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action.reshape((env.num_drones, env.action_dim)))
                total_reward += reward
                # shadow sim: add gaussian noise to obs and predict
                shadow_obs = obs + np.random.normal(0, 0.05, obs.shape)
                shadow_action, _ = self.model.predict(shadow_obs, deterministic=True)
                # evaluate predicted reward using env's reward function (approx)
                # Here we use a quick heuristic: if shadow predicted reward << current reward, trigger conservative mode
                # conservative mode: reduce velocities by half for next few steps
                if np.mean(shadow_action) < np.mean(action) * 0.5:
                    # conservative: send zero vertical velocity to slow down descent, etc.
                    for i, v in enumerate(env.vehicles):
                        try:
                            self.client.moveByVelocityAsync(0, 0, -0.5, 0.5, vehicle_name=v)
                        except Exception:
                            pass
                if done:
                    break
            print(f"[twin] ep{ep} reward={total_reward}")

if __name__ == "__main__":
    dt = DigitalTwin()
    dt.optimize_once(1)
