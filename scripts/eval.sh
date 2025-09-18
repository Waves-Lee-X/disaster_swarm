#!/usr/bin/env bash
# 在训练好的模型上做评估
python - <<'PY'
from stable_baselines3 import PPO
from src.rl.envs.sar_env import SAREnv
model = PPO.load("models/sar_ppo.zip")
env = SAREnv()
for ep in range(5):
    obs = env.reset()
    done = False
    tot = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _ = env.step(action.reshape((env.num_drones, env.action_dim)))
        tot += r
    print("Episode", ep, "reward", tot)
PY
