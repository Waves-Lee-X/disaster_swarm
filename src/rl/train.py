"""
训练脚本：使用 stable-baselines3 PPO 对 SAREnv 进行训练（单策略多 agent flattened obs）。
注意：本示例简化为 single policy 操作，将多 agent 的 obs flatten 作为单一 observation。
在复杂的多智能体设置下建议使用 PettingZoo + MARL 框架。
"""

import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os
import yaml
from src.rl.envs.sar_env import SAREnv

# load hyperparams
with open(os.path.join(os.path.dirname(__file__), "hyperparams.yaml"), "r") as f:
    H = yaml.safe_load(f)

def make_env():
    return SAREnv(num_drones=H['num_drones'], num_victims=H['num_victims'], max_episode_steps=H['max_episode_steps'])

def train(total_timesteps=H['total_timesteps'], model_save="models/sar_ppo"):
    env = make_env()
    os.makedirs("models", exist_ok=True)
    model = PPO('MlpPolicy', env, verbose=1, batch_size=H['batch_size'],
                n_steps=H['n_steps'], learning_rate=H['learning_rate'],
                ent_coef=H['ent_coef'], gamma=H['gamma'])
    checkpoint_callback = CheckpointCallback(save_freq=H['save_freq'], save_path='./models/', name_prefix='sarppo')
    eval_env = make_env()
    eval_callback = EvalCallback(eval_env, best_model_save_path='./models/best/', log_path='./models/eval/', eval_freq=H['eval_freq'], n_eval_episodes=5)
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
    model.save(model_save)
    env.close()

if __name__ == "__main__":
    train()
