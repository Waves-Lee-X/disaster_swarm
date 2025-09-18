#!/usr/bin/env bash
# 运行 RL 训练（Linux / WSL）
source ~/anaconda3/etc/profile.d/conda.sh
conda activate disaster_swarm
python src/rl/train.py
