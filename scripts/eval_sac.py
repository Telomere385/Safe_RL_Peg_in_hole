"""可视化评估训练好的 SAC agent.

运行 (带 IsaacSim 窗口):
    cd ~/IsaacLab && ./isaaclab.sh -p ~/bimanual_peghole/scripts/eval_sac.py

远程运行:
    cd ~/IsaacLab && PUBLIC_IP=100.100.133.86 LIVESTREAM=1 ENABLE_CAMERAS=1 \
        ./isaaclab.sh -p ~/bimanual_peghole/scripts/eval_sac.py
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--agent_path", type=str,
                    default="/home/miao/bimanual_peghole/results/best_agent.msh")
parser.add_argument("--n_episodes", type=int, default=3)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mushroom_rl.core import Core, Agent
from envs.mushroom_wrapper import DualArmPegHoleMushroom
from envs.dual_arm_peg_hole_env import DualArmPegHoleEnvCfg


# ── 网络类定义 (必须与 train_sac.py 一致，Agent.load 反序列化需要) ──
class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features=256, **kwargs):
        super().__init__()
        n_input = input_shape[0]
        n_output = output_shape[0]
        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, n_output)
        nn.init.xavier_uniform_(self._h1.weight)
        nn.init.xavier_uniform_(self._h2.weight)
        nn.init.xavier_uniform_(self._out.weight)

    def forward(self, x, **kwargs):
        x = x.float()
        h = F.relu(self._h1(x))
        h = F.relu(self._h2(h))
        return self._out(h)


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features=256,
                 action_dim=None, **kwargs):
        super().__init__()
        n_input = input_shape[0]
        if action_dim is None:
            action_dim = 14
        n_output = output_shape[0]
        self._h1 = nn.Linear(n_input + action_dim, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, n_output)
        nn.init.xavier_uniform_(self._h1.weight)
        nn.init.xavier_uniform_(self._h2.weight)
        nn.init.xavier_uniform_(self._out.weight)

    def forward(self, state, action, **kwargs):
        state = state.float()
        action = action.float()
        x = torch.cat([state, action], dim=-1)
        h = F.relu(self._h1(x))
        h = F.relu(self._h2(h))
        return self._out(h).squeeze(-1)


def main():
    env_cfg = DualArmPegHoleEnvCfg()
    env_cfg.scene.num_envs = 1
    mdp = DualArmPegHoleMushroom(cfg=env_cfg, gamma=0.99)

    agent = Agent.load(args.agent_path)

    core = Core(agent, mdp)

    print(f"评估 {args.n_episodes} episodes (可视化)...")
    dataset = core.evaluate(n_episodes=args.n_episodes, render=True, quiet=False)

    J = dataset.discounted_return.mean()
    R = dataset.undiscounted_return.mean()
    print(f"\n结果: J(gamma)={J:.3f} | R={R:.3f}")

    mdp.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
