"""SAC 训练脚本 — 双臂末端靠近任务 (pipeline 验证).

运行 (headless):
    cd ~/IsaacLab && ./isaaclab.sh -p ~/bimanual_peghole/scripts/train_sac.py --headless

带可视化:
    cd ~/IsaacLab && ./isaaclab.sh -p ~/bimanual_peghole/scripts/train_sac.py
"""

# ── IsaacLab 启动 (必须在所有 isaaclab import 之前) ──
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# ── 标准库 ──
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── 项目路径 ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── MushroomRL 2.0 ──
from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core

# ── 自定义环境 wrapper ──
from envs.mushroom_wrapper import DualArmPegHoleMushroom
from envs.dual_arm_peg_hole_env import DualArmPegHoleEnvCfg

# ---------------------------------------------------------------------------
# 超参数
# ---------------------------------------------------------------------------
N_EPOCHS = 100
N_STEPS_PER_EPOCH = 1000       # 10 episodes worth (horizon=100)
N_STEPS_PER_FIT = 5            # every 5 env steps do one gradient update
N_EVAL_EPISODES = 3
BATCH_SIZE = 256
INITIAL_REPLAY_SIZE = 1000     # 10 episodes of random exploration
MAX_REPLAY_SIZE = 100000
WARMUP_TRANSITIONS = 1000
TAU = 0.005
LR_ALPHA = 3e-4
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
N_FEATURES = 256
GAMMA = 0.99


# ---------------------------------------------------------------------------
# 网络定义 — MushroomRL 2.0 要求 network 是一个类,
# 构造签名: __init__(self, input_shape, output_shape, **params)
# ---------------------------------------------------------------------------
class ActorNetwork(nn.Module):
    """Actor 网络 (mu 或 sigma 各用一个)."""

    def __init__(self, input_shape, output_shape, n_features=N_FEATURES, **kwargs):
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
    """Critic 网络 Q(s, a).

    MushroomRL 2.0 的 TorchApproximator 会将 (state, action) 作为
    两个独立的位置参数传入 forward, 所以 input_shape 只需要是 (obs_dim,).
    """

    def __init__(self, input_shape, output_shape, n_features=N_FEATURES,
                 action_dim=None, **kwargs):
        super().__init__()
        n_input = input_shape[0]
        if action_dim is None:
            action_dim = 14  # fallback
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("SAC Training — 双臂末端靠近任务 (Pipeline 验证)")
    print("=" * 60)

    # ── 1. 创建环境 ──
    env_cfg = DualArmPegHoleEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    mdp = DualArmPegHoleMushroom(cfg=env_cfg, gamma=GAMMA)

    obs_dim = mdp.info.observation_space.shape[0]  # 37
    act_dim = mdp.info.action_space.shape[0]        # 14

    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim:      {act_dim}")
    print(f"  Horizon:         {mdp.info.horizon}")
    print(f"  Gamma:           {mdp.info.gamma}")

    # ── 2. 构建 SAC agent ──
    actor_mu_params = dict(
        network=ActorNetwork,
        input_shape=(obs_dim,),
        output_shape=(act_dim,),
        n_features=N_FEATURES,
    )

    actor_sigma_params = dict(
        network=ActorNetwork,
        input_shape=(obs_dim,),
        output_shape=(act_dim,),
        n_features=N_FEATURES,
    )

    actor_optimizer = {
        "class": optim.Adam,
        "params": {"lr": LR_ACTOR},
    }

    critic_params = dict(
        network=CriticNetwork,
        input_shape=(obs_dim,),
        output_shape=(1,),
        n_features=N_FEATURES,
        action_dim=act_dim,
        optimizer={
            "class": optim.Adam,
            "params": {"lr": LR_CRITIC},
        },
        loss=F.mse_loss,
    )

    agent = SAC(
        mdp_info=mdp.info,
        actor_mu_params=actor_mu_params,
        actor_sigma_params=actor_sigma_params,
        actor_optimizer=actor_optimizer,
        critic_params=critic_params,
        batch_size=BATCH_SIZE,
        initial_replay_size=INITIAL_REPLAY_SIZE,
        max_replay_size=MAX_REPLAY_SIZE,
        warmup_transitions=WARMUP_TRANSITIONS,
        tau=TAU,
        lr_alpha=LR_ALPHA,
    )

    # ── 3. 创建训练 Core ──
    core = Core(agent, mdp)

    # ── 4. 初始 replay buffer 填充 ──
    print(f"\n填充 replay buffer ({INITIAL_REPLAY_SIZE} steps)...")
    core.learn(n_steps=INITIAL_REPLAY_SIZE, n_steps_per_fit=INITIAL_REPLAY_SIZE)
    print("  Replay buffer 填充完成.")

    # ── 5. 训练循环 ──
    print(f"\n开始训练: {N_EPOCHS} epochs x {N_STEPS_PER_EPOCH} steps")
    print("-" * 60)

    best_J = -np.inf
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    for epoch in range(N_EPOCHS):
        # 训练
        core.learn(
            n_steps=N_STEPS_PER_EPOCH,
            n_steps_per_fit=N_STEPS_PER_FIT,
            quiet=True,
        )

        # 评估
        dataset = core.evaluate(n_episodes=N_EVAL_EPISODES, quiet=True)
        J = dataset.discounted_return.mean()
        R = dataset.undiscounted_return.mean()

        # 保存最优
        if J > best_J:
            best_J = J
            agent.save(str(results_dir / "best_agent.msh"))

        print(f"  Epoch {epoch + 1:3d}/{N_EPOCHS} | "
              f"J(gamma)={J:8.3f} | R={R:8.3f} | best_J={best_J:8.3f}")

    # ── 6. 结束 ──
    print("\n" + "=" * 60)
    print(f"训练完成! Best J(gamma) = {best_J:.3f}")
    print(f"模型保存于: {results_dir / 'best_agent.msh'}")
    print("=" * 60)

    mdp.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
