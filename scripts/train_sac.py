"""SAC 训练脚本 — 双臂末端靠近 (mushroom-rl + VectorCore).

运行:
    conda activate safe_rl
    python scripts/train_sac.py                 # 单 env, 无头, wandb 开启
    python scripts/train_sac.py --num_envs 16   # 16 并行 env
    python scripts/train_sac.py --render        # 打开 IsaacSim 窗口
    python scripts/train_sac.py --no_wandb      # 关闭 wandb 日志
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--render", action="store_true", help="打开 IsaacSim 窗口")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--n_steps_per_epoch", type=int, default=1000)
    p.add_argument("--n_steps_per_fit", type=int, default=64,
                   help="每次 fit 的 env-step 数. 必须 >= num_envs (mushroom "
                        "vectorized dataset 要求 ceil(n_steps_per_fit/num_envs) >= 1)")
    p.add_argument("--n_eval_episodes", type=int, default=3)
    p.add_argument("--wandb_project", type=str, default="bimanual_peghole")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true", help="关闭 wandb 日志")
    return p.parse_args()


# --------------------------------------------------------------------------
# 网络
# --------------------------------------------------------------------------
N_FEATURES = 256


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features=N_FEATURES, **_):
        super().__init__()
        self._h1 = nn.Linear(input_shape[0], n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, output_shape[0])
        for l in (self._h1, self._h2, self._out):
            nn.init.xavier_uniform_(l.weight)

    def forward(self, x, **_):
        h = F.relu(self._h1(x.float()))
        h = F.relu(self._h2(h))
        return self._out(h)


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features=N_FEATURES, action_dim=14, **_):
        super().__init__()
        self._h1 = nn.Linear(input_shape[0] + action_dim, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, output_shape[0])
        for l in (self._h1, self._h2, self._out):
            nn.init.xavier_uniform_(l.weight)

    def forward(self, state, action, **_):
        h = F.relu(self._h1(torch.cat([state.float(), action.float()], dim=-1)))
        h = F.relu(self._h2(h))
        return self._out(h).squeeze(-1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── 创建环境 (内部启动 IsaacSim) ──
    from envs import DualArmPegHoleEnv

    assert args.n_steps_per_fit >= args.num_envs, (
        f"n_steps_per_fit ({args.n_steps_per_fit}) 必须 >= num_envs ({args.num_envs})"
    )

    mdp = DualArmPegHoleEnv(num_envs=args.num_envs, headless=not args.render)
    mdp.seed(args.seed)

    # ── mushroom_rl 模块在 IsaacSim 启动后导入 (避免 carb 冲突) ──
    from mushroom_rl.algorithms.actor_critic import SAC
    from mushroom_rl.core import VectorCore, Logger

    obs_dim = mdp.info.observation_space.shape[0]
    act_dim = mdp.info.action_space.shape[0]

    actor_params = dict(network=ActorNetwork, input_shape=(obs_dim,),
                        output_shape=(act_dim,), n_features=N_FEATURES)
    actor_optimizer = {"class": optim.Adam, "params": {"lr": 3e-4}}
    critic_params = dict(network=CriticNetwork, input_shape=(obs_dim,),
                         output_shape=(1,), n_features=N_FEATURES, action_dim=act_dim,
                         optimizer={"class": optim.Adam, "params": {"lr": 3e-4}},
                         loss=F.mse_loss)

    agent = SAC(
        mdp_info=mdp.info,
        actor_mu_params=actor_params,
        actor_sigma_params=actor_params,
        actor_optimizer=actor_optimizer,
        critic_params=critic_params,
        batch_size=256,
        initial_replay_size=1000,
        max_replay_size=100_000,
        warmup_transitions=1000,
        tau=0.005,
        lr_alpha=3e-4,
    )

    core = VectorCore(agent, mdp)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    logger = Logger("SAC", results_dir=str(results_dir))
    logger.strong_line()
    logger.info(f"obs_dim={obs_dim}  act_dim={act_dim}  horizon={mdp.info.horizon}")

    # ── wandb 初始化 (可选) ──
    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={**vars(args), "algo": "SAC", "n_features": N_FEATURES,
                    "obs_dim": obs_dim, "act_dim": act_dim,
                    "horizon": mdp.info.horizon, "gamma": mdp.info.gamma},
            dir=str(results_dir),
        )
        logger.info(f"wandb run: {wandb_run.url}")

    # ── 1. 初始 replay 填充 ──
    logger.info("填充 replay buffer ...")
    core.learn(n_steps=1000, n_steps_per_fit=1000)

    # ── 2. 训练主循环 ──
    best_J = -np.inf
    total_env_steps = 1000 * args.num_envs  # 初始填充后累计的 env-step
    for epoch in range(args.n_epochs):
        core.learn(n_steps=args.n_steps_per_epoch,
                   n_steps_per_fit=args.n_steps_per_fit, quiet=True)
        total_env_steps += args.n_steps_per_epoch * args.num_envs

        dataset = core.evaluate(n_episodes=args.n_eval_episodes, quiet=True)
        J = torch.mean(dataset.discounted_return).item()
        R = torch.mean(dataset.undiscounted_return).item()
        ep_len = len(dataset) / max(args.n_eval_episodes, 1)

        if J > best_J:
            best_J = J
            agent.save(str(results_dir / "best_agent.msh"))

        logger.epoch_info(epoch + 1, J=J, R=R, best_J=best_J)
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "env_steps": total_env_steps,
                "J": J, "R": R, "best_J": best_J,
                "eval_ep_len": ep_len,
                "alpha": float(agent._alpha.item()) if hasattr(agent, "_alpha") else None,
            }, step=epoch + 1)

    logger.info(f"训练完成. best J = {best_J:.3f}")
    if wandb_run is not None:
        wandb_run.summary["best_J"] = best_J
        wandb_run.finish()
    mdp.stop()


if __name__ == "__main__":
    main()
