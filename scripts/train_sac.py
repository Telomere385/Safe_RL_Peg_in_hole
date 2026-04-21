"""SAC 训练 — 双臂末端靠近 (mushroom-rl + VectorCore).

运行:
    conda activate safe_rl
    python scripts/train_sac.py                        # 16 env, 400 epoch
    python scripts/train_sac.py --utd 16               # UTD=1 (SAC 标准)
    python scripts/train_sac.py --render               # 打开 IsaacSim 窗口
    python scripts/train_sac.py --no_wandb             # 关闭 wandb

注意: num_envs=1 会触发 IsaacSim cloner 的 `*` pattern 失败 → 至少 2.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from networks import ActorNetwork, CriticNetwork


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--render", action="store_true", help="打开 IsaacSim 窗口")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_epochs", type=int, default=400)
    p.add_argument("--n_steps_per_epoch", type=int, default=1000)
    p.add_argument("--n_steps_per_fit", type=int, default=None,
                   help="每次 fit 的 env-step 数, 必须 >= num_envs (默认 = num_envs)")
    p.add_argument("--utd", type=int, default=2,
                   help="每次数据收集后 SAC 梯度步数. 8 → critic overestimation collapse "
                        "(replay 每 epoch 扫 8 遍, critic 过拟合窄 action 分布). 2 更稳")
    p.add_argument("--lr_alpha", type=float, default=3e-4,
                   help="α 自调的 lr (mushroom 默认 3e-4). 原 1e-5 过小 → α 锁住")
    p.add_argument("--target_entropy", type=float, default=-14.0,
                   help="目标 entropy, 默认 -|A|=-14 (双臂各 7 DoF)")
    p.add_argument("--alpha_max", type=float, default=0.05,
                   help="α 上限. 每次 fit 后强制 log_alpha ≤ log(alpha_max). "
                        "v6 α=0.3 时熵奖励 ~2-3/步, 累积 400-600 压过任务 reward [-200,-100]. "
                        "α=0.05 把熵贡献降到 task reward 的 ~10%, 让任务信号主导")
    p.add_argument("--n_eval_episodes", type=int, default=16,
                   help="评估 episode 数. 默认 16, 降低初始位随机性对 eval J/R 的噪声")
    p.add_argument("--rew_pose", type=float, default=None,
                   help="姿态误差权重 w_pose (覆盖 env 默认 0.3). 诊断 run 可 --rew_pose 0 "
                        "跑纯位置任务, 验证 pos-only 是否能收敛 → 隔离 pose reward 的问题")
    p.add_argument("--sanity", action="store_true",
                   help="诊断: trivial 任务 (target=default EE, 极小扰动, 放宽 success). "
                        "若 agent 还学不会 → pipeline 有深层 bug, 不是难度问题. "
                        "若能快速收敛 → 确认是探索/稀疏奖励问题, 需要 curriculum.")
    p.add_argument("--wandb_project", type=str, default="bimanual_peghole")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.n_steps_per_fit is None:
        args.n_steps_per_fit = args.num_envs
    assert args.n_steps_per_fit >= args.num_envs, (
        f"n_steps_per_fit ({args.n_steps_per_fit}) 必须 >= num_envs ({args.num_envs})"
    )

    from envs import DualArmPegHoleEnv
    env_kwargs = dict(num_envs=args.num_envs, headless=not args.render)
    if args.rew_pose is not None:
        env_kwargs["rew_pose"] = args.rew_pose
    if args.sanity:
        env_kwargs.update(
            target_travel_fraction=0.0,   # 目标 = default EE, 原地不动就算赢
            initial_joint_noise=0.05,     # ~3° / joint, EE 偏 ~2-3cm
            success_pos_threshold=0.1,    # 放宽到 10cm (起始就在阈值内)
        )
    mdp = DualArmPegHoleEnv(**env_kwargs)
    mdp.seed(args.seed)

    # IsaacSim 启动后才能导入 mushroom_rl (避免 carb 冲突)
    from mushroom_rl.algorithms.actor_critic import SAC
    from mushroom_rl.core import VectorCore, Logger, Dataset

    obs_dim = mdp.info.observation_space.shape[0]
    act_dim = mdp.info.action_space.shape[0]

    actor_params = dict(network=ActorNetwork, input_shape=(obs_dim,),
                        output_shape=(act_dim,))
    actor_optimizer = {"class": optim.Adam, "params": {"lr": 3e-4}}
    critic_params = dict(network=CriticNetwork, input_shape=(obs_dim,),
                         output_shape=(1,), action_dim=act_dim,
                         optimizer={"class": optim.Adam, "params": {"lr": 3e-4}},
                         loss=F.mse_loss)

    agent = SAC(
        mdp_info=mdp.info,
        actor_mu_params=actor_params,
        actor_sigma_params=actor_params,
        actor_optimizer=actor_optimizer,
        critic_params=critic_params,
        batch_size=256,
        initial_replay_size=10_000,
        max_replay_size=500_000,
        warmup_transitions=10_000,
        tau=0.005,
        lr_alpha=args.lr_alpha,
        use_log_alpha_loss=True,  # 原论文形式, 对 α 爆炸更稳
        target_entropy=args.target_entropy,
    )
    # 注: mushroom 的 StandardizationPreprocessor 对 vectorized env 有 bug
    # (Welford batch 更新把整 batch 当一样本, std 随训练趋于 0, obs 被 clip 成 ±10 垃圾).
    # 暂不使用任何 preprocessor, 直接喂 raw obs (量级混合但 2×256 MLP 能处理).
    core = VectorCore(agent, mdp)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    logger = Logger("SAC", results_dir=str(results_dir))
    logger.strong_line()
    logger.info(f"obs_dim={obs_dim}  act_dim={act_dim}  horizon={mdp.info.horizon}")

    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, name=args.wandb_run_name,
            config={**vars(args), "algo": "SAC",
                    "obs_dim": obs_dim, "act_dim": act_dim,
                    "horizon": mdp.info.horizon, "gamma": mdp.info.gamma},
            dir=str(results_dir),
        )
        logger.info(f"wandb run: {wandb_run.url}")

    # 空 dataset 用来做额外 gradient step: SAC.fit 里 replay.add(empty) 是 no-op,
    # 但 sample + update 照跑. 分配一次复用.
    empty_dataset = Dataset.generate(mdp.info, agent.info, n_steps=1, n_envs=args.num_envs)

    # 预热: 凑够 initial_replay_size 条 transition 再开始训练.
    # VectorCore 的 n_steps 是 vector-step, 每步写 num_envs 条 → 需要的 vector-step 数 = ceil(10000 / num_envs).
    warmup_vec_steps = math.ceil(10_000 / args.num_envs)
    logger.info(f"填充 replay buffer ({warmup_vec_steps} vector-steps × {args.num_envs} envs)...")
    core.learn(n_steps=warmup_vec_steps, n_steps_per_fit=warmup_vec_steps)

    # 训练主循环: 每 epoch 切成 iters_per_epoch 次 (收 fit 块 + utd-1 次空 fit).
    steps_per_iter = args.n_steps_per_fit
    iters_per_epoch = args.n_steps_per_epoch // steps_per_iter  # 整除即可, 余数丢掉
    env_steps_per_epoch = iters_per_epoch * steps_per_iter * args.num_envs
    logger.info(f"utd={args.utd}  iters/epoch={iters_per_epoch}  "
                f"fits/epoch={iters_per_epoch * args.utd}  "
                f"effective UTD={args.utd / args.num_envs:.3f}")

    best_J = -np.inf
    total_env_steps = warmup_vec_steps * args.num_envs
    log_alpha_max = math.log(args.alpha_max)
    absorb_prev = mdp._absorb_count  # warmup 期间的碰撞从 epoch 计数中扣除
    for epoch in range(args.n_epochs):
        for _ in range(iters_per_epoch):
            core.learn(n_steps=steps_per_iter, n_steps_per_fit=steps_per_iter, quiet=True)
            with torch.no_grad():
                agent._log_alpha.clamp_(max=log_alpha_max)
            for _ in range(args.utd - 1):
                agent.fit(empty_dataset)
                with torch.no_grad():
                    agent._log_alpha.clamp_(max=log_alpha_max)
        total_env_steps += env_steps_per_epoch

        # 在 evaluate 之前快照, 只计训练期间的碰撞 (eval 碰撞不算本 epoch)
        absorb_epoch = mdp._absorb_count - absorb_prev

        dataset = core.evaluate(n_episodes=args.n_eval_episodes, quiet=True)
        J = torch.mean(dataset.discounted_return).item()
        R = torch.mean(dataset.undiscounted_return).item()
        ep_len = len(dataset) / args.n_eval_episodes

        if J > best_J:
            best_J = J
            agent.save(str(results_dir / "best_agent.msh"))

        # 下一 epoch 的 train 计数从 eval 结束后开始 (丢弃 eval 期间的碰撞)
        absorb_prev = mdp._absorb_count

        logger.epoch_info(epoch + 1, J=J, R=R, best_J=best_J,
                          absorb_epoch=absorb_epoch)
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1, "env_steps": total_env_steps,
                "J": J, "R": R, "best_J": best_J, "eval_ep_len": ep_len,
                "alpha": agent._alpha.item(),
                "absorb_per_epoch": absorb_epoch,
            }, step=epoch + 1)

    logger.info(f"训练完成. best J = {best_J:.3f}")
    if wandb_run is not None:
        wandb_run.summary["best_J"] = best_J
        wandb_run.finish()
    mdp.stop()


if __name__ == "__main__":
    main()
