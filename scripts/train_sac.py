"""SAC 训练 — 双臂末端到达固定点 (mushroom-rl + VectorCore).

phase 1: 纯位置 reaching, 14 维 joint velocity 动作, 40 维 obs.

运行:
    conda activate safe_rl
    python scripts/train_sac.py
    python scripts/train_sac.py --target_travel_fraction 0.25 --initial_joint_noise 0.05
    python scripts/train_sac.py --left_target -0.62 -0.55 0.69 --right_target -0.62 0.38 0.74
    python scripts/train_sac.py --render
    python scripts/train_sac.py --no_wandb

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
from scripts._eval_utils import (
    compute_hold_metrics,
    deterministic_policy,
    resolve_eval_episode_count,
)


INITIAL_REPLAY_SIZE = 10_000
MAX_REPLAY_SIZE = 500_000
BATCH_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--render", action="store_true", help="打开 IsaacSim 窗口")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_epochs", type=int, default=400)
    p.add_argument("--n_steps_per_epoch", type=int, default=1024,
                   help="每个 epoch 收集的总 env-step 数 (不是 vector-step)")
    p.add_argument("--n_steps_per_fit", type=int, default=None,
                   help="两次 fit 之间收集的总 env-step 数 (默认 = num_envs, 即 1 个 vector-step)")
    p.add_argument("--utd", type=int, default=None,
                   help="每次 fit 块对应的总梯度步数. 默认自动取 n_steps_per_fit, 使 true UTD≈1")
    p.add_argument("--lr_actor", type=float, default=3e-4)
    p.add_argument("--lr_critic", type=float, default=3e-4)
    p.add_argument("--lr_alpha", type=float, default=3e-4)
    p.add_argument("--alpha_max", type=float, default=0.2,
                   help="alpha 上限, 抑制高维动作下 entropy 奖励压过任务 reward")
    p.add_argument("--target_entropy", type=float, default=None,
                   help="目标 entropy. 默认自动取 -act_dim (SAC 标准设置)")
    p.add_argument("--n_eval_episodes", type=int, default=None,
                   help="评估 episode 数. 默认自动取 num_envs, 并要求能被 num_envs 整除, "
                        "避免尾批 inactive env 被 teleport away")
    p.add_argument("--target_travel_fraction", type=float, default=None,
                   help="显式切回 fraction 目标模式时使用的内收比例 f. "
                        "若不传, 默认使用胸前固定目标点")
    p.add_argument("--left_target", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="显式指定左臂固定目标点 (world/env-local frame)")
    p.add_argument("--right_target", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="显式指定右臂固定目标点 (world/env-local frame)")
    p.add_argument("--initial_joint_noise", type=float, default=None,
                   help="覆盖 env 的 reset 关节噪声")
    p.add_argument("--success_pos_threshold", type=float, default=None,
                   help="覆盖 env 的位置成功阈值")
    p.add_argument("--rew_action", type=float, default=None,
                   help="覆盖 env 的动作 L2 惩罚权重")
    p.add_argument("--rew_success", type=float, default=None,
                   help="覆盖 env 的 per-step success bonus (默认 2.0)")
    p.add_argument("--terminal_hold_bonus", type=float, default=None,
                   help="hold-N 步成功后的终结 bonus + episode 终止. "
                        "0 = 关闭 (baseline). >0 启用 absorbing termination.")
    p.add_argument("--hold_success_steps", type=int, default=10,
                   help="eval success 定义 + env 终止阈值: 连续 N 步都在阈值内. "
                        "N=10 ≈ 1s hold (per-step dt≈0.1s). "
                        "若 --terminal_hold_bonus > 0, 这个 N 也是 env absorbing 触发条件.")
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
    if args.utd is None:
        args.utd = args.n_steps_per_fit
    if args.utd < 1:
        raise ValueError("--utd 必须 >= 1")
    if args.n_steps_per_fit < args.num_envs:
        raise ValueError(
            f"n_steps_per_fit ({args.n_steps_per_fit}) 不能小于 num_envs ({args.num_envs})"
        )
    if args.n_steps_per_epoch < args.n_steps_per_fit:
        raise ValueError(
            f"n_steps_per_epoch ({args.n_steps_per_epoch}) 不能小于 "
            f"n_steps_per_fit ({args.n_steps_per_fit})"
        )
    if args.n_steps_per_fit % args.num_envs != 0:
        raise ValueError(
            f"n_steps_per_fit ({args.n_steps_per_fit}) 必须能被 num_envs ({args.num_envs}) 整除, "
            "否则会截断半个 vector-step"
        )
    if args.n_steps_per_epoch % args.n_steps_per_fit != 0:
        raise ValueError(
            f"n_steps_per_epoch ({args.n_steps_per_epoch}) 必须能被 "
            f"n_steps_per_fit ({args.n_steps_per_fit}) 整除, 否则 VectorCore 会在 "
            "epoch 末尾丢弃未 fit 的残余 transition"
        )
    # 注: n_steps_per_epoch % num_envs == 0 由上面两条共同蕴含, 不必单独校验
    args.n_eval_episodes = resolve_eval_episode_count(
        args.n_eval_episodes, args.num_envs, "--n_eval_episodes"
    )

    from envs import (
        DEFAULT_LEFT_CHEST_TARGET,
        DEFAULT_RIGHT_CHEST_TARGET,
        DualArmPegHoleEnv,
    )
    env_kwargs = dict(num_envs=args.num_envs, headless=not args.render)
    if (args.left_target is None) != (args.right_target is None):
        raise ValueError("--left_target 和 --right_target 必须同时提供")
    if args.left_target is not None and args.target_travel_fraction is not None:
        raise ValueError("显式 fixed targets 与 --target_travel_fraction 不能同时使用")

    if args.left_target is not None:
        env_kwargs.update(
            left_target=tuple(args.left_target),
            right_target=tuple(args.right_target),
        )
    elif args.target_travel_fraction is None:
        env_kwargs.update(
            left_target=DEFAULT_LEFT_CHEST_TARGET,
            right_target=DEFAULT_RIGHT_CHEST_TARGET,
        )
    else:
        env_kwargs["target_travel_fraction"] = args.target_travel_fraction
    for key in ("initial_joint_noise", "success_pos_threshold", "rew_action",
                "rew_success", "terminal_hold_bonus"):
        value = getattr(args, key)
        if value is not None:
            env_kwargs[key] = value
    # env 用同一个 hold N 做 absorbing termination + eval metric, 一次性同步
    env_kwargs["success_hold_steps"] = args.hold_success_steps
    mdp = DualArmPegHoleEnv(**env_kwargs)
    mdp.seed(args.seed)

    # IsaacSim 启动后才能导入 mushroom_rl (避免 carb 冲突)
    from mushroom_rl.algorithms.actor_critic import SAC
    from mushroom_rl.core import VectorCore, Logger, Dataset

    obs_dim = mdp.info.observation_space.shape[0]
    act_dim = mdp.info.action_space.shape[0]
    target_entropy = args.target_entropy
    if target_entropy is None:
        target_entropy = -float(act_dim)

    # baseline 不做 obs 归一化, 网络默认不注册 _obs_scale buffer (forward 里 hasattr fallback).
    # 后续若要开归一化, 在这里加 `obs_scale=mdp.get_obs_scale().detach().cpu().tolist()`.
    actor_params = dict(network=ActorNetwork, input_shape=(obs_dim,),
                        output_shape=(act_dim,))
    actor_optimizer = {"class": optim.Adam, "params": {"lr": args.lr_actor}}
    critic_params = dict(network=CriticNetwork, input_shape=(obs_dim,),
                         output_shape=(1,), action_dim=act_dim,
                         optimizer={"class": optim.Adam, "params": {"lr": args.lr_critic}},
                         loss=F.mse_loss)

    agent = SAC(
        mdp_info=mdp.info,
        actor_mu_params=actor_params,
        actor_sigma_params=actor_params,
        actor_optimizer=actor_optimizer,
        critic_params=critic_params,
        batch_size=BATCH_SIZE,
        initial_replay_size=INITIAL_REPLAY_SIZE,
        max_replay_size=MAX_REPLAY_SIZE,
        warmup_transitions=INITIAL_REPLAY_SIZE,
        tau=0.005,
        lr_alpha=args.lr_alpha,
        use_log_alpha_loss=True,  # 原论文形式, 对 α 爆炸更稳
        target_entropy=target_entropy,
    )
    def clamp_alpha(_dataset=None):
        with torch.no_grad():
            agent._log_alpha.clamp_(max=math.log(args.alpha_max))

    # 注: mushroom 的 StandardizationPreprocessor 对 vectorized env 有 bug
    # (Welford batch 更新把整 batch 当一样本, std 随训练趋于 0, obs 被 clip 成 ±10 垃圾).
    # 暂不使用任何 preprocessor, 直接喂 raw obs.
    core = VectorCore(agent, mdp, callbacks_fit=[clamp_alpha])

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    # 杜绝 stale checkpoint: 上一 run 的 best_agent.msh 留下来会让人误以为本次的
    # best 在那里. best_score gate (>0 才存) 在全程没成功时不会保存, 必须先删.
    best_path = results_dir / "best_agent.msh"
    if best_path.exists():
        best_path.unlink()
    logger = Logger("SAC", results_dir=str(results_dir))
    logger.strong_line()
    logger.info(f"清理旧 best checkpoint (本次 run 自建): {best_path}")
    logger.info(f"obs_dim={obs_dim}  act_dim={act_dim}  horizon={mdp.info.horizon}")
    logger.info(f"action_scale={mdp._action_scale:.3f}")
    logger.info(f"target_entropy={target_entropy:.3f}  "
                f"lr_actor={args.lr_actor:.1e}  lr_critic={args.lr_critic:.1e}  "
                f"lr_alpha={args.lr_alpha:.1e}  alpha_max={args.alpha_max:.3f}")
    logger.info(f"n_steps_per_epoch={args.n_steps_per_epoch} env-steps  "
                f"n_steps_per_fit={args.n_steps_per_fit} env-steps  "
                f"num_envs={args.num_envs}")

    mask = torch.ones(args.num_envs, dtype=torch.bool, device=mdp._device)
    obs, _ = mdp.reset_all(mask)
    left_pos_err, right_pos_err, in_thresh_mask = mdp._compute_task_errors(obs)
    in_thresh_rate = float(in_thresh_mask.float().mean())
    logger.info("reset stats: "
                f"in_thresh_rate={in_thresh_rate:.3f}  "
                f"left_pos_err_mean={float(left_pos_err.mean()):.4f}m  "
                f"right_pos_err_mean={float(right_pos_err.mean()):.4f}m")

    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, name=args.wandb_run_name,
            config={**vars(args), "algo": "SAC",
                    "target_entropy_resolved": target_entropy,
                    "obs_dim": obs_dim, "act_dim": act_dim,
                    "horizon": mdp.info.horizon, "gamma": mdp.info.gamma},
            dir=str(results_dir),
        )
        logger.info(f"wandb run: {wandb_run.url}")

    # 空 dataset 用来做额外 gradient step: SAC.fit 里 replay.add(empty) 是 no-op,
    # 但 sample + update 照跑. 分配一次复用.
    empty_dataset = Dataset.generate(mdp.info, agent.info, n_steps=1, n_envs=args.num_envs)

    warmup_vector_steps = math.ceil(INITIAL_REPLAY_SIZE / args.num_envs)
    logger.info("填充 replay buffer: "
                f"{INITIAL_REPLAY_SIZE} env-steps "
                f"(约 {warmup_vector_steps} vector-steps × {args.num_envs} envs)")
    core.learn(n_steps=INITIAL_REPLAY_SIZE, n_steps_per_fit=INITIAL_REPLAY_SIZE)

    fits_per_epoch = args.n_steps_per_epoch // args.n_steps_per_fit
    vector_steps_per_fit = args.n_steps_per_fit / args.num_envs
    vector_steps_per_epoch = args.n_steps_per_epoch / args.num_envs
    effective_utd = args.utd / args.n_steps_per_fit
    logger.info(f"utd={args.utd}  collect-fits/epoch={fits_per_epoch}  "
                f"total-fits/epoch={fits_per_epoch * args.utd}  "
                f"true_UTD={effective_utd:.3f}  "
                f"env-steps/epoch={args.n_steps_per_epoch}  "
                f"vector-steps/fit≈{vector_steps_per_fit:.1f}  "
                f"vector-steps/epoch≈{vector_steps_per_epoch:.1f}")

    best_J = -np.inf
    # best 选择策略两选一:
    # - terminal_hold_bonus > 0 (absorbing 启用): 用 best_J, 因为 J 直接包含 terminal bonus
    #   信号, 量级压过 drift-out 噪声; 而 hold_success_rate × max_hold_mean 会被 hold-N
    #   的 absorbing 钉住 (max_hold 上限 = N), 一旦 sr=1.0 就封顶, 后续更优策略存不下.
    # - terminal_hold_bonus = 0 (baseline): 用 best_score, 因为长 horizon 下 J 受
    #   drift-out 噪声污染, 跨 epoch 比较不稳.
    best_score = -np.inf
    use_J_for_best = mdp._terminal_hold_bonus > 0
    total_env_steps = INITIAL_REPLAY_SIZE
    absorb_prev = mdp._absorb_count  # warmup 期间的碰撞从 epoch 计数中扣除
    for epoch in range(args.n_epochs):
        core.learn(
            n_steps=args.n_steps_per_epoch,
            n_steps_per_fit=args.n_steps_per_fit,
            quiet=True,
        )
        clamp_alpha()
        for _ in range(fits_per_epoch * (args.utd - 1)):
            agent.fit(empty_dataset)
            clamp_alpha()
        total_env_steps += args.n_steps_per_epoch

        # 在 evaluate 之前快照, 只计训练期间的碰撞 (eval 碰撞不算本 epoch)
        absorb_epoch = mdp._absorb_count - absorb_prev

        with deterministic_policy(agent):
            dataset = core.evaluate(n_episodes=args.n_eval_episodes, quiet=True)
        J = torch.mean(dataset.discounted_return).item()
        R = torch.mean(dataset.undiscounted_return).item()
        ep_len = len(dataset) / args.n_eval_episodes
        m = compute_hold_metrics(dataset, mdp, args.hold_success_steps)

        improved_J = J > best_J
        if improved_J:
            best_J = J
        score = m['hold_success_rate'] * m['max_hold_mean']
        improved_score = m['hold_success_rate'] > 0 and score > best_score
        if improved_score:
            best_score = score
        # 选 best: 见上面 use_J_for_best 注释
        save_now = improved_J if use_J_for_best else improved_score
        if save_now:
            agent.save(str(results_dir / "best_agent.msh"))

        # 下一 epoch 的 train 计数从 eval 结束后开始 (丢弃 eval 期间的碰撞)
        absorb_prev = mdp._absorb_count

        logger.epoch_info(epoch + 1, J=J, R=R, best_J=best_J, best_score=best_score,
                          absorb_epoch=absorb_epoch)
        logger.info("eval stats: "
                    f"hold_success_rate={m['hold_success_rate']:.3f} "
                    f"(>= {args.hold_success_steps} consecutive steps)  "
                    f"max_hold_mean={m['max_hold_mean']:.1f}  "
                    f"in_thresh_rate={m['in_thresh_rate']:.3f}  "
                    f"final_in_thresh_rate={m['final_in_thresh_rate']:.3f}  "
                    f"left_err_mean={m['left_pos_err_mean']:.4f}m  "
                    f"right_err_mean={m['right_pos_err_mean']:.4f}m")
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1, "env_steps": total_env_steps,
                "J": J, "R": R, "best_J": best_J, "best_score": best_score,
                "eval_ep_len": ep_len,
                "eval_success_rate": m["hold_success_rate"],
                "eval_max_hold_mean": m["max_hold_mean"],
                "eval_in_thresh_rate": m["in_thresh_rate"],
                "eval_final_in_thresh_rate": m["final_in_thresh_rate"],
                "eval_left_pos_err_mean": m["left_pos_err_mean"],
                "eval_right_pos_err_mean": m["right_pos_err_mean"],
                "alpha": agent._alpha.item(),
                "absorb_per_epoch": absorb_epoch,
            }, step=epoch + 1)

    logger.info(f"训练完成. best J = {best_J:.3f}  best_score = {best_score:.3f}")
    if wandb_run is not None:
        wandb_run.summary["best_J"] = best_J
        wandb_run.summary["best_score"] = best_score
        wandb_run.finish()
    mdp.stop()


if __name__ == "__main__":
    main()
