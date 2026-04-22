"""SAC 训练 — 双臂末端到达固定点 (mushroom-rl + VectorCore).

phase 1: 纯位置 reaching, 14 维 joint velocity 动作, 40 维 obs.

运行:
    conda activate safe_rl
    python scripts/train_sac.py
    python scripts/train_sac.py --target_travel_fraction 0.25 --initial_joint_noise 0.05
    python scripts/train_sac.py --render
    python scripts/train_sac.py --no_wandb

注意: num_envs=1 会触发 IsaacSim cloner 的 `*` pattern 失败 → 至少 2.
"""

import argparse
from contextlib import contextmanager
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

DEFAULT_LEFT_CHEST_TARGET = (-0.6176, -0.75, 0.7391)
DEFAULT_RIGHT_CHEST_TARGET = (-0.6176, 0.73, 0.7391)


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
    p.add_argument("--n_eval_episodes", type=int, default=16,
                   help="评估 episode 数. 默认 16, 降低初始位随机性对 eval J/R 的噪声")
    p.add_argument("--target_travel_fraction", type=float, default=None,
                   help="显式切回 fraction 目标模式时使用的内收比例 f. "
                        "若不传, 默认使用胸前固定目标点")
    p.add_argument("--initial_joint_noise", type=float, default=None,
                   help="覆盖 env 的 reset 关节噪声")
    p.add_argument("--success_pos_threshold", type=float, default=None,
                   help="覆盖 env 的位置成功阈值")
    p.add_argument("--rew_action", type=float, default=None,
                   help="覆盖 env 的动作 L2 惩罚权重")
    p.add_argument("--hold_success_steps", type=int, default=10,
                   help="eval success 定义: 一个 episode 内至少出现连续 N 步都在阈值内. "
                        "N=10 ≈ 1s hold (per-step dt≈0.1s)")
    p.add_argument("--wandb_project", type=str, default="bimanual_peghole")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


@contextmanager
def deterministic_policy(agent):
    policy = agent.policy
    original_draw_action = policy.draw_action

    def draw_action(state, internal_state=None):
        with torch.no_grad():
            mu = policy._mu_approximator.predict(state)
            action = torch.tanh(mu) * policy._delta_a + policy._central_a
        return action.detach(), None

    policy.draw_action = draw_action
    try:
        yield
    finally:
        policy.draw_action = original_draw_action


def evaluate_deterministic(core, agent, n_episodes, quiet):
    with deterministic_policy(agent):
        return core.evaluate(n_episodes=n_episodes, quiet=quiet)


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
    if args.n_steps_per_epoch % args.num_envs != 0:
        raise ValueError(
            f"n_steps_per_epoch ({args.n_steps_per_epoch}) 必须能被 num_envs ({args.num_envs}) 整除"
        )
    if args.n_steps_per_epoch % args.n_steps_per_fit != 0:
        raise ValueError(
            f"n_steps_per_epoch ({args.n_steps_per_epoch}) 必须能被 "
            f"n_steps_per_fit ({args.n_steps_per_fit}) 整除, 否则 VectorCore 会在 "
            "epoch 末尾丢弃未 fit 的残余 transition"
        )

    from envs import DualArmPegHoleEnv
    env_kwargs = dict(num_envs=args.num_envs, headless=not args.render)
    if args.target_travel_fraction is None:
        env_kwargs.update(
            left_target=DEFAULT_LEFT_CHEST_TARGET,
            right_target=DEFAULT_RIGHT_CHEST_TARGET,
        )
    else:
        env_kwargs["target_travel_fraction"] = args.target_travel_fraction
    for key in ("initial_joint_noise", "success_pos_threshold", "rew_action"):
        value = getattr(args, key)
        if value is not None:
            env_kwargs[key] = value
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
    logger = Logger("SAC", results_dir=str(results_dir))
    logger.strong_line()
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

        dataset = evaluate_deterministic(core, agent, args.n_eval_episodes, quiet=True)
        J = torch.mean(dataset.discounted_return).item()
        R = torch.mean(dataset.undiscounted_return).item()
        ep_len = len(dataset) / args.n_eval_episodes
        _, _, _, next_state, _, last = dataset.parse(to="torch")

        # per-step task error & in-threshold mask (整段 trajectory)
        step_left_err, step_right_err, step_in_thresh = mdp._compute_task_errors(next_state)
        last_np = last.cpu().numpy().astype(bool)
        in_thresh_np = step_in_thresh.cpu().numpy().astype(bool)

        # 按 last flag 分段, 统计每个 episode 最长连续 in-threshold 长度
        # "hold success" = 出现长度 >= hold_success_steps 的连续 True 段
        end_indices = np.flatnonzero(last_np)
        ep_max_holds = []
        ep_in_thresh_rates = []
        ep_final_in_thresh = []
        start = 0
        for end in end_indices:
            ep = in_thresh_np[start:end + 1]
            # 最长连续 True
            max_run = 0
            cur = 0
            for flag in ep:
                cur = cur + 1 if flag else 0
                if cur > max_run:
                    max_run = cur
            ep_max_holds.append(max_run)
            ep_in_thresh_rates.append(float(ep.mean()) if len(ep) else 0.0)
            ep_final_in_thresh.append(bool(ep[-1]) if len(ep) else False)
            start = end + 1

        N_hold = args.hold_success_steps
        hold_flags = np.asarray([mh >= N_hold for mh in ep_max_holds], dtype=bool)
        eval_success_rate = float(hold_flags.mean()) if len(hold_flags) else 0.0
        eval_max_hold_mean = float(np.mean(ep_max_holds)) if ep_max_holds else 0.0
        eval_in_thresh_rate = float(np.mean(ep_in_thresh_rates)) if ep_in_thresh_rates else 0.0
        eval_final_in_thresh_rate = (float(np.mean(ep_final_in_thresh))
                                     if ep_final_in_thresh else 0.0)
        eval_left_pos_err_mean = float(step_left_err.mean())
        eval_right_pos_err_mean = float(step_right_err.mean())

        if J > best_J:
            best_J = J
            agent.save(str(results_dir / "best_agent.msh"))

        # 下一 epoch 的 train 计数从 eval 结束后开始 (丢弃 eval 期间的碰撞)
        absorb_prev = mdp._absorb_count

        logger.epoch_info(epoch + 1, J=J, R=R, best_J=best_J,
                          absorb_epoch=absorb_epoch)
        logger.info("eval stats: "
                    f"hold_success_rate={eval_success_rate:.3f} (>= {N_hold} consecutive steps)  "
                    f"max_hold_mean={eval_max_hold_mean:.1f}  "
                    f"in_thresh_rate={eval_in_thresh_rate:.3f}  "
                    f"final_in_thresh_rate={eval_final_in_thresh_rate:.3f}  "
                    f"left_err_mean={eval_left_pos_err_mean:.4f}m  "
                    f"right_err_mean={eval_right_pos_err_mean:.4f}m")
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1, "env_steps": total_env_steps,
                "J": J, "R": R, "best_J": best_J, "eval_ep_len": ep_len,
                "eval_success_rate": eval_success_rate,
                "eval_max_hold_mean": eval_max_hold_mean,
                "eval_in_thresh_rate": eval_in_thresh_rate,
                "eval_final_in_thresh_rate": eval_final_in_thresh_rate,
                "eval_left_pos_err_mean": eval_left_pos_err_mean,
                "eval_right_pos_err_mean": eval_right_pos_err_mean,
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
