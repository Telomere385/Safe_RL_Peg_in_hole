"""Lagrangian SAC 训练 — 双臂 peg-in-hole, mushroom-rl + VectorCore.

跟 train_sac.py 的差异:
- 算法换成 algo.lagrangian_sac.SACLagrangian (在 PROJECT_ROOT/algo/ 下).
- env._create_info_dictionary 必须返回 {"cost": tensor} (已在
  envs/dual_arm_peg_hole_env.py 实现).
- 多了 --cost_limit / --lr_lambda / --lambda_max / --init_log_lambda /
  --gamma_cost flags.
- 每 epoch eval 多算 cost_rate, wandb 加 lambda / cost_q_mean.

Warmstart 注意:
- 从 SAC checkpoint warm-start: **必须 --actor_only_warmstart**, 否则报错
  (SAC 没有 cost critic / lambda, 全量 load 会缺字段).
- 从 SACLagrangian checkpoint warm-start: 可全量 (默认) 或 actor-only.

Cost signal 调标:
- env 的 cost = sphere proxy collision indicator (0/1). PhysX 在当前 USD 上
  不触发 (见 dual_arm_peg_hole_env.py is_absorbing).
- 设 --cost_limit 之前先用 SAC baseline 跑一次 64-ep eval, 记录
  absorb_sphere_per_epoch / n_steps_per_epoch (per-step 触发率), 取其 0.5 倍
  做起步预算. 设 cost_limit=0 会让 λ 一上来就爆.
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
    parse_home_weights,
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
    p.add_argument("--n_steps_per_epoch", type=int, default=1024)
    p.add_argument("--n_steps_per_fit", type=int, default=None)
    p.add_argument("--utd", type=int, default=None)
    p.add_argument("--lr_actor", type=float, default=3e-4)
    p.add_argument("--lr_critic", type=float, default=3e-4)
    p.add_argument("--lr_alpha", type=float, default=3e-4)
    p.add_argument("--alpha_max", type=float, default=0.1)
    p.add_argument("--target_entropy", type=float, default=None)
    p.add_argument("--critic_warmup_transitions", type=int, default=None)
    p.add_argument("--n_eval_episodes", type=int, default=None)

    # ---- Lagrangian 专属 ----------------------------------------------------
    p.add_argument("--cost_limit", type=float, required=True,
                   help="per-step cost 预算 (e.g. 0.01 = 容忍 1% collision rate). "
                        "标定方法见文件头.")
    p.add_argument("--lr_lambda", type=float, default=1e-3,
                   help="Lagrange 乘子学习率. 1e-3~1e-4, 通常比 lr_actor 低.")
    p.add_argument("--lambda_max", type=float, default=100.0,
                   help="λ clamp 上限. λ 冲到几百会把 actor 锁死.")
    p.add_argument("--init_log_lambda", type=float, default=0.0,
                   help="log_λ 初值, 默认 0 → λ_init=1.")
    p.add_argument("--gamma_cost", type=float, default=None,
                   help="cost MDP 折扣. 默认 = env γ. 设 1.0 = average-cost "
                        "(注意此时 cost_limit 直接是 Q_C 量纲, 不是 per-step).")
    # ------------------------------------------------------------------------

    # env 参数 (跟 train_sac.py 同)
    p.add_argument("--initial_joint_noise", type=float, default=None)
    p.add_argument("--preinsert_success_pos_threshold", type=float, default=None)
    p.add_argument("--preinsert_offset", type=float, default=None)
    p.add_argument("--rew_action", type=float, default=None)
    p.add_argument("--rew_home", type=float, default=None)
    p.add_argument("--home_weights", type=parse_home_weights, default=None)
    p.add_argument("--rew_success", type=float, default=None)
    p.add_argument("--rew_pos_success", type=float, default=None)
    p.add_argument("--axis_gate_radius", type=float, default=None)
    p.add_argument("--rew_axis", type=float, default=None)
    p.add_argument("--success_axis_threshold", type=float, default=None)

    # warmstart
    p.add_argument("--load_agent", type=str, default=None,
                   help="warm-start checkpoint. 从 SAC checkpoint 加载必须配 "
                        "--actor_only_warmstart, 否则报错 (SAC 没有 cost critic / λ).")
    p.add_argument("--keep_replay", action="store_true")
    p.add_argument("--actor_only_warmstart", action="store_true",
                   help="仅继承 actor (mu/sigma) 权重. SACLagrangian 从 SAC checkpoint "
                        "warmstart 必开此项. SACLagrangian → SACLagrangian 也建议开 "
                        "(reward 函数 / cost 信号若变, 旧 critic 语义错).")

    # 终止信号
    p.add_argument("--terminal_hold_bonus", type=float, default=None)
    p.add_argument("--hold_success_steps", type=int, default=10)
    p.add_argument("--clearance_hard", type=float, default=None)
    p.add_argument("--proxy_arm_radius", type=float, default=None)
    p.add_argument("--proxy_ee_radius", type=float, default=None)
    p.add_argument("--exclude_ee_from_physx_self_collision", action="store_true")

    # obs
    p.add_argument("--use_axis_resid_obs", action="store_true")

    # wandb
    p.add_argument("--wandb_project", type=str, default="bimanual_peghole")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def compute_cost_metrics(dataset, n_eval_episodes):
    """从 eval flatten dataset 的 info.data["cost"] 算 cost_rate / per-ep cost sum.

    依赖 env._create_info_dictionary 把 cost 写进 step_info; flatten 后顺序与
    reward 对齐 (probe_extra_info_cost.py 验证).
    """
    cost = dataset.info.data.get("cost")
    if cost is None:
        return {"cost_rate": float("nan"), "cost_episode_sum_mean": float("nan")}
    cost_t = cost if isinstance(cost, torch.Tensor) else torch.as_tensor(cost)
    cost_rate = float(cost_t.float().mean())
    # per-episode 和: 总 cost / n_episodes (cost 是 per-step 0/1 -> 总 cost = 触发次数)
    cost_episode_sum_mean = float(cost_t.float().sum()) / max(n_eval_episodes, 1)
    return {"cost_rate": cost_rate, "cost_episode_sum_mean": cost_episode_sum_mean}


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
            f"n_steps_per_fit ({args.n_steps_per_fit}) 必须能被 num_envs 整除"
        )
    if args.n_steps_per_epoch % args.n_steps_per_fit != 0:
        raise ValueError(
            f"n_steps_per_epoch ({args.n_steps_per_epoch}) 必须能被 n_steps_per_fit 整除"
        )
    if args.critic_warmup_transitions is None:
        args.critic_warmup_transitions = INITIAL_REPLAY_SIZE
    if args.critic_warmup_transitions < INITIAL_REPLAY_SIZE:
        raise ValueError(
            f"--critic_warmup_transitions ({args.critic_warmup_transitions}) 必须 >= "
            f"INITIAL_REPLAY_SIZE ({INITIAL_REPLAY_SIZE})"
        )
    if args.cost_limit < 0.0:
        raise ValueError(f"--cost_limit ({args.cost_limit}) 必须 >= 0")
    if args.cost_limit == 0.0:
        print("[WARN] --cost_limit=0 会让 λ 一上来就爆, 一般取 0.5×baseline collision rate.")
    args.n_eval_episodes = resolve_eval_episode_count(
        args.n_eval_episodes, args.num_envs, "--n_eval_episodes"
    )

    from envs import DualArmPegHoleEnv
    env_kwargs = dict(num_envs=args.num_envs, headless=not args.render)
    for key in ("initial_joint_noise", "preinsert_success_pos_threshold",
                "preinsert_offset", "rew_action", "rew_success", "rew_pos_success",
                "rew_axis", "rew_home", "home_weights", "axis_gate_radius",
                "success_axis_threshold", "terminal_hold_bonus",
                "clearance_hard", "proxy_arm_radius", "proxy_ee_radius"):
        value = getattr(args, key)
        if value is not None:
            env_kwargs[key] = value
    if args.use_axis_resid_obs:
        env_kwargs["use_axis_resid_obs"] = True
    if args.exclude_ee_from_physx_self_collision:
        env_kwargs["exclude_ee_from_physx_self_collision"] = True
    env_kwargs["success_hold_steps"] = args.hold_success_steps
    mdp = DualArmPegHoleEnv(**env_kwargs)
    mdp.seed(args.seed)

    # IsaacSim 启动后才能导入 mushroom_rl / algo
    from mushroom_rl.core import Agent, VectorCore, Logger, Dataset
    from algo import SACLagrangian

    obs_dim = mdp.info.observation_space.shape[0]
    act_dim = mdp.info.action_space.shape[0]
    target_entropy = args.target_entropy
    if target_entropy is None:
        target_entropy = -float(act_dim)

    def _cold_create_sac_lag():
        actor_params = dict(network=ActorNetwork, input_shape=(obs_dim,),
                            output_shape=(act_dim,))
        actor_optimizer = {"class": optim.Adam, "params": {"lr": args.lr_actor}}
        critic_params = dict(network=CriticNetwork, input_shape=(obs_dim,),
                             output_shape=(1,), action_dim=act_dim,
                             optimizer={"class": optim.Adam, "params": {"lr": args.lr_critic}},
                             loss=F.mse_loss)
        return SACLagrangian(
            mdp_info=mdp.info,
            actor_mu_params=actor_params,
            actor_sigma_params=actor_params,
            actor_optimizer=actor_optimizer,
            critic_params=critic_params,
            batch_size=BATCH_SIZE,
            initial_replay_size=INITIAL_REPLAY_SIZE,
            max_replay_size=MAX_REPLAY_SIZE,
            warmup_transitions=args.critic_warmup_transitions,
            tau=0.005,
            lr_alpha=args.lr_alpha,
            use_log_alpha_loss=True,
            target_entropy=target_entropy,
            cost_limit=args.cost_limit,
            lr_lambda=args.lr_lambda,
            lambda_max=args.lambda_max,
            init_log_lambda=args.init_log_lambda,
            gamma_cost=args.gamma_cost,
        )

    if args.load_agent is not None:
        load_path = Path(args.load_agent)
        if not load_path.is_file():
            raise FileNotFoundError(f"--load_agent 路径不存在: {load_path}")
        old_agent = Agent.load(str(load_path))
        old_class = type(old_agent).__name__

        if args.actor_only_warmstart:
            agent = _cold_create_sac_lag()
            agent.policy._mu_approximator.set_weights(
                old_agent.policy._mu_approximator.get_weights()
            )
            agent.policy._sigma_approximator.set_weights(
                old_agent.policy._sigma_approximator.get_weights()
            )
            print(f"[WARM-START actor-only] from {old_class} @ {load_path}; "
                  "critic / cost critic / α / λ / replay 全部冷启动.")
            if args.keep_replay:
                print("[WARM-START actor-only] --keep_replay 已忽略.")
            del old_agent
        else:
            if old_class != "SACLagrangian":
                raise RuntimeError(
                    f"--load_agent 是 {old_class}, 全量 warmstart 只能用 SACLagrangian "
                    "checkpoint. 从 SAC checkpoint warmstart 必须加 --actor_only_warmstart."
                )
            agent = old_agent
            print(f"[WARM-START full] 整体加载 SACLagrangian from {load_path}")
            if not args.keep_replay:
                agent._replay_memory.reset()
                print("[WARM-START full] replay buffer 已清空.")
            else:
                print("[WARM-START full] 保留旧 replay buffer.")
    else:
        agent = _cold_create_sac_lag()

    def clamp_alpha(_dataset=None):
        with torch.no_grad():
            agent._log_alpha.clamp_(max=math.log(args.alpha_max))

    core = VectorCore(agent, mdp, callbacks_fit=[clamp_alpha])

    from datetime import datetime
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    run_ts = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    ckpt_dir = results_dir / "checkpoints_lag" / run_ts
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_J_path = ckpt_dir / "best_agent.msh"
    best_hold_path = ckpt_dir / "best_hold.msh"
    final_path = ckpt_dir / "final_agent.msh"
    best_J_path_flat = results_dir / "best_agent_lag.msh"
    best_hold_path_flat = results_dir / "best_hold_lag.msh"
    final_path_flat = results_dir / "final_agent_lag.msh"
    logger = Logger("SACLagrangian", results_dir=str(results_dir))
    logger.strong_line()
    logger.info(f"checkpoint 目录: {ckpt_dir}")
    obs_mode = "axis_resid" if mdp._use_axis_resid_obs else "base"
    logger.info(f"obs_dim={obs_dim} ({obs_mode})  "
                f"act_dim={act_dim}  horizon={mdp.info.horizon}")
    logger.info(f"action_scale={mdp._action_scale:.3f}")
    logger.info(
        "physx_self_collision_group="
        + ("arm_links_only" if mdp._exclude_ee_from_physx_self_collision
           else "arm_links_plus_ee")
    )
    logger.info(f"preinsert_pos_th={mdp._preinsert_success_pos_threshold:.3f}m  "
                f"axis_th={mdp._success_axis_threshold:.3f}  "
                f"w_pos={mdp._w_pos:.3f}  w_axis={mdp._w_axis:.3f}  "
                f"w_pos_success={mdp._w_pos_success:.3f}  "
                f"w_success={mdp._w_success:.3f}")
    if args.load_agent is not None:
        logger.info(f"warm-start: {args.load_agent}")
    logger.info(f"target_entropy={target_entropy:.3f}  "
                f"lr_actor={args.lr_actor:.1e}  lr_critic={args.lr_critic:.1e}  "
                f"lr_alpha={args.lr_alpha:.1e}  alpha_max={args.alpha_max:.3f}")
    gamma_cost_resolved = (args.gamma_cost if args.gamma_cost is not None
                           else mdp.info.gamma)
    logger.info(f"[Lagrangian] cost_limit={args.cost_limit:.4f}  "
                f"lr_lambda={args.lr_lambda:.1e}  lambda_max={args.lambda_max:.1f}  "
                f"init_log_lambda={args.init_log_lambda:.3f}  "
                f"gamma_cost={gamma_cost_resolved:.3f}")
    critic_only_steps = args.critic_warmup_transitions - INITIAL_REPLAY_SIZE
    if critic_only_steps > 0:
        critic_only_epochs = critic_only_steps / args.n_steps_per_epoch
        logger.info(
            f"critic_warmup_transitions={args.critic_warmup_transitions} env-steps "
            f"(replay-fill {INITIAL_REPLAY_SIZE} + critic-only {critic_only_steps} "
            f"≈ {critic_only_epochs:.1f} epoch, actor/α/λ 此期间冻结)"
        )
    logger.info(f"n_steps_per_epoch={args.n_steps_per_epoch}  "
                f"n_steps_per_fit={args.n_steps_per_fit}  num_envs={args.num_envs}")

    mask = torch.ones(args.num_envs, dtype=torch.bool, device=mdp._device)
    obs, _ = mdp.reset_all(mask)
    pos_err, axis_err, in_thresh_mask = mdp._compute_task_errors(obs)
    logger.info("reset stats: "
                f"in_thresh_rate={float(in_thresh_mask.float().mean()):.3f}  "
                f"pos_err_mean={float(pos_err.mean()):.4f}m  "
                f"axis_err_mean={float(axis_err.mean()):.4f}")

    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, name=args.wandb_run_name,
            group=args.wandb_group,
            config={**vars(args), "algo": "SACLagrangian",
                    "target_entropy_resolved": target_entropy,
                    "gamma_cost_resolved": gamma_cost_resolved,
                    "obs_dim": obs_dim, "act_dim": act_dim,
                    "horizon": mdp.info.horizon, "gamma": mdp.info.gamma},
            dir=str(results_dir),
        )
        logger.info(f"wandb run: {wandb_run.url}")

    empty_dataset = Dataset.generate(mdp.info, agent.info, n_steps=1, n_envs=args.num_envs)

    if args.load_agent is not None:
        logger.info("=" * 60)
        logger.info("[EVAL @ epoch 0] warm-start actor BEFORE 任何 fit / 任何 warmup")
        with deterministic_policy(agent):
            ds0 = core.evaluate(n_episodes=args.n_eval_episodes, quiet=True)
        m0 = compute_hold_metrics(ds0, mdp, args.hold_success_steps)
        c0 = compute_cost_metrics(ds0, args.n_eval_episodes)
        logger.info(f"  pos_success_rate={m0['pos_success_rate']:.3f}  "
                    f"pos_err_mean={m0['pos_err_mean']:.4f}m  "
                    f"axis_err_mean={m0['axis_err_mean']:.4f}  "
                    f"hold_success_rate={m0['hold_success_rate']:.3f}")
        logger.info(f"  cost_rate={c0['cost_rate']:.4f}  "
                    f"cost_episode_sum_mean={c0['cost_episode_sum_mean']:.3f}  "
                    f"cost_limit={args.cost_limit:.4f}")
        logger.info("=" * 60)
        if wandb_run is not None:
            wandb_run.log({
                "epoch": 0,
                "warmstart_pos_success_rate": m0["pos_success_rate"],
                "warmstart_pos_err_mean": m0["pos_err_mean"],
                "warmstart_axis_err_mean": m0["axis_err_mean"],
                "warmstart_hold_success_rate": m0["hold_success_rate"],
                "warmstart_cost_rate": c0["cost_rate"],
                "warmstart_cost_episode_sum_mean": c0["cost_episode_sum_mean"],
            }, step=0)

    warmup_vector_steps = math.ceil(INITIAL_REPLAY_SIZE / args.num_envs)
    logger.info(f"填充 replay: {INITIAL_REPLAY_SIZE} env-steps "
                f"(约 {warmup_vector_steps} vector-steps × {args.num_envs} envs)")
    core.learn(n_steps=INITIAL_REPLAY_SIZE, n_steps_per_fit=INITIAL_REPLAY_SIZE)

    fits_per_epoch = args.n_steps_per_epoch // args.n_steps_per_fit
    logger.info(f"utd={args.utd}  fits/epoch={fits_per_epoch}  "
                f"total-fits/epoch={fits_per_epoch * args.utd}")

    best_J = -np.inf
    best_score = -np.inf
    best_hold_rate = -1.0
    best_hold_score = -1.0
    total_env_steps = INITIAL_REPLAY_SIZE
    absorb_prev = mdp._absorb_count
    absorb_physx_prev = mdp._absorb_count_physx
    absorb_sphere_prev = mdp._absorb_count_sphere

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

        absorb_epoch = mdp._absorb_count - absorb_prev
        absorb_physx_epoch = mdp._absorb_count_physx - absorb_physx_prev
        absorb_sphere_epoch = mdp._absorb_count_sphere - absorb_sphere_prev

        with deterministic_policy(agent):
            dataset = core.evaluate(n_episodes=args.n_eval_episodes, quiet=True)
        J = torch.mean(dataset.discounted_return).item()
        R = torch.mean(dataset.undiscounted_return).item()
        ep_len = len(dataset) / args.n_eval_episodes
        m = compute_hold_metrics(dataset, mdp, args.hold_success_steps)
        c = compute_cost_metrics(dataset, args.n_eval_episodes)

        improved_J = J > best_J
        if improved_J:
            best_J = J
            agent.save(str(best_J_path))
            agent.save(str(best_J_path_flat))
        score = m['hold_success_rate'] * m['max_hold_mean']
        improved_score = m['hold_success_rate'] > 0 and score > best_score
        if improved_score:
            best_score = score

        hold_rate = m['hold_success_rate']
        max_hold = m['max_hold_mean']
        improved_hold = (
            hold_rate > best_hold_rate
            or (hold_rate == best_hold_rate and max_hold > best_hold_score)
        )
        if improved_hold and hold_rate > 0:
            best_hold_rate = hold_rate
            best_hold_score = max_hold
            agent.save(str(best_hold_path))
            agent.save(str(best_hold_path_flat))

        absorb_prev = mdp._absorb_count
        absorb_physx_prev = mdp._absorb_count_physx
        absorb_sphere_prev = mdp._absorb_count_sphere

        lambda_val = float(agent._log_lambda.exp().item())
        cost_violation = c['cost_rate'] - args.cost_limit

        logger.epoch_info(epoch + 1, J=J, R=R, best_J=best_J,
                          best_hold=best_hold_rate if best_hold_rate >= 0 else 0.0,
                          best_score=best_score,
                          cost_rate=c['cost_rate'],
                          lam=lambda_val,
                          absorb_epoch=absorb_epoch)
        logger.info("eval stats: "
                    f"hold_success_rate={m['hold_success_rate']:.3f}  "
                    f"max_hold_mean={m['max_hold_mean']:.1f}  "
                    f"in_thresh_rate={m['in_thresh_rate']:.3f}  "
                    f"pos_success_rate={m['pos_success_rate']:.3f}  "
                    f"pos_err_mean={m['pos_err_mean']:.4f}m  "
                    f"axis_err_mean={m['axis_err_mean']:.4f}")
        logger.info(f"  ↳ cost_rate={c['cost_rate']:.4f}  "
                    f"cost_ep_sum={c['cost_episode_sum_mean']:.3f}  "
                    f"violation={cost_violation:+.4f}  "
                    f"λ={lambda_val:.3f}  "
                    f"absorb_sphere={absorb_sphere_epoch}  "
                    f"absorb_physx={absorb_physx_epoch}")

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1, "env_steps": total_env_steps,
                "J": J, "R": R, "best_J": best_J, "best_score": best_score,
                "best_hold_rate": best_hold_rate if best_hold_rate >= 0 else 0.0,
                "best_hold_max_hold_mean": best_hold_score if best_hold_score >= 0 else 0.0,
                "eval_ep_len": ep_len,
                "eval_success_rate": m["hold_success_rate"],
                "eval_max_hold_mean": m["max_hold_mean"],
                "eval_in_thresh_rate": m["in_thresh_rate"],
                "eval_final_in_thresh_rate": m["final_in_thresh_rate"],
                "eval_pos_success_rate": m["pos_success_rate"],
                "eval_pos_err_mean": m["pos_err_mean"],
                "eval_axis_err_mean": m["axis_err_mean"],
                "alpha": agent._alpha.item(),
                # Lagrangian 专属
                "lambda": lambda_val,
                "log_lambda": float(agent._log_lambda.item()),
                "cost_rate": c["cost_rate"],
                "cost_episode_sum_mean": c["cost_episode_sum_mean"],
                "cost_violation": cost_violation,
                "cost_limit": args.cost_limit,
                # absorb 计数
                "absorb_per_epoch": absorb_epoch,
                "absorb_physx_per_epoch": absorb_physx_epoch,
                "absorb_sphere_per_epoch": absorb_sphere_epoch,
            }, step=epoch + 1)

    agent.save(str(final_path))
    agent.save(str(final_path_flat))

    if best_hold_rate < 0:
        best_hold_display = "n/a"
    else:
        best_hold_display = f"{best_hold_rate:.3f} (max_hold_mean={best_hold_score:.1f})"
    logger.info(
        f"训练完成. best J = {best_J:.3f}  "
        f"best_hold_rate = {best_hold_display}  "
        f"final λ = {float(agent._log_lambda.exp().item()):.3f}"
    )
    logger.info(f"checkpoint 写入: {ckpt_dir}/ 下的 "
                f"{best_J_path.name} / {best_hold_path.name} / {final_path.name}. "
                "**eval 时三个都跑一遍**, 注意 best_J 不一定满足 cost_limit, "
                "需手动看 wandb cost_rate 选 safe 子集.")

    if wandb_run is not None:
        wandb_run.summary["best_J"] = best_J
        wandb_run.summary["best_score"] = best_score
        wandb_run.summary["best_hold_rate"] = (
            best_hold_rate if best_hold_rate >= 0 else 0.0
        )
        wandb_run.summary["final_lambda"] = float(agent._log_lambda.exp().item())
        wandb_run.finish()
    mdp.stop()


if __name__ == "__main__":
    main()
