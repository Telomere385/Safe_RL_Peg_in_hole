"""SAC 训练 — 双臂 peg-in-hole preinsert (stage flag 化, mushroom-rl + VectorCore).

obs 默认 32 维 base (joint_pos+joint_vel+pos_vec+axis_dot); 推荐用
--use_axis_resid_obs 切到 34 维 (axis_resid 替换 axis_dot, 当前正式 Stage 1/2
都是 34 维). 同一个 env / 同一条 reward 骨架, stage 用 reward 权重 +
success_axis_threshold 切换:

    Stage 1 = pos-only         --rew_axis 0.0  --success_axis_threshold inf
    Stage 2 = pos + axis 对齐  --rew_axis 0.5  --success_axis_threshold 0.50
                              --axis_gate_radius 0.40 --rew_pos_success 1.0

Stage 2 用 --load_agent path/to/Stage1_checkpoint.msh + --actor_only_warmstart
续训: 只继承 Stage 1 actor 权重, critic/alpha/replay 冷启动. 旧 critic 按旧
reward 学, 用到 Stage 2 reward 上 Q 语义已错, 会把 actor 拽离 Stage 1 manifold.
另外 --critic_warmup_transitions 50000 让冷 critic 先单独学 ~40 epoch.

正式训练命令见 README.md "Stage 1 训练命令" / "Stage 2 训练命令" 段.

注意: num_envs=1 触发 IsaacSim cloner 的 `*` pattern 失败 → 至少 2.
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
    p.add_argument("--n_steps_per_epoch", type=int, default=1024,
                   help="每个 epoch 收集的总 env-step 数 (不是 vector-step)")
    p.add_argument("--n_steps_per_fit", type=int, default=None,
                   help="两次 fit 之间收集的总 env-step 数 (默认 = num_envs, 即 1 个 vector-step)")
    p.add_argument("--utd", type=int, default=None,
                   help="每次 fit 块对应的总梯度步数. 默认自动取 n_steps_per_fit, 使 true UTD≈1")
    p.add_argument("--lr_actor", type=float, default=3e-4,
                   help="cold-start 联合任务推荐降到 1e-4 (避免 noisy critic 拉坏 actor),"
                        "warm-start 用 default 即可.")
    p.add_argument("--lr_critic", type=float, default=3e-4)
    p.add_argument("--lr_alpha", type=float, default=3e-4)
    p.add_argument("--alpha_max", type=float, default=0.1,
                   help="alpha 上限. cold-start 联合任务推荐 0.2 (探索), warm-start 0.1 稳态.")
    p.add_argument("--target_entropy", type=float, default=None,
                   help="目标 entropy. 默认自动取 -act_dim (SAC 标准). "
                        "14-DoF cold-start 联合任务可考虑 -7 (= -act_dim/2), "
                        "让 SAC 倾向稍集中, alpha 不顶 cap.")
    p.add_argument("--critic_warmup_transitions", type=int, default=None,
                   help="actor 开始更新前需要的 replay 容量 (env-steps). 默认 = "
                        "INITIAL_REPLAY_SIZE (10K), 即 critic 仅领先 actor 1 步, "
                        "等价无 critic-only warmup. **actor-only warmstart 推荐设大** "
                        "(e.g. 50000 ≈ 40 epoch), 让冷启动的 critic 先用 warm-start "
                        "actor 收集的 trajectories 单独把 Q 学到合理量级, 再放开 actor, "
                        "避免随机 critic 把转移过来的 actor 推离 Stage 1 manifold. "
                        "必须 >= INITIAL_REPLAY_SIZE.")
    p.add_argument("--n_eval_episodes", type=int, default=None,
                   help="评估 episode 数. 默认自动取 num_envs, 并要求能被 num_envs 整除")
    p.add_argument("--initial_joint_noise", type=float, default=None,
                   help="覆盖 env 的 reset 关节噪声")
    p.add_argument("--preinsert_success_pos_threshold", type=float, default=None,
                   help="覆盖 env 的 preinsert 位置成功阈值 (env 默认 0.10m, 即当前 "
                        "Stage 1/Stage 2 curriculum). 如果显式想跑老 5cm, 传 0.05.")
    p.add_argument("--preinsert_offset", type=float, default=None,
                   help="覆盖 env 的 preinsert offset (默认 0.05m)")
    p.add_argument("--rew_action", type=float, default=None,
                   help="覆盖 env 的动作 L2 惩罚权重")
    p.add_argument("--rew_home", type=float, default=None,
                   help="home regularizer 权重 (joint-range 归一化的 ||q - q_home||²). "
                        "默认 0 关闭. 当前 Stage 1/Stage 2 建议 0.0005 当极弱 tie-breaker.")
    p.add_argument("--home_weights", type=parse_home_weights, default=None,
                   help="home regularizer 的逐关节权重. 接受 7 维单臂权重(自动复制到左右臂)"
                        "或 14 维完整权重, 逗号/空格分隔. 默认全 1. 例如: "
                        "'1,1,1,1,0.5,0.25,0.25'.")
    p.add_argument("--rew_success", type=float, default=None,
                   help="覆盖 env 的 per-step full_success (pos∧axis) bonus (默认 2.0)")
    p.add_argument("--rew_pos_success", type=float, default=None,
                   help="pos-only success bonus (env 默认 0.0). Stage 2 当前正式值 1.0: "
                        "维持 Stage 1 已学的'进 pos 阈值给 bonus'信号, 避免 Stage 2 把 axis 加上后"
                        "Stage 1 成功状态突然失去 bonus 造成 reward 断崖.")
    p.add_argument("--axis_gate_radius", type=float, default=None,
                   help="axis 惩罚的距离门控半径 (m). env 默认 inf = 不门控. "
                        "Stage 2 推荐 0.40: pos_err >= 0.40m 时 axis 项=0, 在 "
                        "[pos_th, 0.40m] 区间线性 ramp, 进 pos_th 后 gate 满.")
    p.add_argument("--rew_axis", type=float, default=None,
                   help="覆盖 env 的 axis_err 权重 (默认 0.0 = Stage 1 pos-only). "
                        "Stage 2 当前正式值 0.5 启用轴对齐惩罚.")
    p.add_argument("--success_axis_threshold", type=float, default=None,
                   help="覆盖 env 的 axis_err success 阈值 (默认 inf = Stage 1 不检查 axis). "
                        "Stage 2 训练阈值 0.50, 严格 eval 用 0.20/0.30. 接受 'inf' 字符串.")
    p.add_argument("--load_agent", type=str, default=None,
                   help="warm-start 路径: 从该 checkpoint 加载 agent (actor/critic/"
                        "optimizer state). obs 维度必须与 checkpoint 匹配 — 32 维 base "
                        "和 34 维 axis_resid 之间不能 warm-start.")
    p.add_argument("--keep_replay", action="store_true",
                   help="warm-start 时保留旧 replay buffer. 默认会清空 — 因为 stage "
                        "切换 (Stage 1→Stage 2) reward 函数变了, 旧 transitions 的 "
                        "reward 标签按旧 reward 算, 留着会拖 critic.")
    p.add_argument("--actor_only_warmstart", action="store_true",
                   help="warm-start 时只继承 actor (mu/sigma 网络) 权重, critic / "
                        "alpha / optimizers / replay buffer 全部冷启动. stage 切换"
                        "(Stage 1→Stage 2 reward shape 变化) 时强烈建议打开 — 否则旧 critic "
                        "按旧 reward 学的 Q 语义会拖坏 actor (Stage 2 一上来 actor 就被拽离 "
                        "Stage 1 learned manifold). 此 flag 打开时 --keep_replay 自动失效.")
    p.add_argument("--terminal_hold_bonus", type=float, default=None,
                   help="hold-N 步成功后的终结 bonus + episode 终止. "
                        "0 = 关闭 (baseline). >0 启用 absorbing termination.")
    p.add_argument("--hold_success_steps", type=int, default=10,
                   help="eval success 定义 + env 终止阈值: 连续 N 步都在阈值内. "
                        "N=10 ≈ 1s hold (per-step dt≈0.1s).")
    p.add_argument("--clearance_hard", type=float, default=None,
                   help="覆盖 env 的 sphere-proxy 自碰撞兜底阈值 (m). 默认 0.0 = 球壳一接触即"
                        "触发 hard absorbing. 关闭时写 --clearance_hard=-inf, 只信 PhysX 力检测.")
    p.add_argument("--proxy_arm_radius", type=float, default=None,
                   help="覆盖 env 的 arm sphere proxy 半径 (默认 0.06m).")
    p.add_argument("--proxy_ee_radius", type=float, default=None,
                   help="覆盖 env 的 EE sphere proxy 半径 (默认 0.04m).")
    p.add_argument("--use_axis_resid_obs", action="store_true",
                   help="agent obs 32 → 34 维: axis_dot[1] 替换成 axis_resid[3] = "
                        "peg_axis + hole_axis (world frame). 模长 ∈ [0, 2], "
                        "0 = 完美反对齐 (preinsert), 2 = 同向 (home pose). "
                        "全程光滑无奇异, 且 axis_err = ||resid||²/2 = 1+dot 与 reward 同语义, "
                        "success_axis_threshold 仍然用旧的 1+dot 量纲 (0.2 ≈ ±37° 锥). "
                        "**注意**: obs 维度变, 32 维 checkpoint 不能 warm-start 到 34 维.")
    p.add_argument("--wandb_project", type=str, default="bimanual_peghole")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
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
            f"n_steps_per_fit ({args.n_steps_per_fit}) 必须能被 num_envs ({args.num_envs}) 整除"
        )
    if args.n_steps_per_epoch % args.n_steps_per_fit != 0:
        raise ValueError(
            f"n_steps_per_epoch ({args.n_steps_per_epoch}) 必须能被 "
            f"n_steps_per_fit ({args.n_steps_per_fit}) 整除"
        )
    if args.critic_warmup_transitions is None:
        args.critic_warmup_transitions = INITIAL_REPLAY_SIZE
    if args.critic_warmup_transitions < INITIAL_REPLAY_SIZE:
        raise ValueError(
            f"--critic_warmup_transitions ({args.critic_warmup_transitions}) 必须 >= "
            f"INITIAL_REPLAY_SIZE ({INITIAL_REPLAY_SIZE}); 否则 replay 还没 initialized "
            "actor gate 已经打开, 设置无意义."
        )
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
    # bool flags: 直接读 args, 不能用 None 哨兵 (action="store_true" 默认 False)
    if args.use_axis_resid_obs:
        env_kwargs["use_axis_resid_obs"] = True
    env_kwargs["success_hold_steps"] = args.hold_success_steps
    mdp = DualArmPegHoleEnv(**env_kwargs)
    mdp.seed(args.seed)

    # IsaacSim 启动后才能导入 mushroom_rl (避免 carb 冲突)
    from mushroom_rl.algorithms.actor_critic import SAC
    from mushroom_rl.core import Agent, VectorCore, Logger, Dataset

    obs_dim = mdp.info.observation_space.shape[0]
    act_dim = mdp.info.action_space.shape[0]
    target_entropy = args.target_entropy
    if target_entropy is None:
        target_entropy = -float(act_dim)

    def _cold_create_sac():
        actor_params = dict(network=ActorNetwork, input_shape=(obs_dim,),
                            output_shape=(act_dim,))
        actor_optimizer = {"class": optim.Adam, "params": {"lr": args.lr_actor}}
        critic_params = dict(network=CriticNetwork, input_shape=(obs_dim,),
                             output_shape=(1,), action_dim=act_dim,
                             optimizer={"class": optim.Adam, "params": {"lr": args.lr_critic}},
                             loss=F.mse_loss)
        return SAC(
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
        )

    if args.load_agent is not None:
        load_path = Path(args.load_agent)
        if not load_path.is_file():
            raise FileNotFoundError(f"--load_agent 路径不存在: {load_path}")
        old_agent = Agent.load(str(load_path))

        if args.actor_only_warmstart:
            # Cold-create 一个新 SAC (匹配当前 Stage 2 reward / env), 然后只把旧 actor
            # (mu / sigma 网络) 的权重拷过来. critic / alpha / optimizers / replay
            # 全部新, 避免旧 critic 用过时 Q 语义拽 actor.
            agent = _cold_create_sac()
            agent.policy._mu_approximator.set_weights(
                old_agent.policy._mu_approximator.get_weights()
            )
            agent.policy._sigma_approximator.set_weights(
                old_agent.policy._sigma_approximator.get_weights()
            )
            print(f"[WARM-START actor-only] 仅继承 Stage 1 actor 权重 from {load_path}; "
                  "critic / alpha / replay 全部冷启动.")
            if args.keep_replay:
                print("[WARM-START actor-only] --keep_replay 已忽略 (replay 强制冷启).")
            del old_agent  # 释放旧 agent 引用
        else:
            # 全量 warm-start: 继承 actor + critic + optimizer + alpha (+ optionally replay).
            # obs 维度必须与 checkpoint 完全一致, 否则 forward 抛 shape 错.
            agent = old_agent
            print(f"[WARM-START full] 整体加载 agent from {load_path}")
            if not args.keep_replay:
                agent._replay_memory.reset()
                print("[WARM-START full] replay buffer 已清空 — 重新走 INITIAL_REPLAY_SIZE 填充. "
                      "若要保留旧 buffer, 加 --keep_replay.")
            else:
                print("[WARM-START full] 保留旧 replay buffer (--keep_replay).")
    else:
        agent = _cold_create_sac()
    def clamp_alpha(_dataset=None):
        with torch.no_grad():
            agent._log_alpha.clamp_(max=math.log(args.alpha_max))

    core = VectorCore(agent, mdp, callbacks_fit=[clamp_alpha])

    from datetime import datetime
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    run_ts = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    ckpt_dir = results_dir / "checkpoints" / run_ts
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # 三种 checkpoint, 各自独立追踪, 互不覆盖:
    #   best_J.msh        - 最高 discounted return J (=best_agent.msh 的别名, 向后兼容)
    #   best_hold.msh     - 最高 hold_success_rate (出现 hold 之后才会写入)
    #   final_agent.msh   - 训练结束时无条件保存最后一个 epoch 的 actor
    # 防止 "终态稳定 hold=1.0 但 J 没破 best_J 上限" 这种情况下 best_agent 锁在
    # 早期低 hold 的快照, 独立 eval 评的不是真实最终策略.
    # ckpt_dir: 按时间归档, 各次运行互不覆盖.
    # results_dir: 直接保存一份, 方便 eval 脚本直接用.
    best_J_path = ckpt_dir / "best_agent.msh"
    best_hold_path = ckpt_dir / "best_hold.msh"
    final_path = ckpt_dir / "final_agent.msh"
    best_J_path_flat = results_dir / "best_agent.msh"
    best_hold_path_flat = results_dir / "best_hold.msh"
    final_path_flat = results_dir / "final_agent.msh"
    logger = Logger("SAC", results_dir=str(results_dir))
    logger.strong_line()
    logger.info(f"checkpoint 目录: {ckpt_dir}")
    obs_mode = "axis_resid" if mdp._use_axis_resid_obs else "base"
    logger.info(f"obs_dim={obs_dim} ({obs_mode})  "
                f"act_dim={act_dim}  horizon={mdp.info.horizon}")
    logger.info(f"action_scale={mdp._action_scale:.3f}")
    logger.info(f"preinsert_pos_th={mdp._preinsert_success_pos_threshold:.3f}m  "
                f"axis_th={mdp._success_axis_threshold:.3f}  "
                f"w_pos={mdp._w_pos:.3f}  w_axis={mdp._w_axis:.3f}  "
                f"w_pos_success={mdp._w_pos_success:.3f}  "
                f"w_success={mdp._w_success:.3f}  "
                f"axis_gate_radius={mdp._axis_gate_radius:.3f}m  "
                f"w_home={mdp._w_home:.4f}  "
                f"preinsert_offset={mdp._preinsert_offset:.3f}m")
    if mdp._w_home > 0:
        logger.info(
            "home_weights="
            + ",".join(f"{float(w):.3g}" for w in mdp._home_weights.detach().cpu())
        )
    if args.load_agent is not None:
        logger.info(f"warm-start: {args.load_agent}")
    logger.info(f"target_entropy={target_entropy:.3f}  "
                f"lr_actor={args.lr_actor:.1e}  lr_critic={args.lr_critic:.1e}  "
                f"lr_alpha={args.lr_alpha:.1e}  alpha_max={args.alpha_max:.3f}")
    critic_only_steps = args.critic_warmup_transitions - INITIAL_REPLAY_SIZE
    if critic_only_steps > 0:
        critic_only_epochs = critic_only_steps / args.n_steps_per_epoch
        logger.info(
            f"critic_warmup_transitions={args.critic_warmup_transitions} env-steps "
            f"(replay-fill {INITIAL_REPLAY_SIZE} + critic-only {critic_only_steps} ≈ "
            f"{critic_only_epochs:.1f} epoch, actor 此期间冻结)"
        )
    else:
        logger.info(
            f"critic_warmup_transitions={args.critic_warmup_transitions} env-steps "
            "(无 critic-only 期, actor 与 critic 几乎同时开启)"
        )
    logger.info(f"n_steps_per_epoch={args.n_steps_per_epoch} env-steps  "
                f"n_steps_per_fit={args.n_steps_per_fit} env-steps  "
                f"num_envs={args.num_envs}")

    mask = torch.ones(args.num_envs, dtype=torch.bool, device=mdp._device)
    obs, _ = mdp.reset_all(mask)
    pos_err, axis_err, in_thresh_mask = mdp._compute_task_errors(obs)
    logger.info("reset stats: "
                f"in_thresh_rate={float(in_thresh_mask.float().mean()):.3f}  "
                f"pos_err_mean={float(pos_err.mean()):.4f}m  "
                f"pos_err_min={float(pos_err.min()):.4f}m  "
                f"pos_err_max={float(pos_err.max()):.4f}m  "
                f"axis_err_mean={float(axis_err.mean()):.4f}  "
                f"axis_err_max={float(axis_err.max()):.4f}")

    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, name=args.wandb_run_name,
            group=args.wandb_group,
            config={**vars(args), "algo": "SAC",
                    "target_entropy_resolved": target_entropy,
                    "obs_dim": obs_dim, "act_dim": act_dim,
                    "horizon": mdp.info.horizon, "gamma": mdp.info.gamma},
            dir=str(results_dir),
        )
        logger.info(f"wandb run: {wandb_run.url}")

    empty_dataset = Dataset.generate(mdp.info, agent.info, n_steps=1, n_envs=args.num_envs)

    # Epoch-0 eval: warm-start actor 在 *任何训练之前* 的表现. 用来确认 actor
    # 权重转移真的生效 (actor-only warmstart 后 pos_err 应该接近 Stage 1 收敛水平).
    # 如果这里 pos_success_rate 已经接近 0, 后面再讨论训练失败就没意义了 —
    # actor 转移本身就没成功.
    if args.load_agent is not None:
        logger.info("=" * 60)
        logger.info("[EVAL @ epoch 0] warm-start actor BEFORE 任何 fit / 任何 warmup")
        with deterministic_policy(agent):
            ds0 = core.evaluate(n_episodes=args.n_eval_episodes, quiet=True)
        m0 = compute_hold_metrics(ds0, mdp, args.hold_success_steps)
        logger.info(f"  pos_success_rate={m0['pos_success_rate']:.3f}  "
                    f"pos_err_mean={m0['pos_err_mean']:.4f}m  "
                    f"axis_err_mean={m0['axis_err_mean']:.4f}  "
                    f"hold_success_rate={m0['hold_success_rate']:.3f}")
        logger.info(f"  conditional (pos_in_thresh count={m0['pos_in_thresh_count']}):  "
                    f"axis_err_in_pos_th_mean={m0['axis_err_in_pos_thresh_mean']:.4f}  "
                    f"axis_err_in_pos_th_min={m0['axis_err_in_pos_thresh_min']:.4f}")
        logger.info("=" * 60)
        if wandb_run is not None:
            wandb_run.log({
                "epoch": 0,
                "warmstart_pos_success_rate": m0["pos_success_rate"],
                "warmstart_pos_err_mean": m0["pos_err_mean"],
                "warmstart_axis_err_mean": m0["axis_err_mean"],
                "warmstart_hold_success_rate": m0["hold_success_rate"],
                "warmstart_axis_err_in_pos_thresh_mean":
                    m0["axis_err_in_pos_thresh_mean"]
                    if m0["pos_in_thresh_count"] > 0 else 0.0,
            }, step=0)

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
    best_score = -np.inf
    # best_hold 用 (hold_success_rate, max_hold_mean) 二级 key:
    # hold_success_rate 是首要指标 (0-1, 大多数 epoch 是 0), max_hold_mean 当 tie-breaker
    # (0-N, hold rate 相同时偏好平均 hold 更长的). 两者都要 > 之前才更新.
    best_hold_rate = -1.0  # 用 -1 而非 0, 让首次出现 hold_rate=0.0 时不会触发"改进"
    best_hold_score = -1.0  # tie-breaker: max_hold_mean
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

        improved_J = J > best_J
        if improved_J:
            best_J = J
            agent.save(str(best_J_path))
            agent.save(str(best_J_path_flat))
        score = m['hold_success_rate'] * m['max_hold_mean']
        improved_score = m['hold_success_rate'] > 0 and score > best_score
        if improved_score:
            best_score = score

        # best_hold 独立追踪: hold_rate 严格超过历史 → 保存; 持平 → 看 max_hold_mean.
        # 这样能捕到 "终态 hold=1.0 但 J 不再涨" 的稳态策略, 不会被早期高 J 低 hold 覆盖.
        hold_rate = m['hold_success_rate']
        max_hold = m['max_hold_mean']
        improved_hold = (
            hold_rate > best_hold_rate
            or (hold_rate == best_hold_rate and max_hold > best_hold_score)
        )
        # 只有 hold_rate > 0 才写 best_hold.msh, 避免早期 0/0 持平也写一遍
        if improved_hold and hold_rate > 0:
            best_hold_rate = hold_rate
            best_hold_score = max_hold
            agent.save(str(best_hold_path))
            agent.save(str(best_hold_path_flat))

        absorb_prev = mdp._absorb_count
        absorb_physx_prev = mdp._absorb_count_physx
        absorb_sphere_prev = mdp._absorb_count_sphere

        logger.epoch_info(epoch + 1, J=J, R=R, best_J=best_J,
                          best_hold=best_hold_rate if best_hold_rate >= 0 else 0.0,
                          best_score=best_score,
                          absorb_epoch=absorb_epoch,
                          absorb_physx=absorb_physx_epoch,
                          absorb_sphere=absorb_sphere_epoch)
        logger.info("eval stats: "
                    f"hold_success_rate={m['hold_success_rate']:.3f} "
                    f"(>= {args.hold_success_steps} consecutive steps)  "
                    f"max_hold_mean={m['max_hold_mean']:.1f}  "
                    f"in_thresh_rate={m['in_thresh_rate']:.3f}  "
                    f"final_in_thresh_rate={m['final_in_thresh_rate']:.3f}  "
                    f"pos_success_rate={m['pos_success_rate']:.3f}  "
                    f"pos_err_mean={m['pos_err_mean']:.4f}m  "
                    f"axis_err_mean={m['axis_err_mean']:.4f}  "
                    f"axis_gate_mean={m['axis_gate_mean']:.3f}  "
                    f"gated_axis_pen={m['gated_axis_penalty_mean']:.3f}")
        # 条件指标: 关键证据是 'pos_in_thresh 时 axis_err 是否下降'.
        # pos_in_thresh_count=0 时 NaN — 用 'n/a' 显示, 避免误读成 0.
        if m['pos_in_thresh_count'] > 0:
            cond_str = (f"axis_err_in_pos_th_mean={m['axis_err_in_pos_thresh_mean']:.4f}  "
                        f"axis_err_in_pos_th_min={m['axis_err_in_pos_thresh_min']:.4f}  "
                        f"axis_gate_in_pos_th_mean={m['axis_gate_in_pos_thresh_mean']:.3f}")
        else:
            cond_str = "axis_err_in_pos_th=n/a (pos_in_thresh_count=0)"
        logger.info(f"  ↳ pos_in_thresh_count={m['pos_in_thresh_count']}  {cond_str}")
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1, "env_steps": total_env_steps,
                "J": J, "R": R, "best_J": best_J, "best_score": best_score,
                "best_hold_rate":
                    best_hold_rate if best_hold_rate >= 0 else 0.0,
                "best_hold_max_hold_mean":
                    best_hold_score if best_hold_score >= 0 else 0.0,
                "eval_ep_len": ep_len,
                "eval_success_rate": m["hold_success_rate"],
                "eval_max_hold_mean": m["max_hold_mean"],
                "eval_in_thresh_rate": m["in_thresh_rate"],
                "eval_final_in_thresh_rate": m["final_in_thresh_rate"],
                "eval_pos_success_rate": m["pos_success_rate"],
                "eval_pos_err_mean": m["pos_err_mean"],
                "eval_axis_err_mean": m["axis_err_mean"],
                "eval_axis_gate_mean": m["axis_gate_mean"],
                "eval_gated_axis_penalty_mean": m["gated_axis_penalty_mean"],
                "eval_pos_in_thresh_count": m["pos_in_thresh_count"],
                "eval_axis_err_in_pos_thresh_mean":
                    m["axis_err_in_pos_thresh_mean"]
                    if m["pos_in_thresh_count"] > 0 else float("nan"),
                "eval_axis_gate_in_pos_thresh_mean":
                    m["axis_gate_in_pos_thresh_mean"]
                    if m["pos_in_thresh_count"] > 0 else float("nan"),
                "alpha": agent._alpha.item(),
                "absorb_per_epoch": absorb_epoch,
                "absorb_physx_per_epoch": absorb_physx_epoch,
                "absorb_sphere_per_epoch": absorb_sphere_epoch,
            }, step=epoch + 1)

    # 训练结束时无条件保存最后一个 epoch 的 actor — 这是 "稳态终态" 的 ground truth,
    # 独立 eval 应当至少评一次它, 跟 best_J / best_hold 对照, 避免 "best_J 锁早期低 hold
    # 快照, 终态 hold=1.0 但因 J 未破上限被永远丢失" 这类 silent failure.
    agent.save(str(final_path))
    agent.save(str(final_path_flat))

    if best_hold_rate < 0:
        best_hold_display = "n/a (no hold success)"
    else:
        best_hold_display = f"{best_hold_rate:.3f} (max_hold_mean={best_hold_score:.1f})"
    logger.info(
        f"训练完成. best J = {best_J:.3f}  "
        f"best_hold_rate = {best_hold_display}  "
        f"best_score = {best_score:.3f}"
    )
    logger.info(f"checkpoint 写入: {ckpt_dir}/ 下的 "
                f"{best_J_path.name} (best J) / "
                f"{best_hold_path.name} (best hold) / {final_path.name} (final). "
                "**eval 时务必对三个都跑一遍**, best_J 不一定 = 最稳策略.")

    if wandb_run is not None:
        wandb_run.summary["best_J"] = best_J
        wandb_run.summary["best_score"] = best_score
        wandb_run.summary["best_hold_rate"] = (
            best_hold_rate if best_hold_rate >= 0 else 0.0
        )
        wandb_run.summary["best_hold_max_hold_mean"] = (
            best_hold_score if best_hold_score >= 0 else 0.0
        )
        wandb_run.finish()
    mdp.stop()


if __name__ == "__main__":
    main()
