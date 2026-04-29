"""M1 → M2 切换前的诊断脚本: 统计当前 M1 policy 在 pos_err 进阈时的 axis_err 分布.

目的:
    M1 训练时 obs 不含姿态、reward 不罚姿态. 切换到 M2 (加 axis 项) 之前需要知道:
    "M1 学出的 reach 行为, 在 pos<0.10 这个圆锥里, 姿态究竟有多差?"

    - 如果 axis_err 平均 ~0.5 (dot ~ -0.5): M1 已隐式带来一些姿态对齐, M2a 可能 30-50 epoch 收敛
    - 如果 axis_err 平均 ~1.5+ (dot ~ +0.5): M2a 是真任务, 预期 100-200 epoch

实现:
    env 已经是 32 维 (axis_dot 在 obs 里), 但 31 维 M1 checkpoint 的 actor 输入层
    是 31 维, 直接喂 32 维会 shape mismatch. 这里做两件事:

    1. monkey-patch _create_observation: 算完 32 维 obs 后顺便从
       _compute_preinsert_errors() 抓全量几何 (pos/axis/radial/axial), 然后**截到
       31 维** (丢掉 axis_dot) 喂给 31-dim M1 agent.

    2. patch mdp.info.observation_space 到 31 维 (slice low/high), 否则
       VectorCore.evaluate 会按 32 维构造 Dataset, env 实际返回 31 维就不一致.

    这是给"用 31-dim M1 checkpoint 在 32-dim env 上跑诊断"专用的 one-shot 脚本.
    M1' (32-dim) 训出来之后就不需要这个 adapter, 直接 eval_sac.py 就能看到
    axis_err_mean.

运行:
    conda activate safe_rl
    python scripts/archive/diagnose_m1_axis.py --headless --num_envs 16 --n_episodes 64

输出会同时打印整体分布和"pos_err < threshold"条件下的子集分布.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# 归档目录在 scripts/archive/, 项目根需 parents[2] 才到 bimanual_peghole/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._eval_utils import deterministic_policy, resolve_eval_episode_count


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_path", type=str,
                   default=str(PROJECT_ROOT / "results/best_agent_M1_31dim_pos10cm.msh"),
                   help="M1 31-dim checkpoint. 默认指向 backup, 避免被后续训练覆盖.")
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--n_episodes", type=int, default=None,
                   help="默认 = num_envs. 必须能被 num_envs 整除.")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--render", action="store_true",
                   help="覆盖 --headless, 显示 IsaacSim 窗口")
    p.add_argument("--initial_joint_noise", type=float, default=None,
                   help="覆盖 reset 噪声. 应与 train 时一致 (默认 0.1)")
    p.add_argument("--preinsert_success_pos_threshold", type=float, default=0.10,
                   help="pos_err 进阈阈值 (诊断条件). 应与 M1 训练时一致.")
    p.add_argument("--preinsert_offset", type=float, default=None)
    p.add_argument("--terminal_hold_bonus", type=float, default=None,
                   help="若 M1 训练时启用, 这里也要传同值, 否则 hold-N 终止行为不同.")
    p.add_argument("--hold_success_steps", type=int, default=10)
    return p.parse_args()


def _percentile(arr, q):
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def _print_dist(name, arr):
    if arr.size == 0:
        print(f"  {name:14s} (空, 没有匹配步数)")
        return
    print(
        f"  {name:14s} n={arr.size:6d}  "
        f"mean={float(arr.mean()):+.4f}  "
        f"median={_percentile(arr, 50):+.4f}  "
        f"min={float(arr.min()):+.4f}  "
        f"max={float(arr.max()):+.4f}  "
        f"std={float(arr.std()):+.4f}  "
        f"p10={_percentile(arr, 10):+.4f}  "
        f"p90={_percentile(arr, 90):+.4f}"
    )


def _print_axis_err_buckets(arr):
    """axis_err ∈ [0, 2], 0=完美对齐, 2=完全反向. 按粗 / 中 / 细分桶."""
    if arr.size == 0:
        print("  axis_err 分桶: 空")
        return
    edges = [0.0, 0.05, 0.2, 0.5, 1.0, 1.5, 2.01]
    labels = [
        "[0.00, 0.05)  axis_th 0.05 (M3 紧)",
        "[0.05, 0.20)  axis_th 0.2  (M2b)",
        "[0.20, 0.50)  axis_th 0.5  (M2a)",
        "[0.50, 1.00)  半锥",
        "[1.00, 1.50)  接近垂直",
        "[1.50, 2.00]  几乎反向 (默认 default 区域)",
    ]
    counts, _ = np.histogram(arr, bins=edges)
    total = arr.size
    print("  axis_err 累计分布 (M2 阈值预估):")
    for label, c in zip(labels, counts):
        pct = 100.0 * c / total
        print(f"    {label:42s}  count={c:6d}  ({pct:5.1f}%)")


def main():
    args = parse_args()
    if args.render:
        args.headless = False
    args.n_episodes = resolve_eval_episode_count(
        args.n_episodes, args.num_envs, "--n_episodes"
    )

    if not Path(args.agent_path).is_file():
        raise FileNotFoundError(
            f"agent_path 不存在: {args.agent_path}\n"
            "如果还没备份 M1, 先跑: cp results/best_agent.msh "
            "results/best_agent_M1_31dim_pos10cm.msh"
        )

    from envs import DualArmPegHoleEnv

    env_kwargs = dict(num_envs=args.num_envs, headless=args.headless)
    for key in ("initial_joint_noise", "preinsert_success_pos_threshold",
                "preinsert_offset", "terminal_hold_bonus"):
        value = getattr(args, key)
        if value is not None:
            env_kwargs[key] = value
    env_kwargs["success_hold_steps"] = args.hold_success_steps
    print(f"[DIAG ENV] {env_kwargs}")
    print(f"[DIAG] agent_path: {args.agent_path}")

    mdp = DualArmPegHoleEnv(**env_kwargs)

    # ---- patch 1: obs 截到 31 维, 适配 31-dim M1 checkpoint ----
    # env 现在返回 32 维 (含 axis_dot), 但加载的 31-dim agent actor 输入层是 31 维.
    # 这里 _create_observation 仍然算完 32 维拿到完整 axis_dot 给 _compute_preinsert_errors,
    # 但返回时丢掉最后一维, 喂给 agent 的就是 31-dim 老格式.
    samples = {
        "pos_err": [],
        "axis_err": [],
        "axis_dot": [],
        "radial_err": [],
        "axial_dist": [],
    }
    sample_failures = {"count": 0, "first_error": None}
    original_create_obs = mdp._create_observation

    def patched_create_obs(raw_obs):
        obs_32 = original_create_obs(raw_obs)
        try:
            errs = mdp._compute_preinsert_errors()
            for key in samples:
                samples[key].append(errs[key].detach().cpu().numpy().copy())
        except Exception as e:
            # 不在 hot path 里 raise (会打断 evaluate), 但记下来, 跑完检查.
            sample_failures["count"] += 1
            if sample_failures["first_error"] is None:
                sample_failures["first_error"] = repr(e)
        return obs_32[..., :31]  # 截掉 axis_dot, 给 31-dim M1 agent

    mdp._create_observation = patched_create_obs

    # ---- patch 2: mdp.info.observation_space 也截到 31 维 ----
    # 否则 VectorCore.evaluate 会按 mdp.info 构造 Dataset (32 维), 而 env 实际
    # 返回 31 维就 shape mismatch. 必须在创建 VectorCore 之前 patch.
    from mushroom_rl.rl_utils.spaces import Box
    old_space = mdp.info.observation_space
    legacy_low = old_space.low[:31].clone()
    legacy_high = old_space.high[:31].clone()
    mdp.info.observation_space = Box(legacy_low, legacy_high, data_type=old_space.data_type)
    print(f"[DIAG] obs_space patched: {old_space.shape} → {mdp.info.observation_space.shape} "
          "(31-dim legacy mode for M1 checkpoint)")

    from mushroom_rl.core import Agent, VectorCore
    agent = Agent.load(args.agent_path)
    core = VectorCore(agent, mdp)

    print(f"[DIAG] 跑 {args.n_episodes} 个 episode (deterministic policy)...")
    with deterministic_policy(agent):
        dataset = core.evaluate(n_episodes=args.n_episodes,
                                render=not args.headless, quiet=False)

    mdp._create_observation = original_create_obs
    mdp.stop()

    # 几何采样失败检查 — 必须在 aggregate 之前 raise, 否则空数组/nan 会让
    # 下面的统计输出看起来"正常"但实际全错.
    if sample_failures["count"] > 0:
        raise RuntimeError(
            f"_compute_preinsert_errors 在 {sample_failures['count']} 个 step 上失败. "
            f"first error: {sample_failures['first_error']}"
        )
    if not samples["pos_err"]:
        raise RuntimeError(
            "几何采样为空 — patched _create_observation 没被调用过. "
            "可能 evaluate 提前退出, 或 monkey-patch 没生效."
        )

    # aggregate
    pos_err = np.concatenate(samples["pos_err"])
    axis_err = np.concatenate(samples["axis_err"])
    axis_dot = np.concatenate(samples["axis_dot"])
    radial_err = np.concatenate(samples["radial_err"])
    axial_dist = np.concatenate(samples["axial_dist"])

    pos_th = args.preinsert_success_pos_threshold
    in_thresh_mask = pos_err < pos_th
    print()
    print(f"[DIAG] 总 step 数 = {pos_err.size},  "
          f"pos_err < {pos_th:.3f} 的 step = {in_thresh_mask.sum()} "
          f"({100.0 * in_thresh_mask.mean():.2f}%)")

    print()
    print("[DIAG 整体分布 — 所有 step]")
    _print_dist("pos_err",    pos_err)
    _print_dist("axis_err",   axis_err)
    _print_dist("axis_dot",   axis_dot)
    _print_dist("radial_err", radial_err)
    _print_dist("axial_dist", axial_dist)

    if in_thresh_mask.any():
        print()
        print(f"[DIAG 子集 — pos_err < {pos_th:.3f} 时的姿态状态]  ← M2 起点决策")
        _print_dist("pos_err",    pos_err[in_thresh_mask])
        _print_dist("axis_err",   axis_err[in_thresh_mask])
        _print_dist("axis_dot",   axis_dot[in_thresh_mask])
        _print_dist("radial_err", radial_err[in_thresh_mask])
        _print_dist("axial_dist", axial_dist[in_thresh_mask])
        print()
        _print_axis_err_buckets(axis_err[in_thresh_mask])

        # 一句话推断
        am = float(axis_err[in_thresh_mask].mean())
        print()
        if am < 0.6:
            verdict = "M1 已隐式带来明显姿态对齐 → M2a 预期较快收敛 (~30-80 epoch)"
        elif am < 1.2:
            verdict = "M1 姿态对齐部分进展 → M2a 预期常规 (~80-150 epoch)"
        else:
            verdict = "M1 几乎完全没学姿态 → M2a 是真任务 (~150-300 epoch), w_axis 起步 1.0 不够大可考虑 2.0"
        print(f"[DIAG VERDICT]  axis_err 子集均值 = {am:+.3f}  →  {verdict}")
    else:
        print()
        print("[DIAG] 警告: 没有 step 满足 pos_err < threshold. M1 没收敛, "
              "或 --preinsert_success_pos_threshold 设得过严.")


if __name__ == "__main__":
    main()
