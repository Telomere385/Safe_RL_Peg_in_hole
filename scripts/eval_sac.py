"""评估训练好的 SAC agent.

运行:
    conda activate safe_rl
    python scripts/eval_sac.py              # 带渲染窗口, 默认验证胸前固定目标
    python scripts/eval_sac.py --headless   # 无窗口验证
    python scripts/eval_sac.py --target_travel_fraction 0.1   # 显式切回 fraction 模式
    python scripts/eval_sac.py --left_target -0.62 -0.55 0.69 --right_target -0.62 0.38 0.74
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._eval_utils import compute_hold_metrics, deterministic_policy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_path", type=str,
                   default=str(PROJECT_ROOT / "results/best_agent.msh"))
    p.add_argument("--n_episodes", type=int, default=16)
    p.add_argument("--num_envs", type=int, default=16,
                   help="与训练保持一致 (16). num_envs=1 会触发 cloner bug.")
    p.add_argument("--headless", action="store_true")
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
                   help="覆盖 env 默认 reset 关节噪声")
    p.add_argument("--success_pos_threshold", type=float, default=None,
                   help="覆盖 env 默认位置成功阈值")
    p.add_argument("--hold_success_steps", type=int, default=10,
                   help="验证 success 定义: episode 内至少出现连续 N 步都在阈值内")
    p.add_argument("--stochastic", action="store_true",
                   help="使用 SAC 采样策略评估. 默认使用 deterministic tanh(mu)")
    return p.parse_args()


def main():
    args = parse_args()

    if (args.left_target is None) != (args.right_target is None):
        raise ValueError("--left_target 和 --right_target 必须同时提供")
    if args.left_target is not None and args.target_travel_fraction is not None:
        raise ValueError("显式 fixed targets 与 --target_travel_fraction 不能同时使用")

    from envs import (
        DEFAULT_LEFT_CHEST_TARGET,
        DEFAULT_RIGHT_CHEST_TARGET,
        DualArmPegHoleEnv,
    )

    env_kwargs = dict(num_envs=args.num_envs, headless=args.headless)
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
    for key in ("initial_joint_noise", "success_pos_threshold"):
        value = getattr(args, key)
        if value is not None:
            env_kwargs[key] = value
    print(f"[EVAL ENV] {env_kwargs}")

    mdp = DualArmPegHoleEnv(**env_kwargs)

    from mushroom_rl.core import Agent, VectorCore
    agent = Agent.load(args.agent_path)
    core = VectorCore(agent, mdp)

    if args.stochastic:
        dataset = core.evaluate(n_episodes=args.n_episodes,
                                render=not args.headless, quiet=False)
    else:
        with deterministic_policy(agent):
            dataset = core.evaluate(n_episodes=args.n_episodes,
                                    render=not args.headless, quiet=False)
    J = torch.mean(dataset.discounted_return).item()
    R = torch.mean(dataset.undiscounted_return).item()
    print(f"J(γ)={J:.3f}  R={R:.3f}")

    m = compute_hold_metrics(dataset, mdp, args.hold_success_steps)
    print(
        f"hold_success_rate={m['hold_success_rate']:.3f} "
        f"(>= {args.hold_success_steps} consecutive steps)  "
        f"max_hold_mean={m['max_hold_mean']:.1f}  "
        f"in_thresh_rate={m['in_thresh_rate']:.3f}  "
        f"final_in_thresh_rate={m['final_in_thresh_rate']:.3f}  "
        f"left_err_mean={m['left_pos_err_mean']:.4f}m  "
        f"right_err_mean={m['right_pos_err_mean']:.4f}m"
    )

    mdp.stop()


if __name__ == "__main__":
    main()
