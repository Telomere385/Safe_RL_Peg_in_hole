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

from scripts._eval_utils import (
    compute_hold_metrics,
    deterministic_policy,
    resolve_eval_episode_count,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_path", type=str,
                   default=str(PROJECT_ROOT / "results/best_agent.msh"))
    p.add_argument("--n_episodes", type=int, default=None,
                   help="评估 episode 数. 默认自动取 num_envs, 并要求能被 num_envs 整除, "
                        "避免尾批 inactive env 被 teleport away")
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
                   help="覆盖 env 默认 reset 关节噪声. 为公平复现 train, 应传与 train 同样的值")
    p.add_argument("--success_pos_threshold", type=float, default=None,
                   help="覆盖 env 默认位置成功阈值. 为公平复现 train, 应传与 train 同样的值")
    p.add_argument("--terminal_hold_bonus", type=float, default=None,
                   help="hold-N 步成功后的终结 bonus + episode 终止. "
                        "**train 时若启用了它, eval 也必须传同样的值**, 否则 absorbing 不触发, "
                        "agent 跑满 horizon 进入未训练区域, 看起来像 drift / 失败.")
    p.add_argument("--hold_success_steps", type=int, default=10,
                   help="验证 success 定义: episode 内至少出现连续 N 步都在阈值内. "
                        "若 --terminal_hold_bonus > 0, 这个 N 也是 env absorbing 触发条件.")
    p.add_argument("--stochastic", action="store_true",
                   help="使用 SAC 采样策略评估. 默认使用 deterministic tanh(mu)")
    return p.parse_args()


def main():
    args = parse_args()
    args.n_episodes = resolve_eval_episode_count(
        args.n_episodes, args.num_envs, "--n_episodes"
    )

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
    for key in ("initial_joint_noise", "success_pos_threshold", "terminal_hold_bonus"):
        value = getattr(args, key)
        if value is not None:
            env_kwargs[key] = value
    # absorbing 终止 N 与 eval metric 共用 (与 train_sac 同步)
    env_kwargs["success_hold_steps"] = args.hold_success_steps
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
