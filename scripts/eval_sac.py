"""评估训练好的 SAC agent (32 维 obs, stage flag 化 reward).

eval 时 --rew_axis / --success_axis_threshold 应与训练时保持一致, 否则
success / hold-N 触发条件不同, 数字没有可比性.

运行:
    conda activate safe_rl
    # M1' (pos-only)
    python scripts/eval_sac.py --headless --num_envs 16 --n_episodes 64 \\
        --agent_path results/best_agent_M1p_32dim_pos10cm.msh \\
        --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50 \\
        --rew_home 0.0005

    # M2 (pos + axis, axis-gate + pos_success bonus)
    python scripts/eval_sac.py --headless --num_envs 16 --n_episodes 64 \\
        --agent_path results/best_agent_M2_axis02.msh \\
        --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50 \\
        --rew_home 0.0005 \\
        --rew_axis 1.0 --success_axis_threshold 0.2 \\
        --rew_pos_success 1.0 --axis_gate_radius 0.40
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
                   help="评估 episode 数. 默认自动取 num_envs, 并要求能被 num_envs 整除")
    p.add_argument("--num_envs", type=int, default=16,
                   help="与训练保持一致 (16). num_envs=1 会触发 cloner bug.")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--initial_joint_noise", type=float, default=None,
                   help="覆盖 env 默认 reset 关节噪声. 应传与 train 相同的值")
    p.add_argument("--preinsert_success_pos_threshold", type=float, default=None,
                   help="覆盖 env 默认 preinsert 位置成功阈值 (env 默认 0.10m). "
                        "应传与 train 相同的值.")
    p.add_argument("--preinsert_offset", type=float, default=None,
                   help="覆盖 env 默认 preinsert offset. 应传与 train 相同的值")
    p.add_argument("--rew_axis", type=float, default=None,
                   help="覆盖 env 的 axis_err 权重. **eval 不算 reward 主项, 但 visualize 会读**, "
                        "保留 CLI 一致性. 默认 0 = M1' 行为.")
    p.add_argument("--rew_pos_success", type=float, default=None,
                   help="覆盖 env 的 pos-only success bonus. 应与 train 一致.")
    p.add_argument("--axis_gate_radius", type=float, default=None,
                   help="覆盖 env 的 axis 距离门控半径. 应与 train 一致.")
    p.add_argument("--rew_home", type=float, default=None,
                   help="覆盖 env 的 home regularizer 权重. 评估 J/R 时应与 train 一致.")
    p.add_argument("--success_axis_threshold", type=float, default=None,
                   help="覆盖 env 默认 axis_err success 阈值. **必须与 train 时一致**, "
                        "否则 hold_success_rate / final_in_thresh_rate 数字不可比.")
    p.add_argument("--terminal_hold_bonus", type=float, default=None,
                   help="hold-N 步成功后的终结 bonus + episode 终止. "
                        "**train 时若启用了它, eval 也必须传同样的值**.")
    p.add_argument("--hold_success_steps", type=int, default=10,
                   help="验证 success 定义: episode 内至少出现连续 N 步都在阈值内.")
    p.add_argument("--clearance_hard", type=float, default=None,
                   help="覆盖 env 的 sphere-proxy 自碰撞兜底阈值. 应与 train 时一致, "
                        "否则碰撞触发率不同, success / J 数字不可比. 关闭写 --clearance_hard=-inf.")
    p.add_argument("--proxy_arm_radius", type=float, default=None,
                   help="覆盖 arm sphere proxy 半径. 应与 train 一致.")
    p.add_argument("--proxy_ee_radius", type=float, default=None,
                   help="覆盖 EE sphere proxy 半径. 应与 train 一致.")
    p.add_argument("--stochastic", action="store_true",
                   help="使用 SAC 采样策略评估. 默认使用 deterministic tanh(mu)")
    return p.parse_args()


def main():
    args = parse_args()
    args.n_episodes = resolve_eval_episode_count(
        args.n_episodes, args.num_envs, "--n_episodes"
    )

    from envs import DualArmPegHoleEnv

    env_kwargs = dict(num_envs=args.num_envs, headless=args.headless)
    for key in ("initial_joint_noise", "preinsert_success_pos_threshold",
                "preinsert_offset", "rew_axis", "rew_pos_success",
                "rew_home", "axis_gate_radius",
                "success_axis_threshold", "terminal_hold_bonus",
                "clearance_hard", "proxy_arm_radius", "proxy_ee_radius"):
        value = getattr(args, key)
        if value is not None:
            env_kwargs[key] = value
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
        f"pos_success_rate={m['pos_success_rate']:.3f}  "
        f"pos_err_mean={m['pos_err_mean']:.4f}m  "
        f"axis_err_mean={m['axis_err_mean']:.4f}  "
        f"axis_gate_mean={m['axis_gate_mean']:.3f}  "
        f"gated_axis_pen={m['gated_axis_penalty_mean']:.3f}"
    )

    mdp.stop()


if __name__ == "__main__":
    main()
