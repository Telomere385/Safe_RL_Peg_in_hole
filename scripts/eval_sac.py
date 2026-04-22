"""评估训练好的 SAC agent.

运行:
    conda activate safe_rl
    python scripts/eval_sac.py              # 带渲染窗口, 默认验证胸前固定目标
    python scripts/eval_sac.py --headless   # 无窗口验证
    python scripts/eval_sac.py --target_travel_fraction 0.1   # 显式切回 fraction 模式
"""

import argparse
from contextlib import contextmanager
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_LEFT_CHEST_TARGET = (-0.6176, -0.75, 0.7391)
DEFAULT_RIGHT_CHEST_TARGET = (-0.6176, 0.73, 0.7391)


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
    p.add_argument("--initial_joint_noise", type=float, default=None,
                   help="覆盖 env 默认 reset 关节噪声")
    p.add_argument("--success_pos_threshold", type=float, default=None,
                   help="覆盖 env 默认位置成功阈值")
    p.add_argument("--hold_success_steps", type=int, default=10,
                   help="验证 success 定义: episode 内至少出现连续 N 步都在阈值内")
    p.add_argument("--stochastic", action="store_true",
                   help="使用 SAC 采样策略评估. 默认使用 deterministic tanh(mu)")
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


def main():
    args = parse_args()

    env_kwargs = dict(num_envs=args.num_envs, headless=args.headless)
    if args.target_travel_fraction is None:
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

    from envs import DualArmPegHoleEnv
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

    _, _, _, next_state, _, last = dataset.parse(to="torch")
    left_err, right_err, in_thresh = mdp._compute_task_errors(next_state)
    last_np = last.cpu().numpy().astype(bool)
    in_thresh_np = in_thresh.cpu().numpy().astype(bool)

    end_indices = np.flatnonzero(last_np)
    ep_max_holds = []
    ep_in_thresh_rates = []
    ep_final_in_thresh = []
    start = 0
    for end in end_indices:
        ep = in_thresh_np[start:end + 1]
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

    hold_flags = np.asarray([mh >= args.hold_success_steps for mh in ep_max_holds], dtype=bool)
    hold_success_rate = float(hold_flags.mean()) if len(hold_flags) else 0.0
    max_hold_mean = float(np.mean(ep_max_holds)) if ep_max_holds else 0.0
    in_thresh_rate = float(np.mean(ep_in_thresh_rates)) if ep_in_thresh_rates else 0.0
    final_in_thresh_rate = float(np.mean(ep_final_in_thresh)) if ep_final_in_thresh else 0.0
    print(
        f"hold_success_rate={hold_success_rate:.3f} (>= {args.hold_success_steps} consecutive steps)  "
        f"max_hold_mean={max_hold_mean:.1f}  "
        f"in_thresh_rate={in_thresh_rate:.3f}  "
        f"final_in_thresh_rate={final_in_thresh_rate:.3f}  "
        f"left_err_mean={float(left_err.mean()):.4f}m  "
        f"right_err_mean={float(right_err.mean()):.4f}m"
    )

    mdp.stop()


if __name__ == "__main__":
    main()
