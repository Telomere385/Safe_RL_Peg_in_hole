"""跑训好的 policy, 在指定 env 第一次达到 hold-N 时冻结 IsaacSim 画面.

eval_sac.py 是数值验证, 这个脚本是肉眼验证. mushroom 的 VectorCore.evaluate
是全自动的, absorb 一触发 episode 立刻 reset, 你根本看不到"agent 稳稳停在
目标点"那一帧. 这里通过 monkey-patch is_absorbing, 让 env 在指定 env 计数器
达到 hold-N 时调 _world.render() 反复刷, 不调 _world.step(), 物理就冻住了.

用法:
    conda activate safe_rl
    python scripts/visualize_policy.py
    python scripts/visualize_policy.py --freeze_seconds 30
    python scripts/visualize_policy.py --viz_env_idx 1     # 看 env 1 而不是 env 0
    python scripts/visualize_policy.py --hold_steps 30     # 等更长的 hold 才冻结

注意:
- 必须 num_envs >= 2 (cloner bug).
- terminal_hold_bonus 强制设 0, 否则 env 自己会在 hold-N absorb 把 episode reset 掉,
  你来不及冻结. 我们手动检测 counter 并冻结.
- 监视 env 0 比较稳: env 0 是 marker spawn 的位置, 红绿球肯定在它的 EE 目标点上.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._eval_utils import deterministic_policy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_path", type=str,
                   default=str(PROJECT_ROOT / "results/best_agent.msh"))
    p.add_argument("--num_envs", type=int, default=2,
                   help="至少 2 (num_envs=1 触发 cloner bug)")
    p.add_argument("--viz_env_idx", type=int, default=0,
                   help="哪个 env 满足 hold-N 时冻结. env 0 才有 marker 球")
    p.add_argument("--success_pos_threshold", type=float, default=0.10,
                   help="跟 train 一致, 默认 0.10")
    p.add_argument("--initial_joint_noise", type=float, default=0.05,
                   help="跟 train 一致, 默认 0.05")
    p.add_argument("--hold_steps", type=int, default=10,
                   help="连续 N 步在阈内即冻结. 默认 10 = 跟 train 的 hold_success_steps 一致")
    p.add_argument("--n_episodes", type=int, default=2,
                   help="VectorCore.evaluate 跑几集. 一般冻结发生在第一集就够看")
    p.add_argument("--freeze_seconds", type=float, default=15.0,
                   help="冻结多少秒, Ctrl-C 可提前退出")
    p.add_argument("--stochastic", action="store_true",
                   help="用 SAC 采样策略而不是 deterministic tanh(mu)")
    return p.parse_args()


def main():
    args = parse_args()

    from envs import (
        DEFAULT_LEFT_CHEST_TARGET,
        DEFAULT_RIGHT_CHEST_TARGET,
        DualArmPegHoleEnv,
    )

    # terminal_hold_bonus=0 让 env 不自动 absorb; 我们读 counter 自己冻结
    mdp = DualArmPegHoleEnv(
        num_envs=args.num_envs,
        headless=False,
        left_target=DEFAULT_LEFT_CHEST_TARGET,
        right_target=DEFAULT_RIGHT_CHEST_TARGET,
        success_pos_threshold=args.success_pos_threshold,
        initial_joint_noise=args.initial_joint_noise,
        success_hold_steps=args.hold_steps,
        terminal_hold_bonus=0.0,
    )

    from mushroom_rl.core import Agent, VectorCore

    agent = Agent.load(args.agent_path)
    print(f"[VIZ] loaded agent from {args.agent_path}")
    print(f"[VIZ] watching env {args.viz_env_idx} for {args.hold_steps} consecutive in_thresh steps")

    # Monkey-patch is_absorbing: env 0 一达到 hold-N 就冻 IsaacSim
    state = {"frozen": False, "step_idx": 0}
    original_is_absorbing = mdp.is_absorbing

    def freeze_on_success(obs):
        result = original_is_absorbing(obs)
        state["step_idx"] += 1
        cnt = int(mdp._consecutive_inthresh[args.viz_env_idx].item())
        if cnt > 0 and state["step_idx"] % 5 == 0:
            left_err, right_err, _ = mdp._last_task_errors
            print(f"  step {state['step_idx']}  env {args.viz_env_idx}  "
                  f"hold count={cnt}  "
                  f"left_err={float(left_err[args.viz_env_idx]):.4f}m  "
                  f"right_err={float(right_err[args.viz_env_idx]):.4f}m")
        if not state["frozen"] and cnt >= args.hold_steps:
            left_err, right_err, _ = mdp._last_task_errors
            print(
                f"\n[FREEZE] env {args.viz_env_idx} reached hold-{args.hold_steps} "
                f"at step {state['step_idx']}\n"
                f"  left_err = {float(left_err[args.viz_env_idx]):.4f}m\n"
                f"  right_err = {float(right_err[args.viz_env_idx]):.4f}m\n"
                f"  freezing for {args.freeze_seconds:.1f}s, Ctrl-C to exit early"
            )
            state["frozen"] = True
            end_t = time.monotonic() + args.freeze_seconds
            try:
                while time.monotonic() < end_t:
                    mdp._world.render()  # render-only, 不 step 物理 → 画面冻结
                    time.sleep(0.05)
            except KeyboardInterrupt:
                pass
            print("[FREEZE] released, evaluation will continue (or end)")
        return result

    mdp.is_absorbing = freeze_on_success

    core = VectorCore(agent, mdp)
    if args.stochastic:
        core.evaluate(n_episodes=args.n_episodes, render=True)
    else:
        with deterministic_policy(agent):
            core.evaluate(n_episodes=args.n_episodes, render=True)

    if not state["frozen"]:
        print(f"\n[VIZ] env {args.viz_env_idx} 在 {args.n_episodes} episode "
              f"({state['step_idx']} 步) 内没达到 hold-{args.hold_steps}.")
        print(f"      可能起手位置组合不利, 或 agent 真的不行. 重跑或 --viz_env_idx 1 试试.")

    mdp.stop()


if __name__ == "__main__":
    main()
