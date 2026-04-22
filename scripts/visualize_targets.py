"""可视化双臂 reaching 目标点, 不训练.

运行:
    conda activate safe_rl
    python scripts/visualize_targets.py
    python scripts/visualize_targets.py --left_target -0.70 -0.45 0.68 --right_target -0.70 0.45 0.68
    python scripts/visualize_targets.py --target_travel_fraction 0.25

说明:
    - 默认打开 Isaac Sim 窗口.
    - 环境会在 env 0 旁生成红/绿目标球 marker.
    - 默认持续运行直到 Ctrl-C; 也可用 --duration 指定秒数后自动退出.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=2,
                   help="至少 2, 避免 IsaacSim cloner 的 num_envs=1 bug")
    p.add_argument("--target_travel_fraction", type=float, default=None,
                   help="显式切回 fraction 目标模式")
    p.add_argument("--left_target", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="显式指定左臂固定目标点")
    p.add_argument("--right_target", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="显式指定右臂固定目标点")
    p.add_argument("--initial_joint_noise", type=float, default=0.0,
                   help="可视化时默认不加 reset 噪声, 更方便看默认姿态与目标点关系")
    p.add_argument("--success_pos_threshold", type=float, default=0.10)
    p.add_argument("--duration", type=float, default=0.0,
                   help=">0 时显示指定秒数后退出; <=0 时持续到 Ctrl-C")
    p.add_argument("--idle_dt", type=float, default=0.02,
                   help="主循环 sleep 间隔, 仅用于降低 CPU 占用")
    return p.parse_args()


def main():
    args = parse_args()

    if (args.left_target is None) != (args.right_target is None):
        raise ValueError("--left_target 和 --right_target 必须同时提供")
    if args.left_target is not None and args.target_travel_fraction is not None:
        raise ValueError("显式 fixed targets 与 --target_travel_fraction 不能同时使用")

    env_kwargs = dict(
        num_envs=args.num_envs,
        headless=False,
        initial_joint_noise=args.initial_joint_noise,
        success_pos_threshold=args.success_pos_threshold,
    )
    from envs import (
        DEFAULT_LEFT_CHEST_TARGET,
        DEFAULT_RIGHT_CHEST_TARGET,
        DualArmPegHoleEnv,
    )

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

    mdp = DualArmPegHoleEnv(**env_kwargs)
    mask = torch.ones(args.num_envs, dtype=torch.bool, device=mdp._device)
    obs, _ = mdp.reset_all(mask)
    left_err, right_err, success = mdp._compute_task_errors(obs)

    print(f"[VIS ENV] {env_kwargs}")
    print(f"[VIS TARGETS] T_L = {mdp._left_target.tolist()}  T_R = {mdp._right_target.tolist()}")
    print(
        "[VIS RESET] "
        f"left_err_mean={float(left_err.mean()):.4f}m  "
        f"right_err_mean={float(right_err.mean()):.4f}m  "
        f"in_thresh_rate={float(success.float().mean()):.3f}"
    )
    if args.duration <= 0:
        print("[VIS] 窗口已打开. 观察完后按 Ctrl-C 退出.")
    else:
        print(f"[VIS] 窗口将保持 {args.duration:.1f}s.")

    start_t = time.monotonic()
    try:
        while True:
            mdp._world.step(render=True)
            if args.duration > 0 and time.monotonic() - start_t >= args.duration:
                break
            time.sleep(args.idle_dt)
    except KeyboardInterrupt:
        pass
    finally:
        mdp.stop()


if __name__ == "__main__":
    main()
