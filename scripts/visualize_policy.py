"""跑训好的 policy, 在指定 env 第一次达到 hold-N 时冻结 IsaacSim 画面.

eval_sac.py 是数值验证, 这个脚本是肉眼验证. mushroom 的 VectorCore.evaluate
是全自动的, absorb 一触发 episode 立刻 reset, 你根本看不到"agent 稳稳停在
preinsert"那一帧. 这里通过 monkey-patch is_absorbing, 让 env 在指定 env
计数器达到 hold-N 时调 _world.render() 反复刷, 不调 _world.step(), 物理就冻住了.

stage 注意: 冻结由 env 的 success counter 触发, 而 success_mask =
(pos<pos_th) ∧ (axis_err<axis_th). Stage 2 视觉验证时**必须**传 --rew_axis 和
--success_axis_threshold 与训练时一致 (Stage 2 推荐 0.50 或 0.30), 否则
axis_th=inf 会让 freeze 在"位置进阈但姿态没达标"时误触发, 看到的不是 Stage 2
真正的成功状态.

正式 Stage 1 / Stage 2 可视化命令见 README.md "可视化命令" 段.

注意:
- 必须 num_envs >= 2 (cloner bug).
- terminal_hold_bonus 强制设 0, 否则 env 自己会在 hold-N absorb 把 episode reset.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._eval_utils import deterministic_policy, parse_home_weights


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_path", type=str,
                   default=str(PROJECT_ROOT / "results/best_agent.msh"))
    p.add_argument("--num_envs", type=int, default=2,
                   help="至少 2 (num_envs=1 触发 cloner bug)")
    p.add_argument("--viz_env_idx", type=int, default=0)
    p.add_argument("--preinsert_success_pos_threshold", type=float, default=0.10,
                   help="跟 train 一致 (env/train/eval 默认 0.10m, Stage 1/Stage 2 curriculum). "
                        "传更紧的值会让 freeze 条件更严, 可能看不到 hold-N 触发.")
    p.add_argument("--initial_joint_noise", type=float, default=0.1,
                   help="跟 env/train 默认 0.1 一致 (Stage 1 训练时的 reset 噪声).")
    p.add_argument("--preinsert_offset", type=float, default=None,
                   help="覆盖 env 的 preinsert offset (默认 0.05m)")
    p.add_argument("--rew_axis", type=float, default=None,
                   help="覆盖 env 的 axis_err 权重. visualize 不算 reward, 但保留 "
                        "CLI 一致性 (env 内部根据这个值打印的 axis 项形式不同).")
    p.add_argument("--rew_home", type=float, default=None,
                   help="覆盖 env 的 home regularizer 权重. visualize 不依赖 reward, "
                        "但保留与 train/eval CLI 一致性.")
    p.add_argument("--home_weights", type=parse_home_weights, default=None,
                   help="home regularizer 的逐关节权重. visualize 不依赖 reward, "
                        "但建议与 train/eval CLI 保持一致.")
    p.add_argument("--success_axis_threshold", type=float, default=None,
                   help="**必须与 train 一致**. Stage 2 训练用 0.50, 严格视觉验证可用 0.30. 不传 = "
                        "默认 inf = Stage 1 行为, 会让 freeze 在 axis 还没对齐时误触发.")
    p.add_argument("--hold_steps", type=int, default=10,
                   help="连续 N 步在阈内即冻结. 默认 10 = 跟 train 的 hold_success_steps 一致")
    p.add_argument("--n_episodes", type=int, default=2,
                   help="VectorCore.evaluate 跑几集. 一般冻结发生在第一集就够看")
    p.add_argument("--freeze_seconds", type=float, default=15.0,
                   help="冻结多少秒, Ctrl-C 可提前退出")
    p.add_argument("--stochastic", action="store_true",
                   help="用 SAC 采样策略而不是 deterministic tanh(mu)")
    p.add_argument("--rew_pos_success", type=float, default=None,
                   help="覆盖 env 的 pos-only success bonus. 应与 train 一致.")
    p.add_argument("--axis_gate_radius", type=float, default=None,
                   help="覆盖 env 的 axis 距离门控半径. 应与 train 一致.")
    p.add_argument("--clearance_hard", type=float, default=None,
                   help="覆盖 env 的 sphere-proxy 自碰撞兜底阈值. 应与 train 一致, "
                        "否则可能出现 train 不撞 / viz 老 reset 的错觉. 关闭写 --clearance_hard=-inf.")
    p.add_argument("--proxy_arm_radius", type=float, default=None,
                   help="覆盖 arm sphere proxy 半径. 应与 train 一致.")
    p.add_argument("--proxy_ee_radius", type=float, default=None,
                   help="覆盖 EE sphere proxy 半径. 应与 train 一致.")
    p.add_argument("--use_axis_resid_obs", action="store_true",
                   help="34 维 obs (axis_resid 替换 axis_dot). 应与 train 一致.")
    return p.parse_args()


def main():
    args = parse_args()
    if not (0 <= args.viz_env_idx < args.num_envs):
        raise ValueError(
            f"--viz_env_idx ({args.viz_env_idx}) 必须落在 [0, {args.num_envs - 1}]"
        )

    from envs import DualArmPegHoleEnv

    env_kwargs = dict(
        num_envs=args.num_envs,
        headless=False,
        preinsert_success_pos_threshold=args.preinsert_success_pos_threshold,
        initial_joint_noise=args.initial_joint_noise,
        success_hold_steps=args.hold_steps,
        terminal_hold_bonus=0.0,
    )
    if args.preinsert_offset is not None:
        env_kwargs["preinsert_offset"] = args.preinsert_offset
    if args.rew_axis is not None:
        env_kwargs["rew_axis"] = args.rew_axis
    if args.rew_home is not None:
        env_kwargs["rew_home"] = args.rew_home
    if args.home_weights is not None:
        env_kwargs["home_weights"] = args.home_weights
    if args.success_axis_threshold is not None:
        env_kwargs["success_axis_threshold"] = args.success_axis_threshold
    for key in ("rew_pos_success", "axis_gate_radius",
                "clearance_hard", "proxy_arm_radius", "proxy_ee_radius"):
        value = getattr(args, key)
        if value is not None:
            env_kwargs[key] = value
    if args.use_axis_resid_obs:
        env_kwargs["use_axis_resid_obs"] = True
    mdp = DualArmPegHoleEnv(**env_kwargs)
    print(f"[VIZ STAGE] pos_th={mdp._preinsert_success_pos_threshold:.3f}m  "
          f"axis_th={mdp._success_axis_threshold:.3f}  "
          f"w_axis={mdp._w_axis:.3f}  "
          f"w_pos_success={mdp._w_pos_success:.3f}  "
          f"w_home={mdp._w_home:.4f}  "
          f"axis_gate_radius={mdp._axis_gate_radius:.3f}m")

    from mushroom_rl.core import Agent, VectorCore

    agent = Agent.load(args.agent_path)
    print(f"[VIZ] loaded agent from {args.agent_path}")
    print(f"[VIZ] watching env {args.viz_env_idx} for {args.hold_steps} consecutive in_thresh steps")

    state = {"frozen": False, "step_idx": 0}
    original_is_absorbing = mdp.is_absorbing

    def freeze_on_success(obs):
        result = original_is_absorbing(obs)
        state["step_idx"] += 1
        cnt = int(mdp._consecutive_inthresh[args.viz_env_idx].item())
        if cnt > 0 and state["step_idx"] % 5 == 0:
            pos_err = mdp._last_pos_err
            axis_err = mdp._last_axis_err
            print(f"  step {state['step_idx']}  env {args.viz_env_idx}  "
                  f"hold count={cnt}  "
                  f"pos_err={float(pos_err[args.viz_env_idx]):.4f}m  "
                  f"axis_err={float(axis_err[args.viz_env_idx]):.4f}")
        if not state["frozen"] and cnt >= args.hold_steps:
            pos_err = mdp._last_pos_err
            axis_err = mdp._last_axis_err
            print(
                f"\n[FREEZE] env {args.viz_env_idx} reached hold-{args.hold_steps} "
                f"at step {state['step_idx']}\n"
                f"  pos_err  = {float(pos_err[args.viz_env_idx]):.4f}m\n"
                f"  axis_err = {float(axis_err[args.viz_env_idx]):.4f}  "
                f"(0 = perfect; success_axis_threshold = {mdp._success_axis_threshold:.3f})\n"
                f"  freezing for {args.freeze_seconds:.1f}s, Ctrl-C to exit early"
            )
            state["frozen"] = True
            end_t = time.monotonic() + args.freeze_seconds
            try:
                while time.monotonic() < end_t:
                    mdp._world.render()
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

    mdp.stop()


if __name__ == "__main__":
    main()
