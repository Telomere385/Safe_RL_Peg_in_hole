"""集群离屏渲染 + 按 episode 保存 mp4 视频.

原理:
  headless=True 启动 Isaac Sim — RTX 渲染管线在 GPU 上跑, 不需要 display.
  【重要】headless 模式下 _set_camera() 建的 viewport render product 会失败:
    "HydraEngine rtx failed creating scene renderer" — 因为 headless 时 viewport
    用的 RTX Hydra Engine 被关掉, 导致 get_data() 里 overscan 参数是 None.
  解决方案: 用 rep.create.camera() 建独立相机 prim + render product, 完全绕开
  viewport。独立 render product 有自己的渲染 context, headless 下正常工作.
  渲染触发用 rep.orchestrator.step() 而不是 _world.render().

多 Agent 录制 (--checkpoint_dir):
  Isaac Sim 只启动一次, env 和相机全程复用, 顺序切换 agent.
  视频文件名: <tag>_<agent_stem>_episode_000.mp4, ...
  env 在每个 agent 开始时完整 reset, 保证干净初态.

依赖:
  opencv-python-headless 已在 Isaac Sim 容器内 (isaac-sim.sif requirements 里有).

用法:
    # 单个 agent (--agent_path)
    python scripts/record_video.py \\
        --agent_path results/best_agent.msh \\
        --n_episodes 4 --output_dir results/videos --tag M1p \\
        --use_axis_resid_obs \\
        --preinsert_success_pos_threshold 0.10 --rew_home 0.0005

    # checkpoint 目录 (--checkpoint_dir, 自动录制目录内所有 .msh)
    python scripts/record_video.py \\
        --checkpoint_dir results/checkpoints/2026-05-07/11-59-24 \\
        --n_episodes 4 --output_dir results/videos --tag ckpt \\
        --use_axis_resid_obs \\
        --preinsert_success_pos_threshold 0.10 --rew_home 0.0005

注意:
  - num_envs 默认 2 (cloner bug 要求 >= 2), 只录 --viz_env_idx 指定的那个 env.
  - terminal_hold_bonus 固定为 0 (录制时不终止成功帧, 展示完整 episode).
  - 渲染分辨率用 --width/--height 控制 (默认 1280x720).
  - FPS 用 --fps 控制 (默认 30).
  - 单 agent 文件名: <tag>_episode_000.mp4
  - 多 agent 文件名: <tag>_<stem>_episode_000.mp4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._eval_utils import deterministic_policy, parse_home_weights


def parse_args():
    p = argparse.ArgumentParser()

    # --- agent 来源: 二选一 ---
    # --agent_path: 录制单个 .msh 文件 (原有行为, 保持向后兼容).
    # --checkpoint_dir: 自动扫描目录内所有 .msh, Isaac Sim 只启动一次顺序录制.
    p.add_argument("--agent_path", type=str,
                   default=str(PROJECT_ROOT / "results/best_agent.msh"),
                   help="单个 agent .msh 路径; 与 --checkpoint_dir 互斥.")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="checkpoint 目录; 自动录制其中所有 .msh 文件.")

    # --- env 参数 (与 eval_sac.py 保持一致) ---
    p.add_argument("--num_envs", type=int, default=2,
                   help="至少 2 (cloner bug). 录制只看 viz_env_idx 那个 env.")
    p.add_argument("--initial_joint_noise", type=float, default=None)
    p.add_argument("--preinsert_success_pos_threshold", type=float, default=None)
    p.add_argument("--preinsert_offset", type=float, default=None)
    p.add_argument("--rew_action", type=float, default=None)
    p.add_argument("--rew_axis", type=float, default=None)
    p.add_argument("--rew_success", type=float, default=None)
    p.add_argument("--rew_pos_success", type=float, default=None)
    p.add_argument("--rew_home", type=float, default=None)
    p.add_argument("--home_weights", type=parse_home_weights, default=None)
    p.add_argument("--axis_gate_radius", type=float, default=None)
    p.add_argument("--success_axis_threshold", type=float, default=None)
    p.add_argument("--terminal_hold_bonus", type=float, default=None,
                   help="录制时忽略此参数; 始终强制为 0 以保留成功帧.")
    p.add_argument("--hold_success_steps", type=int, default=10)
    p.add_argument("--clearance_hard", type=float, default=None)
    p.add_argument("--proxy_arm_radius", type=float, default=None)
    p.add_argument("--proxy_ee_radius", type=float, default=None)
    p.add_argument("--exclude_ee_from_physx_self_collision", action="store_true",
                   help="Stage 3 录制用: PhysX self-collision 分组排除 EE link, "
                        "避免 peg-hole 正常接触被 hard absorbing 截断.")
    p.add_argument("--use_axis_resid_obs", action="store_true")
    p.add_argument("--stochastic", action="store_true",
                   help="SAC 采样策略; 默认 deterministic tanh(mu)")

    # --- 录制参数 ---
    p.add_argument("--n_episodes", type=int, default=4,
                   help="每个 agent 录几个 episode")
    p.add_argument("--viz_env_idx", type=int, default=0,
                   help="只录这个 env 的 episode (其余 env 正常跑但不录)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="视频输出目录. 不指定时: checkpoint_dir 模式自动推导 "
                        "(results/checkpoints/A/B → results/videos/A/B), "
                        "agent_path 模式默认 results/videos.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280,
                   help="渲染宽度 (独立 camera render product 的分辨率)")
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--tag", type=str, default="",
                   help="文件名前缀标签. 单 agent: <tag>_episode_N.mp4; "
                        "多 agent: <tag>_<stem>_episode_N.mp4")

    return p.parse_args()


# ---------------------------------------------------------------------------
# 相机 / 渲染工具
# ---------------------------------------------------------------------------

def _setup_offscreen_camera(width: int, height: int, position, look_at):
    """headless 下建独立离屏相机 + render product.

    headless 模式里 viewport 的 RTX Hydra Engine 被关闭
    ('HydraEngine rtx failed creating scene renderer'), 导致
    _set_camera() 建的 viewport render product 的 get_data() 崩溃
    (overscan 参数是 None).

    这里用 rep.create.camera() 建独立 prim, 有自己的渲染 context,
    不依赖 viewport, headless 下正常工作.

    Returns (annot, rp): annotator 和 render product, 供 _get_frame 使用.
    """
    import omni.replicator.core as rep

    camera = rep.create.camera(
        position=tuple(position),
        look_at=tuple(look_at),
    )
    rp = rep.create.render_product(camera, (width, height))
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach([rp])
    return annot, rp


def _get_frame(annot, width: int, height: int) -> np.ndarray:
    """触发独立相机渲染并返回 BGR uint8 帧 (H, W, 3).

    用 rep.orchestrator.step() 触发 replicator 的渲染 pass,
    和 viewport / _world.render() 完全解耦.
    """
    import omni.replicator.core as rep

    rep.orchestrator.step(rt_subframes=1)
    raw = annot.get_data()

    if hasattr(raw, 'numpy'):
        arr = raw.numpy()
    else:
        arr = np.asarray(raw)

    if arr.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    rgb = arr[..., :3]
    bgr = rgb[..., ::-1].copy()   # RGB → BGR (cv2), copy 消除负步长
    return bgr.astype(np.uint8)


def _make_writer(out_path: Path, fps: int, width: int, height: int):
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter 打不开: {out_path}\n"
            "确认 opencv-python-headless 正常安装 (pip show opencv-python-headless)"
        )
    return writer


# ---------------------------------------------------------------------------
# 单个 agent 录制
# ---------------------------------------------------------------------------

def _record_one_agent(mdp, agent, args, rec_annot, out_dir: Path, prefix: str) -> int:
    """用给定 agent 录制 args.n_episodes 个 episode, 返回实际完成数.

    env 在进入时做全量 reset (env_mask=全 True), 保证每个 agent 的录制
    都从干净的初始状态开始, 不受上一个 agent 遗留状态影响.

    Args:
        mdp:       已初始化的 DualArmPegHoleEnv.
        agent:     已加载的 mushroom_rl Agent.
        args:      parse_args() 返回值 (录制参数).
        rec_annot: 离屏相机 annotator (复用, 不重建).
        out_dir:   视频输出目录.
        prefix:    视频文件名前缀, 例如 "M1p_best_agent_".
    """
    env_mask = torch.ones(args.num_envs, dtype=torch.bool, device=mdp._device)

    # 每个 agent 开始前全量 reset, 避免继承上一轮的物理状态
    obs, _ = mdp.reset_all(env_mask)
    episode_steps = torch.zeros(args.num_envs, dtype=torch.long, device=mdp._device)

    ep_idx = 0
    step_count = 0
    writer = None
    ep_frames = 0

    print(f"[REC] agent '{prefix.rstrip('_')}': 开始录制 {args.n_episodes} 个 episode ...")

    try:
        while ep_idx < args.n_episodes:
            # ---- action ----
            if args.stochastic:
                action, _ = agent.policy.draw_action(obs)
            else:
                with deterministic_policy(agent):
                    action, _ = agent.policy.draw_action(obs)

            # ---- physics step ----
            next_obs, reward, absorbing, _info = mdp.step_all(env_mask, action)
            step_count += 1
            episode_steps[env_mask] += 1
            last = absorbing | (episode_steps >= mdp.info.horizon)

            # ---- 非录制 env 按 VectorCore 语义在 absorbing|horizon 时 reset ----
            other_last = last.clone()
            other_last[args.viz_env_idx] = False
            if other_last.any():
                reset_obs, _ = mdp.reset_all(other_last)
                next_obs = next_obs.clone()
                next_obs[other_last] = reset_obs[other_last]
                episode_steps[other_last] = 0

            # ---- 独立相机渲染 & 抓帧 ----
            frame = _get_frame(rec_annot, args.width, args.height)

            # ---- 首帧时创建 writer ----
            if writer is None:
                out_path = out_dir / f"{prefix}episode_{ep_idx:03d}.mp4"
                writer = _make_writer(out_path, args.fps, args.width, args.height)
                ep_frames = 0
                print(f"[REC]   episode {ep_idx:03d} → {out_path}")

            writer.write(frame)
            ep_frames += 1

            # ---- viz env 的 episode 结束 ----
            if last[args.viz_env_idx].item():
                writer.release()
                writer = None
                print(f"[REC]   episode {ep_idx:03d} 完成: {ep_frames} 帧 "
                      f"(累计步数: {step_count})")
                ep_idx += 1

                if ep_idx < args.n_episodes:
                    obs, _ = mdp.reset_all(env_mask)
                    episode_steps.zero_()
                    continue

            obs = next_obs

            # ---- 保险: 防死循环 (单 agent 的步数上限) ----
            if step_count > args.n_episodes * mdp.info.horizon * 3:
                print("[REC] 警告: 超过预期步数上限, 强制结束当前 agent 录制")
                break

    finally:
        # episode 中途中断时释放 writer, 避免文件损坏
        if writer is not None:
            writer.release()
            print(f"[REC] 中断, 已释放 episode {ep_idx:03d} 的 writer")

    return ep_idx


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not (0 <= args.viz_env_idx < args.num_envs):
        raise ValueError(
            f"--viz_env_idx ({args.viz_env_idx}) 必须在 [0, {args.num_envs-1}] 内"
        )

    # ---- 确定待录制的 (agent_path, 视频前缀) 列表 ----
    if args.checkpoint_dir is not None:
        # 多 agent 模式: 扫描目录内所有 .msh, 按文件名排序保证顺序稳定
        ckpt_dir = Path(args.checkpoint_dir)
        agent_paths = sorted(ckpt_dir.glob("*.msh"))
        if not agent_paths:
            raise FileNotFoundError(
                f"--checkpoint_dir 下没有找到 .msh 文件: {ckpt_dir}"
            )
        # 前缀: <tag>_<stem>_ 或 <stem>_ (tag 为空时)
        def _prefix(p: Path) -> str:
            return f"{args.tag}_{p.stem}_" if args.tag else f"{p.stem}_"
        agent_records = [(p, _prefix(p)) for p in agent_paths]
        print(f"[REC] checkpoint_dir 模式: 发现 {len(agent_records)} 个 agent")
        for p, pfx in agent_records:
            print(f"[REC]   {p.name}  →  前缀 '{pfx}'")
    else:
        # 单 agent 模式: 与原有行为完全一致
        prefix = f"{args.tag}_" if args.tag else ""
        agent_records = [(Path(args.agent_path), prefix)]

    # ---- 确定输出目录 ----
    if args.output_dir is not None:
        # 显式指定时直接使用
        out_dir = Path(args.output_dir)
    elif args.checkpoint_dir is not None:
        # checkpoint_dir 模式: 把路径中的 "checkpoints" 段替换为 "videos"
        # 例: results/checkpoints/2026-05-07/11-59-24
        #   → results/videos/2026-05-07/11-59-24
        ckpt = Path(args.checkpoint_dir)
        parts = ckpt.parts
        if "checkpoints" in parts:
            idx = parts.index("checkpoints")
            out_dir = Path(*parts[:idx]) / "videos" / Path(*parts[idx + 1:])
        else:
            # 路径中没有 "checkpoints" 段时, 在 results/videos 下用目录名兜底
            out_dir = PROJECT_ROOT / "results" / "videos" / ckpt.name
    else:
        out_dir = PROJECT_ROOT / "results" / "videos"

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ env
    # env 只建一次, 多 agent 录制时复用同一个物理场景
    from envs import DualArmPegHoleEnv

    env_kwargs = dict(
        num_envs=args.num_envs,
        headless=True,
        success_hold_steps=args.hold_success_steps,
        terminal_hold_bonus=0.0,  # 强制为 0: 不在成功瞬间截断 episode
    )
    for key in ("initial_joint_noise", "preinsert_success_pos_threshold",
                "preinsert_offset", "rew_action", "rew_axis", "rew_success",
                "rew_pos_success", "rew_home", "home_weights",
                "axis_gate_radius", "success_axis_threshold",
                "clearance_hard", "proxy_arm_radius", "proxy_ee_radius"):
        val = getattr(args, key)
        if val is not None:
            env_kwargs[key] = val
    if args.use_axis_resid_obs:
        env_kwargs["use_axis_resid_obs"] = True
    if args.exclude_ee_from_physx_self_collision:
        env_kwargs["exclude_ee_from_physx_self_collision"] = True

    print("[DBG] creating DualArmPegHoleEnv ...", flush=True)
    mdp = DualArmPegHoleEnv(**env_kwargs)
    print("[DBG] DualArmPegHoleEnv created", flush=True)

    # ------------------------------------------------------------------ 离屏相机
    # 必须在 env (= SimulationApp + world) 初始化后才能调 rep API.
    # 相机建一次, 所有 agent 录制共用同一个 render product.
    print("[DBG] setting up offscreen camera ...", flush=True)
    world_pos, _ = mdp._task.robots.get_world_poses()
    base = world_pos[args.viz_env_idx].detach().cpu().tolist()
    cam_target = [base[0], base[1], base[2] + 0.45]
    cam_eye = [cam_target[0] + 2.0, cam_target[1] - 1.6, cam_target[2] + 1.0]
    print(f"[REC] camera eye={tuple(round(x, 3) for x in cam_eye)} "
          f"target={tuple(round(x, 3) for x in cam_target)}", flush=True)
    rec_annot, _rec_rp = _setup_offscreen_camera(args.width, args.height, cam_eye, cam_target)
    print("[DBG] offscreen camera ready", flush=True)

    print(f"[REC] env ready. headless=True, {args.num_envs} envs, "
          f"recording env {args.viz_env_idx}")
    print(f"[REC] pos_th={mdp._preinsert_success_pos_threshold:.3f}m  "
          f"axis_th={mdp._success_axis_threshold:.3f}  "
          f"render_size={args.width}x{args.height}  fps={args.fps}")

    # ------------------------------------------------------------------ 预热渲染管线
    # 前几帧渲染结果不稳定 (shader 首次编译), warmup 后再开始正式录制.
    import omni.replicator.core as rep

    print("[REC] warming up render pipeline ...", flush=True)
    for _w in range(10):
        print(f"[DBG] warmup step {_w} ...", flush=True)
        rep.orchestrator.step(rt_subframes=1)
        print(f"[DBG] warmup step {_w} done", flush=True)
    print("[DBG] warmup done", flush=True)

    # ------------------------------------------------------------------ 顺序录制每个 agent
    from mushroom_rl.core import Agent

    total_videos = 0
    for agent_path, prefix in agent_records:
        print(f"\n[REC] {'='*60}")
        print(f"[REC] 加载 agent: {agent_path}", flush=True)
        agent = Agent.load(str(agent_path))
        print(f"[REC] agent 加载完成", flush=True)

        done = _record_one_agent(mdp, agent, args, rec_annot, out_dir, prefix)
        total_videos += done
        print(f"[REC] agent '{agent_path.name}' 完成, 录制 {done} 个视频")

    mdp.stop()
    print(f"\n[REC] 全部完成. 共 {total_videos} 个视频保存在: {out_dir}")


if __name__ == "__main__":
    main()
