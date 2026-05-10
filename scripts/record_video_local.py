"""Record a local visualization video for one .msh checkpoint.

This is a lightweight local entry point. It does not use Hydra, Apptainer, or
Slurm. Put checkpoints under results/video_local/checkpoint if you want a
stable project-local place for manual video recording.

Example:
    python scripts/record_video_local.py \
        --checkpoint_path results/video_local/checkpoint/best_agent.msh \
        --n_episodes 1 \
        --use_axis_resid_obs \
        --preinsert_success_pos_threshold 0.10 \
        --rew_home 0.0005 \
        --clearance_hard=-inf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._eval_utils import parse_home_weights
from scripts.record_video import (
    _record_one_agent,
    _setup_offscreen_camera,
)

VIDEO_LOCAL_ROOT = PROJECT_ROOT / "results" / "video_local"
DEFAULT_CHECKPOINT_DIR = VIDEO_LOCAL_ROOT / "checkpoint"
DEFAULT_OUTPUT_DIR = VIDEO_LOCAL_ROOT / "video"


def _project_path(value: str | Path) -> Path:
    """Resolve relative paths from the project root for repeatable CLI use."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record local mp4 video for one mushroom-rl .msh checkpoint."
    )

    p.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help=(
            "Path to the .msh checkpoint. Relative paths are resolved from the "
            f"project root. Suggested location: {DEFAULT_CHECKPOINT_DIR}"
        ),
    )

    p.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Video output directory. Defaults to results/video_local/video.",
    )
    p.add_argument("--tag", type=str, default="", help="Optional filename prefix.")

    # Local rendering defaults.
    p.add_argument("--headless", action="store_true", help="Run without Isaac Sim UI.")
    p.add_argument(
        "--num_envs",
        type=int,
        default=2,
        help="Must be at least 2 because of the cloner issue; only viz_env_idx is recorded.",
    )
    p.add_argument("--viz_env_idx", type=int, default=0)
    p.add_argument("--n_episodes", type=int, default=1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Use SAC sampling policy. Default is deterministic tanh(mu).",
    )

    # Env parameters. Keep these aligned with the checkpoint's training command.
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
    p.add_argument("--hold_success_steps", type=int, default=10)
    p.add_argument("--clearance_hard", type=float, default=None)
    p.add_argument("--proxy_arm_radius", type=float, default=None)
    p.add_argument("--proxy_ee_radius", type=float, default=None)
    p.add_argument(
        "--exclude_ee_from_physx_self_collision",
        action="store_true",
        help=(
            "Stage 3 recording: exclude EE links from PhysX self-collision groups "
            "so peg-hole contact does not trigger hard absorbing."
        ),
    )
    p.add_argument(
        "--use_axis_resid_obs",
        action="store_true",
        help="Use the 34-D axis_resid observation. Must match training.",
    )

    return p.parse_args()


def _build_env_kwargs(args: argparse.Namespace) -> dict:
    env_kwargs = dict(
        num_envs=args.num_envs,
        headless=args.headless,
        success_hold_steps=args.hold_success_steps,
        # Keep recording episodes visually complete instead of cutting at hold-N.
        terminal_hold_bonus=0.0,
    )

    for key in (
        "initial_joint_noise",
        "preinsert_success_pos_threshold",
        "preinsert_offset",
        "rew_action",
        "rew_axis",
        "rew_success",
        "rew_pos_success",
        "rew_home",
        "home_weights",
        "axis_gate_radius",
        "success_axis_threshold",
        "clearance_hard",
        "proxy_arm_radius",
        "proxy_ee_radius",
    ):
        value = getattr(args, key)
        if value is not None:
            env_kwargs[key] = value

    if args.use_axis_resid_obs:
        env_kwargs["use_axis_resid_obs"] = True
    if args.exclude_ee_from_physx_self_collision:
        env_kwargs["exclude_ee_from_physx_self_collision"] = True

    return env_kwargs


def main() -> None:
    args = parse_args()

    if not (0 <= args.viz_env_idx < args.num_envs):
        raise ValueError(
            f"--viz_env_idx ({args.viz_env_idx}) must be in [0, {args.num_envs - 1}]"
        )

    checkpoint_path = _project_path(args.checkpoint_path)
    if checkpoint_path.suffix != ".msh":
        raise ValueError(f"--checkpoint_path must point to a .msh file: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint_dir = DEFAULT_CHECKPOINT_DIR
    out_dir = _project_path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    from envs import DualArmPegHoleEnv

    env_kwargs = _build_env_kwargs(args)
    print(f"[LOCAL REC] checkpoint: {checkpoint_path}")
    print(f"[LOCAL REC] output_dir: {out_dir}")
    print(f"[LOCAL REC] env_kwargs: {env_kwargs}")

    mdp = DualArmPegHoleEnv(**env_kwargs)
    try:
        world_pos, _ = mdp._task.robots.get_world_poses()
        base = world_pos[args.viz_env_idx].detach().cpu().tolist()
        cam_target = [base[0], base[1], base[2] + 0.45]
        cam_eye = [cam_target[0] + 2.0, cam_target[1] - 1.6, cam_target[2] + 1.0]
        print(
            "[LOCAL REC] camera "
            f"eye={tuple(round(x, 3) for x in cam_eye)} "
            f"target={tuple(round(x, 3) for x in cam_target)}"
        )
        rec_annot, _rec_rp = _setup_offscreen_camera(
            args.width, args.height, cam_eye, cam_target
        )

        import omni.replicator.core as rep

        print("[LOCAL REC] warming up render pipeline ...", flush=True)
        for _ in range(args.warmup_steps):
            rep.orchestrator.step(rt_subframes=1)

        from mushroom_rl.core import Agent

        agent = Agent.load(str(checkpoint_path))
        prefix = f"{args.tag}_{checkpoint_path.stem}_" if args.tag else f"{checkpoint_path.stem}_"
        done = _record_one_agent(mdp, agent, args, rec_annot, out_dir, prefix)
    finally:
        mdp.stop()

    print(f"[LOCAL REC] done. Recorded {done} video(s) under: {out_dir}")


if __name__ == "__main__":
    main()
