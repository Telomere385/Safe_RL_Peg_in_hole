"""可视化双臂场景 — 旧 reaching 目标点 + phase 1.5 preinsert 三要素, 不训练.

Phase 1.5 commit 2 新增:
    - peg (红) / hole (绿) 已经从 USDA 挂在左/右末端
    - 本脚本会在 env 0 世界坐标下额外 spawn:
        peg_axis 箭头 (浅红) 从 peg_tip 沿 peg 轴向外 5cm
        hole_axis 箭头 (浅绿) 从 hole_entry 沿 hole 开口方向外 5cm
        preinsert_target 球 (黄)     hole_entry 外 preinsert_offset 处, 预插入站位
    - 主循环每步更新这三个 marker, 它们会跟着左右手腕的运动实时贴在 peg/hole 上

说明:
    - 默认打开 IsaacSim 窗口, 仍然显示旧的红/绿胸前目标球 (reaching 任务 marker).
      这不冲突: 它们在 env 0 固定点, 和 peg_tip/hole_entry/preinsert 是不同的几何对象.
    - 肉眼验收: 黄色 preinsert_target 应该在 hole 开口外 5cm 处沿 hole 轴线的方向.
      peg/hole 的 axis 箭头分别从自身尖端 / 开口中心沿 "+Z of self" 伸出.
      手动转动机器人 (如果可以) 或 reset 多次, 看 marker 是否稳定跟随.

运行:
    conda activate safe_rl
    python scripts/visualize_targets.py
    python scripts/visualize_targets.py --preinsert_offset 0.08
    python scripts/visualize_targets.py --duration 30
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
    p.add_argument("--viz_env_idx", type=int, default=0,
                   help="用哪一个 env 的 peg/hole frame 驱动 marker")
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
    p.add_argument("--preinsert_offset", type=float, default=None,
                   help="覆盖 env 的 preinsert_offset (默认 0.05m)")
    p.add_argument("--duration", type=float, default=0.0,
                   help=">0 时显示指定秒数后退出; <=0 时持续到 Ctrl-C")
    p.add_argument("--idle_dt", type=float, default=0.02,
                   help="主循环 sleep 间隔, 仅用于降低 CPU 占用")
    return p.parse_args()


def _spawn_preinsert_markers(axis_length=0.10, axis_radius=0.004, sphere_radius=0.010):
    """在 /World/viz/ 下 spawn peg_axis / hole_axis 箭头 + preinsert_target 球.

    返回 handles dict (每个元素是 xformOp, 可以 .Set() 更新). 如果 USD stage 拿不到
    就返回 None (静默降级).
    """
    try:
        import omni.usd
        from pxr import UsdGeom, Sdf, Gf, Vt
    except ImportError:
        return None
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None

    def _make_arrow(path, color, length, radius):
        """Xform (translate+orient) -> Cylinder(along +Z, offset +length/2) 子节点.
        Xform 的 orient 决定箭头指向; translate 决定起点 (箭头从起点沿 +Z 伸出)."""
        head_length = 0.25 * length
        shaft_length = length - head_length
        xform_prim = UsdGeom.Xform.Define(stage, Sdf.Path(path))
        xf = UsdGeom.Xformable(xform_prim.GetPrim())
        xf.ClearXformOpOrder()
        t_op = xf.AddTranslateOp()
        r_op = xf.AddOrientOp()
        # 初始化到一个明显"未更新"的位置 (远离工作区), 避免第一帧闪烁在原点.
        t_op.Set(Gf.Vec3d(0.0, 0.0, 10.0))
        r_op.Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

        shaft_path = path + "/Shaft"
        cyl = UsdGeom.Cylinder.Define(stage, Sdf.Path(shaft_path))
        cyl.GetRadiusAttr().Set(radius)
        cyl.GetHeightAttr().Set(shaft_length)
        cyl.GetAxisAttr().Set("Z")
        cyl.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        cxf = UsdGeom.Xformable(cyl.GetPrim())
        cxf.ClearXformOpOrder()
        cxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, shaft_length / 2.0))

        head_path = path + "/Head"
        cone = UsdGeom.Cone.Define(stage, Sdf.Path(head_path))
        cone.GetRadiusAttr().Set(2.2 * radius)
        cone.GetHeightAttr().Set(head_length)
        cone.GetAxisAttr().Set("Z")
        cone.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        hxf = UsdGeom.Xformable(cone.GetPrim())
        hxf.ClearXformOpOrder()
        hxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, shaft_length + head_length / 2.0))
        return t_op, r_op

    def _make_sphere(path, color, radius):
        sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
        sphere.GetRadiusAttr().Set(radius)
        sphere.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        xf = UsdGeom.Xformable(sphere.GetPrim())
        xf.ClearXformOpOrder()
        t_op = xf.AddTranslateOp()
        t_op.Set(Gf.Vec3d(0.0, 0.0, 10.0))
        return t_op

    peg_t, peg_r = _make_arrow("/World/viz/peg_axis_arrow",
                                color=(1.0, 0.35, 0.35),
                                length=axis_length, radius=axis_radius)
    hole_t, hole_r = _make_arrow("/World/viz/hole_axis_arrow",
                                  color=(0.35, 1.0, 0.35),
                                  length=axis_length, radius=axis_radius)
    pre_t = _make_sphere("/World/viz/preinsert_target",
                         color=(1.0, 0.95, 0.15),
                         radius=sphere_radius)

    print(f"[VIS PREINSERT] spawned markers: /World/viz/{{peg_axis_arrow, "
          f"hole_axis_arrow, preinsert_target}}  (axis length={axis_length}m, "
          f"preinsert sphere r={sphere_radius}m)")
    return {"peg_t": peg_t, "peg_r": peg_r,
            "hole_t": hole_t, "hole_r": hole_r,
            "pre_t": pre_t}


def _update_preinsert_markers(frames, handles, env_idx):
    """用指定 env 的帧更新 marker. frames 是 mdp.get_preinsert_frames() 的返回."""
    if handles is None or frames is None:
        return
    from pxr import Gf
    pt = frames["peg_tip_pos"][env_idx].cpu().tolist()
    pq = frames["peg_axis_quat"][env_idx].cpu().tolist()   # wxyz
    he = frames["hole_entry_pos"][env_idx].cpu().tolist()
    hq = frames["hole_axis_quat"][env_idx].cpu().tolist()
    pi = frames["preinsert_target_pos"][env_idx].cpu().tolist()

    handles["peg_t"].Set(Gf.Vec3d(pt[0], pt[1], pt[2]))
    handles["peg_r"].Set(Gf.Quatf(pq[0], Gf.Vec3f(pq[1], pq[2], pq[3])))
    handles["hole_t"].Set(Gf.Vec3d(he[0], he[1], he[2]))
    handles["hole_r"].Set(Gf.Quatf(hq[0], Gf.Vec3f(hq[1], hq[2], hq[3])))
    handles["pre_t"].Set(Gf.Vec3d(pi[0], pi[1], pi[2]))


def _print_preinsert_frames(frames, env_idx):
    """打印指定 env 的 peg_tip / hole_entry / preinsert 数值, 用于肉眼比对."""
    if frames is None:
        print("[VIS PREINSERT] peg/hole 资产未加载 (老 USD?), 跳过 preinsert 可视化")
        return
    pt = frames["peg_tip_pos"][env_idx].tolist()
    pa = frames["peg_axis"][env_idx].tolist()
    he = frames["hole_entry_pos"][env_idx].tolist()
    ha = frames["hole_axis"][env_idx].tolist()
    pi = frames["preinsert_target_pos"][env_idx].tolist()
    align = float(torch.sum(frames["peg_axis"][env_idx] * frames["hole_axis"][env_idx]))
    def _fmt(v):
        return "(" + ", ".join(f"{x:+.4f}" for x in v) + ")"
    print(f"[VIS PREINSERT] env {env_idx} frames (world coords):")
    print(f"  peg_tip      pos={_fmt(pt)}  axis={_fmt(pa)}")
    print(f"  hole_entry   pos={_fmt(he)}  axis={_fmt(ha)}")
    print(f"  preinsert    pos={_fmt(pi)}")
    print(f"  dot(peg_axis, hole_axis) = {align:+.4f}  "
          "(理想插入对齐 = -1; 当前是默认姿态下两臂末端相对几何)")


def _focus_camera_on_env(mdp, env_idx):
    try:
        from isaacsim.core.utils.viewports import set_camera_view
    except ImportError:
        return
    world_pos, _ = mdp._task.robots.get_world_poses()
    base = world_pos[env_idx].detach().cpu().tolist()
    target = [base[0], base[1], base[2] + 0.45]
    eye = [target[0] + 2.0, target[1] - 1.6, target[2] + 1.0]
    set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
    print(f"[VIS CAMERA] env {env_idx} eye={tuple(round(x, 3) for x in eye)} "
          f"target={tuple(round(x, 3) for x in target)}")


def main():
    args = parse_args()
    if not (0 <= args.viz_env_idx < args.num_envs):
        raise ValueError(f"--viz_env_idx ({args.viz_env_idx}) 必须落在 [0, {args.num_envs - 1}]")

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
    if args.preinsert_offset is not None:
        env_kwargs["preinsert_offset"] = args.preinsert_offset

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
    _focus_camera_on_env(mdp, args.viz_env_idx)
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

    # 先 render 一帧打开窗口; preinsert frame 查询本身走 live Fabric pose.
    mdp._world.step(render=True)
    handles = _spawn_preinsert_markers()
    frames = mdp.get_preinsert_frames()
    _print_preinsert_frames(frames, args.viz_env_idx)
    _update_preinsert_markers(frames, handles, args.viz_env_idx)

    if args.duration <= 0:
        print("[VIS] 窗口已打开. 观察完后按 Ctrl-C 退出.")
    else:
        print(f"[VIS] 窗口将保持 {args.duration:.1f}s.")

    start_t = time.monotonic()
    try:
        while True:
            mdp._world.step(render=True)
            # 每帧刷新 preinsert markers (peg/hole 跟着 EE 动, marker 需要跟上)
            frames = mdp.get_preinsert_frames()
            _update_preinsert_markers(frames, handles, args.viz_env_idx)
            if args.duration > 0 and time.monotonic() - start_t >= args.duration:
                break
            time.sleep(args.idle_dt)
    except KeyboardInterrupt:
        pass
    finally:
        mdp.stop()


if __name__ == "__main__":
    main()
