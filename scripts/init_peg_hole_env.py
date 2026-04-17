"""双臂 peg-in-hole 场景初始化 — 静态展示，无控制逻辑。

运行: cd ~/IsaacLab && ./isaaclab.sh -p ~/bimanual_peghole/scripts/init_peg_hole_env.py
"""
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import sys, math, numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.math import quat_apply, quat_mul
from pxr import UsdGeom, Sdf, Gf, Vt

# ── 几何参数 ──
PEG_RADIUS   = 0.008
PEG_HEIGHT   = 0.035
HOLE_OUTER_R = 0.012
HOLE_INNER_R = 0.010
HOLE_HEIGHT  = 0.030

LEFT_FINGER  = 0.004   # 左夹爪左指开口
RIGHT_FINGER = 0.008   # 右夹爪左指开口
PART_X       = -0.0055  # mimic 不对称补偿
PART_Z       = 0.125    # 指间区域中心

# 90° around local X — 使零件轴垂直于手指
_C = math.cos(math.pi / 4)
_S = math.sin(math.pi / 4)

# ── 机器人配置 ──
sys.path.insert(0, "/home/miao/IsaacLab/source/isaaclab_tasks")
from isaaclab_tasks.direct.miao_dual_arm.dual_arm_cfg import DUAL_ARM_CFG

ROBOT_CFG = DUAL_ARM_CFG.replace(
    prim_path="/World/DualArm",
    actuators={
        "left_arm":  ImplicitActuatorCfg(joint_names_expr=["left_arm_A[1-7]"],  stiffness=4000, damping=400),
        "right_arm": ImplicitActuatorCfg(joint_names_expr=["right_arm_A[1-7]"], stiffness=4000, damping=400),
        "left_gripper_L":  ImplicitActuatorCfg(joint_names_expr=["left_hande_robotiq_hande_left_finger_joint"],   stiffness=50,  damping=10),
        "right_gripper_L": ImplicitActuatorCfg(joint_names_expr=["right_hande_robotiq_hande_left_finger_joint"],  stiffness=50,  damping=10),
        "left_gripper_R":  ImplicitActuatorCfg(joint_names_expr=["left_hande_robotiq_hande_right_finger_joint"],  stiffness=400, damping=40),
        "right_gripper_R": ImplicitActuatorCfg(joint_names_expr=["right_hande_robotiq_hande_right_finger_joint"], stiffness=400, damping=40),
    },
)


def create_hollow_cylinder(stage, prim_path, outer_r, inner_r, height, color, num_seg=48):
    """创建底部封闭的空心圆柱 mesh (顶部开口可插入 peg)。"""
    half_h = height / 2
    angles = np.linspace(0, 2 * np.pi, num_seg, endpoint=False)
    ca, sa = np.cos(angles), np.sin(angles)
    N = num_seg

    # 顶点: 0..N-1 底外, N..2N-1 底内, 2N..3N-1 顶外, 3N..4N-1 顶内, 4N 底中心
    points = []
    for ring_r, z in [(outer_r, -half_h), (inner_r, -half_h), (outer_r, half_h), (inner_r, half_h)]:
        for i in range(N):
            points.append(Gf.Vec3f(float(ring_r * ca[i]), float(ring_r * sa[i]), float(z)))
    points.append(Gf.Vec3f(0, 0, float(-half_h)))  # 底部中心点
    center = 4 * N

    fvc, fvi = [], []
    for i in range(N):
        j = (i + 1) % N
        fvc.append(4); fvi.extend([i, j, 2*N+j, 2*N+i])           # 外壁
        fvc.append(4); fvi.extend([N+i, 3*N+i, 3*N+j, N+j])       # 内壁
        fvc.append(4); fvi.extend([2*N+i, 2*N+j, 3*N+j, 3*N+i])   # 顶面环
        fvc.append(4); fvi.extend([i, N+i, N+j, j])                # 底面外环
        fvc.append(3); fvi.extend([center, N+j, N+i])              # 底面封盖

    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(prim_path))
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(fvc))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(fvi))
    mesh.GetSubdivisionSchemeAttr().Set("none")
    mesh.GetDoubleSidedAttr().Set(True)
    mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]))

    xf = UsdGeom.Xformable(mesh.GetPrim())
    xf.ClearXformOpOrder()
    t_op = xf.AddTranslateOp()
    r_op = xf.AddOrientOp()
    t_op.Set(Gf.Vec3d(0, 0, 2))
    return t_op, r_op


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    sim.set_camera_view([3.0, 0.0, 2.5], [0.0, 0.0, 1.2])

    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())

    robot = Articulation(ROBOT_CFG)

    peg = RigidObject(RigidObjectCfg(
        prim_path="/World/Peg",
        spawn=sim_utils.CylinderCfg(
            radius=PEG_RADIUS, height=PEG_HEIGHT, axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.12, 0.10), metallic=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 2)),
    ))

    hole_t_op, hole_r_op = create_hollow_cylinder(
        sim.stage, "/World/Hole", HOLE_OUTER_R, HOLE_INNER_R, HOLE_HEIGHT, (0.20, 0.75, 0.20))

    sim.reset()
    robot.reset()
    peg.reset()

    dev = robot.device
    dt = sim.get_physics_dt()
    rot90 = torch.tensor([[_C, _S, 0.0, 0.0]], device=dev)

    left_ee  = robot.find_bodies("left_hande_robotiq_hande_link")[0][0]
    right_ee = robot.find_bodies("right_hande_robotiq_hande_link")[0][0]
    peg_off  = torch.tensor([[PART_X, 0.0, PART_Z]], device=dev)
    hole_off = torch.tensor([[PART_X, 0.0, PART_Z]], device=dev)

    jnames = robot.data.joint_names
    idx_left_L  = jnames.index("left_hande_robotiq_hande_left_finger_joint")
    idx_right_L = jnames.index("right_hande_robotiq_hande_left_finger_joint")

    target = robot.data.default_joint_pos.clone()
    target[:, idx_left_L]  = LEFT_FINGER
    target[:, idx_right_L] = RIGHT_FINGER

    # 预热 (让手指到位)
    for _ in range(500):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

    print("环境就绪, 关闭窗口退出。")

    while simulation_app.is_running():
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

        lp, lq = robot.data.body_pos_w[:, left_ee], robot.data.body_quat_w[:, left_ee]
        rp, rq = robot.data.body_pos_w[:, right_ee], robot.data.body_quat_w[:, right_ee]

        # Peg
        peg.write_root_pose_to_sim(torch.cat([lp + quat_apply(lq, peg_off), quat_mul(lq, rot90)], dim=-1))
        peg.update(dt)

        # Hole
        hp = (rp + quat_apply(rq, hole_off))[0]
        hq = quat_mul(rq, rot90)[0]
        hole_t_op.Set(Gf.Vec3d(hp[0].item(), hp[1].item(), hp[2].item()))
        hole_r_op.Set(Gf.Quatf(hq[0].item(), Gf.Vec3f(hq[1].item(), hq[2].item(), hq[3].item())))


if __name__ == "__main__":
    main()
    simulation_app.close()
