"""双臂 peg-in-hole 场景 — 关节速度控制测试。

对左右臂的 A1~A7 关节施加正弦速度指令，验证速度控制是否正常工作。
手臂 actuator 的 stiffness=0，纯阻尼模式，使 velocity target 生效。

运行: cd ~/IsaacLab && ./isaaclab.sh -p ~/bimanual_peghole/scripts/test_velocity_control.py
远程运行: cd ~/IsaacLab && PUBLIC_IP=100.100.133.86 LIVESTREAM=1 ENABLE_CAMERAS=1 ./isaaclab.sh -p ~/bimanual_peghole/scripts/test_velocity_control.py
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

LEFT_FINGER  = 0.004
RIGHT_FINGER = 0.008
PART_X       = -0.0055
PART_Z       = 0.125

_C = math.cos(math.pi / 4)
_S = math.sin(math.pi / 4)

# ── 机器人配置 (手臂用速度控制: stiffness=0) ──
sys.path.insert(0, "/home/miao/IsaacLab/source/isaaclab_tasks")
from isaaclab_tasks.direct.miao_dual_arm.dual_arm_cfg import DUAL_ARM_CFG

ROBOT_CFG = DUAL_ARM_CFG.replace(
    prim_path="/World/DualArm",
    actuators={
        "left_arm":  ImplicitActuatorCfg(joint_names_expr=["left_arm_A[1-7]"],  stiffness=0, damping=400),
        "right_arm": ImplicitActuatorCfg(joint_names_expr=["right_arm_A[1-7]"], stiffness=0, damping=400),
        "left_gripper_L":  ImplicitActuatorCfg(joint_names_expr=["left_hande_robotiq_hande_left_finger_joint"],   stiffness=50,  damping=10),
        "right_gripper_L": ImplicitActuatorCfg(joint_names_expr=["right_hande_robotiq_hande_left_finger_joint"],  stiffness=50,  damping=10),
        "left_gripper_R":  ImplicitActuatorCfg(joint_names_expr=["left_hande_robotiq_hande_right_finger_joint"],  stiffness=400, damping=40),
        "right_gripper_R": ImplicitActuatorCfg(joint_names_expr=["right_hande_robotiq_hande_right_finger_joint"], stiffness=400, damping=40),
    },
)

# ── 速度控制参数 ──
VEL_AMP    = 0.3   # rad/s 振幅
VEL_FREQ   = 0.5   # Hz
TEST_JOINTS = [0, 1, 3]  # 选几个关节做正弦运动 (A1, A2, A4)


def create_hollow_cylinder(stage, prim_path, outer_r, inner_r, height, color, num_seg=48):
    """创建底部封闭的空心圆柱 mesh (顶部开口可插入 peg)。"""
    half_h = height / 2
    angles = np.linspace(0, 2 * np.pi, num_seg, endpoint=False)
    ca, sa = np.cos(angles), np.sin(angles)
    N = num_seg

    points = []
    for ring_r, z in [(outer_r, -half_h), (inner_r, -half_h), (outer_r, half_h), (inner_r, half_h)]:
        for i in range(N):
            points.append(Gf.Vec3f(float(ring_r * ca[i]), float(ring_r * sa[i]), float(z)))
    points.append(Gf.Vec3f(0, 0, float(-half_h)))
    center = 4 * N

    fvc, fvi = [], []
    for i in range(N):
        j = (i + 1) % N
        fvc.append(4); fvi.extend([i, j, 2*N+j, 2*N+i])
        fvc.append(4); fvi.extend([N+i, 3*N+i, 3*N+j, N+j])
        fvc.append(4); fvi.extend([2*N+i, 2*N+j, 3*N+j, 3*N+i])
        fvc.append(4); fvi.extend([i, N+i, N+j, j])
        fvc.append(3); fvi.extend([center, N+j, N+i])

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

    # 手臂关节索引
    left_arm_ids  = [jnames.index(f"left_arm_A{i}") for i in range(1, 8)]
    right_arm_ids = [jnames.index(f"right_arm_A{i}") for i in range(1, 8)]

    # 先用位置控制预热让夹爪到位 (速度目标 = 0 保持静止)
    pos_target = robot.data.default_joint_pos.clone()
    pos_target[:, idx_left_L]  = LEFT_FINGER
    pos_target[:, idx_right_L] = RIGHT_FINGER

    vel_target = torch.zeros_like(pos_target)

    for _ in range(500):
        robot.set_joint_position_target(pos_target)
        robot.set_joint_velocity_target(vel_target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

    print("预热完成, 开始速度控制测试 (正弦波驱动 A1, A2, A4)...")
    print("关闭窗口退出。")

    step = 0
    while simulation_app.is_running():
        t = step * dt

        # 构造速度目标: 对选定关节施加正弦速度, 左右臂镜像
        vel_target = torch.zeros_like(pos_target)
        for k, j_local in enumerate(TEST_JOINTS):
            v = VEL_AMP * math.sin(2 * math.pi * VEL_FREQ * t + k * 0.5)
            vel_target[:, left_arm_ids[j_local]]  = v
            vel_target[:, right_arm_ids[j_local]] = -v  # 镜像

        # 夹爪仍用位置控制保持开口
        robot.set_joint_position_target(pos_target)
        robot.set_joint_velocity_target(vel_target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

        # 每 100 步打印关节速度反馈
        if step % 100 == 0:
            actual_vel = robot.data.joint_vel[0]
            cmd_v = vel_target[0, left_arm_ids[TEST_JOINTS[0]]].item()
            act_v = actual_vel[left_arm_ids[TEST_JOINTS[0]]].item()
            print(f"[t={t:6.2f}s] left_A1 cmd_vel={cmd_v:+.3f}  actual_vel={act_v:+.3f} rad/s")

        # Peg 跟随左 EE
        lp, lq = robot.data.body_pos_w[:, left_ee], robot.data.body_quat_w[:, left_ee]
        rp, rq = robot.data.body_pos_w[:, right_ee], robot.data.body_quat_w[:, right_ee]

        peg.write_root_pose_to_sim(torch.cat([lp + quat_apply(lq, peg_off), quat_mul(lq, rot90)], dim=-1))
        peg.update(dt)

        hp = (rp + quat_apply(rq, hole_off))[0]
        hq = quat_mul(rq, rot90)[0]
        hole_t_op.Set(Gf.Vec3d(hp[0].item(), hp[1].item(), hp[2].item()))
        hole_r_op.Set(Gf.Quatf(hq[0].item(), Gf.Vec3f(hq[1].item(), hq[2].item(), hq[3].item())))

        step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
