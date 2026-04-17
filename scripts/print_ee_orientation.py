"""打印默认姿态下左右 EE 的朝向轴, 确认哪个轴是夹爪前方."""
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app

import sys, torch
sys.path.insert(0, "/home/miao/IsaacLab/source/isaaclab_tasks")
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_tasks.direct.miao_dual_arm.dual_arm_cfg import DUAL_ARM_CFG

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
robot = Articulation(DUAL_ARM_CFG.replace(prim_path="/World/Robot"))
sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
sim.reset(); robot.reset()
for _ in range(200):
    robot.write_data_to_sim(); sim.step(); robot.update(0.005)

def quat_rotate(quat, vec):
    """wxyz quaternion rotation."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    vx, vy, vz = vec[0], vec[1], vec[2]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return torch.tensor([
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ])

left_idx = robot.find_bodies("left_hande_robotiq_hande_link")[0][0]
right_idx = robot.find_bodies("right_hande_robotiq_hande_link")[0][0]

for name, idx in [("Left EE", left_idx), ("Right EE", right_idx)]:
    quat = robot.data.body_quat_w[0, idx]  # wxyz
    pos = robot.data.body_pos_w[0, idx]
    x_axis = quat_rotate(quat, torch.tensor([1., 0., 0.]))
    y_axis = quat_rotate(quat, torch.tensor([0., 1., 0.]))
    z_axis = quat_rotate(quat, torch.tensor([0., 0., 1.]))
    print(f"\n{name} (body idx {idx}):")
    print(f"  Position: {[round(v, 4) for v in pos.cpu().tolist()]}")
    print(f"  Quat(wxyz): {[round(v, 4) for v in quat.cpu().tolist()]}")
    print(f"  Local X → World: {[round(v, 4) for v in x_axis.cpu().tolist()]}")
    print(f"  Local Y → World: {[round(v, 4) for v in y_axis.cpu().tolist()]}")
    print(f"  Local Z → World: {[round(v, 4) for v in z_axis.cpu().tolist()]}")

# 两 EE 之间的连线方向
lp = robot.data.body_pos_w[0, left_idx]
rp = robot.data.body_pos_w[0, right_idx]
l2r = rp - lp
l2r_unit = l2r / torch.norm(l2r)
print(f"\nLeft→Right direction: {[round(v, 4) for v in l2r_unit.cpu().tolist()]}")
print(f"EE distance: {torch.norm(l2r).item():.4f} m")

app.close()
