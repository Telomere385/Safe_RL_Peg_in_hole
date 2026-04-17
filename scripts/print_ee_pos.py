"""打印默认姿态下左右末端位置."""
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

left_ee = robot.find_bodies("left_hande_robotiq_hande_link")[0][0]
right_ee = robot.find_bodies("right_hande_robotiq_hande_link")[0][0]
lp = robot.data.body_pos_w[:, left_ee]
rp = robot.data.body_pos_w[:, right_ee]
print(f"Left  EE: {[round(x,4) for x in lp[0].cpu().tolist()]}")
print(f"Right EE: {[round(x,4) for x in rp[0].cpu().tolist()]}")
print(f"Midpoint: {[round(x,4) for x in ((lp+rp)/2)[0].cpu().tolist()]}")
print(f"Root:     {[round(x,4) for x in robot.data.root_pos_w[0].cpu().tolist()]}")
app.close()
