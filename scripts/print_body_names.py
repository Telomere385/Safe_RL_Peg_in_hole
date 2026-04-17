"""打印机器人所有 body 名称."""
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app

import sys
sys.path.insert(0, "/home/miao/IsaacLab/source/isaaclab_tasks")
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_tasks.direct.miao_dual_arm.dual_arm_cfg import DUAL_ARM_CFG

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
robot = Articulation(DUAL_ARM_CFG.replace(prim_path="/World/Robot"))
sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
sim.reset(); robot.reset()
robot.update(0.005)

print("=== Body names ===")
for i, name in enumerate(robot.data.body_names):
    print(f"  [{i:2d}] {name}")

print("\n=== Joint names ===")
for i, name in enumerate(robot.data.joint_names):
    print(f"  [{i:2d}] {name}")

app.close()
