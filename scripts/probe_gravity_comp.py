"""验证 IsaacSim 重力补偿前馈可行性:
1. get_generalized_gravity_forces() 是否返回合理的 τ_g
2. set_joint_efforts(τ_g) + velocity drive (kp=0 v_target=0) 能否让手臂静止住

对照 eval_zero_action.py: 纯 velocity drive 时 5 秒坠落 1m.
加了 G(q) 前馈后: 应当保持在 default pose (pos_err ≈ 0).
"""
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs import DualArmPegHoleEnv

mdp = DualArmPegHoleEnv(num_envs=2, headless=True,
                       target_travel_fraction=0.0,
                       initial_joint_noise=0.0,
                       success_pos_threshold=0.1)

mask = torch.ones(2, dtype=torch.bool, device=mdp._device)
obs, _ = mdp.reset_all(mask)

robots = mdp._task.robots
cj = mdp._task._controlled_joints
print(f"controlled_joints: {cj}")
print(f"n_envs: {mdp._n_envs}")

# 读一次 G(q)
tau_g = robots.get_generalized_gravity_forces(clone=True)
print(f"\ntau_g shape: {tau_g.shape}")
print(f"tau_g[env=0, controlled]: {tau_g[0][cj].detach().cpu().numpy().round(3)}")

# 读 C(q, q̇) q̇
tau_c = robots.get_coriolis_and_centrifugal_forces(clone=True)
print(f"tau_c[env=0, controlled]: {tau_c[0][cj].detach().cpu().numpy().round(3)}")

# 自己跑 30 step, 每步: 发 v=0 命令, 额外 set_joint_efforts(G(q))
act_zero = torch.zeros(2, 14, device=mdp._device)
gamma = mdp.info.gamma

# 我们要绕过 step_all 的默认 apply_action, 手动来:
# 先 reset 让状态干净
mask_all = torch.ones(2, dtype=torch.bool, device=mdp._device)
mdp.reset_all(mask_all)

# 手动仿 step_all 核心循环, 但插入重力补偿
from isaacsim.core.utils.types import ArticulationActions

def get_left_ee_err():
    obs_list = mdp._task.get_observations(clone=True)
    obs_vec = mdp.observation_helper.build_obs(obs_list)
    left_ee = mdp.observation_helper.get_from_obs(obs_vec, "left_ee_pos")[0]
    return torch.norm(left_ee - mdp._left_target).item()

for t in range(30):
    # 每个 sub-step 都补
    for _ in range(mdp._n_intermediate_steps):
        # 1) velocity target = 0
        act = ArticulationActions(joint_indices=cj,
                                  joint_velocities=torch.zeros(2, 14, device=mdp._device))
        robots.apply_action(act)
        # 2) feedforward = G(q) on controlled joints only
        tau_g = robots.get_generalized_gravity_forces(clone=True)
        eff = tau_g[:, cj]
        robots.set_joint_efforts(eff, joint_indices=cj)
        mdp._world.step(render=False)

    if t in (0, 5, 10, 20, 29):
        err = get_left_ee_err()
        print(f"  t={t:2d}  left_ee_pos_err = {err:.5f}m")

mdp.stop()
