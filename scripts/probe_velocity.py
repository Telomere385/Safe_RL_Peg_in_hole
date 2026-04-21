"""独立诊断: velocity 指令到底推不推得动关节?

不经 SAC, 直接给恒定 velocity action, 看 joint_pos 有没有在变.

运行:
    conda activate safe_rl
    python scripts/probe_velocity.py
"""
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs import DualArmPegHoleEnv

mdp = DualArmPegHoleEnv(num_envs=2, headless=True)

mask = torch.ones(2, dtype=torch.bool, device=mdp._device)
obs, _ = mdp.reset_all(mask)

def get_jp_jv(obs, env_idx=0):
    jp = mdp.observation_helper.get_from_obs(obs, "joint_pos")[env_idx]
    jv = mdp.observation_helper.get_from_obs(obs, "joint_vel")[env_idx]
    left_ee = mdp.observation_helper.get_from_obs(obs, "left_ee_pos")[env_idx]
    return jp.detach().cpu().numpy(), jv.detach().cpu().numpy(), left_ee.detach().cpu().numpy()

jp0, jv0, ee0 = get_jp_jv(obs)
print(f"[t=0]  jp_left[0:7]={jp0[:7].round(4)}")
print(f"       jv_left[0:7]={jv0[:7].round(4)}")
print(f"       left_ee_pos  ={ee0.round(4)}")
print()

# 恒定 action: 左臂全 +1, 右臂全 0.
# 经 _preprocess_action → clip+scale=0.4 rad/s velocity target.
# 30 vec-step × n_intermediate=5 × dt=0.02 = 3s 物理时间.
# 预期左臂关节移动 ~ 0.4 * 3 = 1.2 rad (如果 velocity 真起作用, 不撞极限).
act = torch.zeros(2, 14, device=mdp._device)
act[:, :7] = 1.0

for t in range(1, 31):
    obs, r, absorb, _ = mdp.step_all(mask, act)
    if t in (1, 3, 5, 10, 20, 30):
        jp, jv, ee = get_jp_jv(obs)
        djp = jp[:7] - jp0[:7]
        dee = ee - ee0
        print(f"[t={t:2d}] Δjp_left={djp.round(4)}")
        print(f"       jv_left  ={jv[:7].round(4)}")
        print(f"       Δee      ={dee.round(4)}  |Δee|={((dee**2).sum()**0.5):.4f}m")
        print()

mdp.stop()
