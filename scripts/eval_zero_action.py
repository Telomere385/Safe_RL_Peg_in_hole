"""Baseline: 全零 action, 评估 J / R / pos_err 轨迹.

目的: 搞清 "什么都不做" 在 kp=0 velocity drive 下是什么行为.
- 若 zero-action J 比 SAC 好 → agent 学到的反而不如不动, critic/actor 有问题
- 若 pos_err 随时间上升 → 重力在拉, 不动也在恶化, 必须主动补偿
"""
import sys
from pathlib import Path
import argparse
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

p = argparse.ArgumentParser()
p.add_argument("--n_episodes", type=int, default=16)
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--sanity", action="store_true")
p.add_argument("--trace", action="store_true")
args = p.parse_args()

from envs import DualArmPegHoleEnv
env_kwargs = dict(num_envs=args.num_envs, headless=True)
if args.sanity:
    env_kwargs.update(
        target_travel_fraction=0.0,
        initial_joint_noise=0.05,
        success_pos_threshold=0.1,
    )
mdp = DualArmPegHoleEnv(**env_kwargs)

# 手动跑一个 episode (所有 env 并行, 取前 n_episodes 个). 不用 VectorCore 避免搞 Agent 抽象.
mask = torch.ones(args.num_envs, dtype=torch.bool, device=mdp._device)
obs, _ = mdp.reset_all(mask)
gamma = mdp.info.gamma

act_zero = torch.zeros(args.num_envs, 14, device=mdp._device)
n_ep = min(args.n_episodes, args.num_envs)

R = torch.zeros(n_ep, device=mdp._device)
J = torch.zeros(n_ep, device=mdp._device)
done = torch.zeros(n_ep, dtype=torch.bool, device=mdp._device)
T = mdp.info.horizon

for t in range(T):
    obs, r, absorb, _ = mdp.step_all(mask, act_zero)
    # 对未 done 的 env 累加
    r_ep = r[:n_ep]
    active = ~done
    R = R + torch.where(active, r_ep, torch.zeros_like(r_ep))
    J = J + torch.where(active, (gamma ** t) * r_ep, torch.zeros_like(r_ep))
    done = done | absorb[:n_ep]

    if args.trace and t % 10 == 0:
        left_ee = mdp.observation_helper.get_from_obs(obs, "left_ee_pos")[0]
        l_err = torch.norm(left_ee - mdp._left_target).item()
        print(f"  t={t:3d}  pos_err_L={l_err:.4f}m  r[0]={r[0].item():+.3f}  "
              f"done_count={done.sum().item()}")

print(f"\n[ZERO-ACTION]  J={J.mean().item():.3f}  R={R.mean().item():.3f}  "
      f"(over {n_ep} episodes, {done.sum().item()}/{n_ep} absorbed early)")

mdp.stop()
