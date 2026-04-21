"""用确定性策略 (tanh(μ), 不 sample) 评估 best_agent.

对比 eval_sac.py (sample 策略): 差值 ≈ "因采样噪声损失的 return".
若差异巨大 → agent 的 μ 已指向目标, 但 σ 把它推出 success 区 → 需要确定性 eval
或 pushing α 更小 / 目标熵更深.
"""
import sys
from pathlib import Path
import argparse
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

p = argparse.ArgumentParser()
p.add_argument("--agent_path", default=str(PROJECT_ROOT / "results/best_agent.msh"))
p.add_argument("--n_episodes", type=int, default=16)
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--sanity", action="store_true",
               help="若用 --sanity 训练的 agent, eval 也要用相同 env 配置")
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

from mushroom_rl.core import Agent, VectorCore
agent = Agent.load(args.agent_path)

# Monkeypatch policy.draw_action → 用 tanh(μ) 不采样
policy = agent.policy
orig_draw = policy.draw_action

def deterministic_draw(state, internal_state=None):
    mu = policy._mu_approximator.predict(state)
    a_raw = torch.as_tensor(mu, device=policy._delta_a.device, dtype=policy._delta_a.dtype)
    a = torch.tanh(a_raw)
    a_true = a * policy._delta_a + policy._central_a
    return a_true.detach(), None

policy.draw_action = deterministic_draw

core = VectorCore(agent, mdp)
dataset = core.evaluate(n_episodes=args.n_episodes, quiet=False)
J = torch.mean(dataset.discounted_return).item()
R = torch.mean(dataset.undiscounted_return).item()
print(f"\n[DETERMINISTIC]  J={J:.3f}  R={R:.3f}")

# 再跑一次 stochastic 对照
policy.draw_action = orig_draw
dataset = core.evaluate(n_episodes=args.n_episodes, quiet=True)
J_s = torch.mean(dataset.discounted_return).item()
R_s = torch.mean(dataset.undiscounted_return).item()
print(f"[STOCHASTIC]     J={J_s:.3f}  R={R_s:.3f}")
print(f"Δ J = {J - J_s:+.3f}   (正 = 确定性更好, 确认采样噪声是问题)")

mdp.stop()
