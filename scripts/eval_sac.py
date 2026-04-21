"""评估训练好的 SAC agent.

运行:
    conda activate safe_rl
    python scripts/eval_sac.py              # 带渲染窗口
    python scripts/eval_sac.py --headless   # 只看 J/R
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_path", type=str,
                   default=str(PROJECT_ROOT / "results/best_agent.msh"))
    p.add_argument("--n_episodes", type=int, default=3)
    p.add_argument("--num_envs", type=int, default=16,
                   help="与训练保持一致 (16). num_envs=1 会触发 cloner bug.")
    p.add_argument("--headless", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    from envs import DualArmPegHoleEnv
    mdp = DualArmPegHoleEnv(num_envs=args.num_envs, headless=args.headless)

    from mushroom_rl.core import Agent, VectorCore
    agent = Agent.load(args.agent_path)
    core = VectorCore(agent, mdp)

    dataset = core.evaluate(n_episodes=args.n_episodes,
                            render=not args.headless, quiet=False)
    J = torch.mean(dataset.discounted_return).item()
    R = torch.mean(dataset.undiscounted_return).item()
    print(f"J(γ)={J:.3f}  R={R:.3f}")

    mdp.stop()


if __name__ == "__main__":
    main()
