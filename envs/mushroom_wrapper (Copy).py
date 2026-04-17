"""MushroomRL wrapper for the IsaacLab DualArmPegHoleEnv.

Adapts the IsaacLab DirectRLEnv interface (torch, batched) to MushroomRL's
Environment interface (numpy, single-env).

Usage:
    from envs.mushroom_wrapper import DualArmPegHoleMushroom
    env = DualArmPegHoleMushroom(return_cost=True)
    obs = env.reset()
    obs, reward, cost, done, info = env.step(action)
"""

from __future__ import annotations

import numpy as np
import torch

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils.spaces import Box

from .dual_arm_peg_hole_env import DualArmPegHoleEnv, DualArmPegHoleEnvCfg


class DualArmPegHoleMushroom(Environment):
    """Single-env MushroomRL wrapper around DualArmPegHoleEnv (num_envs=1)."""

    def __init__(self, cfg: DualArmPegHoleEnvCfg | None = None, return_cost: bool = True, gamma: float = 0.99):
        self.return_cost = return_cost

        if cfg is None:
            cfg = DualArmPegHoleEnvCfg()
        cfg.scene.num_envs = 1

        self._env = DualArmPegHoleEnv(cfg)

        # Build MushroomRL MDPInfo from IsaacLab env config.
        obs_dim = cfg.observation_space    # 46
        act_dim = cfg.action_space         # 14

        observation_space = Box(low=-5.0, high=5.0, shape=(obs_dim,))
        action_space = Box(low=-1.0, high=1.0, shape=(act_dim,))

        horizon = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))

        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
        super().__init__(mdp_info)

    # ── MushroomRL interface ──────────────────────────────────────────────────

    def reset(self, state=None):
        obs_dict, _extras = self._env.reset()
        obs = self._to_numpy(obs_dict["policy"])
        return obs

    def step(self, action):
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self._env.device).unsqueeze(0)

        obs_dict, reward_t, terminated_t, truncated_t, extras = self._env.step(action_t)

        obs = self._to_numpy(obs_dict["policy"])
        reward = reward_t.item()
        done = terminated_t.item() or truncated_t.item()

        cost = extras.get("cost", torch.zeros(1))
        cost = cost.item() if isinstance(cost, torch.Tensor) else float(cost)

        info = {k: self._to_numpy(v) if isinstance(v, torch.Tensor) else v for k, v in extras.get("log", {}).items()}
        info["cost"] = cost

        if self.return_cost:
            return obs, reward, cost, done, info
        return obs, reward, done, info

    def render(self, record=False):
        frame = self._env.render()
        return frame

    def close(self):
        self._env.close()

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        """Squeeze batch dim and convert to numpy."""
        return t.squeeze(0).detach().cpu().numpy()
