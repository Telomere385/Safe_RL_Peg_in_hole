"""MushroomRL adapter for the bimanual peg-in-hole IsaacLab environment.

Wraps the IsaacLab DirectRLEnv interface (torch, batched) into MushroomRL's
standard environment interface:

    reset(state=None)  -> (obs_np, info_dict)
    step(action_np)    -> (obs_np, reward, absorbing, info_dict)
    render(record)     -> frame
    info               -> MDPInfo(obs_space, action_space, gamma, horizon)
"""

from __future__ import annotations

import numpy as np
import torch

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils.spaces import Box


# ---------------------------------------------------------------------------
# Configuration defaults (mirrored from IsaacLab env cfg, kept here so the
# wrapper can be instantiated even when the IsaacLab env is not yet ready).
# ---------------------------------------------------------------------------
OBS_DIM = 37       # observation dimensionality
ACT_DIM = 14       # action dimensionality (7 joints x 2 arms)
GAMMA = 0.99
EPISODE_LENGTH_S = 10.0
SIM_DT = 0.02     # 50 Hz simulation
DECIMATION = 5     # 50 Hz / 10 Hz = 5
HORIZON = int(EPISODE_LENGTH_S / (SIM_DT * DECIMATION))   # 100 steps


class DualArmPegHoleMushroom(Environment):
    """MushroomRL Environment that wraps the IsaacLab bimanual peg-hole env."""

    def __init__(
        self,
        cfg=None,
        gamma: float = GAMMA,
        obs_low: float = -2.0,
        obs_high: float = 2.0,
    ):
        # ---- build MDPInfo (always available, even without IsaacLab) --------
        observation_space = Box(low=obs_low, high=obs_high, shape=(OBS_DIM,))
        action_space = Box(low=-1.0, high=1.0, shape=(ACT_DIM,))
        mdp_info = MDPInfo(observation_space, action_space, gamma, HORIZON)
        super().__init__(mdp_info)

        # ---- create the underlying IsaacLab env ----------------------------
        self._cfg = cfg
        self._isaaclab_env = None  # lazy init

    # ------------------------------------------------------------------
    # Internal: lazy-create the IsaacLab environment
    # ------------------------------------------------------------------
    def _get_isaaclab_env(self):
        if self._isaaclab_env is None:
            from .dual_arm_peg_hole_env import DualArmPegHoleEnv, DualArmPegHoleEnvCfg

            cfg = self._cfg if self._cfg is not None else DualArmPegHoleEnvCfg()
            cfg.scene.num_envs = 1
            self._isaaclab_env = DualArmPegHoleEnv(cfg)
        return self._isaaclab_env

    # ------------------------------------------------------------------
    # MushroomRL / SafeCore interface
    # ------------------------------------------------------------------
    def reset(self, state=None):
        """Reset the environment.

        Returns:
            tuple: (obs_np, info_dict)
                - obs_np: numpy array of shape (OBS_DIM,)
                - info_dict: episode metadata (empty dict for now)
        """
        env = self._get_isaaclab_env()
        obs_dict, extras = env.reset()
        obs = self._to_numpy(obs_dict["policy"])
        info = {
            k: (self._to_numpy(v) if isinstance(v, torch.Tensor) else v)
            for k, v in extras.items()
        }
        return obs, info

    def step(self, action):
        """Take one step in the environment.

        Args:
            action (np.ndarray): action of shape (ACT_DIM,), values in [-1, 1].

        Returns:
            tuple: (obs, reward, absorbing, info)
                - obs (np.ndarray): next observation, shape (OBS_DIM,)
                - reward (float): scalar reward
                - absorbing (bool): True if the episode terminated or truncated
                - info (dict): extra logging info
        """
        env = self._get_isaaclab_env()

        action_t = torch.as_tensor(
            action, dtype=torch.float32, device=env.device
        ).unsqueeze(0)

        obs_dict, reward_t, terminated_t, truncated_t, extras = env.step(action_t)

        obs = self._to_numpy(obs_dict["policy"])
        reward = float(reward_t.item())

        absorbing = bool(terminated_t.item() or truncated_t.item())

        # cost from the environment (joint-velocity constraint violation)
        cost_t = extras.get("cost", torch.zeros(1))
        cost = float(cost_t.item()) if isinstance(cost_t, torch.Tensor) else float(cost_t)

        # logging info
        log = extras.get("log", {})
        info = {
            k: (self._to_numpy(v) if isinstance(v, torch.Tensor) else v)
            for k, v in log.items()
        }
        info["cost"] = cost

        return obs, reward, absorbing, info

    def render(self, record=False):
        env = self._get_isaaclab_env()
        return env.render(record)

    def close(self):
        if self._isaaclab_env is not None:
            self._isaaclab_env.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        """Squeeze batch dim and convert to float32 numpy."""
        return t.squeeze(0).detach().cpu().float().numpy()
