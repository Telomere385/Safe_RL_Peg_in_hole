"""Smoke test for the MushroomRL wrapper.

Run:
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/bimanual_peghole/scripts/smoke_test_mushroom_wrapper.py
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.mushroom_wrapper import DualArmPegHoleMushroom

def main():
    env = DualArmPegHoleMushroom()

    obs, reset_info = env.reset()
    print("reset type:", type(obs))
    print("reset shape:", obs.shape)
    print("reset info keys:", sorted(reset_info.keys()))

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (37,)
    assert isinstance(reset_info, dict)

    action = np.zeros(14, dtype=np.float32)
    obs_next, reward, done, info = env.step(action)

    print("step obs shape:", obs_next.shape)
    print("reward:", reward, type(reward))
    print("done:", done, type(done))
    print("info keys:", sorted(info.keys()))

    assert isinstance(obs_next, np.ndarray)
    assert obs_next.shape == (37,)
    assert isinstance(reward, float)
    assert isinstance(done, (bool, np.bool_))
    assert isinstance(info, dict)
    assert "cost" in info

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
