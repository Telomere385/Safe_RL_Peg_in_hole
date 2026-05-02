from .dual_arm_peg_hole_env import (
    AGENT_OBS_DIM_AXIS,
    AGENT_OBS_DIM_BASE,
    AGENT_OBS_DIM_ROTVEC,
    DEFAULT_PREINSERT_OFFSET,
    DualArmPegHoleEnv,
)

# 向后兼容: 旧代码可能引用 AGENT_OBS_DIM (32 维默认).
AGENT_OBS_DIM = AGENT_OBS_DIM_BASE

__all__ = [
    "AGENT_OBS_DIM",
    "AGENT_OBS_DIM_BASE",
    "AGENT_OBS_DIM_AXIS",
    "AGENT_OBS_DIM_ROTVEC",
    "DEFAULT_PREINSERT_OFFSET",
    "DualArmPegHoleEnv",
]
