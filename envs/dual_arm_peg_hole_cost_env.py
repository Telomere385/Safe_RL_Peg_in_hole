"""DualArmPegHoleCostEnv — Lagrangian SAC Stage 2 专用子类.

Stage 2 冷启动 reward 设计（参照 Stage 1 骨架，零补丁）:

    r = - w_pos     × pos_err                    # 位置主梯度
        - w_axis    × axis_err                   # 轴向主梯度（无门控，冷启动两项同时开）
        + w_axis_prog × clamp(1-axis_err/2,0,1)  # 轴对齐渐进正奖励
        + w_success × 1[pos_ok AND axis_ok]      # 成功 per-step bonus（不终止）
        - w_action  × ‖a_raw‖²                   # 动作平滑
        - w_home    × Σ ((q−q_home)/range)²      # 均匀 home 正则（tie-breaker）

Stage 1 相比移除的所有遗留补丁:
    - w_pos_success   : Stage 1→2 暖启动桥接，冷启动无需保护已学行为
    - axis_gate_radius: 暖启动补丁，防 axis 项干扰已收敛的位置策略
    - w_joint_limit   : 软关节极限惩罚，安全全权委托给 Lagrangian cost
    - terminal_hold_bonus / hold-N absorbing: 增加 Q-target 边界复杂度，收益有限
    - home_weights 非对称: 针对暖启动手腕姿态调出的非均匀权重

量纲平衡（冷启动 home 位置）:
    pos_err  ≈ 0.4~0.8m  →  w_pos=1.0  → 贡献 −0.4~−0.8
    axis_err ≈ 2.0        →  w_axis=0.5 → 贡献 −1.0（两项幅度相近，互不压制）
    axis_prog ∈ [0,1]     →  默认 0.0；需要正向 shaping 时配置打开
    success  per-step     →  w_success=2.0（清晰可见但不扭曲 Q 估计）
    action   最大≈14      →  w_action=0.005 → 最大 −0.07（正则量级）
    home_dev 典型0.5~1.5  →  w_home=0.001  → 最大 −0.002（tie-breaker 量级）

cost 信号（不变）:
    collision indicator (sphere-proxy OR PhysX)
    → _create_info_dictionary() → dataset.info.data["cost"]
    → ConstrainedReplayMemory → SACLagrangian.fit()
"""

import torch

from envs.dual_arm_peg_hole_env import DualArmPegHoleEnv, _AGENT_OBS_JOINT_POS


class DualArmPegHoleCostEnv(DualArmPegHoleEnv):
    """Stage 2 Lagrangian SAC 环境: 干净 reward + collision cost."""

    _DEFAULT_W_POS     = 1.0
    _DEFAULT_W_AXIS    = 0.5
    _DEFAULT_W_AXIS_PROGRESS = 0.0
    _DEFAULT_W_SUCCESS = 2.0
    _DEFAULT_W_ACTION  = 0.005
    _DEFAULT_W_HOME    = 0.001

    def __init__(
        self,
        rew_pos=_DEFAULT_W_POS,
        rew_axis=_DEFAULT_W_AXIS,
        rew_axis_progress=_DEFAULT_W_AXIS_PROGRESS,
        rew_success=_DEFAULT_W_SUCCESS,
        rew_action=_DEFAULT_W_ACTION,
        rew_home=_DEFAULT_W_HOME,
        **parent_kwargs,
    ):
        """
        Args:
            rew_pos:     pos_err 惩罚系数.       默认 1.0
            rew_axis:    axis_err 惩罚系数.      默认 0.5 (axis_err∈[0,2], 折合满量程≈pos项)
            rew_axis_progress: 轴对齐渐进正奖励系数. 默认 0.0
            rew_success: 成功 per-step bonus.    默认 2.0
            rew_action:  动作 L2 正则系数.        默认 0.005
            rew_home:    Home 偏差正则系数.       默认 0.001 (均匀权重, tie-breaker 量级)
            **parent_kwargs: 透传 DualArmPegHoleEnv (几何/物理/obs/终止参数).
                terminal_hold_bonus 固定为 0.0 — hold-N absorbing 在 Stage 2 关闭.
                parent 的 reward 权重参数若经由 kwargs 传入则无效 (reward() 已覆写).
        """
        self._s2_w_pos     = float(rew_pos)
        self._s2_w_axis    = float(rew_axis)
        self._s2_w_axis_progress = float(rew_axis_progress)
        self._s2_w_success = float(rew_success)
        self._s2_w_action  = float(rew_action)
        self._s2_w_home    = float(rew_home)
        # hold-N absorbing 强制关闭
        parent_kwargs.pop("terminal_hold_bonus", None)
        super().__init__(terminal_hold_bonus=0.0, **parent_kwargs)

    # ------------------------------------------------------------------
    # Stage 2 reward（干净公式，不调父类 _compute_normal_reward）
    # ------------------------------------------------------------------
    def _compute_stage2_reward(self, next_obs):
        pos_err      = self._last_pos_err
        axis_err     = self._last_axis_err
        axis_progress = (1.0 - 0.5 * axis_err).clamp(0.0, 1.0)
        full_success = self._last_success_mask.to(pos_err.dtype)

        joint_pos = next_obs[..., _AGENT_OBS_JOINT_POS]
        action_sq = (self._last_raw_action ** 2).sum(dim=-1)

        joint_range = self._joint_upper - self._joint_lower
        home_dev    = (
            (joint_pos - self._default_joint_pos.unsqueeze(0)) / joint_range.unsqueeze(0)
        )
        home_norm = (home_dev ** 2).sum(dim=-1)

        return (
            - self._s2_w_pos     * pos_err
            - self._s2_w_axis    * axis_err
            + self._s2_w_axis_progress * axis_progress
            + self._s2_w_success * full_success
            - self._s2_w_action  * action_sq
            - self._s2_w_home    * home_norm
        )

    def reward(self, obs, action, next_obs, absorbing):
        return self._reward_scale * self._compute_stage2_reward(next_obs)

    # ------------------------------------------------------------------
    # Cost signal
    # ------------------------------------------------------------------
    def cost(self):
        if self._last_collision_mask is None:
            return torch.zeros(self._n_envs, dtype=torch.float32, device=self._device)
        return self._last_collision_mask.float()

    def _create_info_dictionary(self, obs):
        return {"cost": self.cost()}
