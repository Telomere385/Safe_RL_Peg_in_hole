"""DualArmPegHoleCostEnv — Lagrangian SAC 专用子类.

继承 DualArmPegHoleEnv 的所有几何/物理/obs/终止逻辑，仅覆写 reward()
和新增 _create_info_dictionary()，将碰撞从 reward 信号剥离到 cost 约束信号。

设计原则
---------
标准 SAC (DualArmPegHoleEnv):
    collision → is_absorbing=True, reward = r_min/(1-γ) ≈ -200

Lagrangian SAC (DualArmPegHoleCostEnv):
    collision → is_absorbing=True, reward = shaped reward（不加大负奖励）
                                    cost   = 1.0（per-step 约束信号）

碰撞仍然终止 episode（is_absorbing 不变），但碰撞惩罚完全由 Lagrange 乘子 λ 和
cost critic Q_C 承担，reward critic Q_R 只看任务进展。这使得 SAC vs Lagrangian SAC
的对比实验能严格控制变量：相同 env 骨架、相同 obs、相同终止条件，仅 reward/cost
结构不同。

Cost 信号
---------
_create_info_dictionary() 返回 {"cost": collision_mask.float()}，其中
collision_mask = physx_collision | sphere_proxy_collision（self._last_collision_mask）。
当前 USD 下 PhysX 自碰撞不触发，等价于纯 sphere-proxy 信号；未来若 PhysX 恢复，
cost 会同时包含两种碰撞，行为自然正确。

可用于 reward() 的缓存量（reward 调用前已由 is_absorbing / _create_observation 写入）
---------------------------------------------------------------------------------
任务误差（来自 is_absorbing，每步刷新）:
    self._last_pos_err           [n_envs]        peg_tip 到 preinsert_target 的距离 (m)
    self._last_axis_err          [n_envs]        1 + dot(peg_axis, hole_axis) ∈ [0,2]
                                                 0 = 完美反平行对齐, 2 = 同向 (home 附近)
    self._last_success_mask      [n_envs] bool   full success: pos<pos_th AND axis<axis_th
    self._last_pos_success_mask  [n_envs] bool   pos-only success: pos<pos_th

碰撞信号（来自 is_absorbing，每步刷新）:
    self._last_collision_mask    [n_envs] bool   PhysX | sphere-proxy 碰撞，cost 信号来源
    self._last_min_clearance     [n_envs] float  双臂 sphere proxy 最小间距 (m)，
                                                 <0 表示穿插，可用于连续接近惩罚

终止类型（来自 is_absorbing）:
    self._last_hold_done_mask    [n_envs] bool   hold-N 成功触发的软 absorbing

动作（来自 _preprocess_action，每步刷新）:
    self._last_raw_action        [n_envs, 14]    tanh 后 clip 到 [-1,1] 的原始动作，
                                                 未乘 action_scale，用于 L2 惩罚

关节状态（从 next_obs 切片，reward 入参）:
    next_obs[..., :14]           [n_envs, 14]    joint_pos  左右臂 A1-A7 关节角
    next_obs[..., 14:28]         [n_envs, 14]    joint_vel  左右臂 A1-A7 关节角速度
    next_obs[..., 28:31]         [n_envs, 3]     pos_vec    peg_tip - preinsert_target
    next_obs[..., 31:34]         [n_envs, 3]     axis_resid peg_axis + hole_axis (34 维时)

关节限位（构造时确定，不变）:
    self._joint_lower            [14]            各关节下限 (rad)
    self._joint_upper            [14]            各关节上限 (rad)
    self._default_joint_pos      [14]            home pose (胸前 ready)
    self._home_weights           [14]            home regularizer 逐关节权重

reward 超参（构造时确定，可通过 CLI 覆盖）:
    self._w_pos                  float           位置误差权重 (default 1.0)
    self._w_axis                 float           轴对齐误差权重 (default 0.0 Stage1 / 0.5 Stage2)
    self._w_success              float           full success per-step bonus (default 2.0)
    self._w_pos_success          float           pos-only success bonus (default 0.0 / 1.0)
    self._w_joint_limit          float           关节极限软惩罚权重 (default 0.02)
    self._w_action               float           动作 L2 惩罚权重 (default 0.005)
    self._w_home                 float           home regularizer 权重 (default 0.0)
    self._reward_scale           float           全局 reward 缩放 (default 1.0)
    self._r_min                  float           absorbing reward 基础值 (default -2.0)
    self._terminal_hold_bonus    float           hold-N 成功终止奖励 (default 0.0)
    self._axis_gate_radius       float           axis 惩罚的距离门控半径 (default inf)
    self._preinsert_success_pos_threshold float  pos success 阈值 (default 0.10 m)
    self._success_axis_threshold float           axis success 阈值 (default inf / 0.50)
    self.info.gamma              float           折扣因子 (default 0.99)
    self._action_scale           float           动作缩放系数 (default 0.4 rad/s)
"""

import torch

from envs.dual_arm_peg_hole_env import DualArmPegHoleEnv, _AGENT_OBS_JOINT_POS


class DualArmPegHoleCostEnv(DualArmPegHoleEnv):
    """Lagrangian SAC 版环境：碰撞走 cost 约束，不进 reward.

    覆写 reward() 和新增 _create_info_dictionary()。
    如需全新 reward 设计，直接在 reward() 里用模块 docstring 列出的缓存量实现即可，
    不必调用 _compute_normal_reward()。
    """

    def reward(self, obs, action, next_obs, absorbing):
        """覆写 reward：碰撞步给 normal shaped reward，不给 r_min/(1-γ) 大负奖励.

        collision 的惩罚由 cost 约束（λ·Q_C）承担；reward critic 只学任务进展。
        hold-N absorbing 的 terminal_hold_bonus 行为不变。
        """
        normal = self._compute_normal_reward(next_obs)
        r = torch.where(
            self._last_hold_done_mask,
            normal + self._terminal_hold_bonus,
            normal,
        )
        return self._reward_scale * r

    def _create_info_dictionary(self):
        """每步返回 cost 信号，供 ConstrainedReplayMemory 和 SACLagrangian 使用.

        cost = 1.0 当且仅当该 env 该步发生碰撞（PhysX 或 sphere-proxy），否则 0.0.
        mushroom-rl VectorCore 会将返回 dict 写入 dataset.info.data，
        与 reward 在时间和 env 索引上对齐。
        """
        if self._last_collision_mask is None:
            cost = torch.zeros(self._n_envs, dtype=torch.float32, device=self._device)
        else:
            cost = self._last_collision_mask.float()
        return {"cost": cost}
