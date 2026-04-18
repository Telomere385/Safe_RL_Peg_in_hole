"""双臂末端靠近任务 — 基于 mushroom-rl 的 IsaacSim 向量化环境.

没有任何 isaaclab 依赖. IsaacSim 原生 API 由 mushroom_rl.environments.IsaacSim 托管.

观测 (37 维, 全部按物理量归一化前的原始值):
    joint_pos     (14)   左右臂 7+7 个关节角
    joint_vel     (14)   左右臂 7+7 个关节角速度
    left_ee_pos   (3)    左末端位姿 (env-local)
    right_ee_pos  (3)    右末端位姿 (env-local)
    delta         (3)    left_ee - right_ee  (由 _create_observation 注入)

动作 (14 维): action ∈ [-1, 1] 经 _preprocess_action 缩放为关节速度指令.
碰撞吸收: 左右臂任意 link 之间的接触力 > 阈值 → absorbing, r = r_min / (1 - γ).
"""

import torch

from mushroom_rl.environments import IsaacSim
from mushroom_rl.utils.isaac_sim import ObservationType, ActionType
from mushroom_rl.rl_utils.spaces import Box


USD_PATH = "/home/miao/dual_arm_ws/usd_imports/dual_arm_iiwa/dual_arm_iiwa.usd"

LEFT_ARM_JOINTS = [f"left_arm_A{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"right_arm_A{i}" for i in range(1, 8)]
ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS  # 14

LEFT_EE_PATH = "/left_hande_robotiq_hande_link"
RIGHT_EE_PATH = "/right_hande_robotiq_hande_link"

LEFT_ARM_LINKS = [f"/left_arm_link_{i}" for i in range(1, 8)] + [LEFT_EE_PATH]
RIGHT_ARM_LINKS = [f"/right_arm_link_{i}" for i in range(1, 8)] + [RIGHT_EE_PATH]
ARM_LINKS = LEFT_ARM_LINKS + RIGHT_ARM_LINKS  # 单一碰撞组, 索引 [:8]=左, [8:]=右


class DualArmPegHoleEnv(IsaacSim):
    """双臂末端靠近 — vectorized IsaacSim 环境."""

    def __init__(
        self,
        num_envs=1,
        horizon=100,
        gamma=0.99,
        headless=True,
        device="cuda:0",
        action_scale=1.5,
        initial_joint_noise=0.1,
        collision_force_threshold=10.0,
        reward_absorbing_r_min=-1.0,
        rew_mutual=1.0,
        rew_approach=5.0,
        approach_scale=0.3,
        rew_joint_reg=0.005,
        rew_joint_limit=1.0,
        joint_limit_margin_frac=0.95,
    ):
        self._action_scale = action_scale
        self._initial_joint_noise = initial_joint_noise
        self._collision_threshold = collision_force_threshold
        self._r_min = reward_absorbing_r_min
        self._gamma_cfg = gamma
        self._w_mutual = rew_mutual
        self._w_approach = rew_approach
        self._approach_scale = approach_scale
        self._w_joint_reg = rew_joint_reg
        self._w_joint_limit = rew_joint_limit
        self._joint_limit_margin_frac = joint_limit_margin_frac

        observation_spec = [
            ("joint_pos", "", ObservationType.JOINT_POS, ARM_JOINTS),
            ("joint_vel", "", ObservationType.JOINT_VEL, ARM_JOINTS),
            ("left_ee_pos", LEFT_EE_PATH, ObservationType.BODY_POS, None),
            ("right_ee_pos", RIGHT_EE_PATH, ObservationType.BODY_POS, None),
        ]

        # 双碰撞组: 左右臂分离, RigidContactView 可用 group-to-group 的过滤.
        collision_groups = [("arm_L", LEFT_ARM_LINKS), ("arm_R", RIGHT_ARM_LINKS)]

        super().__init__(
            usd_path=USD_PATH,
            actuation_spec=ARM_JOINTS,
            observation_spec=observation_spec,
            backend="torch",
            device=device,
            collision_between_envs=False,
            num_envs=num_envs,
            env_spacing=4.0,
            gamma=gamma,
            horizon=horizon,
            timestep=0.02,
            n_intermediate_steps=5,
            action_type=ActionType.VELOCITY,
            collision_groups=collision_groups,
            headless=headless,
        )

        # 额外观测: delta = left_ee - right_ee
        self.observation_helper.add_obs("delta", 3, -5.0, 5.0)
        obs_low, obs_high = self.observation_helper.obs_limits
        self._mdp_info.observation_space = Box(obs_low, obs_high, data_type=obs_high.dtype)

        # 动作空间覆写为 [-1, 1]^14  (SAC 的 tanh 策略用)
        one = torch.ones(len(ARM_JOINTS), device=device)
        self._mdp_info.action_space = Box(-one, one, data_type=one.dtype)

        # 关节位限, 用于 reward 中的 joint-limit 惩罚
        # get_joint_pos_limits 返回 shape (2, n_joints) — [0]=lower, [1]=upper
        limits = self._task.get_joint_pos_limits()
        self._joint_lower = limits[0]
        self._joint_upper = limits[1]
        self._default_joint_pos = self._task.robots.get_joints_default_state().positions[0][
            self._task._controlled_joints
        ].clone()

    # ------------------------------------------------------------------
    # 动作处理
    # ------------------------------------------------------------------
    def _preprocess_action(self, action):
        # action ∈ [-1, 1] → 关节速度指令 rad/s
        return torch.clamp(action, -1.0, 1.0) * self._action_scale

    # ------------------------------------------------------------------
    # 观测注入: delta
    # ------------------------------------------------------------------
    def _create_observation(self, obs):
        left_ee = self.observation_helper.get_from_obs(obs, "left_ee_pos")
        right_ee = self.observation_helper.get_from_obs(obs, "right_ee_pos")
        delta_idx = self.observation_helper.obs_idx_map["delta"]
        obs[:, delta_idx] = left_ee - right_ee
        return obs

    # ------------------------------------------------------------------
    # 终止判定: 左右臂接触
    # ------------------------------------------------------------------
    def is_absorbing(self, obs):
        # 左右臂互撞: 接触力矩阵最大模大于阈值时吸收.
        return self._check_collision("arm_L", "arm_R", self._collision_threshold,
                                     dt=self._timestep)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def reward(self, obs, action, next_obs, absorbing):
        left_ee = self.observation_helper.get_from_obs(next_obs, "left_ee_pos")
        right_ee = self.observation_helper.get_from_obs(next_obs, "right_ee_pos")
        joint_pos = self.observation_helper.get_from_obs(next_obs, "joint_pos")

        dist = torch.norm(left_ee - right_ee, dim=-1)
        approach = torch.exp(-dist / self._approach_scale)

        joint_dev = torch.sum((joint_pos - self._default_joint_pos) ** 2, dim=-1)

        margin = self._joint_upper * self._joint_limit_margin_frac
        over = torch.clamp(torch.abs(joint_pos) - margin, min=0.0)
        joint_limit_pen = torch.sum(over ** 2, dim=-1)

        normal = (
            -self._w_mutual * dist
            + self._w_approach * approach
            - self._w_joint_reg * joint_dev
            - self._w_joint_limit * joint_limit_pen
        )

        # 吸收态: 一次性给 r_min / (1 - γ), 等价于吸收态中每步获得 r_min
        absorbing_r = self._r_min / (1.0 - self._gamma_cfg)
        return torch.where(absorbing, torch.full_like(normal, absorbing_r), normal)

    # ------------------------------------------------------------------
    # Reset 初始化
    # ------------------------------------------------------------------
    def setup(self, env_indices, obs):
        n = len(env_indices)
        noise = self._initial_joint_noise * (
            2.0 * torch.rand(n, len(ARM_JOINTS), device=self._device) - 1.0
        )
        joint_pos = self._default_joint_pos.unsqueeze(0) + noise
        joint_vel = torch.zeros_like(joint_pos)
        self._write_data("joint_pos", joint_pos, env_indices)
        self._write_data("joint_vel", joint_vel, env_indices)
