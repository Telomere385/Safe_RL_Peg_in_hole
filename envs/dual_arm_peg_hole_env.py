"""IsaacLab DirectRLEnv for bimanual peg-in-hole — 简化版 (末端靠近任务).

第一阶段目标: 让左右臂末端夹爪尽可能靠近，验证整个 RL pipeline 可以跑通。
动作空间: 14 维关节速度指令 (7 joints × 2 arms)
观测空间: 关节位置(14) + 关节速度(14) + 左EE位置(3) + 右EE位置(3) + EE距离差(3) = 37 维

碰撞检测: 使用 ContactSensor 监测自碰撞，碰撞 → absorbing state → r_abs
    absorbing state value: Q_abs = r_abs / (1 - γ)
"""

from __future__ import annotations

import sys
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

# ── 机器人基础配置 ──
sys.path.insert(0, "/home/miao/IsaacLab/source/isaaclab_tasks")
from isaaclab_tasks.direct.miao_dual_arm.dual_arm_cfg import DUAL_ARM_CFG


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@configclass
class DualArmPegHoleEnvCfg(DirectRLEnvCfg):
    """双臂末端靠近任务配置."""

    # env
    decimation = 5              # 50Hz / 10Hz = 5 sim steps per control step
    episode_length_s = 10.0
    action_space = 14           # 7 joints × 2 arms
    observation_space = 37      # joint_pos(14) + joint_vel(14) + left_ee(3) + right_ee(3) + delta(3)
    state_space = 0

    # simulation: 50 Hz 仿真, 10 Hz 控制
    sim: SimulationCfg = SimulationCfg(
        dt=0.02,                # 1/50 = 0.02s → 50 Hz 仿真
        render_interval=5,
    )

    # robot — 速度控制 (stiffness=0, 纯阻尼)
    robot_cfg: ArticulationCfg = DUAL_ARM_CFG.replace(
        prim_path="/World/envs/env_.*/DualArm",
        spawn=DUAL_ARM_CFG.spawn.replace(
            activate_contact_sensors=True,  # ContactSensor 需要此项
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,  # 启用自碰撞检测
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                fix_root_link=True,
            ),
        ),
        actuators={
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_A[1-7]"],
                stiffness=0.0,
                damping=400.0,
            ),
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["right_arm_A[1-7]"],
                stiffness=0.0,
                damping=400.0,
            ),
            # 夹爪固定位置控制 (不参与 RL)
            "left_gripper_L": ImplicitActuatorCfg(
                joint_names_expr=["left_hande_robotiq_hande_left_finger_joint"],
                stiffness=50.0,
                damping=10.0,
            ),
            "right_gripper_L": ImplicitActuatorCfg(
                joint_names_expr=["right_hande_robotiq_hande_left_finger_joint"],
                stiffness=50.0,
                damping=10.0,
            ),
            "left_gripper_R": ImplicitActuatorCfg(
                joint_names_expr=["left_hande_robotiq_hande_right_finger_joint"],
                stiffness=400.0,
                damping=40.0,
            ),
            "right_gripper_R": ImplicitActuatorCfg(
                joint_names_expr=["right_hande_robotiq_hande_right_finger_joint"],
                stiffness=400.0,
                damping=40.0,
            ),
        },
    )

    # 碰撞传感器 — 监测所有 body 的接触力
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/DualArm/.*",
        history_length=2,
        update_period=0.0,  # 每个 sim step 更新
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # 末端执行器 body 名称
    left_ee_body_name: str = "left_hande_robotiq_hande_link"
    right_ee_body_name: str = "right_hande_robotiq_hande_link"

    # 手臂关节名称 (只控制手臂, 不控制夹爪)
    left_arm_joint_names: list = [f"left_arm_A{i}" for i in range(1, 8)]
    right_arm_joint_names: list = [f"right_arm_A{i}" for i in range(1, 8)]

    # 动作缩放: action ∈ [-1,1] × action_scale = 速度指令 (rad/s)
    # KUKA iiwa7 关节速度上限 1.5~2.4 rad/s, 取 1.5 保证安全范围
    action_scale: float = 1.5

    # KUKA iiwa7 关节限位 (rad), 对称 ±
    # A1: ±170°, A2: ±120°, A3: ±170°, A4: ±120°, A5: ±170°, A6: ±120°, A7: ±175°
    joint_limits_rad: list = [2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054]
    # 接近限位时开始衰减速度指令的缓冲区 (rad)
    joint_limit_buffer: float = 0.1

    # ── reward 配置 ──
    rew_weight_mutual: float = 1.0       # 两末端互相靠近 (主目标, -dist)
    rew_weight_approach: float = 5.0     # 接近指数奖励峰值: +w * exp(-dist/scale)
    approach_scale: float = 0.3          # 指数衰减尺度 (m); dist=scale 时 bonus≈w/e
    rew_weight_joint_reg: float = 0.005  # 关节偏离默认位置的惩罚 (轻微正则)
    rew_weight_height: float = 0.0       # EE 低于底座高度的惩罚 (禁用)
    rew_weight_facing: float = 0.0       # EE 朝向对齐奖励 (暂时禁用)
    rew_weight_joint_limit: float = 1.0  # 接近关节极限的惩罚
    joint_limit_margin_frac: float = 0.95  # 关节限位惩罚触发阈值 (限位的百分比)
    robot_base_height: float = 0.83      # 机器人底座高度

    # ── absorbing state (碰撞) 配置 ──
    # 碰撞检测力阈值 (N): 过低会被仿真初始化噪声误触发
    collision_force_threshold: float = 10.0
    # r_min: 碰撞惩罚基准值
    # 碰撞时一次性给 r_min / (1 - γ), 然后 terminate
    # 等价于在 absorbing state 中每步获得 r_min, Q_abs = r_min / (1 - γ)
    # 例如 r_min=-1.0, γ=0.99 → 碰撞 reward = -1 / 0.01 = -100
    reward_absorbing_r_min: float = -1.0
    reward_gamma: float = 0.99  # discount factor, 用于计算 absorbing reward

    # reset 时关节位置随机扰动范围
    initial_joint_noise: float = 0.1  # rad


# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------
class DualArmPegHoleEnv(DirectRLEnv):
    """双臂末端靠近任务 — 带碰撞检测和 absorbing state."""

    cfg: DualArmPegHoleEnvCfg

    def __init__(self, cfg: DualArmPegHoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ── 关节索引 ──
        joint_names = self.robot.data.joint_names
        self._left_arm_ids = [joint_names.index(n) for n in self.cfg.left_arm_joint_names]
        self._right_arm_ids = [joint_names.index(n) for n in self.cfg.right_arm_joint_names]
        self._arm_joint_ids = self._left_arm_ids + self._right_arm_ids  # 14 个
        # 用于 _apply_action 的向量化索引
        self._arm_joint_ids_tensor = torch.tensor(self._arm_joint_ids, device=self.device, dtype=torch.long)

        # ── 末端执行器 body 索引 ──
        self._left_ee_idx = self.robot.find_bodies(self.cfg.left_ee_body_name)[0][0]
        self._right_ee_idx = self.robot.find_bodies(self.cfg.right_ee_body_name)[0][0]

        # ── 夹爪固定位置目标 ──
        self._gripper_target = self.robot.data.default_joint_pos.clone()

        # ── 默认关节位置 (用于 regularization) ──
        self._default_arm_pos = self.robot.data.default_joint_pos[:, self._arm_joint_ids].clone()

        # ── 碰撞 body 索引 ──
        # 监测手臂 link 的碰撞 (不含夹爪指尖, 因为末端靠近时指尖可能轻触)
        self._collision_body_ids, _ = self._contact_sensor.find_bodies(".*arm.*")

        # ── 关节限位张量 (左右臂相同限位, 共 14 维) ──
        limits = torch.tensor(self.cfg.joint_limits_rad * 2, device=self.device)  # (14,)
        self._joint_upper = limits.unsqueeze(0)   # (1, 14)
        self._joint_lower = -limits.unsqueeze(0)  # (1, 14)

        # ── absorbing state flag ──
        self._in_absorbing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # actions: (num_envs, 14), 范围 [-1, 1]
        self.actions = actions.clone().clamp(-1.0, 1.0) * self.cfg.action_scale

    def _apply_action(self) -> None:
        # 手臂: 速度控制 + 关节限位强制执行
        arm_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]  # (N, 14)
        arm_vel_cmd = self.actions.clone()  # (N, 14), 已在 _pre_physics_step 中 clamp

        # 当关节接近上/下限时，禁止继续往越限方向运动
        buf = self.cfg.joint_limit_buffer
        # 接近上限 → 正速度衰减到 0
        near_upper = torch.clamp((arm_pos - (self._joint_upper - buf)) / buf, 0.0, 1.0)
        arm_vel_cmd = arm_vel_cmd - torch.clamp(arm_vel_cmd, min=0.0) * near_upper
        # 接近下限 → 负速度衰减到 0
        near_lower = torch.clamp(((self._joint_lower + buf) - arm_pos) / buf, 0.0, 1.0)
        arm_vel_cmd = arm_vel_cmd + torch.clamp(-arm_vel_cmd, min=0.0) * near_lower

        vel_target = torch.zeros_like(self.robot.data.joint_pos)
        vel_target[:, self._arm_joint_ids_tensor] = arm_vel_cmd
        self.robot.set_joint_velocity_target(vel_target)

        # 夹爪: 位置控制保持固定
        self.robot.set_joint_position_target(self._gripper_target)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _quat_rotate_vec(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """用四元数(wxyz)旋转向量. quat: (N,4), vec: (3,) → (N,3)."""
        w, x, y, z = quat[:, 0:1], quat[:, 1:2], quat[:, 2:3], quat[:, 3:4]
        # vec broadcast to (N, 3)
        vx, vy, vz = vec[0], vec[1], vec[2]
        # quaternion rotation: v' = q * v * q^-1
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        rx = vx + w * tx + (y * tz - z * ty)
        ry = vy + w * ty + (z * tx - x * tz)
        rz = vz + w * tz + (x * ty - y * tx)
        return torch.cat([rx, ry, rz], dim=-1)

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------
    def _check_collision(self) -> torch.Tensor:
        """检测自碰撞, 返回 (num_envs,) bool tensor."""
        # net_forces_w_history: (num_envs, history, num_bodies, 3)
        net_forces = self._contact_sensor.data.net_forces_w_history
        # 取手臂 body, 在历史维度上取最大力
        # force norm per body per history step
        arm_forces = net_forces[:, :, self._collision_body_ids]  # (N, T, B_arm, 3)
        force_norms = torch.norm(arm_forces, dim=-1)              # (N, T, B_arm)
        max_force = torch.max(torch.max(force_norms, dim=1)[0], dim=1)[0]  # (N,)
        return max_force > self.cfg.collision_force_threshold

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]   # (num_envs, 14)
        joint_vel = self.robot.data.joint_vel[:, self._arm_joint_ids]   # (num_envs, 14)

        left_ee_pos = self.robot.data.body_pos_w[:, self._left_ee_idx]   # (num_envs, 3)
        right_ee_pos = self.robot.data.body_pos_w[:, self._right_ee_idx]  # (num_envs, 3)

        # 转换到机器人本体坐标系 (减去 scene origin)
        left_ee_local = left_ee_pos - self.scene.env_origins
        right_ee_local = right_ee_pos - self.scene.env_origins

        delta = left_ee_local - right_ee_local  # (num_envs, 3)

        # 归一化: 让所有特征尺度一致 ∈ ~[-1, 1]
        joint_pos_norm = joint_pos / self._joint_upper           # ÷ 关节限位 → [-1, 1]
        joint_vel_norm = joint_vel / self.cfg.action_scale       # ÷ 最大速度 → [-1, 1]
        ee_scale = 1.5  # 工作空间半径约 1.5m
        left_ee_norm = left_ee_local / ee_scale
        right_ee_norm = right_ee_local / ee_scale
        delta_norm = delta / ee_scale

        obs = torch.cat([joint_pos_norm, joint_vel_norm, left_ee_norm, right_ee_norm, delta_norm], dim=-1)
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        left_ee_pos = self.robot.data.body_pos_w[:, self._left_ee_idx]
        right_ee_pos = self.robot.data.body_pos_w[:, self._right_ee_idx]

        # ── 检测碰撞, 更新 absorbing flag ──
        collision = self._check_collision()
        # 一旦进入 absorbing state, 不再退出 (直到 reset)
        self._in_absorbing = self._in_absorbing | collision

        # ── 正常状态下的 reward ──
        # 1) 两末端互相靠近: 线性距离惩罚 + 指数接近奖励
        #    - 线性项提供远距离的方向性梯度
        #    - 指数项在近距离提供强正激励, 对抗 collision penalty 的过度避让
        dist_mutual = torch.norm(left_ee_pos - right_ee_pos, dim=-1)
        approach_bonus = torch.exp(-dist_mutual / self.cfg.approach_scale)

        # 2) 关节偏离默认位置的惩罚
        joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
        joint_deviation = torch.sum((joint_pos - self._default_arm_pos) ** 2, dim=-1)

        # 3) EE 高度惩罚
        base_z = self.scene.env_origins[:, 2] + self.cfg.robot_base_height
        left_below = torch.clamp(base_z - left_ee_pos[:, 2], min=0.0)
        right_below = torch.clamp(base_z - right_ee_pos[:, 2], min=0.0)
        height_penalty = left_below + right_below

        # 4) EE 朝向对齐 (权重为 0 时跳过计算)
        if self.cfg.rew_weight_facing > 0.0:
            left_ee_quat = self.robot.data.body_quat_w[:, self._left_ee_idx]
            right_ee_quat = self.robot.data.body_quat_w[:, self._right_ee_idx]
            left_z = self._quat_rotate_vec(left_ee_quat, torch.tensor([0., 0., 1.], device=self.device))
            right_z = self._quat_rotate_vec(right_ee_quat, torch.tensor([0., 0., 1.], device=self.device))
            l2r = right_ee_pos - left_ee_pos
            l2r_norm = l2r / (torch.norm(l2r, dim=-1, keepdim=True) + 1e-6)
            align_left = torch.sum(left_z * l2r_norm, dim=-1)
            align_right = torch.sum(right_z * (-l2r_norm), dim=-1)
            facing_reward = align_left + align_right
        else:
            facing_reward = torch.zeros(self.num_envs, device=self.device)

        # 5) 关节接近极限的惩罚 (只在真正接近时触发)
        margin = self._joint_upper * self.cfg.joint_limit_margin_frac
        over_margin = torch.clamp(torch.abs(joint_pos) - margin, min=0.0)
        joint_limit_penalty = torch.sum(over_margin ** 2, dim=-1)

        normal_reward = (
            -self.cfg.rew_weight_mutual * dist_mutual
            + self.cfg.rew_weight_approach * approach_bonus
            - self.cfg.rew_weight_joint_reg * joint_deviation
            - self.cfg.rew_weight_height * height_penalty
            + self.cfg.rew_weight_facing * facing_reward
            - self.cfg.rew_weight_joint_limit * joint_limit_penalty
        )

        # ── absorbing state: 碰撞时一次性给 r_min / (1 - γ), 然后 terminate ──
        # 这等价于在 absorbing state 中每步获得 r_min
        # 因为 terminated 后 bootstrap=0, Q_terminal = r_min/(1-γ)
        absorbing_reward = self.cfg.reward_absorbing_r_min / (1.0 - self.cfg.reward_gamma)
        reward = torch.where(
            self._in_absorbing,
            torch.full_like(normal_reward, absorbing_reward),
            normal_reward,
        )

        # ── extras ──
        self.extras["cost"] = collision.float()
        self.extras["log"] = {
            "ee_distance": dist_mutual,
            "approach_bonus": approach_bonus,
            "joint_deviation": joint_deviation,
            "height_penalty": height_penalty,
            "facing_reward": facing_reward,
            "joint_limit_penalty": joint_limit_penalty,
            "collision": collision.float(),
            "in_absorbing": self._in_absorbing.float(),
            "reward": reward,
        }

        return reward

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # terminated: 碰撞进入 absorbing state
        terminated = self._in_absorbing

        # truncated: episode 超时
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # 清除 absorbing state flag
        self._in_absorbing[env_ids] = False

        # 关节位置 = 默认位置 + 小随机扰动 (仅手臂关节)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        noise = self.cfg.initial_joint_noise
        arm_noise = sample_uniform(
            -noise, noise,
            (len(env_ids), len(self._arm_joint_ids)),
            joint_pos.device,
        )
        joint_pos[:, self._arm_joint_ids_tensor] += arm_noise

        joint_vel = torch.zeros_like(joint_pos)

        # 写入 sim
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
