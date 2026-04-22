"""双臂末端"位置到达"任务 (phase 1) — mushroom-rl IsaacSim 向量化环境.

phase 1 目标: 左右末端各自到达固定在 world frame 的绝对位置目标. 不涉及任何
姿态控制, 不涉及 peg/hole 几何. 长期目标 (peg-in-hole 装配) 在后续 phase 引入.

每臂控全部 7 DoF (A1-A7), 14 维 joint velocity 动作.

观测 (40 维):
    joint_pos          (14) 左右臂 A1-A7 关节角
    joint_vel          (14) 左右臂 A1-A7 关节角速度
    left_ee_pos        (3)  左末端位置 (env-local)
    right_ee_pos       (3)  右末端位置 (env-local)
    left_rel_goal_pos  (3)  T_L - left_ee
    right_rel_goal_pos (3)  T_R - right_ee

动作 (14):
    action ∈ [-1,1]^14 → joint velocity 指令 (rad/s), 系数 action_scale.

Reward:
    - w_pos * (||left_ee - T_L|| + ||right_ee - T_R||)
    - w_joint_limit * joint_limit_norm
    - w_action * sum(a_i^2)                 # raw action, 解耦 action_scale
    + w_success * 1[||·||_L < pos_th ∧ ||·||_R < pos_th]

Target 生成 (在 __init__ 末尾 eager-init 一次冻结, 固定在 world frame):
    - 默认使用对称的前下方固定目标点:
      T_L = [-0.80, -0.40, 0.62]
      T_R = [-0.80,  0.40, 0.62]
    - 若未显式给 fixed target, 则回退到 fraction 模式:
      left_default, right_default = env-平均的 default EE 位置
      center = (left_default + right_default) / 2
      T_L = left_default  + f · (center - left_default)
      T_R = right_default + f · (center - right_default)
      f = target_travel_fraction ∈ [0, 1]; 0 = 原地, 1 = 中点.

终止:
    - 碰撞 (左右臂接触力 > 阈值): 吸收 r = r_min / (1 - γ)
    - 成功 不终止: w_success 变成每步 dwell bonus, optimal policy 会学成
      "reach → hold", 而非 "蹭进阈值立刻终止跑路" 的 boundary-hugging 策略.

Eager-init 后在 env 0 旁边 spawn 红绿可视化球作为目标位置 marker.
"""

from pathlib import Path

import torch

from mushroom_rl.environments import IsaacSim
from mushroom_rl.utils.isaac_sim import ObservationType, ActionType
from mushroom_rl.rl_utils.spaces import Box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USD_PATH = PROJECT_ROOT / "assets" / "usd" / "dual_arm_iiwa" / "dual_arm_iiwa.usd"

CONTROLLED_IDX = (1, 2, 3, 4, 5, 6, 7)  # A1-A7, 7 DoF/臂
LEFT_ARM_JOINTS = [f"left_arm_A{i}" for i in CONTROLLED_IDX]
RIGHT_ARM_JOINTS = [f"right_arm_A{i}" for i in CONTROLLED_IDX]
ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS  # 14

LEFT_ARM_LINKS = [f"/left_arm_link_{i}" for i in range(1, 8)]
RIGHT_ARM_LINKS = [f"/right_arm_link_{i}" for i in range(1, 8)]

LEFT_EE_PATH = "/left_hande_robotiq_hande_link"
RIGHT_EE_PATH = "/right_hande_robotiq_hande_link"

LEFT_ARM_GROUP = LEFT_ARM_LINKS + [LEFT_EE_PATH]
RIGHT_ARM_GROUP = RIGHT_ARM_LINKS + [RIGHT_EE_PATH]

DEFAULT_LEFT_CHEST_TARGET = (-0.80, -0.40, 0.62)
DEFAULT_RIGHT_CHEST_TARGET = (-0.80, 0.40, 0.62)


class DualArmPegHoleEnv(IsaacSim):
    def __init__(
        self,
        num_envs=2,
        horizon=150,
        gamma=0.99,
        headless=True,
        device="cuda:0",
        action_scale=0.4,
        initial_joint_noise=0.1,
        collision_force_threshold=10.0,
        reward_absorbing_r_min=-2.0,
        reward_scale=1.0,
        rew_pos=1.0,
        rew_success=2.0,
        rew_joint_limit=0.02,
        rew_action=0.005,
        success_pos_threshold=0.075,
        joint_limit_margin_frac=0.8,
        target_travel_fraction=0.5,
        left_target=DEFAULT_LEFT_CHEST_TARGET,
        right_target=DEFAULT_RIGHT_CHEST_TARGET,
        usd_path=None,
    ):
        self._action_scale = action_scale
        self._initial_joint_noise = initial_joint_noise
        self._collision_threshold = collision_force_threshold
        self._r_min = reward_absorbing_r_min
        self._reward_scale = reward_scale
        self._w_pos = rew_pos
        self._w_success = rew_success
        self._w_joint_limit = rew_joint_limit
        self._w_action = rew_action
        self._success_pos_threshold = success_pos_threshold
        self._joint_limit_margin_frac = joint_limit_margin_frac
        self._target_travel_fraction = target_travel_fraction
        self._fixed_left_target = left_target
        self._fixed_right_target = right_target
        self._usd_path = Path(usd_path) if usd_path is not None else DEFAULT_USD_PATH
        if not self._usd_path.is_file():
            raise FileNotFoundError(
                "找不到机器人 USD 资产文件: "
                f"{self._usd_path}\n"
                "请确认仓库内存在 assets/usd/dual_arm_iiwa/dual_arm_iiwa.usd，"
                "或在构造 DualArmPegHoleEnv 时显式传入 usd_path。"
            )
        self._left_target = None
        self._right_target = None
        # is_absorbing 与 reward 在同一 next_obs 上背靠背调用, 缓存避免重复计算
        self._last_collision_mask = None
        self._last_task_errors = None
        # _preprocess_action → reward 链路里缓存 pre-scale 的 raw action,
        # 用于 L2 惩罚. 这样 w_action 和 action_scale 解耦, 改 action_scale 时
        # 惩罚强度语义不变.
        self._last_raw_action = None

        observation_spec = [
            ("joint_pos", "", ObservationType.JOINT_POS, ARM_JOINTS),
            ("joint_vel", "", ObservationType.JOINT_VEL, ARM_JOINTS),
            ("left_ee_pos", LEFT_EE_PATH, ObservationType.BODY_POS, None),
            ("right_ee_pos", RIGHT_EE_PATH, ObservationType.BODY_POS, None),
        ]
        collision_groups = [("arm_L", LEFT_ARM_GROUP), ("arm_R", RIGHT_ARM_GROUP)]

        super().__init__(
            usd_path=str(self._usd_path),
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
            # 多 env 默认相机站在群中心, 画面混乱; 拉远到群外上方俯瞰整网格.
            camera_position=(20, -15, 10),
            camera_target=(5, 0, 0.5),
        )

        # 关节位限 + 默认位
        limits = self._task.get_joint_pos_limits()
        self._joint_lower, self._joint_upper = limits[0], limits[1]
        self._default_joint_pos = self._task.robots.get_joints_default_state().positions[0][
            self._task._controlled_joints
        ].clone()

        # USD iiwa 默认 position-drive (kp ~ 5e5); velocity 控制必须把 kp 置 0,
        # 否则 reset 里 set_joint_positions 会一并写 pos_target, 高 kp 把关节钉
        # 回 reset 点, velocity 指令完全推不动.
        robots = self._task.robots
        cj = self._task._controlled_joints
        zero_kps = torch.zeros(self._n_envs, len(ARM_JOINTS), device=device)
        _, cur_kds = robots.get_gains(joint_indices=cj, clone=True)
        robots.set_gains(kps=zero_kps, kds=cur_kds, joint_indices=cj)
        self._cj = cj
        self._robots = robots

        # 累计碰撞终止次数 (train_sac 每 epoch 读取 + 上一 epoch 值做差分 → 本 epoch 新增).
        self._absorb_count = 0

        # Eager-init targets 在所有 step_all/teleport_away 之前: super().__init__()
        # 已经跑过 self._world.reset(), 16 个 env 的 robot 都在 USD default pose
        # (EE 在 world z≈0.73). 读一次 fresh obs 冻结绝对目标, 之后完全免疫
        # teleport_away 污染 (n_episodes < num_envs 时 inactive env 会被抬到 z=50).
        # 读 obs 前先 step 一次: BODY_POS view 在 _world.reset() 后未必同步物理状态.
        self._world.step(render=False)
        init_obs = self.observation_helper.build_obs(self._task.get_observations(clone=True))
        self._init_targets(
            self.observation_helper.get_from_obs(init_obs, "left_ee_pos"),
            self.observation_helper.get_from_obs(init_obs, "right_ee_pos"),
        )
        self._spawn_target_markers()

    def _modify_mdp_info(self, mdp_info):
        # action: [-1,1]^14, SAC tanh policy 直接映射
        device = mdp_info.action_space.low.device
        dtype = mdp_info.action_space.low.dtype
        one = torch.ones(len(ARM_JOINTS), device=device, dtype=dtype)
        mdp_info.action_space = Box(-one, one, data_type=dtype)

        # observation: 在 raw 34 dim 之外追加 6 dim 相对目标位置
        obs_low, obs_high = self.observation_helper.obs_limits
        goal_low = torch.full((6,), -5.0, device=obs_low.device, dtype=obs_low.dtype)
        goal_high = torch.full((6,), 5.0, device=obs_high.device, dtype=obs_high.dtype)
        new_obs_low = torch.cat([obs_low, goal_low], dim=0)
        new_obs_high = torch.cat([obs_high, goal_high], dim=0)
        mdp_info.observation_space = Box(new_obs_low, new_obs_high, data_type=new_obs_high.dtype)
        return mdp_info

    def _preprocess_action(self, action):
        action = torch.as_tensor(action, device=self._device, dtype=self._joint_lower.dtype)
        clipped = torch.clamp(action, -1.0, 1.0)
        self._last_raw_action = clipped
        return clipped * self._action_scale

    def _create_observation(self, obs):
        left_ee = self.observation_helper.get_from_obs(obs, "left_ee_pos")
        right_ee = self.observation_helper.get_from_obs(obs, "right_ee_pos")
        left_rel = self._left_target.unsqueeze(0) - left_ee
        right_rel = self._right_target.unsqueeze(0) - right_ee
        return torch.cat([obs, left_rel, right_rel], dim=-1)

    def _init_targets(self, left_ee, right_ee):
        """优先使用固定胸前目标; 否则从 env-平均 default EE 位置 + fraction 冻结目标."""
        if (self._fixed_left_target is None) != (self._fixed_right_target is None):
            raise ValueError("left_target 和 right_target 必须同时提供或同时为 None")

        left_default = left_ee.mean(dim=0).detach().clone()
        right_default = right_ee.mean(dim=0).detach().clone()

        if self._fixed_left_target is not None:
            self._left_target = torch.as_tensor(
                self._fixed_left_target, device=left_default.device, dtype=left_default.dtype
            ).clone()
            self._right_target = torch.as_tensor(
                self._fixed_right_target, device=right_default.device, dtype=right_default.dtype
            ).clone()
            print("[TARGETS] fixed symmetric front-low targets\n"
                  f"  T_L = {self._left_target.tolist()}  (default {left_default.tolist()})\n"
                  f"  T_R = {self._right_target.tolist()}  (default {right_default.tolist()})")
            return

        center = 0.5 * (left_default + right_default)
        f = self._target_travel_fraction
        self._left_target = left_default + f * (center - left_default)
        self._right_target = right_default + f * (center - right_default)
        print(f"[TARGETS] fraction={f}\n"
              f"  T_L = {self._left_target.tolist()}  (default {left_default.tolist()})\n"
              f"  T_R = {self._right_target.tolist()}  (default {right_default.tolist()})")

    def _spawn_target_markers(self):
        """env 0 旁 spawn 球形 USD marker. BODY_POS obs 是 env-local, 加回 env 0 world offset."""
        try:
            import omni.usd
            from pxr import UsdGeom, Sdf, Gf, Vt
        except ImportError:
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        try:
            world_pos_tensor, _ = self._task.robots.get_world_poses()
            env0 = world_pos_tensor[0].detach().cpu()
            env0_offset = (float(env0[0]), float(env0[1]), float(env0[2]))
        except Exception as e:
            print(f"[VIZ] env 0 offset query 失败 ({e}), marker 放在 world 原点")
            env0_offset = (0.0, 0.0, 0.0)
        print(f"[VIZ] env 0 world offset = {env0_offset}")

        def add_marker(prefix, pos, color):
            world_pos = (float(pos[0]) + env0_offset[0],
                         float(pos[1]) + env0_offset[1],
                         float(pos[2]) + env0_offset[2])
            sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(f"{prefix}_sphere"))
            sphere.GetRadiusAttr().Set(0.03)
            sphere.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
            xf = UsdGeom.Xformable(sphere.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*world_pos))

        add_marker("/World/target_L", self._left_target, (1.0, 0.15, 0.15))
        add_marker("/World/target_R", self._right_target, (0.15, 1.0, 0.15))

    def _compute_task_errors(self, obs):
        """Returns (left_pos_err, right_pos_err, success_mask). Targets 已冻结."""
        left_ee = self.observation_helper.get_from_obs(obs, "left_ee_pos")
        right_ee = self.observation_helper.get_from_obs(obs, "right_ee_pos")
        left_pos_err = torch.norm(left_ee - self._left_target, dim=-1)
        right_pos_err = torch.norm(right_ee - self._right_target, dim=-1)
        success_mask = ((left_pos_err < self._success_pos_threshold)
                        & (right_pos_err < self._success_pos_threshold))
        return left_pos_err, right_pos_err, success_mask

    def is_absorbing(self, obs):
        collision = self._check_collision("arm_L", "arm_R", self._collision_threshold,
                                          dt=self._timestep)
        self._absorb_count += int(collision.sum().item())
        self._last_collision_mask = collision

        # success 不再触发 absorbing; 但仍需算 task errors 给 reward 用 (dwell bonus).
        self._last_task_errors = self._compute_task_errors(obs)
        return collision

    def reward(self, obs, action, next_obs, absorbing):
        joint_pos = self.observation_helper.get_from_obs(next_obs, "joint_pos")
        # is_absorbing 已在同一 next_obs 上算过, 直接复用
        left_pos_err, right_pos_err, success_mask = self._last_task_errors
        success = success_mask.to(left_pos_err.dtype)

        # 关节极限软惩罚: 超 margin 才计, 单关节撞极限贡献 ~1, 正常区域 0
        joint_range = self._joint_upper - self._joint_lower
        joint_center = 0.5 * (self._joint_upper + self._joint_lower)
        excess = torch.clamp(
            (torch.abs((joint_pos - joint_center) / (0.5 * joint_range))
             - self._joint_limit_margin_frac)
            / (1.0 - self._joint_limit_margin_frac),
            min=0.0, max=1.0,
        )
        joint_limit_norm = torch.sum(excess ** 2, dim=-1)

        # action L2: 用 cache 的 raw action (pre-scale), 和 action_scale 解耦
        action_sq = (self._last_raw_action ** 2).sum(dim=-1)

        normal = (
            -self._w_pos * (left_pos_err + right_pos_err)
            - self._w_joint_limit * joint_limit_norm
            - self._w_action * action_sq
            + self._w_success * success
        )

        # collision 是唯一 absorbing 源, 其 reward 盖成 r_min/(1-γ); success 不终止
        absorbing_r = self._r_min / (1.0 - self.info.gamma)
        r = torch.where(
            self._last_collision_mask, torch.full_like(normal, absorbing_r), normal
        )
        return self._reward_scale * r

    def setup(self, env_indices, obs):
        n = len(env_indices)
        noise = self._initial_joint_noise * (
            2.0 * torch.rand(n, len(ARM_JOINTS), device=self._device) - 1.0
        )
        joint_pos = self._default_joint_pos.unsqueeze(0) + noise
        self._write_data("joint_pos", joint_pos, env_indices)
        self._write_data("joint_vel", torch.zeros_like(joint_pos), env_indices)

        # set_joint_positions 只写 DOF buffer; 不 step 的话 BODY_POS view 还是
        # reset 前的值, reset_all 读到的 EE 位姿是 stale 的.
        self._world.step(render=False)

    def _simulation_pre_step(self):
        """每 intermediate step 物理前施加重力补偿前馈.

        kp=0 velocity drive 只有阻尼项 (kd·(v_target - v_current)), 对恒定重力
        干扰是结构性欠阻尼 — zero-action 3s 手臂坠落 ~1m. 真机 iiwa (KUKA Sunrise
        velocity mode) 底层跑重力补偿; sim 里我们用 get_generalized_gravity_forces()
        拿到 G(q) 作为前馈 effort. agent 的 velocity action 含义不变, 只是不再需要
        从零学 G(q) 这个 7-DoF 非线性映射.
        """
        tau_g = self._robots.get_generalized_gravity_forces(clone=False)
        self._robots.set_joint_efforts(tau_g[:, self._cj], joint_indices=self._cj)
