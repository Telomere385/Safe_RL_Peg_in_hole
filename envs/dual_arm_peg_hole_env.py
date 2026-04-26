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
import re

import torch

from mushroom_rl.environments import IsaacSim
from mushroom_rl.utils.isaac_sim import ObservationType, ActionType
from mushroom_rl.rl_utils.spaces import Box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Phase 1.5 commit 2: 默认切到带 peg/hole 视觉资产的 composed USDA.
# Check A 已验证过两份 USD 在 articulation 层是物理 no-op, 所以现有
# phase 1 训练 checkpoint 在这里仍然可以 1:1 复用.
DEFAULT_USD_PATH = (
    PROJECT_ROOT / "assets" / "usd" / "dual_arm_iiwa" / "dual_arm_iiwa_with_peghole.usda"
)

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

# 相对 peg_tip / hole_entry prim 的稳定路径后缀. 找 cloned env 里对应 prim 用.
PEG_TIP_SUFFIX = "left_hande_robotiq_hande_link/Peg/peg_tip"
HOLE_ENTRY_SUFFIX = "right_hande_robotiq_hande_link/Hole/hole_entry"

# 预插入站位: hole_entry 沿 hole_axis 方向后退 preinsert_offset 距离. 默认 5cm.
DEFAULT_PREINSERT_OFFSET = 0.05


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
        rew_inthresh_vel=0.0,
        success_hold_steps=10,
        terminal_hold_bonus=0.0,
        success_pos_threshold=0.075,
        joint_limit_margin_frac=0.8,
        target_travel_fraction=0.5,
        left_target=DEFAULT_LEFT_CHEST_TARGET,
        right_target=DEFAULT_RIGHT_CHEST_TARGET,
        preinsert_offset=DEFAULT_PREINSERT_OFFSET,
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
        # 阈值内的关节速度惩罚: 杀死 "到达后立刻飘走" 的 drift-out 行为. 0 = 关闭.
        self._w_inthresh_vel = rew_inthresh_vel
        # hold-N absorbing 设计: 连续 N 步在阈内即触发 episode 终止 + terminal bonus.
        # terminal_hold_bonus=0 时整个机制关闭 (Step 2 baseline 行为).
        # bonus 量级估算: ≥ b × (1-γ^(H-N))/(1-γ) × 安全余量, b=per-step success bonus.
        # 例 (b=0.5, γ=0.99, H=150, N=10): 公式 ≈ 50, 取 150 给 3× 余量.
        self._success_hold_steps = int(success_hold_steps)
        self._terminal_hold_bonus = float(terminal_hold_bonus)
        self._absorbing_terminal_active = self._terminal_hold_bonus > 0.0
        self._success_pos_threshold = success_pos_threshold
        self._joint_limit_margin_frac = joint_limit_margin_frac
        self._target_travel_fraction = target_travel_fraction
        self._fixed_left_target = left_target
        self._fixed_right_target = right_target
        self._preinsert_offset = float(preinsert_offset)
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
        # _view 由 peg/hole prim 发现块按需赋值; 老 USD 没 peg/hole 就保持 None,
        # get_preinsert_frames() 据此返回 None.
        self._peg_tip_view = None
        self._hole_entry_view = None

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

        # hold-N 计数器 (per env). 每个 env 一个 long, success_mask=True 累加, False 清零.
        # is_absorbing 里更新, reward 里读 cache (_last_hold_done_mask).
        # setup() 在 episode reset 时把对应 env 的 counter 清零.
        self._consecutive_inthresh = torch.zeros(
            self._n_envs, dtype=torch.long, device=self._device
        )
        self._last_hold_done_mask = None

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

        # Phase 1.5 commit 2: 发现 peg/hole 的 cloned prim 路径.
        # 如果加载的是老 USD (无 peg/hole), 两个列表会是空的, get_preinsert_frames()
        # 会返回 None. 这时候旧流程 (reaching reward/obs) 不受影响.
        self._peg_tip_paths = self._find_cloned_prim_paths(PEG_TIP_SUFFIX)
        self._hole_entry_paths = self._find_cloned_prim_paths(HOLE_ENTRY_SUFFIX)
        if self._peg_tip_paths and self._hole_entry_paths:
            from isaacsim.core.prims import XFormPrim

            self._peg_tip_view = XFormPrim(
                self._peg_tip_paths, name="peg_tip_frame_view", reset_xform_properties=False, usd=False
            )
            self._hole_entry_view = XFormPrim(
                self._hole_entry_paths, name="hole_entry_frame_view", reset_xform_properties=False, usd=False
            )
            self._peg_tip_view.initialize()
            self._hole_entry_view.initialize()
            print(
                f"[PREINSERT] peg/hole 视觉资产已加载, "
                f"cloned 实例数: peg_tip={len(self._peg_tip_paths)}, "
                f"hole_entry={len(self._hole_entry_paths)}"
            )
        else:
            print(
                "[PREINSERT] 注意: stage 里没找到 peg_tip / hole_entry prim. "
                "get_preinsert_frames() 将返回 None. 若不是故意加载无-peghole USD, "
                "请确认 --usd 或 DEFAULT_USD_PATH 指向 dual_arm_iiwa_with_peghole.usda"
            )

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

        # task errors 算给 reward 用 (dwell bonus + hold-N counter)
        self._last_task_errors = self._compute_task_errors(obs)
        _, _, success_mask = self._last_task_errors

        # 更新 per-env 的连续 in-threshold 计数: in 阈值则 +1, 否则清零
        self._consecutive_inthresh = torch.where(
            success_mask,
            self._consecutive_inthresh + 1,
            torch.zeros_like(self._consecutive_inthresh),
        )

        # hold-N absorbing 仅当 terminal_hold_bonus > 0 时启用. baseline (=0) 行为完全保留.
        if self._absorbing_terminal_active:
            hold_done = self._consecutive_inthresh >= self._success_hold_steps
        else:
            hold_done = torch.zeros_like(collision)
        self._last_hold_done_mask = hold_done

        return collision | hold_done

    def reward(self, obs, action, next_obs, absorbing):
        joint_pos = self.observation_helper.get_from_obs(next_obs, "joint_pos")
        joint_vel = self.observation_helper.get_from_obs(next_obs, "joint_vel")
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

        # 阈值内的关节速度惩罚: 只在已经 reach 时罚速度, 强迫 agent 学会停下来.
        # 不在阈值时为 0, 不影响初始 reaching 阶段的探索.
        joint_vel_sq = (joint_vel ** 2).sum(dim=-1)
        inthresh_vel_pen = success * joint_vel_sq

        normal = (
            -self._w_pos * (left_pos_err + right_pos_err)
            - self._w_joint_limit * joint_limit_norm
            - self._w_action * action_sq
            - self._w_inthresh_vel * inthresh_vel_pen
            + self._w_success * success
        )

        # 三路 reward 选择:
        #   collision  → r_min/(1-γ) (硬 absorbing, 失败终结)
        #   hold-N done → normal + terminal_hold_bonus (软 absorbing, 成功终结)
        #   其他       → normal
        # 嵌套 where 让 collision 优先 (同时为 True 时按 collision 处理).
        absorbing_r = self._r_min / (1.0 - self.info.gamma)
        r = torch.where(
            self._last_collision_mask,
            torch.full_like(normal, absorbing_r),
            torch.where(
                self._last_hold_done_mask,
                normal + self._terminal_hold_bonus,
                normal,
            ),
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

        # 重置正在 reset 的 env 的 hold counter, 避免上一 episode 累计的步数泄漏到下一个
        idx_tensor = torch.as_tensor(env_indices, device=self._device, dtype=torch.long)
        self._consecutive_inthresh[idx_tensor] = 0

        # set_joint_positions 只写 DOF buffer; 不 step 的话 BODY_POS view 还是
        # reset 前的值, reset_all 读到的 EE 位姿是 stale 的.
        self._world.step(render=False)

    def get_obs_scale(self):
        """40 维 fixed-divisor 归一化向量, 与 observation_space 一一对应.

        对应顺序 (与 _create_observation 输出一致):
            joint_pos[14], joint_vel[14], left_ee[3], right_ee[3],
            left_rel_goal[3], right_rel_goal[3]

        缩放选择 (大致让每维标准差落到 ~1):
            joint_pos: 半关节范围 (~1.5-3 rad/joint)
            joint_vel: 2.0 rad/s (iiwa velocity-mode 典型上限)
            ee / rel: 1.0 m (机器人 reach + 目标点距离量级)

        网络在 forward 第一行除以这个 scale, env 内部 obs 仍是物理单位 (米/弧度),
        所以 _compute_task_errors 等内部读 obs 的代码不受影响.
        """
        device = self._joint_lower.device
        dtype = torch.float32  # 网络一律用 float32, 这里直接对齐
        joint_pos_scale = 0.5 * (self._joint_upper - self._joint_lower).to(dtype)
        joint_vel_scale = torch.full_like(joint_pos_scale, 2.0)
        ee_scale = torch.full((6,), 1.0, device=device, dtype=dtype)
        rel_scale = torch.full((6,), 1.0, device=device, dtype=dtype)
        return torch.cat([joint_pos_scale, joint_vel_scale, ee_scale, rel_scale])

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

    # ------------------------------------------------------------------
    # Phase 1.5 commit 2: preinsert pose helpers
    # ------------------------------------------------------------------
    # 这一组只暴露"读"接口, 不参与 reward / obs / is_absorbing. 训练语义仍然
    # 是 phase 1 的 reaching. 任务切换留给 commit 3+.

    @staticmethod
    def _quat_apply(q_wxyz, v):
        """用单位四元数 q (wxyz 约定) 旋转向量 v. 支持广播.

        Args:
            q_wxyz: [..., 4]
            v:      [..., 3]
        Returns:
            [..., 3] 旋转后的向量.
        """
        w = q_wxyz[..., 0]
        x = q_wxyz[..., 1]
        y = q_wxyz[..., 2]
        z = q_wxyz[..., 3]
        vx = v[..., 0]
        vy = v[..., 1]
        vz = v[..., 2]
        # 标准公式: v' = v + 2*w*(q_xyz × v) + 2*(q_xyz × (q_xyz × v))
        tx = 2 * (y * vz - z * vy)
        ty = 2 * (z * vx - x * vz)
        tz = 2 * (x * vy - y * vx)
        rx = vx + w * tx + (y * tz - z * ty)
        ry = vy + w * ty + (z * tx - x * tz)
        rz = vz + w * tz + (x * ty - y * tx)
        return torch.stack([rx, ry, rz], dim=-1)

    @staticmethod
    def _quat_mul(q1_wxyz, q2_wxyz):
        """四元数乘法, 支持广播, 输入输出都用 (w, x, y, z)."""
        w1, x1, y1, z1 = q1_wxyz.unbind(dim=-1)
        w2, x2, y2, z2 = q2_wxyz.unbind(dim=-1)
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dim=-1)

    def _find_cloned_prim_paths(self, relative_suffix):
        """遍历 stage 找所有以 relative_suffix 结尾的 prim path, 按字符串排序.

        这样不依赖 cloner 的精确命名模式; 只要 peg_tip/hole_entry 的相对路径在
        USDA 里是稳定的, 就能发现所有 cloned env 里的副本.
        """
        try:
            import omni.usd
        except ImportError:
            return []
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return []
        suffix = "/" + relative_suffix.lstrip("/")
        paths = []
        for prim in stage.Traverse():
            p = str(prim.GetPath())
            if p.endswith(suffix):
                paths.append(p)
        def env_sort_key(path):
            m = re.search(r"/env_(\d+)(?:/|$)", path)
            return (int(m.group(1)) if m else -1, path)
        paths.sort(key=env_sort_key)
        return paths

    def _query_world_pose_quat(self, prim_view):
        """用 IsaacSim live xform view 查 prim 的世界位姿.

        Returns:
            pos [N, 3], quat [N, 4] (wxyz), 都在 self._device 上, dtype 与 joint limits 一致.
        """
        dtype = self._joint_lower.dtype
        pos, quat = prim_view.get_world_poses(usd=False)
        pos = torch.as_tensor(pos, device=self._device, dtype=dtype)
        quat = torch.as_tensor(quat, device=self._device, dtype=dtype)
        return pos, quat

    def get_preinsert_frames(self):
        """返回 peg_tip / hole_entry / preinsert_target 的世界帧位姿.

        这是 phase 1.5 **纯读** helper — 不改 reward/obs/is_absorbing, 不影响训练
        语义. 留给 visualize_targets (commit 2) 和后续 commit 3 的 task error
        计算使用.

        Returns None 当 peg/hole 不在 stage 里 (用户加载了老 USD).
        否则返回 dict of batched tensors (都是世界坐标, batch=num_envs):
            peg_tip_pos          [N, 3]
            peg_tip_quat         [N, 4]  wxyz
            peg_axis             [N, 3]  unit, 就是 peg 本地 +Z 在世界里的方向
            hole_entry_pos       [N, 3]
            hole_entry_quat      [N, 4]
            hole_axis            [N, 3]  unit, hole 开口朝向 (hole 本地 +Z 在世界)
            preinsert_target_pos [N, 3]  hole_entry_pos + preinsert_offset * hole_axis
            preinsert_target_quat[N, 4]  = hole_entry_quat (同一帧)

        几何含义: peg/hole 都是圆柱, 它们的 local +Z 即对称轴. peg 从 preinsert_target
        沿 -hole_axis 方向往 hole_entry 插入 -> 理想装配时
        dot(peg_axis, hole_axis) ≈ -1 (轴反平行, 面对面).
        """
        if self._peg_tip_view is None or self._hole_entry_view is None:
            return None
        if (len(self._peg_tip_paths) != self._n_envs
                or len(self._hole_entry_paths) != self._n_envs):
            print(
                "[PREINSERT] prim 数量与 num_envs 不一致, "
                f"peg_tip={len(self._peg_tip_paths)} "
                f"hole_entry={len(self._hole_entry_paths)} "
                f"num_envs={self._n_envs}"
            )
            return None

        peg_tip_pos, peg_tip_quat = self._query_world_pose_quat(self._peg_tip_view)
        hole_entry_pos, hole_entry_quat = self._query_world_pose_quat(self._hole_entry_view)

        unit_z = torch.zeros_like(peg_tip_pos)
        unit_z[..., 2] = 1.0
        flip_x_180 = torch.zeros_like(peg_tip_quat)
        flip_x_180[..., 1] = 1.0
        peg_axis_quat = self._quat_mul(peg_tip_quat, flip_x_180)
        peg_axis = self._quat_apply(peg_axis_quat, unit_z)
        hole_axis_quat = hole_entry_quat
        hole_axis = self._quat_apply(hole_entry_quat, unit_z)

        preinsert_target_pos = hole_entry_pos + self._preinsert_offset * hole_axis
        preinsert_target_quat = hole_entry_quat.clone()

        return {
            "peg_tip_pos": peg_tip_pos,
            "peg_tip_quat": peg_tip_quat,
            "peg_axis_quat": peg_axis_quat,
            "peg_axis": peg_axis,
            "hole_entry_pos": hole_entry_pos,
            "hole_entry_quat": hole_entry_quat,
            "hole_axis_quat": hole_axis_quat,
            "hole_axis": hole_axis,
            "preinsert_target_pos": preinsert_target_pos,
            "preinsert_target_quat": preinsert_target_quat,
        }
