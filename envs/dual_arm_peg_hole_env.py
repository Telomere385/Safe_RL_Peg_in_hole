"""双臂 peg-in-hole preinsert 任务 — mushroom-rl IsaacSim 向量化环境.

阶段 (stage flag 化, 同一个 env / 同一个 obs / 同一条 reward 骨架):
    M1' = pos-only           rew_axis=0,    success_axis_threshold=inf
    M2a = pos + 粗轴对齐      rew_axis=1.0,  success_axis_threshold=0.5
    M2b = pos + 紧轴对齐      rew_axis=1.0,  success_axis_threshold=0.2
    M3+ = 在 obs 里再加
          radial/axial 维度 (留给后续 commit, 此版本不放).

切换 stage 不改 env 结构, 只改 reward 权重和 success_axis_threshold. 这样
M1' → M2a → M2b 之间可以 warm-start (--load_agent), 网络输入维度恒为 32.

每臂控全部 7 DoF (A1-A7), 14 维 joint velocity 动作.

观测 (32 维):
    joint_pos          (14) 左右臂 A1-A7 关节角
    joint_vel          (14) 左右臂 A1-A7 关节角速度
    pos_vec            (3)  peg_tip - preinsert_target
    axis_dot           (1)  dot(peg_axis, hole_axis) ∈ [-1, +1]
                            -1 = 完美轴反平行 (理想对齐).
                            放标量 (而非完整 peg_axis/hole_axis 6 维): 一维已经
                            把"对齐到什么程度"的梯度信号给出来; 完整向量与 EE
                            quat 强冗余, 徒增维度.

动作 (14):
    action ∈ [-1,1]^14 → joint velocity 指令 (rad/s), 系数 action_scale.

Reward (统一骨架):
    - w_pos     * pos_err                             # ||peg_tip - preinsert_target||
    - w_axis    * axis_err                            # 1 + dot(peg_axis, hole_axis), 0 = ideal
    - w_joint_limit * joint_limit_norm
    - w_action  * sum(a_i^2)                          # raw action, 解耦 action_scale
    + w_success * 1[success]                          # per-step dwell bonus, 不终止
    success = (pos_err < pos_th) ∧ (axis_err < axis_th)
              # axis_th=inf 时退化为 pos-only — 这就是 M1' 的语义

终止:
    - 自碰撞 (双信号 OR, 任一触发即吸收 r = r_min / (1 - γ)):
        * PhysX 接触力 > collision_force_threshold
        * sphere-proxy clearance < clearance_hard (PhysX 在 1cm-5cm 边缘失明
          的几何兜底, 默认 clearance_hard=0.0 即球壳一接触就算碰撞)
    - hold-N (success 连续 N 步): 软 absorbing + terminal_hold_bonus (可选, =0 关闭)
    - success 本身不终止 (沿用 phase 1 结论, 避免 Q-target 边界断崖, 见
      feedback_bimanual_reward_shaping.md Rule 1)

PEG/HOLE 几何 — 解析式 frame (不依赖 XFormPrim):
    peg/hole 是视觉-only 的 USD over, 挂在左/右 EE 下, 没有 RigidBodyAPI /
    MassAPI / CollisionAPI (build_peghole_usd.py 验证). 它们的 pose 完全由
    EE link 的世界位姿 + 一个常量本地偏移决定:
        peg_tip_world  = LeftEE_pos  + R(LeftEE_quat)  · PEG_TIP_OFFSET_IN_LEFTEE
        peg_axis_world =                R(LeftEE_quat)  · PEG_AXIS_IN_LEFTEE
        hole_entry / hole_axis 同理 (RightEE).
    所以训练 headless 下完全不需要 XFormPrim/Fabric flush — 只要 BODY_POS +
    BODY_ROT 是 fresh 的, 帧就是对的. visualize_* 也走同一条解析路径,
    XFormPrim 不再使用.
"""

import math
from pathlib import Path

import torch

from mushroom_rl.environments import IsaacSim
from mushroom_rl.utils.isaac_sim import ObservationType, ActionType
from mushroom_rl.rl_utils.spaces import Box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USD_PATH = (
    PROJECT_ROOT / "assets" / "usd" / "dual_arm_iiwa" / "dual_arm_iiwa_with_peghole.usda"
)

CONTROLLED_IDX = (1, 2, 3, 4, 5, 6, 7)  # A1-A7, 7 DoF/臂
LEFT_ARM_JOINTS = [f"left_arm_A{i}" for i in CONTROLLED_IDX]
RIGHT_ARM_JOINTS = [f"right_arm_A{i}" for i in CONTROLLED_IDX]
ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS  # 14

# Home pose — 取代 USD 自带的 zero 默认位姿. 双臂在胸前略弯肘的 ready 姿态,
# 让 reset center 落在任务相关区域, 减少 M1' 早期探索的无效扇区.
# 镜像约定: 奇数 joint (绕"竖向"轴) 取反, 偶数 joint (hinge) 同号. 镜像若不对
# (右臂方向反了 / 甩出去), 调换 A1 正负号通常即可.
HOME_JOINT_POS = (
    # left arm  A1     A2      A3      A4     A5   A6   A7
    -2.568,  -0.250, -0.078,  0.814,  0.0, 0.0,  0.010,
    # right arm A1     A2      A3      A4     A5   A6   A7
    +2.568,  -0.250, +0.078,  0.814,  0.0, 0.0, -0.010,
)
assert len(HOME_JOINT_POS) == 14, "HOME_JOINT_POS 必须 14 维 (左 7 + 右 7)"

LEFT_ARM_LINKS = [f"/left_arm_link_{i}" for i in range(1, 8)]
RIGHT_ARM_LINKS = [f"/right_arm_link_{i}" for i in range(1, 8)]

LEFT_EE_PATH = "/left_hande_robotiq_hande_link"
RIGHT_EE_PATH = "/right_hande_robotiq_hande_link"

LEFT_ARM_GROUP = LEFT_ARM_LINKS + [LEFT_EE_PATH]
RIGHT_ARM_GROUP = RIGHT_ARM_LINKS + [RIGHT_EE_PATH]

# ---- Sphere-proxy clearance 几何常量 -------------------------------------
# 把每条机械臂离散成一组球, 球心从 articulation BODY_POS 拿. 两侧球心两两算
# clearance = ||c_L - c_R|| - r_L - r_R, 取 min. 这是 PhysX 接触力检测之外的
# 几何 proxy, 用于阻止双臂 cross-over (PhysX 力检测在 1cm-5cm 边缘失明).
#
# 每侧 19 球 = 8 关节球 (link_0..link_7) + 7 中点球 (相邻 link 直线中点)
#            + 4 EE 球 (coupler, hande_link, left_finger, right_finger).
# 中点用 0.5*(BODY_POS[i] + BODY_POS[i+1]); iiwa link 大体直筒, 直线中点
# 与 mesh 几何中心差距小, 不需要查 inertial / visual mesh, 完全只读 BODY_POS.
LEFT_ARM_JOINT_BODY_NAMES = [f"left_arm_link_{i}" for i in range(0, 8)]   # 8 球
RIGHT_ARM_JOINT_BODY_NAMES = [f"right_arm_link_{i}" for i in range(0, 8)]
LEFT_EE_PROXY_BODY_NAMES = [
    "left_hande_robotiq_hande_coupler",
    "left_hande_robotiq_hande_link",
    "left_hande_robotiq_hande_left_finger",
    "left_hande_robotiq_hande_right_finger",
]
RIGHT_EE_PROXY_BODY_NAMES = [
    "right_hande_robotiq_hande_coupler",
    "right_hande_robotiq_hande_link",
    "right_hande_robotiq_hande_left_finger",
    "right_hande_robotiq_hande_right_finger",
]
# 半径起步值: arm 6cm (link 直径 ~6-10cm 给 margin), ee 3cm (hande/finger 直径 ~4-6cm).
ARM_PROXY_RADIUS = 0.06
EE_PROXY_RADIUS = 0.03

# Peg/Hole 几何与挂载常量 — 必须与 build_peghole_usd.py 保持一致.
# 推导 (USD 应用顺序: orient 后 translate, 见 build script:
#   xformOpOrder = ["xformOp:translate", "xformOp:orient"]):
#   Peg/Hole 在 EE 帧里 = T(PART_X, 0, PART_Z) ∘ R_x(+90°)
#   peg_tip 在 Peg 局部为 (0, 0, +PEG_HEIGHT/2)
#   hole_entry 在 Hole 局部为 (0, 0, +HOLE_HEIGHT/2)
# 所以 peg_tip 在 LeftEE 帧:  R_x(+90°)·(0,0,h/2) + (PART_X, 0, PART_Z)
#                          = (PART_X, -h/2, PART_Z)
_PART_X = -0.0055
_PART_Z = 0.125
_PEG_HEIGHT = 0.035
_HOLE_HEIGHT = 0.030
_PEG_TIP_LOCAL_Z = 0.5 * _PEG_HEIGHT       # 0.0175
_HOLE_ENTRY_LOCAL_Z = 0.5 * _HOLE_HEIGHT   # 0.015

PEG_TIP_OFFSET_IN_LEFTEE = (_PART_X, -_PEG_TIP_LOCAL_Z, _PART_Z)
HOLE_ENTRY_OFFSET_IN_RIGHTEE = (_PART_X, -_HOLE_ENTRY_LOCAL_Z, _PART_Z)

# peg_axis: 沿用旧 XFormPrim 实现的 sign convention (peg_tip_quat * R_x(180°) 后
# apply 到 +Z), 等价于 R(LeftEE_quat) · (0, +1, 0). 这样 axis_err = 1+dot 的
# 0 = ideal-alignment 语义在新旧实现间一致, 避免 phase 1.5 visualize 验收过的
# preinsert_target / dot 数值因为符号约定突变.
PEG_AXIS_IN_LEFTEE = (0.0, +1.0, 0.0)
# hole_axis: 直接 R(RightEE_quat) · (0,0,1) 经 R_x(+90°) = (0, -1, 0).
HOLE_AXIS_IN_RIGHTEE = (0.0, -1.0, 0.0)

# peg_axis_quat / hole_axis_quat — 给 visualize_targets 的箭头 orient 用.
# peg_axis_quat = LeftEE_quat ∘ R_x(+90°) ∘ R_x(180°) = LeftEE_quat ∘ R_x(+270°)
# R_x(270°) 的 wxyz quat: (cos 135°, sin 135°, 0, 0) = (-√2/2, +√2/2, 0, 0)
_C45 = math.cos(math.pi / 4)
_S45 = math.sin(math.pi / 4)
PEG_AXIS_QUAT_OFFSET = (-_S45, _C45, 0.0, 0.0)   # R_x(270°), 沿用旧 flip 约定
HOLE_AXIS_QUAT_OFFSET = (_C45, _S45, 0.0, 0.0)   # R_x(+90°), 无 flip

# Agent obs 索引切片 — reward / is_absorbing 直接按位读, 不再走 obs_helper
# (obs_helper 的 idx_map 对应 raw obs 而非 agent obs).
# 32 维布局对所有 stage (M1' / M2a / M2b) 通用; M3+ 才会再加 radial/axial 维度.
_AGENT_OBS_JOINT_POS = slice(0, 14)
_AGENT_OBS_JOINT_VEL = slice(14, 28)
_AGENT_OBS_POS_VEC = slice(28, 31)
_AGENT_OBS_AXIS_DOT = slice(31, 32)
AGENT_OBS_DIM = 32

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
        rew_axis=0.0,
        rew_success=2.0,
        rew_joint_limit=0.02,
        rew_action=0.005,
        success_hold_steps=10,
        terminal_hold_bonus=0.0,
        preinsert_success_pos_threshold=0.10,
        success_axis_threshold=float("inf"),
        joint_limit_margin_frac=0.8,
        preinsert_offset=DEFAULT_PREINSERT_OFFSET,
        # Sphere-proxy 自碰撞兜底 (PhysX 失明区补丁, 全 stage 通用):
        #   min_clearance < clearance_hard 时与 PhysX 力 OR, 触发 hard absorbing.
        #   default 0.0 = 球壳一接触就算碰撞. 设 -inf 即关闭.
        clearance_hard=0.0,
        proxy_arm_radius=ARM_PROXY_RADIUS,
        proxy_ee_radius=EE_PROXY_RADIUS,
        usd_path=None,
    ):
        self._action_scale = action_scale
        self._initial_joint_noise = initial_joint_noise
        self._collision_threshold = collision_force_threshold
        self._r_min = reward_absorbing_r_min
        self._reward_scale = reward_scale
        self._w_pos = rew_pos
        # rew_axis 默认 0 = M1' 行为 (axis 项消失). M2a/M2b 通过 CLI 打开.
        self._w_axis = rew_axis
        self._w_success = rew_success
        self._w_joint_limit = rew_joint_limit
        self._w_action = rew_action
        # hold-N absorbing 设计 (沿用 phase 1): 连续 N 步在阈内即终止 + bonus.
        # bonus=0 时整个机制关闭 (baseline 行为).
        self._success_hold_steps = int(success_hold_steps)
        self._terminal_hold_bonus = float(terminal_hold_bonus)
        self._absorbing_terminal_active = self._terminal_hold_bonus > 0.0
        self._preinsert_success_pos_threshold = float(preinsert_success_pos_threshold)
        # success_axis_threshold 默认 inf = success 不检查 axis (M1' 行为).
        # M2 时通过 CLI 设成 0.5 / 0.2. 用 inf 而不是 None 让 success_mask 表达式
        # 不需要 None-check 分支, 永远是干净的 (pos<pos_th) & (axis_err<axis_th).
        self._success_axis_threshold = float(success_axis_threshold)
        self._joint_limit_margin_frac = joint_limit_margin_frac
        self._preinsert_offset = float(preinsert_offset)
        # Sphere-proxy 兜底参数. clearance_hard 允许 -inf (=关闭); 半径必须有限正数.
        self._clearance_hard = float(clearance_hard)
        self._proxy_arm_radius = float(proxy_arm_radius)
        self._proxy_ee_radius = float(proxy_ee_radius)
        if not (math.isfinite(self._proxy_arm_radius) and self._proxy_arm_radius > 0.0):
            raise ValueError(
                f"proxy_arm_radius 必须是有限正数, 传入 {proxy_arm_radius}"
            )
        if not (math.isfinite(self._proxy_ee_radius) and self._proxy_ee_radius > 0.0):
            raise ValueError(
                f"proxy_ee_radius 必须是有限正数, 传入 {proxy_ee_radius}"
            )
        self._usd_path = Path(usd_path) if usd_path is not None else DEFAULT_USD_PATH
        if not self._usd_path.is_file():
            raise FileNotFoundError(
                "找不到机器人 USD 资产文件: "
                f"{self._usd_path}\n"
                "请确认仓库内存在 assets/usd/dual_arm_iiwa/dual_arm_iiwa_with_peghole.usda，"
                "或在构造 DualArmPegHoleEnv 时显式传入 usd_path。"
            )

        # is_absorbing 与 reward 在同一 next_obs 上背靠背调用, 缓存避免重复计算.
        # _create_observation 每步会刷新这些 cache.
        self._last_collision_mask = None
        self._last_pos_err = None
        self._last_axis_err = None
        self._last_success_mask = None
        # sphere-proxy clearance: is_absorbing 里每步算并 cache,
        # _last_min_clearance < clearance_hard 即触发 hard absorbing.
        self._last_min_clearance = None
        # _preprocess_action → reward 链路里缓存 pre-scale 的 raw action,
        # 用于 L2 惩罚. 这样 w_action 和 action_scale 解耦.
        self._last_raw_action = None

        observation_spec = [
            ("joint_pos", "", ObservationType.JOINT_POS, ARM_JOINTS),
            ("joint_vel", "", ObservationType.JOINT_VEL, ARM_JOINTS),
            ("left_ee_pos", LEFT_EE_PATH, ObservationType.BODY_POS, None),
            ("right_ee_pos", RIGHT_EE_PATH, ObservationType.BODY_POS, None),
            ("left_ee_rot", LEFT_EE_PATH, ObservationType.BODY_ROT, None),
            ("right_ee_rot", RIGHT_EE_PATH, ObservationType.BODY_ROT, None),
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
            camera_position=(20, -15, 10),
            camera_target=(5, 0, 0.5),
        )

        # 关节位限 + 默认位
        limits = self._task.get_joint_pos_limits()
        self._joint_lower, self._joint_upper = limits[0], limits[1]
        # 不用 USD 自带的 zero pose, 改用 HOME_JOINT_POS (胸前 ready), 避免 reset
        # center 落在 zero 全展开姿态, M1' 早期探索浪费在无效扇区.
        self._default_joint_pos = torch.tensor(
            HOME_JOINT_POS, device=device, dtype=self._joint_lower.dtype
        )
        # fail-fast: home pose 必须落在每个关节的 [lower, upper] 内, 不然 reset
        # 那一步 PhysX 会把它 clamp 到边界, 与设计意图不符.
        if torch.any(self._default_joint_pos < self._joint_lower) or torch.any(
            self._default_joint_pos > self._joint_upper
        ):
            bad = (
                (self._default_joint_pos < self._joint_lower)
                | (self._default_joint_pos > self._joint_upper)
            ).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(
                f"HOME_JOINT_POS 越界: 关节索引 {bad} 不在 [lower, upper] 内. "
                f"lower={self._joint_lower.tolist()}  upper={self._joint_upper.tolist()}  "
                f"home={self._default_joint_pos.tolist()}"
            )

        # USD iiwa 默认 position-drive (kp ~ 5e5); velocity 控制必须把 kp 置 0,
        # 否则 reset 里 set_joint_positions 会一并写 pos_target, 高 kp 把关节钉
        # 回 reset 点.
        robots = self._task.robots
        cj = self._task._controlled_joints
        zero_kps = torch.zeros(self._n_envs, len(ARM_JOINTS), device=device)
        _, cur_kds = robots.get_gains(joint_indices=cj, clone=True)
        robots.set_gains(kps=zero_kps, kds=cur_kds, joint_indices=cj)
        self._cj = cj
        self._robots = robots

        # 累计自碰撞终止次数 (train_sac 每 epoch 读取并差分).
        # _absorb_count       — 总数 (任一信号触发, 反映 episode 终止次数)
        # _absorb_count_physx — PhysX 力检测触发数
        # _absorb_count_sphere— sphere-proxy clearance 触发数
        # PhysX 与 sphere 可同步触发, 所以 physx + sphere ≥ total.
        self._absorb_count = 0
        self._absorb_count_physx = 0
        self._absorb_count_sphere = 0

        # hold-N 计数器 (per env)
        self._consecutive_inthresh = torch.zeros(
            self._n_envs, dtype=torch.long, device=self._device
        )
        self._last_hold_done_mask = None

        # 解析式 frame 用的常量 (LeftEE 局部坐标), 一次 build, broadcast 用
        dtype = self._joint_lower.dtype
        dev = self._device
        self._peg_tip_offset = torch.tensor(PEG_TIP_OFFSET_IN_LEFTEE, device=dev, dtype=dtype)
        self._hole_entry_offset = torch.tensor(HOLE_ENTRY_OFFSET_IN_RIGHTEE, device=dev, dtype=dtype)
        self._peg_axis_local = torch.tensor(PEG_AXIS_IN_LEFTEE, device=dev, dtype=dtype)
        self._hole_axis_local = torch.tensor(HOLE_AXIS_IN_RIGHTEE, device=dev, dtype=dtype)
        self._peg_axis_quat_offset = torch.tensor(
            PEG_AXIS_QUAT_OFFSET, device=dev, dtype=dtype
        )
        self._hole_axis_quat_offset = torch.tensor(
            HOLE_AXIS_QUAT_OFFSET, device=dev, dtype=dtype
        )

        # peg/hole 资产存在性 fail-fast 检查 (phase 1.5 commit 2 的 print 降级删除)
        self._verify_peghole_prims_exist()

        # Sphere-proxy 索引 + 半径 tensor 解析. 必须在 super().__init__() 之后,
        # body_names 才存在. is_absorbing 每步调 _compute_min_clearance().
        self._build_sphere_proxy_indices()

        # 同步一次物理状态, 避免 reset_all 后第一帧 BODY_POS / BODY_ROT 是 stale
        self._world.step(render=False)

    # ------------------------------------------------------------------
    # mushroom hooks
    # ------------------------------------------------------------------
    def _modify_mdp_info(self, mdp_info):
        # action: [-1,1]^14, SAC tanh policy 直接映射
        device = mdp_info.action_space.low.device
        dtype = mdp_info.action_space.low.dtype
        one = torch.ones(len(ARM_JOINTS), device=device, dtype=dtype)
        mdp_info.action_space = Box(-one, one, data_type=dtype)

        # observation: 32 维 agent obs (见模块 docstring + _AGENT_OBS_* 切片).
        # 不能用 self._joint_lower / _joint_upper 在这里取 — 它们在 super().__init__
        # 之后才赋值, 而 mushroom 在 super 里就调本函数. obs_helper 已经构造完毕,
        # 走它的 obs_limits + obs_idx_map 切出 joint 段.
        raw_low, raw_high = self.observation_helper.obs_limits
        jp_idx = self.observation_helper.obs_idx_map["joint_pos"]
        jv_idx = self.observation_helper.obs_idx_map["joint_vel"]
        jp_low = raw_low[jp_idx].to(dtype)
        jp_high = raw_high[jp_idx].to(dtype)
        jv_low = raw_low[jv_idx].to(dtype)
        jv_high = raw_high[jv_idx].to(dtype)
        pos_lo = torch.full((3,), -5.0, device=jp_low.device, dtype=dtype)
        pos_hi = torch.full((3,), 5.0, device=jp_low.device, dtype=dtype)
        axis_lo = torch.full((1,), -1.0, device=jp_low.device, dtype=dtype)
        axis_hi = torch.full((1,), 1.0, device=jp_low.device, dtype=dtype)
        new_obs_low = torch.cat([jp_low, jv_low, pos_lo, axis_lo], dim=0)
        new_obs_high = torch.cat([jp_high, jv_high, pos_hi, axis_hi], dim=0)
        mdp_info.observation_space = Box(new_obs_low, new_obs_high, data_type=dtype)
        return mdp_info

    def _preprocess_action(self, action):
        action = torch.as_tensor(action, device=self._device, dtype=self._joint_lower.dtype)
        clipped = torch.clamp(action, -1.0, 1.0)
        self._last_raw_action = clipped
        return clipped * self._action_scale

    def _create_observation(self, obs):
        """raw obs (42 dim) → agent obs (32 dim).

        raw 布局 (与 observation_spec 顺序一致):
            joint_pos[14] joint_vel[14] left_ee_pos[3] right_ee_pos[3]
            left_ee_rot[4] right_ee_rot[4]
        agent obs 布局 (见 _AGENT_OBS_* 切片):
            joint_pos[14] joint_vel[14] pos_vec[3] axis_dot[1]
        """
        joint_pos = self.observation_helper.get_from_obs(obs, "joint_pos")
        joint_vel = self.observation_helper.get_from_obs(obs, "joint_vel")
        left_ee = self.observation_helper.get_from_obs(obs, "left_ee_pos")
        right_ee = self.observation_helper.get_from_obs(obs, "right_ee_pos")
        left_quat = self.observation_helper.get_from_obs(obs, "left_ee_rot")
        right_quat = self.observation_helper.get_from_obs(obs, "right_ee_rot")

        peg_tip = left_ee + self._quat_apply(left_quat, self._peg_tip_offset)
        hole_entry = right_ee + self._quat_apply(right_quat, self._hole_entry_offset)
        peg_axis = self._quat_apply(left_quat, self._peg_axis_local)
        hole_axis = self._quat_apply(right_quat, self._hole_axis_local)
        peg_axis = peg_axis / peg_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        hole_axis = hole_axis / hole_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        preinsert_target = hole_entry + self._preinsert_offset * hole_axis
        pos_vec = peg_tip - preinsert_target

        # axis_dot ∈ [-1, +1], -1 = 完美对齐. axis_err = 1 + axis_dot ∈ [0, 2].
        axis_dot = (peg_axis * hole_axis).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        axis_err = 1.0 + axis_dot.squeeze(-1)

        # cache 给 is_absorbing / reward 复用 (避免再算一遍 quat 旋转).
        # M3+ 想加 radial_vec / axial_dist 时, 在这里 cache 即可.
        self._cached_pos_vec = pos_vec
        self._cached_axis_err = axis_err

        return torch.cat([joint_pos, joint_vel, pos_vec, axis_dot], dim=-1)

    def _compute_task_errors(self, agent_obs):
        """从 agent obs 切片重建 (pos_err, axis_err, success_mask).

        train_sac.py 的 reset stats 和 _eval_utils.compute_hold_metrics 都通过
        这条接口拉指标; agent obs 已经包含 pos_vec 和 axis_dot, 所以不需要再走
        _create_observation 重新查 EE pose.

        success 用 stage flag 控制:
            success_axis_threshold=inf 时 axis 项恒 True, 退化为 pos-only (M1' 行为).
            success_axis_threshold=0.5/0.2 时变成 pos ∧ axis (M2a/M2b).
        """
        pos_vec = agent_obs[..., _AGENT_OBS_POS_VEC]
        pos_err = torch.norm(pos_vec, dim=-1)
        # axis_dot 已经在 obs 里 [-1, +1]; axis_err = 1 + axis_dot ∈ [0, 2].
        axis_dot = agent_obs[..., _AGENT_OBS_AXIS_DOT].squeeze(-1)
        axis_err = 1.0 + axis_dot
        success_mask = (
            (pos_err < self._preinsert_success_pos_threshold)
            & (axis_err < self._success_axis_threshold)
        )
        return pos_err, axis_err, success_mask

    def is_absorbing(self, obs):
        physx_collision = self._check_collision("arm_L", "arm_R", self._collision_threshold,
                                                dt=self._timestep)
        # sphere-proxy 兜底: 双臂 19 球 vs 19 球的最小 clearance 跌破 clearance_hard
        # 也算 collision. clearance_hard=-inf 时此项恒 False, 退化为纯 PhysX.
        min_clearance, _ = self._compute_min_clearance()
        self._last_min_clearance = min_clearance
        if math.isfinite(self._clearance_hard):
            sphere_collision = min_clearance < self._clearance_hard
        else:
            sphere_collision = torch.zeros_like(physx_collision)
        collision = physx_collision | sphere_collision
        # 两个 bucket 可同时触发 (一步同时撞), 分别累加便于诊断哪个信号在主导;
        # _absorb_count 仍按 OR 后的 collision 累加 (= 实际 absorb 次数).
        self._absorb_count_physx += int(physx_collision.sum().item())
        self._absorb_count_sphere += int(sphere_collision.sum().item())
        self._absorb_count += int(collision.sum().item())
        self._last_collision_mask = collision

        # _create_observation 已 cache pos_vec / axis_err; 在这里只 compose success.
        # axis_th=inf (M1') 时 axis 项恒 True, success 退化为 pos-only.
        pos_err = torch.norm(self._cached_pos_vec, dim=-1)
        axis_err = self._cached_axis_err
        success_mask = (
            (pos_err < self._preinsert_success_pos_threshold)
            & (axis_err < self._success_axis_threshold)
        )
        self._last_pos_err = pos_err
        self._last_axis_err = axis_err
        self._last_success_mask = success_mask

        # 更新 per-env 的连续 in-threshold 计数
        self._consecutive_inthresh = torch.where(
            success_mask,
            self._consecutive_inthresh + 1,
            torch.zeros_like(self._consecutive_inthresh),
        )

        if self._absorbing_terminal_active:
            hold_done = self._consecutive_inthresh >= self._success_hold_steps
        else:
            hold_done = torch.zeros_like(collision)
        self._last_hold_done_mask = hold_done

        return collision | hold_done

    def reward(self, obs, action, next_obs, absorbing):
        joint_pos = next_obs[..., _AGENT_OBS_JOINT_POS]
        pos_err = self._last_pos_err
        axis_err = self._last_axis_err
        success = self._last_success_mask.to(pos_err.dtype)

        # 关节极限软惩罚: 超 margin 才计
        joint_range = self._joint_upper - self._joint_lower
        joint_center = 0.5 * (self._joint_upper + self._joint_lower)
        excess = torch.clamp(
            (torch.abs((joint_pos - joint_center) / (0.5 * joint_range))
             - self._joint_limit_margin_frac)
            / (1.0 - self._joint_limit_margin_frac),
            min=0.0, max=1.0,
        )
        joint_limit_norm = torch.sum(excess ** 2, dim=-1)

        # action L2: pre-scale raw action, 与 action_scale 解耦
        action_sq = (self._last_raw_action ** 2).sum(dim=-1)

        # rew_axis=0 时 axis 项消失, 这就是 M1' 行为. M2 通过 CLI 把它打开.
        normal = (
            -self._w_pos * pos_err
            - self._w_axis * axis_err
            - self._w_joint_limit * joint_limit_norm
            - self._w_action * action_sq
            + self._w_success * success
        )

        # 三路 reward 选择: collision (硬 absorbing) > hold-N (软 absorbing) > normal
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

        idx_tensor = torch.as_tensor(env_indices, device=self._device, dtype=torch.long)
        self._consecutive_inthresh[idx_tensor] = 0

        # set_joint_positions 只写 DOF buffer; 不 step 的话 BODY_POS / BODY_ROT
        # view 还是 reset 前的值, reset_all 读到的 EE pose 是 stale.
        self._world.step(render=False)

    def get_obs_scale(self):
        """32 维 fixed-divisor 归一化向量, 与 agent obs 一一对应.

        缩放选择 (大致让每维标准差落到 ~1):
            joint_pos: 半关节范围
            joint_vel: 2.0 rad/s
            pos_vec:   0.3m (preinsert 目标距离量级 ~ <0.5m)
            axis_dot:  1.0  (本身已经在 [-1, +1], 直接除 1 不变)
        """
        device = self._joint_lower.device
        dtype = torch.float32
        joint_pos_scale = 0.5 * (self._joint_upper - self._joint_lower).to(dtype)
        joint_vel_scale = torch.full_like(joint_pos_scale, 2.0)
        pos_scale = torch.full((3,), 0.3, device=device, dtype=dtype)
        axis_scale = torch.full((1,), 1.0, device=device, dtype=dtype)
        return torch.cat([joint_pos_scale, joint_vel_scale, pos_scale, axis_scale])

    def _simulation_pre_step(self):
        """每 intermediate step 前注入重力补偿 effort.

        kp=0 velocity drive 只有阻尼项, 对恒定重力是结构性欠阻尼. 真机 iiwa
        velocity mode 底层跑重力补偿, sim 里我们用 G(q) 作为前馈 effort,
        agent 不需要从零学这个 7-DoF 非线性映射.
        """
        tau_g = self._robots.get_generalized_gravity_forces(clone=False)
        self._robots.set_joint_efforts(tau_g[:, self._cj], joint_indices=self._cj)

    # ------------------------------------------------------------------
    # 解析式 peg / hole frame helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _quat_apply(q_wxyz, v):
        """用单位四元数 q (wxyz) 旋转向量 v. 支持 [N,4]×[N,3] 或 [N,4]×[3] 广播.

        返回 [N, 3].
        """
        if v.dim() == 1:
            v = v.unsqueeze(0).expand(q_wxyz.shape[0], -1)
        w = q_wxyz[..., 0]
        x = q_wxyz[..., 1]
        y = q_wxyz[..., 2]
        z = q_wxyz[..., 3]
        vx = v[..., 0]
        vy = v[..., 1]
        vz = v[..., 2]
        tx = 2 * (y * vz - z * vy)
        ty = 2 * (z * vx - x * vz)
        tz = 2 * (x * vy - y * vx)
        rx = vx + w * tx + (y * tz - z * ty)
        ry = vy + w * ty + (z * tx - x * tz)
        rz = vz + w * tz + (x * ty - y * tx)
        return torch.stack([rx, ry, rz], dim=-1)

    @staticmethod
    def _quat_mul(q1_wxyz, q2_wxyz):
        """四元数乘法 (wxyz), 支持广播."""
        if q2_wxyz.dim() == 1:
            q2_wxyz = q2_wxyz.unsqueeze(0).expand(q1_wxyz.shape[0], -1)
        w1, x1, y1, z1 = q1_wxyz.unbind(dim=-1)
        w2, x2, y2, z2 = q2_wxyz.unbind(dim=-1)
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dim=-1)

    def _verify_peghole_prims_exist(self):
        """fail fast: peg_tip / hole_entry prim 不在 stage 直接 raise.

        老的 phase 1.5 实现是 print 后继续跑; 现在主线已经是 peg-in-hole, 加载
        无 peg/hole 的 USD 必然导致 _create_observation 用错误的常量 offset
        生成无意义的 frame. 早死早超生.
        """
        try:
            import omni.usd
        except ImportError:
            return  # 单元测试或非 IsaacSim 路径下跳过
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage 还没初始化 — peg/hole 检查无法执行")
        peg_found = hole_found = False
        for prim in stage.Traverse():
            p = str(prim.GetPath())
            if p.endswith("/Peg/peg_tip"):
                peg_found = True
            if p.endswith("/Hole/hole_entry"):
                hole_found = True
            if peg_found and hole_found:
                return
        raise RuntimeError(
            "stage 里找不到 peg_tip / hole_entry prim. "
            f"usd_path={self._usd_path}\n"
            "M0+ 要求加载带 peg/hole 视觉资产的 USDA. "
            "用 dual_arm_iiwa_with_peghole.usda, 或重跑 build_peghole_usd.py 生成."
        )

    def get_preinsert_frames(self):
        """返回 peg_tip / hole_entry / preinsert_target 的世界帧位姿 (env-local).

        训练循环里 _create_observation 已经算并缓存了同样的量; 但 visualize_*
        的主循环可能在两次 step 之间没经过 _create_observation, cache 会 stale.
        所以这里强制重新查一次 fresh raw obs 再算.

        Returns batched dict (batch=num_envs):
            peg_tip_pos          [N, 3]
            peg_tip_quat         [N, 4]  wxyz, = LeftEE_quat
            peg_axis             [N, 3]  unit, R(LeftEE_quat) · PEG_AXIS_IN_LEFTEE
            peg_axis_quat        [N, 4]  让 +Z apply 后等于 peg_axis 的 quat
            hole_entry_pos       [N, 3]
            hole_entry_quat      [N, 4]  wxyz, = RightEE_quat
            hole_axis            [N, 3]
            hole_axis_quat       [N, 4]
            preinsert_target_pos [N, 3]
            preinsert_target_quat[N, 4]
        """
        raw = self.observation_helper.build_obs(self._task.get_observations(clone=True))
        left_ee = self.observation_helper.get_from_obs(raw, "left_ee_pos")
        right_ee = self.observation_helper.get_from_obs(raw, "right_ee_pos")
        left_quat = self.observation_helper.get_from_obs(raw, "left_ee_rot")
        right_quat = self.observation_helper.get_from_obs(raw, "right_ee_rot")

        peg_tip = left_ee + self._quat_apply(left_quat, self._peg_tip_offset)
        hole_entry = right_ee + self._quat_apply(right_quat, self._hole_entry_offset)
        peg_axis = self._quat_apply(left_quat, self._peg_axis_local)
        hole_axis = self._quat_apply(right_quat, self._hole_axis_local)
        peg_axis = peg_axis / peg_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        hole_axis = hole_axis / hole_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        peg_axis_quat = self._quat_mul(left_quat, self._peg_axis_quat_offset)
        hole_axis_quat = self._quat_mul(right_quat, self._hole_axis_quat_offset)

        preinsert_target_pos = hole_entry + self._preinsert_offset * hole_axis
        preinsert_target_quat = hole_axis_quat.clone()

        return {
            "peg_tip_pos": peg_tip,
            "peg_tip_quat": left_quat,
            "peg_axis_quat": peg_axis_quat,
            "peg_axis": peg_axis,
            "hole_entry_pos": hole_entry,
            "hole_entry_quat": right_quat,
            "hole_axis_quat": hole_axis_quat,
            "hole_axis": hole_axis,
            "preinsert_target_pos": preinsert_target_pos,
            "preinsert_target_quat": preinsert_target_quat,
        }

    def _compute_preinsert_errors(self, frames=None):
        """完整 preinsert 几何误差 — 给 visualize_* 的诊断输出用, 不进 reward.

        训练 reward / is_absorbing 走 _compute_task_errors (从 cached agent obs
        切片读); 这条路径独立查 fresh raw obs, 适合 visualize/diagnose 主循环.
        success_mask 用与训练一致的 (pos<pos_th) ∧ (axis<axis_th) 表达式, axis_th
        默认 inf 时退化为 pos-only.
        """
        if frames is None:
            frames = self.get_preinsert_frames()
        peg_tip = frames["peg_tip_pos"]
        hole_entry = frames["hole_entry_pos"]
        preinsert_target = frames["preinsert_target_pos"]
        peg_axis = frames["peg_axis"]
        hole_axis = frames["hole_axis"]

        pos_vec = peg_tip - preinsert_target
        pos_err = torch.norm(pos_vec, dim=-1)

        axis_dot = torch.sum(peg_axis * hole_axis, dim=-1).clamp(-1.0, 1.0)
        axis_err = 1.0 + axis_dot

        d = peg_tip - hole_entry
        axial_dist = torch.sum(d * hole_axis, dim=-1)
        radial_vec = d - axial_dist.unsqueeze(-1) * hole_axis
        radial_err = torch.norm(radial_vec, dim=-1)

        success_mask = (
            (pos_err < self._preinsert_success_pos_threshold)
            & (axis_err < self._success_axis_threshold)
        )

        return {
            "pos_vec": pos_vec,
            "pos_err": pos_err,
            "axis_dot": axis_dot,
            "axis_err": axis_err,
            "axial_dist": axial_dist,
            "radial_vec": radial_vec,
            "radial_err": radial_err,
            "success_mask": success_mask,
        }

    # ------------------------------------------------------------------
    # Sphere-proxy clearance (PhysX 自碰撞兜底, 全 stage 通用)
    # ------------------------------------------------------------------
    def _build_sphere_proxy_indices(self):
        """从 articulation body_names 解析每侧 19 球需要的 body 索引.

        构造完成后:
            self._left_arm_joint_idx   [8]   left_arm_link_0..link_7 在 body_names 里的位置
            self._right_arm_joint_idx  [8]
            self._left_ee_proxy_idx    [4]   coupler / hande_link / l_finger / r_finger
            self._right_ee_proxy_idx   [4]
            self._proxy_radii_per_side [19]  arm 段 15 球 + EE 段 4 球, 半径来自 env 参数
        """
        body_names = list(self._task.robots.body_names)

        def _resolve_all(names):
            missing = [n for n in names if n not in body_names]
            if missing:
                raise RuntimeError(
                    "build_sphere_proxy_indices: body_names 里缺这些 link: "
                    f"{missing}\navailable: {body_names}"
                )
            return [body_names.index(n) for n in names]

        device = self._device
        self._left_arm_joint_idx = torch.as_tensor(
            _resolve_all(LEFT_ARM_JOINT_BODY_NAMES), device=device, dtype=torch.long
        )
        self._right_arm_joint_idx = torch.as_tensor(
            _resolve_all(RIGHT_ARM_JOINT_BODY_NAMES), device=device, dtype=torch.long
        )
        self._left_ee_proxy_idx = torch.as_tensor(
            _resolve_all(LEFT_EE_PROXY_BODY_NAMES), device=device, dtype=torch.long
        )
        self._right_ee_proxy_idx = torch.as_tensor(
            _resolve_all(RIGHT_EE_PROXY_BODY_NAMES), device=device, dtype=torch.long
        )
        # 每侧 19 球的半径 (顺序: 8 关节 + 7 中点 + 4 EE)
        n_arm = len(LEFT_ARM_JOINT_BODY_NAMES)     # 8
        n_mid = n_arm - 1                          # 7 段中点
        n_ee = len(LEFT_EE_PROXY_BODY_NAMES)       # 4
        radii = torch.empty(n_arm + n_mid + n_ee, device=device, dtype=torch.float32)
        radii[:n_arm + n_mid] = self._proxy_arm_radius
        radii[n_arm + n_mid:] = self._proxy_ee_radius
        self._proxy_radii_per_side = radii         # [19]
        self._n_proxies_per_side = n_arm + n_mid + n_ee

    def _gather_side_proxies(self, body_pos, joint_idx, ee_idx):
        """body_pos: [n_envs, n_bodies, 3] → [n_envs, 19, 3] sphere proxy 球心.

        球心顺序: 8 关节 + 7 中点 + 4 EE, 与 self._proxy_radii_per_side 对齐.
        """
        joints = body_pos[:, joint_idx, :]                   # [n_envs, 8, 3]
        mids = 0.5 * (joints[:, :-1, :] + joints[:, 1:, :])  # [n_envs, 7, 3]
        ee = body_pos[:, ee_idx, :]                          # [n_envs, 4, 3]
        return torch.cat([joints, mids, ee], dim=1)          # [n_envs, 19, 3]

    def _compute_min_clearance(self):
        """sphere-proxy 双臂 clearance, 跨所有 env vectorized.

            clearance_ij = ||c_L_i - c_R_j|| - r_L_i - r_R_j
            min_clearance = clearance.min over (i, j)  → [n_envs]

        Returns:
            min_clearance: [n_envs]   每个 env 当前最小双臂 clearance (m).
                                       <0 表示两侧 sphere proxy 已经穿插.
            info: dict
                "min_pair_left_idx":  [n_envs]  long, 0..18 (球索引 per side)
                "min_pair_right_idx": [n_envs]  long
                "left_proxies":  [n_envs, 19, 3]  球心位置 (env-local world)
                "right_proxies": [n_envs, 19, 3]
        """
        physics_view = self._task.robots._physics_view
        xforms = physics_view.get_link_transforms()       # [n_envs, n_bodies, 7] (xyz+quat)
        xforms_t = torch.as_tensor(xforms, device=self._device, dtype=torch.float32)
        if xforms_t.dim() == 2:
            n_bodies = len(self._task.robots.body_names)
            xforms_t = xforms_t.view(self._n_envs, n_bodies, -1)
        body_pos = xforms_t[..., :3]                      # [n_envs, n_bodies, 3]

        left = self._gather_side_proxies(
            body_pos, self._left_arm_joint_idx, self._left_ee_proxy_idx
        )
        right = self._gather_side_proxies(
            body_pos, self._right_arm_joint_idx, self._right_ee_proxy_idx
        )
        # [n_envs, 19, 19, 3] → [n_envs, 19, 19] center-to-center
        diff = left.unsqueeze(2) - right.unsqueeze(1)
        dist = diff.norm(dim=-1)
        rL = self._proxy_radii_per_side.view(1, -1, 1)    # [1, 19, 1]
        rR = self._proxy_radii_per_side.view(1, 1, -1)    # [1, 1, 19]
        clearance = dist - rL - rR                         # [n_envs, 19, 19]

        n = self._n_proxies_per_side
        flat = clearance.view(self._n_envs, -1)
        min_vals, min_flat = flat.min(dim=1)               # [n_envs]
        left_idx = min_flat // n
        right_idx = min_flat % n

        info = {
            "min_pair_left_idx": left_idx,
            "min_pair_right_idx": right_idx,
            "left_proxies": left,
            "right_proxies": right,
        }
        return min_vals, info
