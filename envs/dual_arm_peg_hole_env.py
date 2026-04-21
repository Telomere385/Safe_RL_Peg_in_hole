"""双臂末端"装配就绪位姿"任务 (phase 1) — mushroom-rl IsaacSim 向量化环境.

phase 1 目标: 左右末端到达固定在 world frame 的绝对 (pos, quat) 目标,
            为后续 peg/hole 装配做准备. 不涉及任何 peg/hole 几何.

每臂控全部 7 DoF (A1-A7), 提供完整 6-DoF 末端可达性 + 1 冗余.

观测 (42 维):
    joint_pos     (14) 左右臂 A1-A7 关节角
    joint_vel     (14) 左右臂 A1-A7 关节角速度
    left_ee_pos   (3)  左末端位置 (env-local)
    right_ee_pos  (3)  右末端位置 (env-local)
    left_ee_rot   (4)  左末端四元数 (wxyz)
    right_ee_rot  (4)  右末端四元数 (wxyz)

动作 (14): action ∈ [-1,1] → velocity 指令 (rad/s), 系数 action_scale.

Reward (每个 EE 独立锚定到 world frame 的绝对 6-DoF 目标 — 完全无对称简并):
    - w_pos  * (||left_ee - T_L_pos|| + ||right_ee - T_R_pos||)   位置误差
    - w_pose * (quat_dist(q_L, T_L_rot) + quat_dist(q_R, T_R_rot)) 姿态误差
                 quat_dist(a,b) = 1 - |a·b|  ∈ [0,1], 不受 ±q 双覆盖影响.
                 旧式 1 - (a·b)² 在 θ=π (a·b=0) 处梯度消失, policy 卡在 plateau;
                 1 - |a·b| 线性, 只在 a·b=0 有 cusp (非极值点), 处处有下降方向.
    - w_joint_limit * joint_limit_norm                            关节极限软惩罚
    + w_success * 1[‖·‖_L<pos_th ∧ ‖·‖_R<pos_th ∧ pose_err<pose_th]

Target 生成 (在 __init__ 末尾 eager-init 一次冻结, 固定在 world frame):
    left_default, right_default = env-平均的 default EE 位置
    center = (left_default + right_default) / 2
    T_L_pos = left_default  + f · (center - left_default)   # f = target_travel_fraction
    T_R_pos = right_default + f · (center - right_default)
    T_L_rot = (min-rotation from z_L_default to dir(T_R_pos - T_L_pos)) · q_L_default
    T_R_rot = (min-rotation from z_R_default to dir(T_L_pos - T_R_pos)) · q_R_default
→ 左 z 轴指向 T_R, 右 z 轴指向 T_L (夹爪对开). 其它 2 DoF 由 default quat 决定.
→ 目标是 world frame 的常量, 对称性完全打破: 无 hug/face-away/body-center 简并.
终止:
    - 碰撞 (左右臂接触力 > 阈值): 吸收 r = r_min / (1 - γ)
    - 成功 (位姿都达标): 吸收, 该步 reward 照常发放 (含 w_success 奖励)
Eager-init 后在 env 0 的 world 原点位置 spawn 可视化 marker:
    红球 + 红细圆柱 → T_L_pos + T_L_rot z 轴;  绿球 + 绿细圆柱 → T_R_pos + T_R_rot z 轴.
"""

import torch

from mushroom_rl.environments import IsaacSim
from mushroom_rl.utils.isaac_sim import ObservationType, ActionType
from mushroom_rl.rl_utils.spaces import Box


USD_PATH = "/home/miao/dual_arm_ws/usd_imports/dual_arm_iiwa/dual_arm_iiwa.usd"

CONTROLLED_IDX = (1, 2, 3, 4, 5, 6, 7)  # A1-A7, 7 DoF/臂, 完整可达 SE(3)
LEFT_ARM_JOINTS = [f"left_arm_A{i}" for i in CONTROLLED_IDX]
RIGHT_ARM_JOINTS = [f"right_arm_A{i}" for i in CONTROLLED_IDX]
ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS  # 14

# 碰撞组仍覆盖完整链路: 固定关节的 link 也会撞
LEFT_ARM_ALL_LINKS = [f"/left_arm_link_{i}" for i in range(1, 8)]
RIGHT_ARM_ALL_LINKS = [f"/right_arm_link_{i}" for i in range(1, 8)]

LEFT_EE_PATH = "/left_hande_robotiq_hande_link"
RIGHT_EE_PATH = "/right_hande_robotiq_hande_link"

LEFT_ARM_LINKS = LEFT_ARM_ALL_LINKS + [LEFT_EE_PATH]
RIGHT_ARM_LINKS = RIGHT_ARM_ALL_LINKS + [RIGHT_EE_PATH]


class DualArmPegHoleEnv(IsaacSim):
    def __init__(
        self,
        num_envs=1,
        horizon=150,
        gamma=0.99,
        headless=True,
        device="cuda:0",
        action_scale=0.4,
        initial_joint_noise=0.3,
        collision_force_threshold=10.0,
        reward_absorbing_r_min=-2.0,
        reward_scale=1.0,
        rew_pos=1.0,
        rew_pose=0.3,
        rew_success=2.0,
        rew_joint_limit=0.02,
        success_pos_threshold=0.075,
        success_pose_threshold=0.5,
        joint_limit_margin_frac=0.8,
        target_travel_fraction=0.5,
    ):
        self._action_scale = action_scale
        self._initial_joint_noise = initial_joint_noise
        self._collision_threshold = collision_force_threshold
        self._r_min = reward_absorbing_r_min
        self._reward_scale = reward_scale
        self._w_pos = rew_pos
        self._w_pose = rew_pose
        self._w_success = rew_success
        self._w_joint_limit = rew_joint_limit
        self._success_pos_threshold = success_pos_threshold
        self._success_pose_threshold = success_pose_threshold
        self._joint_limit_margin_frac = joint_limit_margin_frac
        self._target_travel_fraction = target_travel_fraction
        # 在 __init__ 末尾 eager-init: 用 self._world.reset() 后的干净 default EE
        # 冻结 world-frame 目标 + spawn viz markers. 不 lazy-init 以避免 step_all
        # 里的 teleport_away (inactive env z=50) 污染均值: 当 n_episodes < num_envs
        # (如 eval_sac.py --n_episodes 3 vs 16 envs), get_mask 只 active 3 env,
        # teleport_away 把 13 env 抬到 z=50, 首次 is_absorbing 里 left_ee.mean(dim=0)
        # 变成 (3*0.73 + 13*50)/16 ≈ 41 → 目标与 marker 全飘空中, J 崩.
        self._left_target = None
        self._right_target = None
        self._left_target_rot = None
        self._right_target_rot = None
        # 缓存本步 collision mask, reward() 里只对 collision 盖 r_min (success 吸收给正常 reward)
        self._last_collision_mask = None
        # 缓存本步 task errors (is_absorbing 和 reward 背靠背在同一 next_obs 上调用, 避免重复计算)
        self._last_task_errors = None

        observation_spec = [
            ("joint_pos", "", ObservationType.JOINT_POS, ARM_JOINTS),
            ("joint_vel", "", ObservationType.JOINT_VEL, ARM_JOINTS),
            ("left_ee_pos", LEFT_EE_PATH, ObservationType.BODY_POS, None),
            ("right_ee_pos", RIGHT_EE_PATH, ObservationType.BODY_POS, None),
            ("left_ee_rot", LEFT_EE_PATH, ObservationType.BODY_ROT, None),
            ("right_ee_rot", RIGHT_EE_PATH, ObservationType.BODY_ROT, None),
        ]
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
            # GeneralTask 默认相机 (5,0,4)→(0,0,0) 是单 env 设置. 16 env + spacing=4
            # 时 robot 群大致在 x∈[0,12], y∈[-6,6], 默认相机直接站在群中心, 画面混乱.
            # 拉远到群外上方俯瞰整个 4x4 网格.
            camera_position=(20, -15, 10),
            camera_target=(5, 0, 0.5),
        )

        # 动作空间改为 [-1, 1]^|A| (给 SAC tanh policy)
        one = torch.ones(len(ARM_JOINTS), device=device)
        self._mdp_info.action_space = Box(-one, one, data_type=one.dtype)

        # 关节位限 + 默认位 (reward 里用)
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
        # 缓存给 _simulation_pre_step 用 (重力补偿前馈)
        self._cj = cj
        self._robots = robots

        # 累计碰撞终止次数 (train_sac 每 epoch 读取 + 上一 epoch 值做差分 → 本 epoch 新增).
        self._absorb_count = 0

        # Eager-init targets 在所有 step_all/teleport_away 之前: super().__init__()
        # 已经跑过 self._world.reset(), 16 个 env 的 robot 都在 USD default pose
        # (EE 在 world z≈0.73). 读一次 fresh obs 冻结绝对目标, 之后 _compute_task_errors
        # 不再 lazy-init, 完全免疫 teleport_away 污染.
        # 读 obs 前先 step 一次: BODY_POS view 在 _world.reset() 后未必同步物理状态
        # (与 setup() 里的 world.step 同理), 不 step 可能读到 stale 位姿.
        self._world.step(render=False)
        init_obs = self.observation_helper.build_obs(self._task.get_observations(clone=True))
        self._init_targets(
            self.observation_helper.get_from_obs(init_obs, "left_ee_pos"),
            self.observation_helper.get_from_obs(init_obs, "right_ee_pos"),
            self.observation_helper.get_from_obs(init_obs, "left_ee_rot"),
            self.observation_helper.get_from_obs(init_obs, "right_ee_rot"),
        )
        self._spawn_target_markers()

    @staticmethod
    def _quat_rotate_z(q):
        """q: (..., 4) wxyz → 世界系下的 z 轴 (R @ [0,0,1]): (..., 3)."""
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return torch.stack([
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ], dim=-1)

    @staticmethod
    def _quat_multiply(a, b):
        """Hamilton 乘 a ⊗ b, 输入 wxyz, 形状 (..., 4)."""
        aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return torch.stack([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ], dim=-1)

    @staticmethod
    def _quat_shortest_arc(v_from, v_to, eps=1e-8):
        """把单位向量 v_from 旋到 v_to 的最短弧四元数 (wxyz). 反平行情况不考虑."""
        v_from = v_from / (torch.norm(v_from, dim=-1, keepdim=True) + eps)
        v_to = v_to / (torch.norm(v_to, dim=-1, keepdim=True) + eps)
        dot = (v_from * v_to).sum(dim=-1, keepdim=True)
        axis = torch.linalg.cross(v_from, v_to, dim=-1)
        q = torch.cat([1.0 + dot, axis], dim=-1)
        return q / (torch.norm(q, dim=-1, keepdim=True) + eps)

    def _preprocess_action(self, action):
        return torch.clamp(action, -1.0, 1.0) * self._action_scale

    def _init_targets(self, left_ee, right_ee, left_rot, right_rot):
        """首次调用: 从 env-平均 default pose + fraction 冻结出绝对 6-DoF 目标."""
        left_default = left_ee.mean(dim=0).detach().clone()
        right_default = right_ee.mean(dim=0).detach().clone()
        left_rot_default = left_rot.mean(dim=0).detach().clone()
        right_rot_default = right_rot.mean(dim=0).detach().clone()
        left_rot_default = left_rot_default / torch.norm(left_rot_default)
        right_rot_default = right_rot_default / torch.norm(right_rot_default)

        center = 0.5 * (left_default + right_default)
        f = self._target_travel_fraction
        self._left_target = left_default + f * (center - left_default)
        self._right_target = right_default + f * (center - right_default)

        # f=0 应退化成 "目标 = default 6-DoF pose". 若仍强制 z 轴彼此相对，
        # 会出现位置不动但姿态目标突变的非连续行为，--sanity 会被错误地变成
        # "位置在阈值内、姿态永远不在阈值内" 的假 trivial task。
        if abs(float(f)) < 1e-8:
            self._left_target_rot = left_rot_default
            self._right_target_rot = right_rot_default
        else:
            # 目标 z 轴: 左 EE z 指向 T_R, 右 EE z 指向 T_L (夹爪对开)
            z_L_target = self._right_target - self._left_target
            z_L_target = z_L_target / torch.norm(z_L_target)
            z_R_target = -z_L_target

            # 最小弧旋转 · default quat → target quat (保持其余 2 DoF 接近 default)
            z_L_default = self._quat_rotate_z(left_rot_default)
            z_R_default = self._quat_rotate_z(right_rot_default)
            delta_L = self._quat_shortest_arc(z_L_default, z_L_target)
            delta_R = self._quat_shortest_arc(z_R_default, z_R_target)
            self._left_target_rot = self._quat_multiply(delta_L, left_rot_default)
            self._right_target_rot = self._quat_multiply(delta_R, right_rot_default)
        self._left_target_rot = self._left_target_rot / torch.norm(self._left_target_rot)
        self._right_target_rot = self._right_target_rot / torch.norm(self._right_target_rot)

        print(f"[TARGETS] fraction={f}  center={center.tolist()}\n"
              f"  T_L_pos = {self._left_target.tolist()}  T_L_rot = {self._left_target_rot.tolist()}\n"
              f"  T_R_pos = {self._right_target.tolist()}  T_R_rot = {self._right_target_rot.tolist()}\n"
              f"  (left_default_pos={left_default.tolist()} right_default_pos={right_default.tolist()})")

    def _spawn_target_markers(self):
        """在 env 0 的机器人旁边放 USD 可视化标记: 球 + z 轴短细圆柱.
        BODY_POS obs 是 env-local, 要把 env 0 的 world offset 加回去."""
        try:
            import omni.usd
            from pxr import UsdGeom, Sdf, Gf, Vt
        except ImportError:
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        # env 0 的 world 原点: 机器人 articulation root world 位置 (robot 在 env-local 0,0,0)
        try:
            world_pos_tensor, _ = self._task.robots.get_world_poses()
            env0 = world_pos_tensor[0].detach().cpu()
            env0_offset = (float(env0[0]), float(env0[1]), float(env0[2]))
        except Exception as e:
            print(f"[VIZ] env 0 offset query 失败 ({e}), marker 放在 world 原点")
            env0_offset = (0.0, 0.0, 0.0)
        print(f"[VIZ] env 0 world offset = {env0_offset}")

        def add_marker(prefix, pos, quat, color):
            world_pos = (float(pos[0]) + env0_offset[0],
                         float(pos[1]) + env0_offset[1],
                         float(pos[2]) + env0_offset[2])

            sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(f"{prefix}_sphere"))
            sphere.GetRadiusAttr().Set(0.03)
            sphere.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
            xf = UsdGeom.Xformable(sphere.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*world_pos))

            cyl = UsdGeom.Cylinder.Define(stage, Sdf.Path(f"{prefix}_zaxis"))
            cyl.GetRadiusAttr().Set(0.008)
            cyl.GetHeightAttr().Set(0.20)
            cyl.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
            xf2 = UsdGeom.Xformable(cyl.GetPrim())
            xf2.ClearXformOpOrder()
            xf2.AddTranslateOp().Set(Gf.Vec3d(*world_pos))
            xf2.AddOrientOp().Set(
                Gf.Quatf(float(quat[0]),
                         Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3]))))

        add_marker("/World/target_L", self._left_target, self._left_target_rot,
                   (1.0, 0.15, 0.15))
        add_marker("/World/target_R", self._right_target, self._right_target_rot,
                   (0.15, 1.0, 0.15))

    def _compute_task_errors(self, obs):
        """Returns (left_pos_err, right_pos_err, pose_err, success_mask_bool).
        Targets 已在 __init__ eager-init, 这里只读不改."""
        left_ee = self.observation_helper.get_from_obs(obs, "left_ee_pos")
        right_ee = self.observation_helper.get_from_obs(obs, "right_ee_pos")
        left_rot = self.observation_helper.get_from_obs(obs, "left_ee_rot")
        right_rot = self.observation_helper.get_from_obs(obs, "right_ee_rot")

        left_pos_err = torch.norm(left_ee - self._left_target, dim=-1)
        right_pos_err = torch.norm(right_ee - self._right_target, dim=-1)
        left_rot_n = left_rot / (torch.norm(left_rot, dim=-1, keepdim=True) + 1e-8)
        right_rot_n = right_rot / (torch.norm(right_rot, dim=-1, keepdim=True) + 1e-8)
        left_rot_err = 1.0 - torch.abs((left_rot_n * self._left_target_rot).sum(dim=-1))
        right_rot_err = 1.0 - torch.abs((right_rot_n * self._right_target_rot).sum(dim=-1))
        pose_err = left_rot_err + right_rot_err  # [0,2]

        success_mask = ((left_pos_err < self._success_pos_threshold)
                        & (right_pos_err < self._success_pos_threshold)
                        & (pose_err < self._success_pose_threshold))
        return left_pos_err, right_pos_err, pose_err, success_mask

    def is_absorbing(self, obs):
        collision = self._check_collision("arm_L", "arm_R", self._collision_threshold,
                                          dt=self._timestep)
        self._absorb_count += int(collision.sum().item())
        self._last_collision_mask = collision

        errs = self._compute_task_errors(obs)
        self._last_task_errors = errs
        return collision | errs[3]

    def reward(self, obs, action, next_obs, absorbing):
        joint_pos = self.observation_helper.get_from_obs(next_obs, "joint_pos")
        # is_absorbing 已在同一 next_obs 上算过, 直接复用
        left_pos_err, right_pos_err, pose_err, success_mask = self._last_task_errors
        success = success_mask.to(pose_err.dtype)

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

        normal = (
            -self._w_pos * (left_pos_err + right_pos_err)
            - self._w_pose * pose_err
            - self._w_joint_limit * joint_limit_norm
            + self._w_success * success
        )

        # 只对 collision 盖 r_min; success 吸收照给 normal (含 w_success 奖励)
        absorbing_r = self._r_min / (1.0 - self.info.gamma)
        collision = self._last_collision_mask
        if collision is None:
            r = normal
        else:
            r = torch.where(collision, torch.full_like(normal, absorbing_r), normal)
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
        拿到 G(q), 作为前馈 effort 施加. agent 的 velocity action 含义不变, 只是
        不再需要从零学 G(q) 这个 7-DoF 非线性映射.
        验证脚本: scripts/probe_gravity_comp.py (3s 漂移 <0.3mm vs 不补偿 0.95m).
        """
        tau_g = self._robots.get_generalized_gravity_forces(clone=False)
        self._robots.set_joint_efforts(tau_g[:, self._cj], joint_indices=self._cj)
