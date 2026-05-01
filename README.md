# bimanual_peghole

双臂 KUKA iiwa 在 IsaacSim 里做 peg-in-hole 的 RL 控制. 整体路线分四步:

```
Step 1   位置接近        peg_tip 进入预定义的相对位置区域
Step 2   位置 + 姿态对齐  peg_tip 学到完整 "preinsert pose" (← 当前阶段)
Step 3   预插入 + 低速接近 在 Step 2 基础上加相对速度约束
Step 4   peg-in-hole     横向对齐 / 轴向推进 / 倾斜最小 / 接触力惩罚
                         (Lagrangian SAC, peg/hole 加 CollisionAPI)
```

当前主线只到 **Step 2**: peg/hole 是视觉-only USD over (无 RigidBodyAPI / MassAPI /
CollisionAPI), 左末端挂 peg、右末端挂 hole, 学 `peg_tip` → `preinsert_target =
hole_entry + 5cm · hole_axis`. Step 2 内部用 stage flag 进一步切档:

```
              rew_axis    success_axis_threshold     语义
M1'  pos-only  0.0          inf                       (32 维 baseline, 等价老 strict-pos-only)
M2a  pos+粗轴  1.0          0.5                       (≈ ±60° 锥)
M2b  pos+紧轴  1.0          0.2                       (≈ ±37° 锥)
```

`success_axis_threshold = inf` 时 success_mask 退化成 pos-only, `-w_axis · axis_err`
在 `rew_axis = 0` 时为 0 — 同一个 env、同一个 32 维 obs、同一条 reward 骨架,
M1' → M2a → M2b 之间 `--load_agent` warm-start 不需要重训.

peg/hole frame 用解析式 (`EE_pose ⊗ const_offset`) 不依赖 XFormPrim 的 Fabric flush,
headless 训练永不 stale.

## 环境

```bash
conda env create -f environment.yml   # 创建 safe_rl
conda activate safe_rl
```

依赖:
- `mushroom-rl` (dev 分支), `torch==2.7.0`, `numpy==1.26.0`, `wandb` 由 `environment.yml` 安装
- 目标机器需要 IsaacSim; 没有就 `pip install isaacsim`
- 机器人 USD 资产随仓库提供:
  - `assets/usd/dual_arm_iiwa/dual_arm_iiwa.usd` — 原始 iiwa (env 不直接加载, 历史保留)
  - `assets/usd/dual_arm_iiwa/dual_arm_iiwa_with_peghole.usda` — **当前唯一支持**,
    在原始 iiwa 上挂了视觉 peg/hole 与 `peg_tip` / `hole_entry` 参考帧
  - 加载无 peg/hole 的旧 USD 会被 `_verify_peghole_prims_exist` 直接 raise

## 目录结构

```
envs/
  dual_arm_peg_hole_env.py   # IsaacSim 子类: 14 DoF velocity, 32 维 obs (含 axis_dot),
                             # 解析式 peg/hole frame, stage flag 化 reward,
                             # PhysX + sphere-proxy 双信号自碰撞兜底
  __init__.py                # 导出 DualArmPegHoleEnv / AGENT_OBS_DIM /
                             # DEFAULT_PREINSERT_OFFSET

networks.py                  # SAC actor / critic MLP (input → 256 → 256 → out)

scripts/
  train_sac.py               # SAC 训练 (VectorCore, 默认 num_envs=16); --load_agent 续训
  eval_sac.py                # 加载 best_agent.msh 评估
  visualize_targets.py       # 不训练, 看 peg(红)/hole(绿)/preinsert(黄) marker
  visualize_policy.py        # 跑训好的 policy, hold-N 满足时冻结画面
  _eval_utils.py             # deterministic policy + hold-N success 指标
  archive/
    check_peghole_asset.py   # USD 资产 articulation no-op 校验 (改 USD 时再跑)
    README.md

assets/
  usd/dual_arm_iiwa/
    dual_arm_iiwa.usd               # 原始机器人 (历史保留)
    dual_arm_iiwa_with_peghole.usda # 当前唯一加载: robot + 视觉 peg/hole
    build_peghole_usd.py            # 重新生成 composed USDA
    configuration/*.usd             # 机器人分层资产

results/                     # 训练产物 (best_agent.msh / SAC logs / wandb), gitignore
environment.yml
```

## 任务设定

- **动作**: `a ∈ [-1,1]^14` → joint velocity `rad/s`, 系数 `action_scale=0.4`,
  控制周期 `0.1s` (`timestep=0.02 × n_intermediate=5`).
- **观测 (32 维)**:
  ```
  joint_pos[14] + joint_vel[14] + pos_vec[3] + axis_dot[1]
  pos_vec  = peg_tip - preinsert_target           # env-local
  axis_dot = dot(peg_axis, hole_axis) ∈ [-1,+1]   # -1 = 完美轴反平行
  ```
  axis_dot 一维标量 (而非完整 peg_axis/hole_axis 6 维): 一维已把 "对齐到什么程度"
  的梯度给出来; 完整向量与 EE quat 强冗余, 徒增维度. radial / axial 分量留到 Step 3+.
- **Peg / Hole frame** (env-local, 解析式):
  ```
  peg_tip   = LeftEE_pos  + R(LeftEE_quat)  · (-0.0055, -0.0175, 0.125)
  hole_entry= RightEE_pos + R(RightEE_quat) · (-0.0055, -0.015,  0.125)
  peg_axis  =                R(LeftEE_quat)  · (0, +1, 0)
  hole_axis =                R(RightEE_quat) · (0, -1, 0)
  ```
  常量来自 `build_peghole_usd.py` 的 `PART_X / PART_Z + R_x(+90°)` 推导.
  绕过 XFormPrim → Fabric flush 链路, headless / `render=False` 也保证 fresh.

### Reward (统一骨架)

```
- w_pos     · pos_err                          # ||peg_tip - preinsert_target||
- w_axis    · axis_err                         # 1 + dot(peg_axis, hole_axis), 0 = ideal
- w_joint_limit · joint_limit_norm             # 软极限, 进 margin 后才计 (default 0.02)
- w_action  · ||raw a||²                       # pre-scale action, 与 action_scale 解耦 (default 0.005)
+ w_success · 1[success]                       # per-step dwell bonus, 不终止 (default 2.0)
success = (pos_err < pos_th) ∧ (axis_err < axis_th)
```

- `rew_axis = 0` 时 axis 项消失 (M1'); `success_axis_threshold = inf` 时 success
  退化为 pos-only.
- success 本身**不**做 absorbing — 只给每步 dwell bonus, 避免边界 hugging 的
  Q-target 断崖. 要切 hold-N absorbing, 给 `--terminal_hold_bonus > 0`.

### 终止 (absorbing)

- **自碰撞 — 双信号 OR**, 任一触发即吸收, reward 盖成 `r_min/(1-γ) ≈ -200`:
  - PhysX 接触力: `arm_L vs arm_R > collision_force_threshold` (default 10 N)
  - **Sphere-proxy clearance**: 双臂各 19 球 (8 关节球 + 7 中点球 + 4 EE 球),
    球心两两算 `clearance = ||c_L - c_R|| - r_L - r_R`, 取 min.
    `min_clearance < clearance_hard` 即触发 (default `clearance_hard = 0.0`,
    球壳一接触就算碰撞). PhysX 力检测在 1cm-5cm 边缘失明, sphere proxy 是几何兜底.
- **hold-N**: success 连续 N 步即软 absorbing + `terminal_hold_bonus`. `bonus=0`
  时机制关闭 (baseline 行为).

### Eval success

Episode 内出现长度 `≥ hold_success_steps` (default 10 ≈ 1s) 的连续 in-threshold
段. in-threshold 与 reward 用同一个 `success_mask`, 所以 M2 的 hold-N 同时要求
pos 和 axis 都进.

## 训练 / 评估命令

三阶段 curriculum, 每一阶段从上一阶段 checkpoint warm-start:

```bash
# Step 1 / M1': 32 维 baseline (axis 项关闭)
python scripts/train_sac.py --no_wandb --n_epochs 100 \
    --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50
cp results/best_agent.msh results/best_agent_M1p_32dim_pos10cm.msh

# Step 2a / M2a: 加 axis reward (粗对齐, ±60° 锥)
python scripts/train_sac.py --no_wandb --n_epochs 150 \
    --load_agent results/best_agent_M1p_32dim_pos10cm.msh \
    --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50 \
    --rew_axis 1.0 --success_axis_threshold 0.5
cp results/best_agent.msh results/best_agent_M2a_axis05.msh

# Step 2b / M2b: 收紧到 ±37° 锥
python scripts/train_sac.py --no_wandb --n_epochs 100 \
    --load_agent results/best_agent_M2a_axis05.msh \
    --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50 \
    --rew_axis 1.0 --success_axis_threshold 0.2

# 评估 (CLI 必须与训练一致, 否则 success 触发条件不同, 数字不可比)
python scripts/eval_sac.py --headless --num_envs 16 --n_episodes 64 \
    --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50 \
    --rew_axis 1.0 --success_axis_threshold 0.5
```

`--load_agent` 默认会清空 replay buffer (旧 reward 标签不能带过来), 加 `--keep_replay`
保留 (一般只在 debug 时用).

### 可视化

```bash
# 看 peg(红)/hole(绿)/preinsert(黄) marker, 不训练
python scripts/visualize_targets.py
python scripts/visualize_targets.py --preinsert_offset 0.08 --duration 20
python scripts/visualize_targets.py --n_resets 30   # 看 reset 分布

# 跑训好的 policy, hold-N 满足时冻结画面 (M2 必须传 --success_axis_threshold!)
python scripts/visualize_policy.py \
    --preinsert_success_pos_threshold 0.10 \
    --rew_axis 1.0 --success_axis_threshold 0.5
```

### 资产校验 (改 USD 时跑)

```bash
python scripts/archive/check_peghole_asset.py \
    --usd assets/usd/dual_arm_iiwa/dual_arm_iiwa_with_peghole.usda
```

## CLI 参数全表

### `train_sac.py` / `eval_sac.py` 共有 (env passthrough)

| 参数 | 默认 | 说明 |
|---|---|---|
| `--num_envs` | 16 (eval) / 16 (train) | vectorized 环境数, **必须 ≥ 2** (cloner bug) |
| `--initial_joint_noise` | env=0.1 | reset 时关节角的 ±范围 |
| `--preinsert_success_pos_threshold` | env=0.10 | success 的 pos_err 阈值 (m) |
| `--preinsert_offset` | env=0.05 | hole_entry 沿 hole_axis 的 preinsert 距离 (m) |
| `--rew_axis` | env=0.0 | axis_err 权重. M1' 用 0, M2a/M2b 用 1.0 |
| `--success_axis_threshold` | env=inf | axis_err 的 success 阈值. M1' inf, M2a 0.5, M2b 0.2 |
| `--terminal_hold_bonus` | env=0.0 | hold-N 软 absorbing bonus, =0 关闭机制 |
| `--hold_success_steps` | 10 | 连续 N 步 in-threshold 才算 success (≈1s) |
| `--clearance_hard` | env=0.0 | sphere proxy 自碰撞兜底阈值. 0 = 球壳触碰即碰撞, `-inf` = 关闭 |
| `--proxy_arm_radius` | env=0.06 | arm sphere proxy 半径 (m) |
| `--proxy_ee_radius` | env=0.03 | EE sphere proxy 半径 (m) |

### `train_sac.py` 独有

| 参数 | 默认 | 说明 |
|---|---|---|
| `--n_epochs` | 400 | 训练 epoch 数 |
| `--n_steps_per_epoch` | 1024 | 每 epoch 收集的 env-step 总数 |
| `--n_steps_per_fit` | num_envs | 两次 fit 之间收集的 env-step 总数 (默认 = 1 个 vector-step) |
| `--utd` | n_steps_per_fit | 每个 fit block 的总梯度步, 默认让 true UTD ≈ 1 |
| `--lr_actor` / `--lr_critic` / `--lr_alpha` | 3e-4 | Adam 学习率 |
| `--alpha_max` | 0.2 | SAC alpha 上限 (高 dim 动作下抑制 entropy 压过任务 reward) |
| `--target_entropy` | -act_dim | 默认 SAC 标准设置, 14 维动作 → -14 |
| `--rew_action` | env=0.005 | 动作 L2 惩罚权重 |
| `--rew_success` | env=2.0 | per-step success bonus |
| `--load_agent` | None | warm-start checkpoint 路径 |
| `--keep_replay` | False | warm-start 时保留旧 replay buffer (默认清空) |
| `--n_eval_episodes` | num_envs | 每 epoch 末 eval 的 episode 数, 必须能被 num_envs 整除 |
| `--render` | False | 打开 IsaacSim 窗口 (无此 flag 即 headless) |
| `--seed` | 42 | torch + numpy seed |
| `--no_wandb` | False | 关 wandb logging |
| `--wandb_project` / `--wandb_run_name` | bimanual_peghole / None | wandb 标识 |

### `eval_sac.py` 独有

| 参数 | 默认 | 说明 |
|---|---|---|
| `--agent_path` | results/best_agent.msh | 要 load 的 .msh checkpoint |
| `--n_episodes` | num_envs | eval episode 数, 必须能被 num_envs 整除 |
| `--headless` | False | 关 IsaacSim 窗口 |
| `--stochastic` | False | 用采样策略, 默认 deterministic tanh(μ) |

### `visualize_*.py`

| 脚本 | 关键参数 |
|---|---|
| `visualize_targets.py` | `--num_envs` (≥2) `--viz_env_idx` `--initial_joint_noise` `--preinsert_success_pos_threshold` `--success_axis_threshold` `--preinsert_offset` `--duration` `--n_resets` |
| `visualize_policy.py` | `--agent_path` `--num_envs` `--viz_env_idx` `--n_episodes` `--freeze_seconds` `--hold_steps` `--rew_axis` `--success_axis_threshold` `--preinsert_offset` `--initial_joint_noise` `--stochastic` |

### 产物位置

- `results/best_agent.msh` — 按 epoch-eval 的 J / hold-score 取最大保存
- `results/SAC/` — mushroom-rl Logger 输出
- `results/wandb/` — wandb run 目录

## 已知约束 / 坑

- `num_envs=1` 触发 IsaacSim cloner `*` pattern bug, **至少用 2**.
- velocity 控制要求 `kp=0` (env `__init__` 里强制置零), 否则 reset 写的
  `pos_target` 会把关节钉住.
- `setup()` / `__init__` 末尾各 `world.step()` 一次, 让 BODY_POS / BODY_ROT view
  同步物理状态 (否则 reset 后第一步 obs stale).
- `_simulation_pre_step` 注入 `G(q)` 重力补偿 effort, 让 agent 不需要从零学这个
  非线性映射; 与 velocity drive 加性叠加.
- 不要装 `mushroom_rl.rl_utils.preprocessors.StandardizationPreprocessor`,
  Welford 在 vectorized env 下 std 会衰减到 0 → obs 被 clip 成 ±10 垃圾.
- `train_sac.py` / `eval_sac.py` 的 eval episode 数要能被 `num_envs` 整除
  (默认直接取 `num_envs`).
- success 不做 absorbing — 只给每步 dwell bonus, 避免边界 hugging Q-target 断崖.
  要切 hold-N absorbing, `--terminal_hold_bonus > 0`.
- **stage 切换 (M1' → M2a → M2b) 默认清空 replay buffer**: 旧 transitions 的 reward
  标签按旧 reward 算的, 留下来会拖 critic. 显式 `--keep_replay` 可保留 (debug 用).
- peg/hole 是**视觉 only**: 不产生接触力, 也不会触发 `arm_L`/`arm_R` collision
  group. Step 4 才会给 peg 加 `CollisionAPI` 并设计 collision group.
- **sphere proxy 默认开启**, `clearance_hard=0.0` 球壳触碰即终止. 探索阶段如果
  trip 太频繁拉爆 reset, 把它放松到 `-0.01` 或 `-0.02`; 极端调试时 `--clearance_hard -inf`
  关掉, 退回纯 PhysX.

## 后续阶段 (规划)

- **Step 3** — preinsert 后加相对速度约束 (低速接近), peg 还是视觉-only.
  obs 可能再加 axial/radial 几维, 网络输入 shape 变化时 cold start.
- **Step 4** — peg-in-hole 真插入. 给 peg 加 `CollisionAPI`, reward 改成
  横向对齐 + 轴向推进 + 倾斜最小 + 非法接触/大接触力惩罚. 用 Lagrangian SAC
  把接触力作为 cost, 由对偶变量自适应惩罚, cost 才有真正物理意义.
