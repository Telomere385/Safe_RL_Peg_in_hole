# bimanual_peghole

双臂 KUKA iiwa 在 IsaacSim 里做 peg-in-hole 的 RL 控制。最终目标是用
**Lagrangian SAC** 处理装配约束；当前阶段：

- **M1' / M2 (当前)** — 普通 SAC + **stage flag 化的 preinsert 任务**。
  peg/hole 仍然是视觉-only (无 CollisionAPI)，左末端挂 peg、右末端挂
  hole；目标是把 `peg_tip` 拉到
  `preinsert_target = hole_entry + 5cm · hole_axis`，姿态对齐由 `axis_dot`
  obs + `axis_err` reward 项控制。

  同一个 env、同一个 32 维 obs、同一条 reward 骨架，stage 仅由
  `--rew_axis` 和 `--success_axis_threshold` 切换；M1'→M2a→M2b 之间
  `--load_agent` warm-start，不需要 cold start。peg/hole frame 用解析式
  (`EE_pose ⊗ const_offset`)，不依赖 XFormPrim 的 Fabric flush，headless
  训练永不 stale。

历史阶段：phase 1 的 reaching 任务 (左右 EE 各自到固定胸前点) 和早期
**31 维 strict pos-only M1** 已退役，主线不再保留兼容；如需回看见 git
历史 (`git log --grep=reaching` / `git log --grep="31.dim"`)。

## 环境

```bash
conda env create -f environment.yml   # 创建 safe_rl
conda activate safe_rl
```

依赖：

- `mushroom-rl` (dev 分支)、`torch==2.7.0`、`numpy==1.26.0`、`wandb`
  由 `environment.yml` 安装
- IsaacSim 仍需在目标机器上可用；若环境里没有，补装 `pip install isaacsim`
- 机器人 USD 资产随仓库提供：
  - `assets/usd/dual_arm_iiwa/dual_arm_iiwa.usd` — 原始 iiwa (env 不再直接用)
  - `assets/usd/dual_arm_iiwa/dual_arm_iiwa_with_peghole.usda` — **当前唯一支持**,
    在原始 iiwa 上挂了视觉 peg/hole 与 `peg_tip` / `hole_entry` 参考帧
  - 加载无 peg/hole 的旧 USD 会被 `_verify_peghole_prims_exist` 直接 raise

## 目录结构

```
envs/
  dual_arm_peg_hole_env.py   # IsaacSim 子类: 14 DoF velocity, 32 维 obs (含
                             #   axis_dot), 解析式 peg/hole frame, stage flag
                             #   化 reward (rew_axis / success_axis_threshold)
  __init__.py                # 导出 DualArmPegHoleEnv / AGENT_OBS_DIM /
                             #   DEFAULT_PREINSERT_OFFSET

networks.py                  # SAC actor / critic MLP (input → 256 → 256 → out)

scripts/
  train_sac.py               # SAC 训练 (VectorCore, 默认 num_envs=16);
                             #   --load_agent 续训, 默认清 replay
  eval_sac.py                # 加载 best_agent.msh 评估
  visualize_targets.py       # 不训练, 看 peg(红)/hole(绿)/preinsert(黄) marker
  visualize_policy.py        # 跑训好的 policy, hold-N 满足时冻结画面;
                             #   M2 时需传 --success_axis_threshold
  diagnose_m1_axis.py        # 一次性脚本: 用 31 维 M1 老 checkpoint 在 32 维
                             #   env 里跑, 统计 pos<th 时 axis_err 分布
  check_peghole_asset.py     # 资产探针 (DOF / G(q) / prim 计数, 仅支持新 USDA)
  _eval_utils.py             # deterministic policy + hold-N success 指标

assets/
  usd/
    dual_arm_iiwa/
      dual_arm_iiwa.usd               # 原始机器人 (历史保留)
      dual_arm_iiwa_with_peghole.usda # 当前唯一加载: robot + 视觉 peg/hole
      build_peghole_usd.py            # 重新生成 composed USDA
      configuration/*.usd             # 机器人分层资产

results/                     # 训练产物 (best_agent.msh / SAC logs / wandb)
environment.yml
```

## Stage flag 设计

```
                          rew_axis    success_axis_threshold
M1'  pos-only             0.0         inf                       (32 维 baseline)
M2a  pos + 粗轴对齐        1.0         0.5                       (≈ ±60° 锥)
M2b  pos + 紧轴对齐        1.0         0.2                       (≈ ±37° 锥)
```

`success_axis_threshold = inf` 时 success_mask 退化成 pos-only，`-w_axis * axis_err`
在 `rew_axis = 0` 时数学上为 0，所以 M1' 和老 strict-pos-only 在 reward 量级上等价。
M2a/M2b 通过 `--load_agent` 续训上一个 stage 的 checkpoint。

## 运行

```bash
# 一次性诊断: 用 31 维 M1 老 checkpoint 看 pos<th 时 axis_err 分布
# (脚本自动把 obs 截到 31 维, 仅在还没训 M1' 时使用)
python scripts/diagnose_m1_axis.py --headless --num_envs 16 --n_episodes 64

# M1': 32 维 baseline, axis 项关闭 (rew_axis 默认 0, axis_th 默认 inf)
python scripts/train_sac.py --no_wandb --n_epochs 100 \
    --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50
cp results/best_agent.msh results/best_agent_M1p_32dim_pos10cm.msh

# M2a: 从 M1' warm-start, 加 axis reward (粗对齐)
# 默认会清空 replay buffer (旧 reward 数据不能带过来), 加 --keep_replay 可保留
python scripts/train_sac.py --no_wandb --n_epochs 150 \
    --load_agent results/best_agent_M1p_32dim_pos10cm.msh \
    --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50 \
    --rew_axis 1.0 --success_axis_threshold 0.5
cp results/best_agent.msh results/best_agent_M2a_axis05.msh

# M2b: 从 M2a 收紧到 ±37° 锥
python scripts/train_sac.py --no_wandb --n_epochs 100 \
    --load_agent results/best_agent_M2a_axis05.msh \
    --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50 \
    --rew_axis 1.0 --success_axis_threshold 0.2

# 评估 (CLI 必须与训练时一致, 否则 success 触发条件不同, 数字不可比)
python scripts/eval_sac.py --headless --num_envs 16 --n_episodes 64 \
    --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50 \
    --rew_axis 1.0 --success_axis_threshold 0.5

# 可视化 marker (不训练)
python scripts/visualize_targets.py
python scripts/visualize_targets.py --preinsert_offset 0.08 --duration 20
python scripts/visualize_targets.py --n_resets 30   # 看 reset 分布

# 可视化 policy 在 hold-N 时冻结 (M2 必须传 --success_axis_threshold!)
python scripts/visualize_policy.py \
    --preinsert_success_pos_threshold 0.10 \
    --rew_axis 1.0 --success_axis_threshold 0.5

# 资产健康检查 (仅支持新 USDA, env 现在 fail-fast 拒绝旧 USD)
python scripts/check_peghole_asset.py \
    --usd assets/usd/dual_arm_iiwa/dual_arm_iiwa_with_peghole.usda
```

### 产物位置

- `results/best_agent.msh` — 按 epoch-eval 的 J / hold-score 取最大保存
- `results/SAC/` — mushroom-rl Logger 输出
- `results/wandb/` — wandb run 目录

## 任务设定

- **动作**: `a ∈ [-1,1]^14` → joint velocity `rad/s`, 系数 `action_scale=0.4`,
  控制周期 `0.1s` (`timestep=0.02 × n_intermediate=5`)
- **观测 (32 维)**:
  ```
  joint_pos[14] + joint_vel[14] + pos_vec[3] + axis_dot[1]
  pos_vec  = peg_tip - preinsert_target          # env-local
  axis_dot = dot(peg_axis, hole_axis) ∈ [-1,+1]  # -1 = 完美轴反平行
  ```
  axis_dot 一维标量 (而非完整 peg_axis/hole_axis 6 维): 一维已经把"对齐到什么
  程度"的梯度信号给出来; 完整向量与 EE quat 强冗余, 徒增维度。radial / axial
  分量留到 M3+ 再放。
- **Peg / Hole frame** (env-local): 解析式
  ```
  peg_tip   = LeftEE_pos  + R(LeftEE_quat)  · (-0.0055, -0.0175, 0.125)
  hole_entry= RightEE_pos + R(RightEE_quat) · (-0.0055, -0.015,  0.125)
  hole_axis =                R(RightEE_quat) · (0, -1, 0)
  peg_axis  =                R(LeftEE_quat)  · (0, +1, 0)
  ```
  常量来自 `build_peghole_usd.py` 的 `PART_X / PART_Z + R_x(+90°)` 推导。
  这绕过 XFormPrim → Fabric flush 链路, headless / `render=False` 也保证 fresh。
- **Reward (统一骨架)**:
  ```
  - w_pos     · pos_err                          # ||peg_tip - preinsert_target||
  - w_axis    · axis_err                         # 1 + dot(peg_axis, hole_axis), 0 = ideal
  - w_joint_limit · joint_limit_norm             # 软极限, 进 margin 后才计
  - w_action  · ||raw a||²                       # pre-scale action, 与 action_scale 解耦
  + w_success · 1[success]                       # per-step dwell bonus, 不终止
  success = (pos_err < pos_th) ∧ (axis_err < axis_th)
  ```
  `rew_axis = 0` 时 axis 项消失 (M1'); `success_axis_threshold = inf` 时
  success 退化为 pos-only。collision (`arm_L` vs `arm_R` 自碰撞) 是唯一硬
  absorbing, reward 盖成 `r_min/(1-γ) ≈ -200`。
- **Eval success**: episode 内出现长度 `≥ hold_success_steps (default 10 ≈ 1s)`
  的连续 in-threshold 段。in-threshold 与 reward 用同一个 success_mask, 所以
  M2 的 hold-N 同时要求 pos 和 axis 都进。

## 已知约束 / 坑

- `num_envs=1` 触发 IsaacSim cloner `*` pattern bug，**至少用 2**。
- velocity 控制要求 `kp=0` (env `__init__` 里强制置零)，否则 reset 写的
  `pos_target` 会把关节钉住。
- `setup()` / `__init__` 末尾各 `world.step()` 一次，让 BODY_POS / BODY_ROT view
  同步物理状态 (否则 reset 后第一步 obs stale)。
- `_simulation_pre_step` 注入 `G(q)` 重力补偿 effort，让 agent 不需要从零学
  这个非线性映射；与 velocity drive 加性叠加。
- 不要装 `mushroom_rl.rl_utils.preprocessors.StandardizationPreprocessor`，
  Welford 在 vectorized env 下 std 会衰减到 0 → obs 被 clip 成 ±10 垃圾。
- `train_sac.py` / `eval_sac.py` 的 eval episode 数要与 `num_envs` 对齐
  (默认直接取 `num_envs`)。
- `success` 不做 absorbing — 只给每步 dwell bonus，避免边界 hugging 的
  Q-target 断崖。如要切 hold-N absorbing, 给 `--terminal_hold_bonus > 0`。
- **stage 切换 (M1'→M2a→M2b) 默认清空 replay buffer**: 旧 transitions 的
  reward 标签按旧 reward 算的, 留下来会拖 critic. 显式 `--keep_replay`
  可保留 (一般只在调试时用)。
- **31 维老 M1 checkpoint 不能 warm-start 到 32 维 env**: actor 输入层
  shape 不匹配, `Agent.load` 后 forward 直接抛错. 必须重训 M1'
  (`results/best_agent_M1_31dim_pos10cm.msh` 仅供 `diagnose_m1_axis.py`
  一次性诊断使用)。
- peg/hole 是**视觉 only**：不产生接触力，也不会触发 `arm_L`/`arm_R`
  collision group。M3 才会给 peg 加 `CollisionAPI` 并设计 collision group。

## 后续阶段 (规划)

- **M3** — 加相对速度项 + 几何插入 (`axial_dist` 推进 + radial 控制 + analytic
  illegal-insertion penalty), 给 peg 加真实 CollisionAPI。obs 会再加几维
  (radial/axial), 此时网络输入 shape 变化, 又一次 cold start。
- **M4** — Lagrangian SAC: 把 collision force / 接触力作为 cost, 由对偶变量
  自适应惩罚; 此时 cost 才有真正的物理意义。
