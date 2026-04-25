# bimanual_peghole

双臂 KUKA iiwa 在 IsaacSim 里做 peg-in-hole 的 RL 控制。最终目标是用
**Lagrangian SAC** 处理装配约束；当前工作分两段：

- **Phase 1** — 普通 SAC + reaching，把整条 pipeline (env / 控制 / reward /
  eval / 可视化 / Hydra 启动) 打通，目标是两臂末端到达固定胸前点。
- **Phase 1.5 (当前)** — 把 peg/hole 作为**视觉 only** 的子节点挂在左/右夹
  爪上，生成稳定的 `peg_tip` / `hole_entry` 参考帧，并提供纯读接口
  `get_preinsert_frames()` / `_compute_preinsert_errors()`。MDP 语义
  (obs / reward / is_absorbing) 暂时仍是 phase 1 的 reaching，
  phase 1.5 只改资产和"读"接口，不改训练语义，便于 commit-by-commit 验收。

## 环境

```bash
conda env create -f environment.yml   # 创建 safe_rl
conda activate safe_rl
```

依赖：

- `mushroom-rl==2.0.0rc1`、`torch==2.7.0`、`numpy==1.26.0`、`wandb`
  由 `environment.yml` 安装
- IsaacSim 仍需在目标机器上可用；若环境里没有，补装 `pip install isaacsim`，
  或者在有 GPU 的节点上用 Apptainer 容器 (见下)
- `hydra-core` / `hydra-submitit-launcher` 只有在用 `scripts/train_hydra.py`
  把 job 提交到 Slurm 时才必须
- 机器人 USD 资产随仓库提供：
  - `assets/usd/dual_arm_iiwa/dual_arm_iiwa.usd` — 原始 iiwa
  - `assets/usd/dual_arm_iiwa/dual_arm_iiwa_with_peghole.usda` — **当前默认**,
    在上面挂了视觉 peg/hole 与 `peg_tip` / `hole_entry` 参考帧

## 目录结构

```
envs/
  dual_arm_peg_hole_env.py   # IsaacSim 子类: 14 DoF velocity, 40 维 obs,
                             #   reaching reward, preinsert 纯读 helpers
  __init__.py                # 导出 DualArmPegHoleEnv 与默认常量

networks.py                  # SAC actor / critic MLP (input → 256 → 256 → out)

scripts/
  train_sac.py               # SAC 训练 (VectorCore, 默认 num_envs=16)
  eval_sac.py                # 加载 best_agent.msh 评估
  visualize_targets.py       # 不训练, 看 reaching 目标点 + peg/hole 预插入帧
  check_peghole_asset.py     # 对比老 USD vs 新 USDA, 验证 articulation no-op
  train_hydra.py             # Hydra + Apptainer/Slurm 启动器, 包裹 train_sac.py
  _eval_utils.py             # deterministic policy + hold-N success 指标

conf/                        # Hydra 配置 (仅 train_hydra.py 使用)
  train.yaml                 # 顶层: project 路径 + defaults
  experiment/default.yaml    # train_sac.py 的 CLI 参数 (num_envs, n_epochs ...)
  apptainer/default.yaml     # Apptainer 容器 / bind / nv
  hydra/launcher/
    slurm_apptainer.yaml     # hydra-submitit slurm launcher 默认值

assets/
  usd/
    dual_arm_iiwa/
      dual_arm_iiwa.usd               # 原始机器人
      dual_arm_iiwa_with_peghole.usda # 默认: robot + 视觉 peg/hole (no-op)
      build_peghole_usd.py            # 重新生成 composed USDA
      configuration/*.usd             # 机器人分层资产 (base/physics/robot/sensor)

results/                     # 训练产物 (best_agent.msh / SAC logs / wandb)
environment.yml
```

## 运行

### 本地直接跑

```bash
# 训练 (默认胸前对称固定点, num_envs=16, 400 epochs, wandb 启用)
python scripts/train_sac.py

# 自定义目标
python scripts/train_sac.py --left_target -0.62 -0.55 0.69 \
                            --right_target -0.62 0.38 0.74

# 关掉 wandb / 打开 IsaacSim 窗口
python scripts/train_sac.py --no_wandb --render

# 评估最优 checkpoint
python scripts/eval_sac.py                 # 带渲染
python scripts/eval_sac.py --headless

# 可视化: 看目标球 + peg(红)/hole(绿)/preinsert(黄) 实时 marker (不训练)
python scripts/visualize_targets.py
python scripts/visualize_targets.py --preinsert_offset 0.08 --duration 20

# 资产健康检查: 对比老 USD vs 新 USDA 的 articulation 数值必须完全一致
python scripts/check_peghole_asset.py --usd assets/usd/dual_arm_iiwa/dual_arm_iiwa.usd \
    > /tmp/check_old.txt
python scripts/check_peghole_asset.py --usd assets/usd/dual_arm_iiwa/dual_arm_iiwa_with_peghole.usda \
    > /tmp/check_new.txt
diff /tmp/check_old.txt /tmp/check_new.txt   # 只应有 [USD] / [PRIM_*] 行差异
```

### 通过 Hydra + Apptainer (HPC / 容器内跑)

`scripts/train_hydra.py` 读 `conf/train.yaml` 合成 override 后的配置，把它
映射回 `train_sac.py` 的 CLI 参数，然后在 Apptainer 容器里执行。支持本地
一次性跑，也支持通过 `hydra-submitit-launcher` 提交 Slurm。

```bash
# 单次本地运行 (overrides 形如 <cfg_group>.<key>=value)
python scripts/train_hydra.py \
    train.n_epochs=100 \
    train.num_envs=8

# 提交到 Slurm
python scripts/train_hydra.py \
    hydra/launcher=submitit_slurm \
    hydra.launcher.partition=gpua40 \
    hydra.launcher.gres=gpu:1

# Hydra multirun 扫参: 每一组 override 一个独立 Slurm job
python scripts/train_hydra.py --multirun \
    train.n_epochs=50 train.no_wandb=true train.num_envs=16 \
    hydra/launcher=submitit_slurm \
    hydra.launcher.partition=gpua40 hydra.launcher.gres=gpu:1 \
    hydra.launcher.timeout_min=120
```

需要本地改 `conf/train.yaml` 里 `project.host_project_root` /
`project.container_project_root` / `project.isaac_python`，以及
`conf/apptainer/default.yaml` 里 `image` 和 `binds`，与目标 HPC 一致。

### 产物位置

- `results/best_agent.msh` — 按 epoch-eval 的 J 取最大保存
- `results/SAC/` — mushroom-rl Logger 输出
- `results/wandb/` — wandb run 目录
- `hydra_output/…` — Hydra 跑的 single-run / multirun 目录 (仅 `train_hydra.py` 路径)

## Phase 1 任务设定 (当前训练语义)

- **动作**: `a ∈ [-1,1]^14` → joint velocity `rad/s`, 系数 `action_scale=0.4`,
  控制周期 `0.1s` (`timestep=0.02 × n_intermediate=5`)
- **观测 (40 维)**: 14 joint pos + 14 joint vel + 2×3 EE pos + 2×3 (T - EE) 相对目标
- **目标 (默认)**: `T_L = (-0.80, -0.40, 0.62)`, `T_R = (-0.80, 0.40, 0.62)`,
  在 `__init__` 末尾冻结到 world frame, 之后免疫 `teleport_away` 污染
- **Reward**:
  ```
  - w_pos · (||·||_L + ||·||_R)
  - w_joint_limit · joint_limit_norm     # 软极限, 进 margin 后才计
  - w_action · ||raw a||²                # 用 pre-scale action, 与 action_scale 解耦
  + w_success · 1[||·||_L < th ∧ ||·||_R < th]   # per-step dwell bonus, 不终止
  ```
  collision 是唯一 absorbing 源, reward 盖成 `r_min/(1-γ) ≈ -200`。
- **Eval success**: episode 内出现长度 `≥ hold_success_steps (default 10 ≈ 1s)`
  的连续 in-threshold 段。这种 hold-N 指标比"末步成功"对边界震荡更鲁棒。

## Phase 1.5 已交付 (视觉 + 纯读接口)

- **Commit 1 / 资产层**: `build_peghole_usd.py` 生成
  `dual_arm_iiwa_with_peghole.usda` — 通过 USD `references` 引用原始 robot,
  再用 `over` 在左/右 EE 下挂 `Peg` / `Hole` prim。**不加**
  `RigidBodyAPI` / `MassAPI` / `CollisionAPI`：几何、动力学、碰撞集合全不变，
  由 `check_peghole_asset.py` diff 验证两份 USD 的 `G(q)` / default pose
  / controlled-joint index 完全一致。
- **Commit 2 / env 接入**: env 默认加载带 peghole 的 USDA；发现 cloned
  env 里所有 `peg_tip` / `hole_entry` prim，建 XFormPrim view，暴露
  `get_preinsert_frames()` 返回 peg/hole 世界位姿、轴向，以及
  `preinsert_target = hole_entry + preinsert_offset · hole_axis` (默认 5cm)。
  `visualize_targets.py` 画出 peg_axis 箭头 (浅红)、hole_axis 箭头 (浅绿)、
  preinsert 目标球 (黄)，主循环每帧刷新。
- **Commit 3 / 误差 helper**: `_compute_preinsert_errors()` 返回
  `pos_err = ||peg_tip - preinsert_target||` 和
  `axis_err = 1 + dot(peg_axis, hole_axis) ∈ [0,2]` (0 = 轴反平行 = 理想对齐)，
  以及 `pose_success_mask`。**仍不进 reward/obs/is_absorbing**，只是
  `visualize_targets` 里打印出来对照 frames 自洽 —— commit 4+ 才会把
  MDP 语义切换过去。

## 已知约束 / 坑

- `num_envs=1` 触发 IsaacSim cloner `*` pattern bug，**至少用 2**。
- velocity 控制要求 `kp=0` (env `__init__` 里强制置零)，否则 reset 写的
  `pos_target` 会把关节钉住。
- `setup()` / `__init__` 末尾各 `world.step()` 一次，让 `BODY_POS` view 同步
  物理状态 (否则 reset 后第一步 obs stale)。
- `_simulation_pre_step` 注入 `G(q)` 重力补偿 effort，让 agent 不需要从零学
  这个非线性映射；与 velocity drive 加性叠加。
- 不要装 `mushroom_rl.rl_utils.preprocessors.StandardizationPreprocessor`，
  Welford 在 vectorized env 下 std 会衰减到 0 → obs 被 clip 成 ±10 垃圾。
- 跨 env 聚合 (`mean(dim=0)` 类) 必须在 `super().__init__()` 之后、第一次
  `step_all` 之前做，否则 inactive env 被 teleport 到 `z=50` 污染均值。
- `success` 不做 absorbing — 只给每步 dwell bonus，避免边界 hugging
  (eval rate 在 1↔0 间剧烈振荡) 的 Q-target 断崖。
- peg/hole 是**视觉 only**：不产生接触力，也不会触发 `arm_L`/`arm_R`
  collision group。如果未来要让 peg 插入产生物理反馈，需要给 Peg 补
  `CollisionAPI` 并设计 collision group 方案。

## Phase 2 (规划)

- 把 `preinsert_target_pos` + `hole_axis` 接进 obs / reward
  (commit 4+)，训练 agent 把 `peg_tip` 对齐到 `hole_entry` 外缘。
- 引入真实的 peg/hole 接触（给 Peg 加 `CollisionAPI`、调姿态约束）。
- 算法换成 **Lagrangian SAC**：把 collision / 接触力 / 姿态误差等从硬
  `r_min` 改成 cost 信号，由对偶变量自适应惩罚。
