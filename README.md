# bimanual_peghole

双臂 KUKA iiwa 在 IsaacSim 里做 peg-in-hole 的 RL 控制。最终目标是用
**Lagrangian SAC** 处理装配约束；当前 **phase 1** 用普通 SAC 训末端到达
固定点，验证整条 pipeline (env / 控制 / reward / eval / 可视化)。

## 环境

```bash
conda env create -f environment.yml   # 创建 safe_rl
conda activate safe_rl
```

依赖：

- `mushroom-rl` 2.0.0rc1 (`dev` 分支), 可编辑安装在 `~/mushroom-rl`
- IsaacSim (随 mushroom-rl 的 `IsaacSim` env class 走 `pip install isaacsim`)
- USD: `~/dual_arm_ws/usd_imports/dual_arm_iiwa/dual_arm_iiwa.usd`

## 文件

```
envs/
  dual_arm_peg_hole_env.py   # IsaacSim 子类: 14 DoF velocity, 40 维 obs, reach reward
  __init__.py                # 导出 DualArmPegHoleEnv 与默认目标常量
networks.py                  # SAC actor / critic MLP (input → 256 → 256 → out)
scripts/
  train_sac.py               # SAC 训练 (VectorCore, 默认 num_envs=16)
  eval_sac.py                # 加载 best_agent.msh 评估
  visualize_targets.py       # 不训练, 只在窗口里看默认姿态 vs 目标点
  _eval_utils.py             # deterministic policy + hold-N success 指标
environment.yml
```

## 运行

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

# 只看目标点位置 (不训练)
python scripts/visualize_targets.py
```

`results/best_agent.msh` 是按 epoch eval 的 J 取最大保存的。wandb 输出
落在 `results/wandb/`。

## Phase 1 任务设定

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

## Phase 2 (规划)

- 引入 peg/hole 几何与姿态约束
- 算法换成 Lagrangian SAC，把 collision / 接触力等从硬 `r_min` 改成 cost
  信号，由对偶变量自适应惩罚
