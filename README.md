# bimanual_peghole

双臂 KUKA iiwa 在 IsaacSim 中做 peg-in-hole 的 SAC 训练 (mushroom-rl 2.0).

主线分三个 stage:

- **Stage 1 — preinsert 位置**: 学会把 `peg_tip` 推到 `hole_entry - 0.05m * hole_axis`
  附近. 不要求轴对齐. **已通过** (5-seed sweep).
- **Stage 2 — preinsert 位置 + 轴对齐**: 在 Stage 1 actor 上 warm-start, 加 axis
  alignment reward. **已通过** (oneshot 配方, seed 0, 2026-05-09).
- **Stage 3 — 真实插入**: peg 进 hole, 用 `axial / radial` 几何量定义 success.
  **下一步**. collider 已在 USD asset 里, 待做的是 insertion obs/reward/absorbing
  + 启用 EE self-collision exclude.

当前 setup (post-Davide / collision-aware) 跟 2026-05-08 之前有几个根本性差别,
旧的 M2 / S1_axisresid_repro / S2p* checkpoint 已经删除, **不能拿来对比**:

1. `networks.py` 用 Davide 建议的 `xavier_uniform_(gain = calculate_gain(act)/10)`
   + `bias=0`. ReLU 隐藏层 gain ≈ 0.1414, linear 输出层 gain = 0.1.
2. 双臂 sphere-proxy 自碰撞 hard absorbing 已合进 env 主线 reward.
3. hold-N (success-on-dwell) absorbing 默认关闭 (`terminal_hold_bonus=0`).
   episode 只走 horizon, `_consecutive_inthresh` 仅作 eval 指标.
4. **peg/hole 已带 collision proxy** (USD asset
   `assets/usd/dual_arm_iiwa/dual_arm_iiwa_with_peghole.usda` line 47/141):
   peg = invisible Cylinder, hole wall = invisible Cube ring, 都有
   `PhysicsCollisionAPI + PhysxCollisionAPI`. **但故意没有 RigidBodyAPI/MassAPI**
   (`build_peghole_usd.py:8-15`): 它们是 EE link 的附属 collision shape, 不是
   独立刚体, 不进 articulation 计算 — 不要建议加 RigidBody 重做物理. USD 是
   ground truth; `envs/dual_arm_peg_hole_env.py:55-60` docstring 已同步到这个
   事实. Stage 1/2 的 reward/success 不读 contact 或 insertion progress;
   Stage 3 才把这些信号纳入 obs/reward/absorbing.

---

## Stage 1 (已通过)

### Verified recipe

| 参数 | 值 | 备注 |
|---|---|---|
| `lr_actor` | 1e-4 | |
| `lr_critic` | 3e-4 | |
| `lr_alpha` | 3e-4 | |
| `target_entropy` | -7 | 默认 -14 太松 (14-DoF) |
| `alpha_max` | 0.05 | 防 entropy 项淹没精度信号 |
| `n_epochs` | 50 | |
| `n_steps_per_epoch` | 1024 | |
| `n_steps_per_fit` | 16 | |
| `num_envs` | 16 | |
| `n_eval_episodes` | 16 | |
| `--use_axis_resid_obs` | true | 34 维 obs |
| `preinsert_success_pos_threshold` | 0.10m | |
| `success_axis_threshold` | inf | Stage 1 = pos only |
| `rew_axis` | 0.0 | Stage 1 关 |
| `rew_home` | 0.0005 | weak home tie-breaker |
| `terminal_hold_bonus` | 0 | hold-N absorbing **关** |
| `hold_success_steps` | 10 | 仅 eval 指标用 |

`networks.py` 见 `_GAIN_RELU` / `_GAIN_LINEAR` 定义.

### 5-seed sweep 结果 (2026-05-08)

每个 seed 单独跑 50 ep, single seed ≈ 12 min, 总 ~60 min.
`peak_max_hold` 是 50 epoch 内真实最高值 (parse 自 output.log), **不要看 wandb
summary 的 `best_hold_max_hold_mean`** — 它的语义是"首次达到 best_hold_rate
那个 epoch 的 max_hold_mean", 会系统性低估慢热 seed (例如 seed 7 真实 peak
15.1, summary 显示 8.6).

| seed | best_J | best_hold_rate | peak_max_hold | sphere_absorb | first ep <0.30m | first ep ≥10 hold |
|---|---:|---:|---:|---:|---:|---:|
| 0   | **53.01** | **1.000** | **58.8** | 39 | 3  | 18 |
| 42  | 7.45  | 0.875 | 40.6 | 60 | 5  | 5  |
| 999 | 4.96  | 0.812 | 20.5 | 54 | 36 | 13 |
| 123 | -4.38 | 0.688 | 16.6 | **78** | 3  | 3  |
| 7   | -12.14| 0.375 | 15.1 | **73** | 27 | 27 |

**结论:**
- 5/5 seeds 都学到了 (peak max_hold 全部 ≥ 15.1, 没完全失败)
- 3/5 达到 hold_rate ≥ 0.8
- seed 0 完全 solve task: hold_rate=1.0, max_hold=58.8 步 (5.88s 稳定 hold)
- **不是 outlier 模式, 是 continuous spectrum**: peak max_hold 58.8 → 15.1
  平滑下降. sphere absorb 与突破延迟正相关 (seed 7/123 sphere absorb 最多 →
  突破延迟最大). 50 ep 可能不够慢热 seed 收敛, Stage 2 直接 100 ep.
- post-peak 塌方 (final J 普遍负, final pos_err 0.3-0.6m): 选 best_hold
  ckpt 部署, 不要 final.

### 推荐部署 / warm-start checkpoint

```text
results/S1_B_davideinit_seed0_best_agent.msh
```

(epoch 38, best_J=53.0, hold_rate=1.0, max_hold=58.8)

`best_hold.msh` (epoch 37 或 38 tie-break) 跟 best_agent 行为接近, 都可作 warm-start.

### Stage 1 训练命令

```bash
cd ~/bimanual_peghole && conda activate safe_rl

python scripts/train_sac.py \
  --num_envs 16 --n_epochs 50 --n_steps_per_epoch 1024 --n_steps_per_fit 16 --n_eval_episodes 16 \
  --use_axis_resid_obs \
  --preinsert_success_pos_threshold 0.10 --success_axis_threshold inf \
  --rew_axis 0.0 --rew_home 0.0005 \
  --lr_actor 1e-4 --alpha_max 0.05 --target_entropy -7 \
  --terminal_hold_bonus 0 --hold_success_steps 10 \
  --seed 0 \
  --wandb_run_name S1_B_davideinit_seed0
```

训练结束**立即**复制 ckpt 防止下次训练覆盖:

```bash
cp results/best_agent.msh results/S1_B_davideinit_seed0_best_agent.msh
cp results/best_hold.msh   results/S1_B_davideinit_seed0_best_hold.msh
cp results/final_agent.msh results/S1_B_davideinit_seed0_final.msh
```

### Stage 1 eval

```bash
python scripts/eval_sac.py \
  --headless --num_envs 16 --n_episodes 128 \
  --agent_path results/S1_B_davideinit_seed0_best_agent.msh \
  --use_axis_resid_obs \
  --preinsert_success_pos_threshold 0.10 \
  --success_axis_threshold inf \
  --rew_axis 0.0 --rew_home 0.0005 \
  --terminal_hold_bonus 0
```

### Stage 1 可视化

```bash
python scripts/visualize_policy.py \
  --agent_path results/S1_B_davideinit_seed0_best_agent.msh \
  --num_envs 4 --viz_env_idx 0 --n_episodes 8 --freeze_seconds 30 \
  --use_axis_resid_obs \
  --preinsert_success_pos_threshold 0.10 \
  --success_axis_threshold inf \
  --rew_axis 0.0 --rew_home 0.0005
```

---

## Stage 2 (已通过, oneshot 配方)

### 目标

在 Stage 1 actor 上 warm-start, 学会 peg/hole 轴对齐, 保留 Stage 1 preinsert 位置能力.

### Verified recipe (oneshot, seed 0, 2026-05-09)

走过 3 次实验. 关键 learning: **`success_axis_threshold` 是 reward 函数开关, 不是
metric**. 训练阈值过严 (0.20 / 0.30) cliff bonus 触发率 < 1%, reward landscape
跟"完全没 cliff" 等价 — exp 1 (axis_th=0.20) 跟 exp 2 (axis_th=0.30) 的 J trajectory
4 位小数完全一致 (deterministic seed). **必须把训练阈值调到 cliff 能密集触发的水平**
(实测 0.40 → 训练时 hold≈80%), agent 才有 sustain signal 学.

| 参数 | 值 | 跟 Stage 1 的差别 |
|---|---|---|
| `--load_agent` | `S1_B_davideinit_seed0_best_agent.msh` | warm-start |
| `--actor_only_warmstart` | true | critic / replay 冷启动 |
| `--critic_warmup_transitions` | 50000 | 给冷 critic ~50 ep 单独学 |
| `success_axis_threshold` | **0.40** | **不是 final goal**, 是 cliff 触发阈值 |
| `axis_gate_radius` | 0.40 | 远处不施 axis 压力 |
| `rew_axis` | **1.0** | Stage 1 是 0.0; oneshot 比常规 0.5 加倍, 推 axis mean 收紧 |
| `rew_pos_success` | 1.0 | Stage 1 anchor (防 actor 忘位置) |
| `rew_success` | **4.0** | cliff bonus, 比常规 2.0 加倍 (sustain signal 更强) |
| `rew_home` | 0.0005 | tie-breaker |
| `home_weights` | 1,1,1,1,0.75,0.5,0.5 | 放松腕部 |
| `lr_actor` | **5e-5** | Stage 1 是 1e-4; 减半防 epoch 86 塌方 |
| `alpha_max` | **0.05** | Stage 1 水平; 0.10 会塌方 |
| `target_entropy` | **-10** | Stage 1 是 -7; 更负更 deterministic |
| `n_epochs` | 200 | 配合 lr 减半时间预算 2x |
| 其余 | 沿用 Stage 1 | `terminal_hold_bonus=0`, init, num_envs=16 |

### 实验对比 (2026-05-09 全部 seed=0)

| | exp 1 (axis_th=0.20) | exp 2 (axis_th=0.30) | exp 3 oneshot (axis_th=0.40) |
|---|---:|---:|---:|
| n_epochs | 100 | 100 | **200** |
| rew_axis / rew_success | 0.5 / 2.0 | 0.5 / 2.0 | **1.0 / 4.0** |
| lr_actor / alpha_max / te | 1e-4 / 0.10 / -7 | 1e-4 / 0.10 / -7 | **5e-5 / 0.05 / -10** |
| best epoch | 85 | 94 | **194** |
| best_J | 26.7 | 22.0 | **189** |
| hold@train_th | 0 | 0.25 | **1.00** |
| max_hold_mean | 0 | 5.5 步 | **105.7 步** |
| axis_in_pos_mean | 0.63 | 0.63 | **0.41** |
| axis_in_pos_min (floor) | 0.27 | 0.27 | 0.29 |
| pos_success | 0.90 | 0.80 | 0.89 |
| catastrophic collapse | ep 86 J→-65 | ep 86 J→-65 | **无 (best ep 194 J=189; ep 196 J=144, ep 200 final J=164: 后段有抖动但无 -65 级 collapse, hold 全程保持 1.0)** |

oneshot 配方治了三件事: (a) catastrophic collapse (lr 5e-5 + alpha 0.05 让
200 ep 没再出现 ep 86 那种 J→-65 突崩, 后段虽有抖动但 hold 保持 1.0), (b) cliff
触发过稀问题 (axis_th 0.40 让 cliff 密集触发), (c) axis 中心收紧
(rew_axis/rew_success 加倍). 代价:
pos_err +11% (0.097m → 0.108m). **best 在 epoch 194, 不要直接用 final**.

### 关键 limit (cliff reward 结构性, 不是 bug)

1. **axis floor 没下移** (0.27 → 0.29). cliff reward 训 distribution 中心,
   不训左尾. 同 ckpt eval @ axis_th=0.30 hold_rate=3% (跟 exp 1 ckpt 完全相同).
   agent 学的是 "sustain near training threshold", 不会自发越过阈值收紧.
2. **Stage 2 axis 数字没物理意义** (Stage 1/2 reward 不利用 contact;
   collider 在 USD 里但 reward 没读). axis_mean=0.41 → tip 偏 5cm, Stage 3
   启用 contact-aware reward 后第一次推进必撞 hole rim. 不要把 Stage 2 数字
   当 task solve 的指标. Stage 3 启用 contact 后 ckpt 会被重新 force-trained.
3. **改 axis_th 改的是 reward 函数, 不是 metric**. eval 时同 ckpt 不同 axis_th
   metric 数字差 50 倍 (`hold_rate` 从 3% → 100%, `J` 从 19 → 186), 但 agent 行为
   完全相同 (`pos_err`, `axis_err_mean` 不变). `J` 是 reward 函数属性, 不是 policy
   属性.

收紧 axis floor 的可选路径 (按推荐度排序):
- (主线) Stage 3 启用 contact-aware reward + `--exclude_ee_from_physx_self_collision`, 让 axial-advancement 物理反馈接力收紧
- Stage 2b curriculum: 当前 ckpt warm-start, axis_th 0.40 → 0.30 → 0.20 渐进
- 改 reward 结构: hard cliff → soft sigmoid `exp(-(axis_err/scale)²)`. 改 env 代码

### 推荐部署 / Stage 3 warm-start checkpoint

```text
results/S2_oneshot_ep194_seed0_best_agent.msh
```

(epoch 194, J=189, hold@0.40=1.00, max_hold=105.7 步, axis_in_pos_mean=0.41,
pos_success=0.89, pos_err_mean=0.108m)

`S2_oneshot_ep194_seed0_best_hold.msh` 跟 best_agent 同 epoch (best_score=105.69
跟 best_J 同时出现), 行为相同.

### Stage 2 训练命令

```bash
cd ~/bimanual_peghole && conda activate safe_rl

python scripts/train_sac.py \
  --num_envs 16 --n_epochs 200 --n_steps_per_epoch 1024 --n_steps_per_fit 16 --n_eval_episodes 16 \
  --use_axis_resid_obs \
  --load_agent results/S1_B_davideinit_seed0_best_agent.msh --actor_only_warmstart --critic_warmup_transitions 50000 \
  --preinsert_success_pos_threshold 0.10 --success_axis_threshold 0.40 --axis_gate_radius 0.40 \
  --rew_axis 1.0 --rew_pos_success 1.0 --rew_success 4.0 --rew_home 0.0005 \
  --home_weights 1,1,1,1,0.75,0.5,0.5 \
  --lr_actor 5e-5 --alpha_max 0.05 --target_entropy -10 \
  --terminal_hold_bonus 0 --hold_success_steps 10 \
  --seed 0 \
  --wandb_run_name S2_oneshot_axis040_rewx2_lowlr_seed0
```

训练结束**立即**复制 ckpt:

```bash
cp results/best_agent.msh results/S2_oneshot_ep194_seed0_best_agent.msh
cp results/best_hold.msh   results/S2_oneshot_ep194_seed0_best_hold.msh
cp results/final_agent.msh results/S2_oneshot_ep200_seed0_final.msh
```

### Stage 2 eval

```bash
python scripts/eval_sac.py \
  --headless --num_envs 16 --n_episodes 32 \
  --agent_path results/S2_oneshot_ep194_seed0_best_agent.msh \
  --use_axis_resid_obs \
  --preinsert_success_pos_threshold 0.10 --success_axis_threshold 0.40 --axis_gate_radius 0.40 \
  --rew_axis 1.0 --rew_pos_success 1.0 --rew_success 4.0 --rew_home 0.0005 \
  --home_weights 1,1,1,1,0.75,0.5,0.5 \
  --terminal_hold_bonus 0 --hold_success_steps 10
```

### Stage 2 可视化 (去掉 `--headless`)

```bash
python scripts/eval_sac.py \
  --num_envs 16 --n_episodes 32 \
  --agent_path results/S2_oneshot_ep194_seed0_best_agent.msh \
  --use_axis_resid_obs \
  --preinsert_success_pos_threshold 0.10 --success_axis_threshold 0.40 --axis_gate_radius 0.40 \
  --rew_axis 1.0 --rew_pos_success 1.0 --rew_success 4.0 --rew_home 0.0005 \
  --home_weights 1,1,1,1,0.75,0.5,0.5 \
  --terminal_hold_bonus 0 --hold_success_steps 10
```

### Stage 2 待解决问题 (推迟到 Stage 3 之后)

- Single seed=0, 没做多 seed sweep. Davide 警告 "single seed 看 peak 没意义",
  但 Stage 3 主线优先, sweep 推迟. 进 Stage 3 时仍用 seed=0 sanity check.
- axis floor 卡在 0.27-0.29, cliff reward 不训 floor. 主线方案是让 Stage 3
  的 axial-advancement 物理反馈接力收紧.

---

## Stage 3 plan (下一步, obs/reward/absorbing 尚未实施)

### 目标

peg 真正进入 hole 一段距离 (`axial < -d_min`, 例如 `d_min = 2cm`), 不要求插到底.

### Pre-Stage-3 必须做的 env 代码工作

> ⚠️ collider **已经在 USD asset 里** (peg = invisible Cylinder, hole walls =
> Cube ring, 都有 `PhysicsCollisionAPI + PhysxCollisionAPI`). 不要重新加 collider,
> 也**不要**给 peg/hole 加 `RigidBodyAPI / MassAPI` — `build_peghole_usd.py:8-15`
> 明说有意不加, 否则会改变 articulation DoF / 质量矩阵.

真正待做:

1. **启用 EE self-collision exclude**: 训练时传
   `--exclude_ee_from_physx_self_collision` (env line 223 已实现这个开关). 否则
   正常 peg-hole 接触会被双臂自碰撞 hard absorb 误杀.
2. **加 obs**: `axial_tip / radial_err / peg-tip in hole frame` (env 已 cache 在
   `_compute_preinsert_errors`, 但未进 obs).
3. **改 reward**: signed axial potential + insertion bonus + flag, 保留 Stage 2
   shaping 作 anchor (5 条 trap 见下).
4. **改 absorbing**: 加 stage3_success absorbing (axial < -d_min), hold-N 仍关闭.
5. **(可选 sanity) 用 Stage 2 ckpt + `--exclude_ee_from_physx_self_collision`
   跑短期 eval**, 验证 contact 报告正常 (peg 撞 hole rim 而非穿透 / 起飞), 再开
   始改 reward.

### 关键设计要点 (memory `feedback_bimanual_stage3_traps.md` 已记录)

1. **Stage 3 reward = Stage 2 shaping 叠加 + insertion 增量**, 不要替换. 保留
   `+w_pos_success` 给 warm-start actor 留住"preinsert 是好状态"的山头.
2. **Absorbing 仅由 stage3_success 触发**, hold-N 不要重新打开. 否则 agent 被
   锁在 preinsert 拿 hold bonus 跨不过 6cm 死区.
3. **signed axial approach potential** 消除 `axial ∈ [-0.06, 0]` 的 progress 死区
   (clipped clamp 到 0 会让这段无信号).
4. **Stage 3 obs 显式加 `axial / radial_max / peg-tip in hole frame`**, 别让
   actor 反推几何信号.
5. **peg_radial_sample_offsets 不要拉到 -0.10/-0.20** (peg 只 7cm).

### 新增几何量 (env 已 cache 在 `_compute_preinsert_errors`, 但未进 reward)

```text
d            = peg_tip - hole_entry
axial_tip    = d · hole_axis            # 进 hole 时 < 0
radial_vec   = d - axial_tip * hole_axis
radial_err   = ||radial_vec||
```

### 一次性插入 bonus + flag (防 reward hacking)

```python
# __init__:
self._inserted_once = torch.zeros(N_envs, dtype=torch.bool, device=...)

# setup() reset:
self._inserted_once[idx_tensor] = False

# is_absorbing:
inserted_now = self._cached_axial_tip < -insert_depth_threshold
insert_event = inserted_now & (~self._inserted_once)
self._inserted_once |= inserted_now

# reward:
+ w_insert_bonus * insert_event.float()
```

每 episode 这条 bonus 上界 = `w_insert_bonus`, 抖动无收益.

---

## 几何和观测

预插入目标:

```text
preinsert_target = hole_entry + preinsert_offset * hole_axis
pos_vec          = peg_tip - preinsert_target
pos_err          = ||pos_vec||
axis_err         = 1 + dot(peg_axis, hole_axis)
```

`axis_err` 越小越好. 理想反向对齐时 `dot=-1`, 所以 `axis_err=0`.

主线使用 `--use_axis_resid_obs`:

```text
obs_dim = 34
obs = joint_pos[14] + joint_vel[14] + pos_vec[3] + axis_resid[3]
axis_resid = peg_axis + hole_axis
axis_err   = ||axis_resid||^2 / 2 = 1 + dot(peg_axis, hole_axis)
```

`axis_resid` 不能简单替换为 `cross(peg_axis, hole_axis)`. cross 在平行/反平行时
都为 0, 无法区分最差同向和最好反向.

## Reward 公式

正常 step reward:

```text
r_normal =
    - w_pos         * pos_err
    - w_axis        * gate(pos_err) * axis_err
    - w_joint_limit * joint_limit_norm
    - w_action      * ||a_raw||^2
    - w_home        * sum_i home_weight_i * ((q_i - q_home_i) / joint_range_i)^2
    + w_pos_success * 1[pos_err < pos_th]
    + w_success     * 1[(pos_err < pos_th) and (axis_err < axis_th)]
```

axis gate:

```text
if axis_gate_radius is finite:
    gate(pos_err) = clamp(
        (axis_gate_radius - pos_err) / (axis_gate_radius - pos_th),
        0, 1
    )
else:
    gate(pos_err) = 1
```

`axis_gate_radius=0.40` 含义: pos_err > 40cm 时 axis 惩罚关, 进 [pos_th, 0.40m]
后线性打开, 进位置阈值后 gate=1. **`axis_gate_radius=inf` 等价不门控**, Stage 1
默认即此.

absorbing:

```text
- collision (PhysX 力 OR sphere-proxy 几何 < clearance_hard):
    r = reward_absorbing_r_min / (1 - gamma)        # ≈ -200 默认
    episode absorbing = True

- hold-N (success 连续 hold_success_steps 步):
    若 terminal_hold_bonus > 0:
        r = r_normal + terminal_hold_bonus
        episode absorbing = True
    若 terminal_hold_bonus == 0 (默认):
        不 absorb, 只更新 _consecutive_inthresh 给 eval

- success 本身不终止 (避免 Q-target 边界断崖)
```

## 保留结果

```text
# Stage 1 ckpt
results/S1_B_davideinit_seed0_best_agent.msh        # Stage 2 warm-start 来源
results/S1_B_davideinit_seed0_best_hold.msh
results/S1_B_davideinit_seed0_final.msh
results/S1_B_davideinit_seed42_best_agent.msh        # 备查
results/S1_B_davideinit_seed42_best_hold.msh
results/S1_B_davideinit_seed42_final.msh
results/S1_ablation_A_with_hold_absorb_*.msh         # ablation A 对照
results/S1_ablation_B_no_hold_absorb_*.msh           # ablation B 对照

# Stage 2 ckpt (2026-05-09 oneshot)
results/S2_oneshot_ep194_seed0_best_agent.msh       # ★ Stage 3 warm-start 用
results/S2_oneshot_ep194_seed0_best_hold.msh        # 同 epoch (best_score 跟 best_J 同时)
results/S2_oneshot_ep200_seed0_final.msh            # 最末状态 J=164, pos 略退化
results/S2_axis020_seed0_best_agent_ep85.msh        # exp 1 旧 baseline (axis_in_pos_mean=0.63), 留作对照
```

顶层 `best_agent.msh / best_hold.msh / final_agent.msh` 是上一次训练的产物,
**会被下次训练覆盖**, 不要直接依赖.

Stage 1 5-seed sweep 中 seed 7 / 123 / 999 的 ckpt 没保留 (hold_rate < 0.85 的 seed
删了节省空间). 如果要从其它 seed warm-start, 重跑对应 seed 即可.

Stage 2 实验 2 (axis_th=0.30 重训) ckpt 没单独备份 (训完直接被 oneshot 覆盖了),
但 trajectory 跟 exp 1 完全相同 (deterministic seed), 性能也跟 exp 1 等价. 不用
重跑.

## 已知 stale code / config (不要直接信)

进 Stage 3 前最好先 cleanup 这几处 (或至少知道避开):

- **`conf/experiment/phase2.yaml`** 还是 2026-05 之前的旧 hyperparams
  (`alpha_max=0.10`, `target_entropy=-7`, `lr_actor=1e-4`, `rew_axis=0.5`,
  `rew_success=2.0`, `success_axis_threshold=0.50`, `load_agent` 指向已删除的
  `S1_axisresid_home_uniform_repro_best_hold.msh`). **不要走 Hydra/YAML 路径
  跑 Stage 2**, 用上面 README 的 CLI 命令是当前 verified 配方.
- **`conf/experiment/phase1.yaml:40`** 仍强制 `--clearance_hard=-inf` (关闭
  sphere-proxy hard absorbing). 当前主线是 `clearance_hard=0.0` 默认开. 如果
  从 Hydra 跑 phase 1 复现, 会得到 collision-blind 的旧版本.
- **`scripts/train_sac_lagrangian.py:204`** 写 `from algo import SACLagrangian`,
  但仓库实际目录是 `algorithm/`. import 失败. Stage 3 如果想试 Lagrangian SAC
  需要先修这个 import (跟当前训练主线无关, 主线走 `train_sac.py`).

## 历史

- 2026-05-09: Stage 2 oneshot 配方完成 (200 ep, axis_th=0.40, rew x2, lr 5e-5,
  alpha 0.05, te -10). 比之前 axis_th=0.20 / 0.30 实验 J 高 7x, max_hold 从 0
  → 105.7 步, axis_in_pos_mean 从 0.63 → 0.41. 关键 learning: cliff reward
  训中心不训 floor; success_axis_threshold 是 reward 函数开关不是 metric;
  post-peak 塌方靠 lr↓ + alpha↓ 治. 详见 Stage 2 章节.
- 2026-05-08: 引入 Davide init + sphere-proxy + hold-N off, Stage 1 5-seed
  sweep 完成. 删除 2026-05-04/05/06/07 的 pre-collision ckpt (S1_axisresid_*,
  S2_*, M2_*, S2p*, S3_warmstart_baseline 等), 不再作为 baseline 比较.
- 2026-05-04 之前: 早期 M1/M2 调参, 没有 sphere-proxy collision, 不可比.
