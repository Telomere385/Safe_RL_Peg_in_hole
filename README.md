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

---

# bimanual_peghole_lagrangianSAC

## 基本说明

双臂 KUKA iiwa 在 IsaacSim 中做 peg-in-hole 的 Lagrangian SAC 安全约束训练 (mushroom-rl 2.0).
基于 `bimanual_peghole` SAC 主线, 将碰撞约束从 reward 剥离为独立 cost 信号, 用 Lagrange 乘子
λ 动态控制安全预算.

标准 SAC 把碰撞编码为巨大负奖励 (`r_min/(1-γ) ≈ -200`), reward critic 和安全信号混在一起难以分别调参.
Lagrangian SAC 把碰撞从 reward 里剥出来, 用独立 cost critic Q_C 学习, 用 Lagrange 乘子 λ 动态
控制安全预算, reward critic 只专注任务进展.

### 与 SAC 的核心差异

| 方面 | SAC (`train_sac.py`) | Lagrangian SAC (`train_sac_lagrangian.py`) |
|------|----------------------|--------------------------------------------|
| Replay buffer | 标准 `(s,a,r,s',absorb,last)` | `ConstrainedReplayMemory` 多存一列 cost `c` |
| Critic 数量 | 2 个 reward critic | 4 个: reward critic ×2 + cost critic ×2 |
| 碰撞处理 | reward = `r_min/(1-γ) ≈ -200`, episode 终止 | reward = shaped (不加大负奖励), cost = 1.0, episode 仍终止 |
| Actor loss | `(α·logπ − Q_R).mean()` | `(α·logπ − Q_R + λ·Q_C).mean()` |
| 环境类 | `DualArmPegHoleEnv` | `DualArmPegHoleCostEnv` |

### cost_limit 标定方法

`--cost_limit` 是训练最关键的超参. 设 0 会让 λ 一上来就爆; 设太高约束形同虚设.

推荐流程:

1. 用现有 SAC checkpoint 跑一次短 eval (64 ep 即可):
   ```bash
   python scripts/eval_sac.py --headless --num_envs 16 --n_episodes 64 \
     --agent_path results/<sac_checkpoint>.msh [... 同训练的 env 参数 ...]
   ```
2. 读输出的 `absorb_sphere` 计数, 算 per-step collision rate:
   ```
   collision_rate = absorb_sphere_per_epoch / n_steps_per_epoch
   ```
3. 取 0.5× 作为起步预算: `--cost_limit = 0.5 × collision_rate`
4. 训练中看 wandb `cost_violation = cost_rate − cost_limit`:
   - 长期正 → λ 持续上升, 正常收紧
   - 长期负 → cost_limit 过松, 考虑降低
   - λ 冲到 `lambda_max` 且 `cost_violation` 仍正 → cost_limit 太严或任务太难

### 关键超参数说明

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `--cost_limit` | (必填) | `0.5 × baseline_rate` | per-step cost 预算; 见上方标定流程 |
| `--lr_lambda` | `1e-3` | `1e-4 ~ 1e-3` | 比 `lr_actor` 低 1–10×; 太高 λ 震荡, 太低约束收紧慢 |
| `--lambda_max` | `100.0` | `50 ~ 200` | λ 上限; 频繁冲顶说明 cost_limit 太严或任务太难 |
| `--init_log_lambda` | `0.0` | `0.0 ~ 2.0` | λ 初值 = `exp(init_log_lambda)`; 从 SAC ckpt warmstart 可适当调高 |
| `--gamma_cost` | `None` (= env γ) | `None` 或 `0.95~0.99` | None 复用 env γ=0.99; 设小一点让约束更短视、收紧更快 |

### 训练命令

从 SAC checkpoint warm-start (actor-only, 推荐起点):

```bash
cd ~/bimanual_peghole && conda activate safe_rl

python scripts/train_sac_lagrangian.py \
  --num_envs 16 --n_epochs 200 --n_steps_per_epoch 1024 --n_steps_per_fit 16 --n_eval_episodes 16 \
  --use_axis_resid_obs \
  --load_agent results/<sac_checkpoint>.msh --actor_only_warmstart \
  --critic_warmup_transitions 50000 \
  --preinsert_success_pos_threshold 0.10 --success_axis_threshold <axis_th> \
  --rew_axis <rew_axis> --rew_pos_success 1.0 --rew_success <rew_success> --rew_home 0.0005 \
  --lr_actor 5e-5 --lr_alpha 3e-4 --alpha_max 0.05 --target_entropy -10 \
  --cost_limit <0.5×baseline_collision_rate> \
  --lr_lambda 1e-3 --lambda_max 100 --init_log_lambda 0.0 \
  --seed 0 \
  --wandb_run_name <run_name>
```

训练结束**立即**备份 ckpt (顶层文件会被下次训练覆盖):

```bash
cp results/best_agent_lag.msh  results/<run_name>_best_agent.msh
cp results/best_hold_lag.msh   results/<run_name>_best_hold.msh
cp results/final_agent_lag.msh results/<run_name>_final.msh
```

### Eval 时看什么

| 指标 | 健康表现 | 异常信号 |
|------|----------|----------|
| `cost_rate` | 逐渐收敛到 ≤ `cost_limit` | 长期 >> cost_limit → λ 失控或任务太难 |
| `cost_violation` | 先正后收敛到 ≤ 0 | 持续正 + λ 冲顶 → cost_limit 太严 |
| `lambda` (λ) | 平稳增长后趋于稳定 | 爆到 `lambda_max` → 上调 `lambda_max` 或放松 `cost_limit` |
| `J` | 不低于 SAC baseline 太多 | 崩到 baseline 一半以下 → `lr_lambda` 太高或 `cost_limit` 太严 |
| `hold_success_rate` | 与 SAC baseline 可比 | 大幅下降 → λ 过大压制了 actor |

> `best_agent_lag.msh` 是最高 J 的 ckpt, **不保证满足 cost_limit** (高 J 可能在 λ
> 收紧之前就出现). 部署前在 wandb `cost_rate` 时间线上核查对应 epoch 是否达标.

### Warmstart 路径

**SAC → Lagrangian SAC** (跨算法, 必须 `--actor_only_warmstart`):

```bash
--load_agent results/<sac_ckpt>.msh --actor_only_warmstart
# critic / cost critic / α / λ / replay 全部冷启动; --keep_replay 此时被忽略
```

**Lagrangian SAC → Lagrangian SAC** (同算法全量, 保留旧 critic 和 replay):

```bash
--load_agent results/<lag_ckpt>.msh
# 加 --keep_replay 可保留旧 replay buffer (reward/cost 函数未变时才合理)
```

**Lagrangian SAC → Lagrangian SAC (actor-only)** (reward 或 cost 函数有改动时):

```bash
--load_agent results/<lag_ckpt>.msh --actor_only_warmstart
```

---

## Stage 1 训练

### 训练命令 (Hydra, 冷启动)

```bash
python scripts/train_hydra.py experiment@train=phase1_lagrangian
```

无需 SAC checkpoint，直接冷启动。`conf/experiment/phase1_lagrangian.yaml` 已包含所有超参，
`--load_agent` 未设置即走 `_cold_create_sac_lag()`。

完整超参见 `conf/experiment/phase1_lagrangian.yaml`，关键值：

| 参数 | 值 | 备注 |
|------|-----|------|
| `lr_actor` | 1e-4 | 冷启动保守值 |
| `alpha_max` | 0.05 | 防 entropy 项压制精度信号 |
| `target_entropy` | -7 | = -act_dim/2, 冷启动 14-DoF |
| `cost_limit` | 0.02 | TODO: 用 SAC baseline eval 标定 |
| `lr_lambda` | 1e-3 | |
| `lambda_max` | 100.0 | |
| `init_log_lambda` | 0.0 | → λ_init = 1.0 |
| `gamma_cost` | null (= env γ = 0.99) | |
| `clearance_hard` | 0.0 | sphere-proxy 开启 |
| `n_epochs` | 100 | |

### λ 的作用与行为分析

**λ 的作用：**

Actor loss = `α·logπ − Q_R + λ·Q_C`

λ 是 Lagrange 乘子，量化"当前约束有多紧"：
- λ 大 → actor 被迫远离高 Q_C 的动作，主动压 cost
- λ → 0 → actor loss 退化为纯 SAC，cost critic 对 actor 零影响

λ 更新规则 (`lagrangian_sac.py:292`)：

```
log_λ += lr_λ × violation
violation = Q_C × (1 − γ_c) − cost_limit
```

violation > 0（超预算）→ λ 上升；violation < 0（有余量）→ λ 下降。

**λ 为何衰减至下限（4.5e-5 = e^{-10}）：**

冷启动时分两个阶段：

- **epoch 1–4（Q_C 低估期）**：cost critic 未收敛，Q_C ≈ 0，导致 `Q_C×(1-γ_c) < cost_limit`，
  violation 看起来为负 → λ 持续下降。此时 eval 实测 cost_rate 已超标（epoch 3: 0.030，
  epoch 4: 0.038），但 λ 更新用的是 replay buffer 中的 Q_C 估计，冷启动阶段低估导致 λ 反常下降。

- **epoch 5+ （约束真正满足期）**：policy 突然学会到达 preinsert 区域（epoch 5: hold_rate=0.938），
  真实 cost_rate 归零。violation 恒为 `-cost_limit = -0.02`，λ 持续下降直到触碰
  lower clamp `log_λ = -10`（`lagrangian_sac.py:300`），停在 λ ≈ 4.5e-5。

**λ ≈ 0 的影响：**

训练从 epoch 5 起等价于标准 SAC，Lagrangian 约束机制完全休眠：

```
actor loss ≈ α·logπ − Q_R + 4.5e-5 · Q_C  ≈  α·logπ − Q_R
```

**是否需要 debug：** 否，λ 行为是正确的数学结果（约束满足、cost_limit 有余量则 λ
应下降）。崩溃（epoch 16/20）发生时 λ 已为 0，与 Lagrangian 机制无关。

**真实崩溃原因：** UTD=16 配合小 replay buffer（epoch 10 时仅约 20K transitions）
导致 Q 值过估计，actor 梯度爆炸。标准 SAC policy collapse，不是 Lagrangian 问题。

**冷启动 Q_C 低估的缓解方法（供参考）：**
- 增大 `--critic_warmup_transitions`（如 50K），让 cost critic 先收敛再放开 λ 更新
- `init_log_lambda` 不要设太高（当前 0.0 = λ_init=1.0 已属于偏高，容易被早期负 violation 快速压下）

### 实验记录 (run-20260510_120554-7bkfr82a, 2026-05-10, 训练中)

seed=42，冷启动，100 epoch（log 已记录至 epoch 56，训练仍在进行）。

**训练曲线概述：**

| epoch 区间 | J 范围 | hold_rate | cost_rate | λ | 事件 |
|---|---:|---:|---:|---:|---|
| 1–4 | -26.9 → -8.2 | 0–0.125 | 0.016–0.038 | 0.36→0.017 | 学习初期，cost_rate 超标但 Q_C 低估，λ 反常下降 |
| 5–10 | 4.8 → 112.0 | 0.938→1.000 | 0.000 | 0.006→≈0 | policy 突破，hold_rate 跳至 1.0，λ 降至下限 |
| 11–15 | 86.6 → 110.9 | 1.000 | 0.000 | ≈4.5e-5 | 平台期，max_hold_mean 稳定在 96–133 步 |
| 16 | **−22.0** | 0.250 | 0.000 | ≈4.5e-5 | **第一次 policy collapse**（UTD=16 + 小 replay buffer Q 过估） |
| 17–19 | 18.2 → 76.6 | 0.938→1.000 | 0.000 | ≈4.5e-5 | 部分恢复 |
| 20–21 | **−54.7 → −51.8** | 0.000 | 0.000 | ≈4.5e-5 | **第二次 policy collapse，更严重** |
| 22–32 | 2.1 → 107.4 | 0.750→1.000 | 0.000–0.005 | ≈4.5e-5 | 完全恢复，逐步爬回平台 |
| 33–44 | 115.8 → **117.7** | 1.000 | 0.000 | ≈4.5e-5 | **超越崩溃前 peak**，epoch 44 创新高 best_J=117.67 |
| 45–56 | 109.6 → 116.3 | 1.000 | 0.000 | ≈4.5e-5 | 稳定平台期（进行中） |

**关键指标（截至 epoch 56）：**

| 指标 | 值 | epoch |
|------|-----|-------|
| best_J | **117.67** | **44** |
| best_hold_rate | 1.000 | 8 |
| best_hold_max_hold_mean | **136.5 步** | **44** |
| cost_rate @ peak | 0.000 | — |
| λ @ peak | ≈4.5e-5 | — |
| 第一次崩溃 | J=−22.0 | 16 |
| 第二次崩溃 | J=−54.7 | 20 |
| 超越崩溃前 peak | epoch 33（J=115.8 > 前 peak 112.0）| 33 |

**checkpoint 位置：**

```text
results/checkpoints_lag/2026-05-10/12-02-15/best_agent.msh   # best_J=117.67, epoch 44 (持续更新)
results/checkpoints_lag/2026-05-10/12-02-15/best_hold.msh    # best hold_rate=1.0
results/checkpoints_lag/2026-05-10/12-02-15/final_agent.msh  # 训练结束后写入
```

**结论与注意事项：**

- cost_rate 从 epoch 5 起持续为 0，满足 `cost_limit=0.02` 的安全约束
- λ 降至 lower bound（≈4.5e-5），Lagrangian 机制从 epoch 5 起等价于纯 SAC
- policy collapse 出现两次（epoch 16/20），但**均完全恢复，且 epoch 33 起超越崩溃前 peak**——与 SAC Phase 1 不同，这里 replay buffer 中的 bad transitions 被稀释后 policy 重新收敛
- 部署选 `best_agent.msh`（当前 epoch 44，训练结束后可能继续更新），不要用 `final_agent.msh`
- 可视化命令：

```bash
python scripts/train_hydra.py --multirun experiment@train=record_checkpoint \
    train.checkpoint_dir=results/checkpoints_lag/2026-05-10/12-02-15 \
    train.tag=lag_phase1
```
