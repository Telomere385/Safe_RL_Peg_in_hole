# bimanual_peghole

双臂 KUKA iiwa 在 IsaacSim 中做 peg-in-hole 的 SAC 训练。当前主线已经完成
Stage 1 和 Stage 2，Stage 3 的插入 reward 还没有正式开始实现。

当前任务仍然是 **preinsert pose**，也就是让左臂 peg tip 到达右臂 hole entry
前方 `preinsert_offset=0.05m` 的预插入目标，并保持成功状态。当前 peg/hole 仍是
visual-only 几何，不使用真实接触插入作为 reward。

## 当前阶段

### Stage 1: preinsert 位置

Stage 1 的目标是先学会把 `peg_tip` 放到预插入位置附近，不要求轴对齐。

- 观测使用 `--use_axis_resid_obs`，即 34 维观测。
- `rew_axis=0.0`，axis 惩罚关闭。
- `success_axis_threshold=inf`，success 只看位置阈值。
- 位置阈值是 `preinsert_success_pos_threshold=0.10m`。
- 使用很弱的 home regularizer: `rew_home=0.0005`。
- Stage 1 使用 uniform home weights，也就是不传 `--home_weights`。

当前正式 Stage 1 checkpoint:

- `results/S1_axisresid_home_uniform_repro_best_agent.msh`
- `results/S1_axisresid_home_uniform_repro_best_hold.msh`

该 run 的 wandb 名称是 `S1_axisresid_home_uniform_repro_50ep`。训练中 best
达到 `best_J=33.6395`、`best_hold_rate=1.0`。

2026-05-06 上传前用下面的 README 命令重新跑了一次 Stage 1 复现实验，run 链接:

```text
https://wandb.ai/miaoxu010522-lund/bimanual_peghole/runs/xsywfqb8
```

该复现实验与正式结果对齐: epoch 15 达到 `best_J=33.6395`、
`best_hold_rate=1.0`、`best_score=10.0`，训练结束时 `best_hold_rate=1.0`
仍被正确保留。最终 epoch 的策略退化，因此后续 Stage 2 warm-start 应使用
`best_agent.msh` / `best_hold.msh` 复制出的正式 checkpoint，不使用 `final_agent.msh`。

### Stage 2: preinsert 位置 + 轴对齐

Stage 2 从 Stage 1 actor warm-start，在保留位置能力的基础上学习 peg/hole 轴对齐。

- 只继承 Stage 1 的 actor: `--actor_only_warmstart`。
- critic、alpha、replay buffer 冷启动。
- 必须使用较长 critic warmup: `--critic_warmup_transitions 50000`。
- 当前正式宽松训练阈值是 `success_axis_threshold=0.50`。
- 轴 reward 使用距离门控: `axis_gate_radius=0.40m`。
- 当前使用 `rew_axis=0.5`、`rew_pos_success=1.0`、`rew_success=2.0`。
- Stage 2 使用非 uniform home weights，给末端腕部更多自由度:
  `1,1,1,1,0.75,0.5,0.5`。

当前正式 Stage 2 checkpoint:

- `results/S2_from_uniformS1_whome_axis05_gate040_cwarm50k_best_agent.msh`
- `results/S2_from_uniformS1_whome_axis05_gate040_cwarm50k_best_hold.msh`
- `results/S2_from_uniformS1_whome_axis05_gate040_cwarm50k_final.msh`

当前最推荐使用的是:

```text
results/S2_from_uniformS1_whome_axis05_gate040_cwarm50k_best_agent.msh
```

原因是它在 128 episodes 正式 eval 中不仅能通过训练阈值 `axis_th=0.50`，也能通过
更严格的 `axis_th=0.30`。`final` checkpoint 已经退化，不应作为后续 warm-start
或展示结果使用。

Stage 2 当前结论:

| checkpoint | eval 阈值 | hold_success_rate | pos_success_rate | pos_err_mean | axis_err_mean |
|---|---:|---:|---:|---:|---:|
| `S2..._best_agent` | 0.50 | 1.000 | 0.778 | 0.1713m | 0.9084 |
| `S2..._best_agent` | 0.30 | 1.000 | 0.831 | 0.1434m | 0.7646 |
| `S2..._best_agent` | 0.20 | 0.648 | 0.739 | 0.1289m | 0.5658 |
| `S2..._best_hold` | 0.50 | 0.992 | 0.880 | 0.1270m | 0.8352 |
| `S2..._best_hold` | 0.30 | 0.000 | 0.753 | 0.1240m | 0.7154 |
| `S2..._final` | 0.50 | 0.102 | 0.264 | 0.5331m | 0.9321 |

## 几何和观测

预插入目标:

```text
preinsert_target = hole_entry + preinsert_offset * hole_axis
pos_vec          = peg_tip - preinsert_target
pos_err          = ||pos_vec||
axis_err         = 1 + dot(peg_axis, hole_axis)
```

`axis_err` 越小越好。理想反向对齐时 `dot=-1`，所以 `axis_err=0`。

当前主线使用 `--use_axis_resid_obs`:

```text
obs_dim = 34
obs = joint_pos[14] + joint_vel[14] + pos_vec[3] + axis_resid[3]
axis_resid = peg_axis + hole_axis
axis_err   = ||axis_resid||^2 / 2 = 1 + dot(peg_axis, hole_axis)
```

注意 `axis_resid` 不能简单替换成 `cross(peg_axis, hole_axis)`。cross 在平行和反平行
时都为 0，无法区分最差同向和最好反向。后续如果要增强姿态观测，可以考虑在保留
`axis_resid` 的基础上额外加入 cross。

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

其中:

```text
pos_th  = preinsert_success_pos_threshold
axis_th = success_axis_threshold
```

axis gate:

```text
if axis_gate_radius is finite:
    gate(pos_err) = clamp(
        (axis_gate_radius - pos_err) / (axis_gate_radius - pos_th),
        0,
        1
    )
else:
    gate(pos_err) = 1
```

`axis_gate_radius=0.40` 的含义是: 离 preinsert 目标 40cm 以外不施加 axis 惩罚，
进入 `[pos_th, 0.40m]` 后线性打开 axis 惩罚，进位置阈值后 gate 为 1。

success 和 absorbing:

```text
full_success = (pos_err < pos_th) and (axis_err < axis_th)
hold_success = full_success 连续 hold_success_steps 步
```

如果 `terminal_hold_bonus > 0`，达到 hold success 时:

```text
r = r_normal + terminal_hold_bonus
episode absorbing = True
```

如果触发双臂自碰撞:

```text
r = reward_absorbing_r_min / (1 - gamma)
episode absorbing = True
```

当前默认 `reward_absorbing_r_min=-2.0`、`gamma=0.99`，所以碰撞 absorbing reward
约为 `-200`。

## Stage 1 训练命令

下面是当前正式 Stage 1 复现实验的等价完整命令。wandb config 中实际只显式传了
部分参数；这里把关键默认值也展开，便于复现。

```bash
cd /home/miao/bimanual_peghole
conda activate safe_rl

python scripts/train_sac.py \
  --num_envs 16 \
  --n_epochs 50 \
  --n_steps_per_epoch 1024 \
  --n_steps_per_fit 16 \
  --n_eval_episodes 16 \
  --use_axis_resid_obs \
  --preinsert_success_pos_threshold 0.10 \
  --success_axis_threshold inf \
  --rew_axis 0.0 \
  --rew_home 0.0005 \
  --terminal_hold_bonus 50 \
  --wandb_run_name S1_axisresid_home_uniform_repro_50ep
```

训练结束后立刻复制 run-named checkpoint，避免下一次训练覆盖顶层文件:

```bash
cp results/best_agent.msh results/S1_axisresid_home_uniform_repro_best_agent.msh
cp results/best_hold.msh results/S1_axisresid_home_uniform_repro_best_hold.msh
```

## Stage 2 训练命令

这是当前正式成功的 Stage 2 命令。关键点是 actor-only warm-start 加
`critic_warmup_transitions=50000`。

```bash
cd /home/miao/bimanual_peghole
conda activate safe_rl

python scripts/train_sac.py \
  --num_envs 16 \
  --n_epochs 200 \
  --n_steps_per_epoch 1024 \
  --n_steps_per_fit 16 \
  --n_eval_episodes 16 \
  --load_agent results/S1_axisresid_home_uniform_repro_best_hold.msh \
  --actor_only_warmstart \
  --critic_warmup_transitions 50000 \
  --use_axis_resid_obs \
  --preinsert_success_pos_threshold 0.10 \
  --success_axis_threshold 0.50 \
  --axis_gate_radius 0.40 \
  --rew_axis 0.5 \
  --rew_pos_success 1.0 \
  --rew_success 2.0 \
  --rew_home 0.0005 \
  --home_weights 1,1,1,1,0.75,0.5,0.5 \
  --lr_actor 1e-4 \
  --target_entropy -7 \
  --alpha_max 0.1 \
  --terminal_hold_bonus 50 \
  --wandb_run_name S2_from_uniformS1_whome_axis05_gate040_cwarm50k
```

训练结束后复制:

```bash
cp results/best_agent.msh results/S2_from_uniformS1_whome_axis05_gate040_cwarm50k_best_agent.msh
cp results/best_hold.msh results/S2_from_uniformS1_whome_axis05_gate040_cwarm50k_best_hold.msh
cp results/final_agent.msh results/S2_from_uniformS1_whome_axis05_gate040_cwarm50k_final.msh
```

不要直接依赖:

```text
results/best_agent.msh
results/best_hold.msh
results/final_agent.msh
```

这些顶层文件会被下一次训练清理或覆盖。

## Eval 命令

### Stage 1 eval

```bash
cd /home/miao/bimanual_peghole
conda activate safe_rl

python scripts/eval_sac.py \
  --headless \
  --num_envs 16 \
  --n_episodes 128 \
  --agent_path results/S1_axisresid_home_uniform_repro_best_hold.msh \
  --use_axis_resid_obs \
  --preinsert_success_pos_threshold 0.10 \
  --success_axis_threshold inf \
  --rew_axis 0.0 \
  --rew_home 0.0005 \
  --terminal_hold_bonus 50
```

### Stage 2 正式 eval

训练阈值 `axis_th=0.50`:

```bash
cd /home/miao/bimanual_peghole
conda activate safe_rl

python scripts/eval_sac.py \
  --headless \
  --num_envs 16 \
  --n_episodes 128 \
  --agent_path results/S2_from_uniformS1_whome_axis05_gate040_cwarm50k_best_agent.msh \
  --use_axis_resid_obs \
  --preinsert_success_pos_threshold 0.10 \
  --success_axis_threshold 0.50 \
  --axis_gate_radius 0.40 \
  --rew_axis 0.5 \
  --rew_pos_success 1.0 \
  --rew_success 2.0 \
  --rew_home 0.0005 \
  --home_weights 1,1,1,1,0.75,0.5,0.5 \
  --terminal_hold_bonus 50
```

严格轴 margin eval，只改 `--success_axis_threshold`:

```bash
# axis_th = 0.30
python scripts/eval_sac.py \
  --headless \
  --num_envs 16 \
  --n_episodes 128 \
  --agent_path results/S2_from_uniformS1_whome_axis05_gate040_cwarm50k_best_agent.msh \
  --use_axis_resid_obs \
  --preinsert_success_pos_threshold 0.10 \
  --success_axis_threshold 0.30 \
  --axis_gate_radius 0.40 \
  --rew_axis 0.5 \
  --rew_pos_success 1.0 \
  --rew_success 2.0 \
  --rew_home 0.0005 \
  --home_weights 1,1,1,1,0.75,0.5,0.5 \
  --terminal_hold_bonus 50

# axis_th = 0.20
python scripts/eval_sac.py \
  --headless \
  --num_envs 16 \
  --n_episodes 128 \
  --agent_path results/S2_from_uniformS1_whome_axis05_gate040_cwarm50k_best_agent.msh \
  --use_axis_resid_obs \
  --preinsert_success_pos_threshold 0.10 \
  --success_axis_threshold 0.20 \
  --axis_gate_radius 0.40 \
  --rew_axis 0.5 \
  --rew_pos_success 1.0 \
  --rew_success 2.0 \
  --rew_home 0.0005 \
  --home_weights 1,1,1,1,0.75,0.5,0.5 \
  --terminal_hold_bonus 50
```

## 可视化命令

`visualize_policy.py` 是肉眼检查姿态用的，不计算正式 reward。它不需要
`--rew_success`，也不要传 `--rew_success`。

### Stage 1 可视化

```bash
cd /home/miao/bimanual_peghole
conda activate safe_rl

python scripts/visualize_policy.py \
  --agent_path results/S1_axisresid_home_uniform_repro_best_hold.msh \
  --num_envs 2 \
  --viz_env_idx 0 \
  --n_episodes 16 \
  --freeze_seconds 60 \
  --use_axis_resid_obs \
  --preinsert_success_pos_threshold 0.10 \
  --success_axis_threshold inf \
  --rew_axis 0.0 \
  --rew_home 0.0005
```

### Stage 2 可视化

先看当前推荐 checkpoint:

```bash
cd /home/miao/bimanual_peghole
conda activate safe_rl

python scripts/visualize_policy.py \
  --agent_path results/S2_from_uniformS1_whome_axis05_gate040_cwarm50k_best_agent.msh \
  --num_envs 2 \
  --viz_env_idx 0 \
  --n_episodes 16 \
  --freeze_seconds 60 \
  --use_axis_resid_obs \
  --preinsert_success_pos_threshold 0.10 \
  --success_axis_threshold 0.30 \
  --axis_gate_radius 0.40 \
  --rew_axis 0.5 \
  --rew_pos_success 1.0 \
  --rew_home 0.0005 \
  --home_weights 1,1,1,1,0.75,0.5,0.5
```

如果只是复现训练时的成功判定，把 `--success_axis_threshold 0.30` 改成 `0.50`。

## 保留结果

当前本地只保留正式 Stage 1/Stage 2 结果和对应 wandb run。

保留的 `.msh`:

```text
results/S1_axisresid_home_uniform_repro_best_agent.msh
results/S1_axisresid_home_uniform_repro_best_hold.msh
results/S2_from_uniformS1_whome_axis05_gate040_cwarm50k_best_agent.msh
results/S2_from_uniformS1_whome_axis05_gate040_cwarm50k_best_hold.msh
results/S2_from_uniformS1_whome_axis05_gate040_cwarm50k_final.msh
```

保留的本地 wandb run:

```text
results/wandb/run-20260505_102834-j9f3mwf0  # S1_axisresid_home_uniform_repro_50ep
results/wandb/run-20260505_122125-2iejpthc  # S2_from_uniformS1_whome_axis05_gate040_cwarm50k
```

旧的 M1/M2 调参 checkpoint、失败的 S2p1 debug run、顶层 `best_agent.msh` /
`best_hold.msh` 已不作为正式结果使用。

## 下一步

Stage 2 的当前结果可以作为 Stage 3 reward 设计的 warm-start baseline，但不要继续
在 Stage 2 上做零散补丁式调参。Stage 3 应该重新定义插入阶段的几何量，例如
`radial_err`、`axial_dist`、`axis_err`，并明确 success/hold 条件后再开始训练。
