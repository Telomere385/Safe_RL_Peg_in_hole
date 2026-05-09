"""train_sac.py / eval_sac.py / visualize_policy.py 共用的 eval 工具.

- deterministic_policy: SAC 评估时把 tanh-Gaussian 策略替换成 tanh(mu).
- compute_hold_metrics: 从 evaluate() 返回的 flatten 后的 dataset 算 hold-N
  success 与每步 in-threshold / pos_err / axis_err 统计.
  in_thresh = (pos_err < pos_th) ∧ (axis_err < axis_th); axis_th=inf 退化为
  pos-only (Stage 1 行为). 同时单独报 pos_success_rate (只看 pos<pos_th, 反映
  Stage 1 已学技能保住没) 与 axis_gate_mean (axis 项实际被门控到几成).
- parse_home_weights: argparse type=, 接受 7/14 维 float 列表 (逗号或空格分隔).
- resolve_eval_episode_count: 让 eval episode 数与 num_envs 对齐.
"""

import argparse
import math
from contextlib import contextmanager

import numpy as np
import torch


def parse_home_weights(value):
    """argparse type=: 接受 7 维(单臂, 自动复制到左右臂)或 14 维 float 列表,
    逗号或空格分隔. 例如 '1,1,1,1,0.5,0.25,0.25'.
    """
    raw = value.replace(",", " ").split()
    try:
        weights = tuple(float(x) for x in raw)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "--home_weights 必须是逗号或空格分隔的 float 列表"
        ) from e
    if len(weights) not in (7, 14):
        raise argparse.ArgumentTypeError(
            f"--home_weights 必须是 7 维(单臂)或 14 维, 当前 {len(weights)} 维"
        )
    bad = [i for i, w in enumerate(weights) if not math.isfinite(w) or w < 0.0]
    if bad:
        raise argparse.ArgumentTypeError(
            f"--home_weights 必须是有限非负数, 非法索引 {bad}"
        )
    return weights


@contextmanager
def deterministic_policy(agent):
    policy = agent.policy
    original_draw_action = policy.draw_action

    def draw_action(state, internal_state=None):
        with torch.no_grad():
            mu = policy._mu_approximator.predict(state)
            action = torch.tanh(mu) * policy._delta_a + policy._central_a
        return action.detach(), None

    policy.draw_action = draw_action
    try:
        yield
    finally:
        policy.draw_action = original_draw_action


def compute_hold_metrics(dataset, mdp, hold_n_steps):
    """`hold_success` = episode 内出现长度 >= N 的连续 in-threshold 段.

    in-threshold = (pos_err < pos_th) ∧ (axis_err < axis_th). axis_th=inf 时
    退化为 pos-only (Stage 1 行为).

    依赖 VectorCore.evaluate 末尾返回的 dataset 是 flatten 后的 1D, 每个 env 的
    transitions 在结果里连续, 所以 `flatnonzero(last)` + 顺序切片就能正确分段.
    """
    _, _, _, next_state, _, last = dataset.parse(to="torch")
    pos_err, axis_err, in_thresh = mdp._compute_task_errors(next_state)
    pos_th = mdp._preinsert_success_pos_threshold
    pos_in_thresh = pos_err < pos_th
    # axis-gate 跟 reward 同公式; gate_radius=inf 时全 1.
    if math.isfinite(mdp._axis_gate_radius):
        denom = max(mdp._axis_gate_radius - pos_th, 1e-6)
        axis_gate = ((mdp._axis_gate_radius - pos_err) / denom).clamp(0.0, 1.0)
    else:
        axis_gate = torch.ones_like(pos_err)

    last_np = last.cpu().numpy().astype(bool)
    in_thresh_np = in_thresh.cpu().numpy().astype(bool)

    end_indices = np.flatnonzero(last_np)
    ep_max_holds, ep_in_thresh_rates, ep_final_in_thresh = [], [], []
    start = 0
    for end in end_indices:
        ep = in_thresh_np[start:end + 1]
        max_run, cur = 0, 0
        for flag in ep:
            cur = cur + 1 if flag else 0
            if cur > max_run:
                max_run = cur
        ep_max_holds.append(max_run)
        ep_in_thresh_rates.append(float(ep.mean()) if len(ep) else 0.0)
        ep_final_in_thresh.append(bool(ep[-1]) if len(ep) else False)
        start = end + 1

    hold_flags = np.asarray([mh >= hold_n_steps for mh in ep_max_holds], dtype=bool)
    # 条件指标: 只在 pos_in_thresh 的 timesteps 上算 axis 相关统计.
    # 如果 axis_err_in_pos_thresh_mean 长期 1.0+ 而 pos_success_rate>0, 就坐实
    # "进 pos_th 但 axis 学不会" — 这是 obs 信号不足 (axis_dot 标量缺方向信息)
    # 而不是 reward 量级问题, reward 调参治不了, 必须加 axis 向量 obs.
    pos_in_thresh_count = int(pos_in_thresh.sum().item())
    if pos_in_thresh_count > 0:
        axis_err_in_pos_thresh_mean = float(axis_err[pos_in_thresh].mean())
        axis_err_in_pos_thresh_min = float(axis_err[pos_in_thresh].min())
        axis_gate_in_pos_thresh_mean = float(axis_gate[pos_in_thresh].mean())
    else:
        axis_err_in_pos_thresh_mean = float("nan")
        axis_err_in_pos_thresh_min = float("nan")
        axis_gate_in_pos_thresh_mean = float("nan")

    return {
        "hold_success_rate": float(hold_flags.mean()) if len(hold_flags) else 0.0,
        "max_hold_mean": float(np.mean(ep_max_holds)) if ep_max_holds else 0.0,
        "in_thresh_rate": float(np.mean(ep_in_thresh_rates)) if ep_in_thresh_rates else 0.0,
        "final_in_thresh_rate": float(np.mean(ep_final_in_thresh)) if ep_final_in_thresh else 0.0,
        "pos_err_mean": float(pos_err.mean()),
        "axis_err_mean": float(axis_err.mean()),
        # Stage 1 pos 技能保住程度 — 不被 full success (pos∧axis) 掩盖.
        "pos_success_rate": float(pos_in_thresh.float().mean()),
        # axis 项实际被门控的程度. ≈0 = 还远, axis 不施压; ≈1 = 进 pos 阈, axis 满压.
        "axis_gate_mean": float(axis_gate.mean()),
        # 真正进 reward 的 axis 惩罚量级 (gate * axis_err), 反映 axis 信号强度.
        "gated_axis_penalty_mean": float((axis_gate * axis_err).mean()),
        # 条件指标 — 只看 pos_in_thresh 帧:
        "pos_in_thresh_count": pos_in_thresh_count,
        "axis_err_in_pos_thresh_mean": axis_err_in_pos_thresh_mean,
        "axis_err_in_pos_thresh_min": axis_err_in_pos_thresh_min,
        "axis_gate_in_pos_thresh_mean": axis_gate_in_pos_thresh_mean,
    }


def compute_cost_metrics(dataset, n_eval_episodes):
    """从 eval flatten dataset 的 info.data["cost"] 算 cost_rate / per-ep cost sum.

    依赖 env._create_info_dictionary 把 cost 写进 step_info; flatten 后顺序与
    reward 对齐. cost = 0/1 per-step collision indicator.
    """
    import torch
    cost = dataset.info.data.get("cost")
    if cost is None:
        return {"cost_rate": float("nan"), "cost_episode_sum_mean": float("nan")}
    cost_t = cost if isinstance(cost, torch.Tensor) else torch.as_tensor(cost)
    cost_rate = float(cost_t.float().mean())
    cost_episode_sum_mean = float(cost_t.float().sum()) / max(n_eval_episodes, 1)
    return {"cost_rate": cost_rate, "cost_episode_sum_mean": cost_episode_sum_mean}


def resolve_eval_episode_count(requested_episodes, num_envs, arg_name):
    """评估 episode 数与 vectorized env 对齐.

    `VectorCore.evaluate(n_episodes=...)` 在尾批不足时会把 inactive env
    teleport away. 为了避免渲染里出现"飞天"机器人, 统一要求 eval episode 数
    是 num_envs 的整数倍.
    """
    if requested_episodes is None:
        return num_envs
    if requested_episodes < num_envs:
        raise ValueError(
            f"{arg_name} ({requested_episodes}) 不能小于 num_envs ({num_envs}). "
            "否则 evaluate 的 inactive env 会被 teleport away."
        )
    if requested_episodes % num_envs != 0:
        raise ValueError(
            f"{arg_name} ({requested_episodes}) 必须能被 num_envs ({num_envs}) 整除, "
            "否则最后一批会留下 inactive env 被 teleport away."
        )
    return requested_episodes
