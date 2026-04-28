"""train_sac.py 与 eval_sac.py 共用的 eval 工具.

- deterministic_policy: SAC 评估时把 tanh-Gaussian 策略替换成 tanh(mu).
- compute_hold_metrics: 从 evaluate() 返回的 flatten 后的 dataset 算 hold-N
  success 与每步 in-threshold / pos_err / axis_err 统计.
  in_thresh = (pos_err < pos_th) ∧ (axis_err < axis_th); axis_th=inf 退化为
  pos-only (M1' 行为).
"""

from contextlib import contextmanager

import numpy as np
import torch


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
    退化为 pos-only (M1' 行为).

    依赖 VectorCore.evaluate 末尾返回的 dataset 是 flatten 后的 1D, 每个 env 的
    transitions 在结果里连续, 所以 `flatnonzero(last)` + 顺序切片就能正确分段.
    """
    _, _, _, next_state, _, last = dataset.parse(to="torch")
    pos_err, axis_err, in_thresh = mdp._compute_task_errors(next_state)

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
    return {
        "hold_success_rate": float(hold_flags.mean()) if len(hold_flags) else 0.0,
        "max_hold_mean": float(np.mean(ep_max_holds)) if ep_max_holds else 0.0,
        "in_thresh_rate": float(np.mean(ep_in_thresh_rates)) if ep_in_thresh_rates else 0.0,
        "final_in_thresh_rate": float(np.mean(ep_final_in_thresh)) if ep_final_in_thresh else 0.0,
        "pos_err_mean": float(pos_err.mean()),
        "axis_err_mean": float(axis_err.mean()),
    }


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
