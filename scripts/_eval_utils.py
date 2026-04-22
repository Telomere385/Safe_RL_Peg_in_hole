"""train_sac.py 与 eval_sac.py 共用的 eval 工具.

- deterministic_policy: SAC 评估时把 tanh-Gaussian 策略替换成 tanh(mu).
- compute_hold_metrics: 从 evaluate() 返回的 flatten 后的 dataset 算 hold-N success
  与每步 in-threshold/位置误差统计.
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

    依赖 VectorCore.evaluate 末尾返回的 dataset 是 flatten 后的 1D, 每个 env 的
    transitions 在结果里连续 (pack_padded_sequence 按 env 优先排列), 所以
    `flatnonzero(last)` + 顺序切片就能正确分段.
    """
    _, _, _, next_state, _, last = dataset.parse(to="torch")
    left_err, right_err, in_thresh = mdp._compute_task_errors(next_state)

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
        "left_pos_err_mean": float(left_err.mean()),
        "right_pos_err_mean": float(right_err.mean()),
    }
