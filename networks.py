"""SAC actor/critic 网络定义 — train_sac.py 与 eval_sac.py 共用.

放在顶层模块 (而不是 __main__) 是为了让 agent pickle 在 train/eval 两端
都能解析到同一个类路径 `networks.ActorNetwork`.

obs 归一化设计 (可选):
    构造时若传 `obs_scale=[...]`, 注册成 buffer, forward 第一行除以它.
    不传则**不注册 buffer**, forward 跳过归一化. 这样:
    - baseline 实验 (不归一化): 不传 obs_scale, 等价于原版网络
    - 归一化实验: 传 env.get_obs_scale() 的输出, fixed-divisor, 不学统计量
    - 老 pickle (无 buffer 字段): forward 里 hasattr 判定后自动 fallback
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features=256, obs_scale=None, **_):
        super().__init__()
        self._h1 = nn.Linear(input_shape[0], n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, output_shape[0])
        for l in (self._h1, self._h2, self._out):
            nn.init.xavier_uniform_(l.weight)
        if obs_scale is not None:
            self.register_buffer(
                "_obs_scale", torch.as_tensor(obs_scale, dtype=torch.float32)
            )

    def forward(self, x, **_):
        x = x.float()
        if hasattr(self, "_obs_scale"):
            x = x / self._obs_scale
        h = F.relu(self._h1(x))
        h = F.relu(self._h2(h))
        return self._out(h)


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, action_dim, n_features=256,
                 obs_scale=None, **_):
        super().__init__()
        self._h1 = nn.Linear(input_shape[0] + action_dim, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, output_shape[0])
        for l in (self._h1, self._h2, self._out):
            nn.init.xavier_uniform_(l.weight)
        if obs_scale is not None:
            self.register_buffer(
                "_obs_scale", torch.as_tensor(obs_scale, dtype=torch.float32)
            )

    def forward(self, state, action, **_):
        state = state.float()
        if hasattr(self, "_obs_scale"):
            state = state / self._obs_scale
        h = F.relu(self._h1(torch.cat([state, action.float()], dim=-1)))
        h = F.relu(self._h2(h))
        return self._out(h).squeeze(-1)
