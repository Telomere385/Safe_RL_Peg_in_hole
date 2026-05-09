"""SAC actor/critic 网络定义 — train_sac.py 与 eval_sac.py 共用.

放在顶层模块 (而不是 __main__) 是为了让 agent pickle 在 train/eval 两端
都能解析到同一个类路径 `networks.ActorNetwork`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Davide Tateo (mushroom-rl) Discord 建议的 init: 用激活相关的 calculate_gain
# 再除以 10. 对 ReLU 层 gain ≈ √2/10 ≈ 0.1414, 对 linear 输出层 gain = 1/10 = 0.1.
# 目的是 seed consistency, 让不同 seed 的初始 policy 更接近.
_GAIN_RELU = nn.init.calculate_gain("relu") / 10.0      # ≈ 0.1414
_GAIN_LINEAR = nn.init.calculate_gain("linear") / 10.0  # = 0.1


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features=256, **_):
        super().__init__()
        self._h1 = nn.Linear(input_shape[0], n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, output_shape[0])
        nn.init.xavier_uniform_(self._h1.weight, gain=_GAIN_RELU)
        nn.init.xavier_uniform_(self._h2.weight, gain=_GAIN_RELU)
        nn.init.xavier_uniform_(self._out.weight, gain=_GAIN_LINEAR)
        for l in (self._h1, self._h2, self._out):
            nn.init.zeros_(l.bias)

    def forward(self, x, **_):
        x = x.float()
        h = F.relu(self._h1(x))
        h = F.relu(self._h2(h))
        return self._out(h)


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, action_dim, n_features=256, **_):
        super().__init__()
        self._h1 = nn.Linear(input_shape[0] + action_dim, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, output_shape[0])
        nn.init.xavier_uniform_(self._h1.weight, gain=_GAIN_RELU)
        nn.init.xavier_uniform_(self._h2.weight, gain=_GAIN_RELU)
        nn.init.xavier_uniform_(self._out.weight, gain=_GAIN_LINEAR)
        for l in (self._h1, self._h2, self._out):
            nn.init.zeros_(l.bias)

    def forward(self, state, action, **_):
        state = state.float()
        h = F.relu(self._h1(torch.cat([state, action.float()], dim=-1)))
        h = F.relu(self._h2(h))
        return self._out(h).squeeze(-1)
