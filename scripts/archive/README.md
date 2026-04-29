# scripts/archive/ — 一次性诊断脚本归档

主线 (train/eval/visualize/_eval_utils) 不依赖这里。这些脚本是各 stage
切换前一次性验证用的；结论已沉淀进主 README。**保留在这里是为了将来改资产
/ 改 sphere proxy / 重新验证 PhysX 失明时能复用，不是为日常运行**。

| 脚本 | 写它的 stage | 验证了什么 |
|---|---|---|
| `diagnose_m1_axis.py` | M1 → M2 切换前 | 31 维 M1 checkpoint 在 32 维 env 跑 axis_err 分布 (回答 "M1 隐式带来的姿态对齐有多差") |
| `diagnose_m2b_clearance.py` | M2b → M2c 切换前 | sphere proxy 真 clearance vs PhysX 接触力检测的差距, M2c curriculum 起步阈值 |
| `check_peghole_asset.py` | M0 期 | 新 composed USDA 是 articulation 层 no-op (`diff` 旧/新 USD 的 DOF / G(q) / default_pos) |

**何时再跑它们**:
- 重做 sphere proxy (改半径 / 改球数 / 加 capsule) → `diagnose_m2b_clearance.py` 重新评估 cross-over 比例
- 改 USD 资产 (重新生成 `dual_arm_iiwa_with_peghole.usda`) → `check_peghole_asset.py` 验证 articulation 物理 no-op
- 想重启 31 维 M1 baseline → `diagnose_m1_axis.py` (但当前主线已不用 31 维)

CLI 默认路径假设你 `cd /home/miao/bimanual_peghole` 后 `python scripts/archive/<name>.py`。
