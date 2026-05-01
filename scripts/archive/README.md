# scripts/archive/ — 一次性诊断脚本归档

主线 (train/eval/visualize/_eval_utils) 不依赖这里。**保留在这里是为了将来
改 USD 资产时复用, 不是为日常运行**。

| 脚本 | 写它的 stage | 验证了什么 |
|---|---|---|
| `check_peghole_asset.py` | M0 期 | 新 composed USDA 是 articulation 层 no-op (`diff` 旧/新 USD 的 DOF / G(q) / default_pos) |

**何时再跑它**:
- 改 USD 资产 (重新生成 `dual_arm_iiwa_with_peghole.usda`) → `check_peghole_asset.py` 验证 articulation 物理 no-op

CLI 默认路径假设你 `cd /home/miao/bimanual_peghole` 后 `python scripts/archive/<name>.py`。
