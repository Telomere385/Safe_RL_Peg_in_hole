"""Commit 1 资产健康检查: 验证新 composed USDA 在 articulation 层 **物理 no-op**
同时把 peg/hole 视觉 prim 挂进了 stage.

用法
----
# 基线: 老 USD
python scripts/archive/check_peghole_asset.py --usd assets/usd/dual_arm_iiwa/dual_arm_iiwa.usd \
    > /tmp/check_old.txt

# 新 composed USDA
python scripts/archive/check_peghole_asset.py --usd assets/usd/dual_arm_iiwa/dual_arm_iiwa_with_peghole.usda \
    > /tmp/check_new.txt

# 对比 articulation 数值 (这几行必须完全一样)
diff /tmp/check_old.txt /tmp/check_new.txt
# 预期 diff: 仅 [USD] 行和 [PRIM] 行不同; [NUM_DOF] / [DEFAULT_*] / [G0_*] / [CONTROLLED_JOINTS_IDX] 行完全一致.

# 肉眼验证 peg/hole 位置 (开窗口, 默认 5s 后自动退出)
python scripts/archive/check_peghole_asset.py \
    --usd assets/usd/dual_arm_iiwa/dual_arm_iiwa_with_peghole.usda \
    --render --duration 20

不依赖仓库的改动: env 代码本身已经支持 ``usd_path=`` kwarg, 这里只是把新 USDA
从外部传进去, 不修改 DEFAULT_USD_PATH.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# 归档目录在 scripts/archive/, 项目根需 parents[2] 才到 bimanual_peghole/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--usd", type=str, required=True,
                   help="USD 文件的路径 (仓库相对或绝对均可)")
    p.add_argument("--num_envs", type=int, default=2,
                   help=">=2, 避免 num_envs=1 的 cloner bug")
    p.add_argument("--render", action="store_true",
                   help="打开 IsaacSim 窗口做视觉检查")
    p.add_argument("--duration", type=float, default=5.0,
                   help="--render 下窗口保持的秒数 (<=0 表示到 Ctrl-C)")
    # 用 parse_known_args 而不是 parse_args: IsaacSim/SimulationApp 或 shell
    # 偶尔会把奇怪的额外参数 (包括空字符串) 塞进 sys.argv, 不用硬失败.
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[ARGV_UNKNOWN] sys.argv={sys.argv!r}  unknown={unknown!r}")
    return args


def main():
    args = parse_args()

    usd_path = Path(args.usd)
    if not usd_path.is_absolute():
        usd_path = (PROJECT_ROOT / usd_path).resolve()

    from envs import DualArmPegHoleEnv

    mdp = DualArmPegHoleEnv(
        num_envs=args.num_envs,
        headless=not args.render,
        initial_joint_noise=0.0,
        usd_path=str(usd_path),
    )

    robots = mdp._task.robots
    cj = mdp._task._controlled_joints

    # ---- articulation 数值探针 ----
    # env __init__ 末尾已经 world.step() 过, 姿态已稳定. 因为 initial_joint_noise=0,
    # 两次运行 (旧 USD vs 新 USDA) 应该走完全相同的初始化路径, G(q) 应逐位相等,
    # 除非 peg/hole 真的漏进了动力学 (即 commit 1 的 no-op 承诺被打破).
    num_dof = int(robots.num_dof) if hasattr(robots, "num_dof") else len(mdp._default_joint_pos)
    default_pos = robots.get_joints_default_state().positions[0].detach().cpu()
    G0 = robots.get_generalized_gravity_forces(clone=True).detach().cpu()

    # ---- stage 里 peg/hole 计数 (不用硬编码 cloner 路径) ----
    peg_n = hole_n = tip_n = entry_n = 0
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            for prim in stage.Traverse():
                name = prim.GetName()
                if name == "Peg":
                    peg_n += 1
                elif name == "Hole":
                    hole_n += 1
                elif name == "peg_tip":
                    tip_n += 1
                elif name == "hole_entry":
                    entry_n += 1
    except Exception as e:
        print(f"[PRIM_SCAN_SKIP] {e}")

    print(f"[USD]                    {usd_path}")
    print(f"[NUM_DOF]                {num_dof}")
    print(f"[CONTROLLED_JOINTS_IDX]  {list(map(int, cj))}")
    # 用 6 位有效数字; 够 diff 出 G 变化, 又对纯数值噪声不敏感
    print(f"[DEFAULT_JOINT_POS_norm] {float(default_pos.norm()):.6f}")
    print(f"[DEFAULT_JOINT_POS_sum]  {float(default_pos.sum()):.6f}")
    print(f"[G0_norm]                {float(G0[0].norm()):.6f}")
    print(f"[G0_sum]                 {float(G0[0].sum()):.6f}")
    print(f"[PRIM_PEG_COUNT]         {peg_n}     (expect {args.num_envs} w/ peghole usda, 0 w/ plain)")
    print(f"[PRIM_HOLE_COUNT]        {hole_n}")
    print(f"[PRIM_PEG_TIP_COUNT]     {tip_n}")
    print(f"[PRIM_HOLE_ENTRY_COUNT]  {entry_n}")

    if args.render:
        print(f"[RENDER] window open for {args.duration:.1f}s; 肉眼看 peg(红) 在左指间 / hole(绿) 在右指间, 轴向垂直手指开合面")
        t0 = time.monotonic()
        try:
            while args.duration <= 0 or time.monotonic() - t0 < args.duration:
                mdp._world.step(render=True)
        except KeyboardInterrupt:
            pass

    mdp.stop()


if __name__ == "__main__":
    main()
