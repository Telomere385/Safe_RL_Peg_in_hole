"""M2b/M2c clearance 诊断 — 用 sphere proxy 真 clearance 评估 cross-over 程度.

历史背景: M2b checkpoint 数值通过 (hold_success_rate ≥ 0.9), 但可视化里两臂交叠.
原因 PhysX 接触力检测对"贴近但未穿透"失明 (验证: @1N collision rate = 0%).
本脚本现在用 env._compute_min_clearance() 的 19 球 proxy (减半径) 算真 clearance,
对比单纯 link origin 距离, 给出 M2c curriculum 起步建议.

主指标: success 帧 min_clearance 累计 (≥+5cm / +2cm / 0cm / -2cm), 由此输出
M2c.1 / .2 / .3 起步参数 (见 verdict 段). link origin 距离仅作为对照保留 ——
它会被半径欺骗 (5cm origin 距离对应 -1cm 真 clearance).

采集 (per-step × per-env, 全 num_envs 聚合):
    pos_err / axis_err / success_mask              # 来自 _compute_task_errors
    min_link_origin_dist (11×11 link 原点)         # 对照, 不是真 clearance
    min_clearance (19 球 sphere proxy, 已减半径)   # M2c reward / hard absorbing 的真信号
    ee_dist (hande_link ↔ hande_link)
    physx_collision_at_10N / at_1N                 # 印证 PhysX 失明

实现: monkey-patch _create_observation 每步收集. 不改 env, 完全只读.

用法:
    conda activate safe_rl
    python scripts/archive/diagnose_m2b_clearance.py --headless --num_envs 16 --n_episodes 32
    python scripts/archive/diagnose_m2b_clearance.py --agent_path results/best_agent_M2b_axis02.msh
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# 归档目录在 scripts/archive/, 项目根需 parents[2] 才到 bimanual_peghole/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._eval_utils import deterministic_policy, resolve_eval_episode_count


# 扩展版: arm_link_1..7 + coupler + hande_link + 2 根 finger (每侧 11 个 link).
# 与 env.LEFT_ARM_GROUP 比多了 coupler 和 4 根 finger — finger 是这次新加的关键,
# 因为 EE↔EE 距离 ~11cm 时 finger 视觉体可能撞到对侧 hande/finger.
_LEFT_LINK_PATHS = (
    [f"/left_arm_link_{i}" for i in range(1, 8)]
    + ["/left_hande_robotiq_hande_coupler"]
    + ["/left_hande_robotiq_hande_link"]
    + ["/left_hande_robotiq_hande_left_finger",
       "/left_hande_robotiq_hande_right_finger"]
)
_RIGHT_LINK_PATHS = (
    [f"/right_arm_link_{i}" for i in range(1, 8)]
    + ["/right_hande_robotiq_hande_coupler"]
    + ["/right_hande_robotiq_hande_link"]
    + ["/right_hande_robotiq_hande_left_finger",
       "/right_hande_robotiq_hande_right_finger"]
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_path", type=str,
                   default=str(PROJECT_ROOT / "results/best_agent_M2b_axis02.msh"))
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--n_episodes", type=int, default=None,
                   help="默认 = num_envs.")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--render", action="store_true",
                   help="覆盖 --headless")
    # M2b stage flags — 必须与训练一致, 否则 success_mask 触发条件不同
    p.add_argument("--preinsert_success_pos_threshold", type=float, default=0.10)
    p.add_argument("--success_axis_threshold", type=float, default=0.2)
    p.add_argument("--rew_axis", type=float, default=2.0,
                   help="参与 env reward 算 J/R, 但 clearance 诊断本身不看 reward. "
                        "为与训练 logger 一致, 默认 2.0 与 M2a/M2b 训练保持一致.")
    p.add_argument("--terminal_hold_bonus", type=float, default=50.0,
                   help="必须与 train 一致, 否则 absorbing 行为不同.")
    p.add_argument("--hold_success_steps", type=int, default=10)
    p.add_argument("--proxy_arm_radius", type=float, default=None,
                   help="M2c sphere proxy 的 arm/link 半径 (m). 默认 0.06.")
    p.add_argument("--proxy_ee_radius", type=float, default=None,
                   help="M2c sphere proxy 的 EE/finger 半径 (m). 默认 0.03.")
    # 诊断阈值 — verdict 现在用 sphere proxy clearance (mc) 多阈值评估, 不再依赖
    # 单一 link_dist 阈值. CLI 参数留 collision_force_thresh_low 即可.
    p.add_argument("--collision_force_thresh_low", type=float, default=1.0,
                   help="额外打一个低阈值 PhysX collision, 看是不是因为 10N 太高漏抓.")
    return p.parse_args()


def _percentile(arr, q):
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def _print_dist(name, arr, unit=""):
    if arr.size == 0:
        print(f"  {name:22s} (空)")
        return
    print(
        f"  {name:22s} n={arr.size:6d}  "
        f"mean={float(arr.mean()):+.4f}{unit}  "
        f"median={_percentile(arr, 50):+.4f}{unit}  "
        f"min={float(arr.min()):+.4f}{unit}  "
        f"max={float(arr.max()):+.4f}{unit}  "
        f"p10={_percentile(arr, 10):+.4f}{unit}  "
        f"p90={_percentile(arr, 90):+.4f}{unit}"
    )


def main():
    args = parse_args()
    if args.render:
        args.headless = False
    args.n_episodes = resolve_eval_episode_count(
        args.n_episodes, args.num_envs, "--n_episodes"
    )
    if not Path(args.agent_path).is_file():
        raise FileNotFoundError(f"agent_path 不存在: {args.agent_path}")

    from envs import DualArmPegHoleEnv

    env_kwargs = dict(
        num_envs=args.num_envs,
        headless=args.headless,
        preinsert_success_pos_threshold=args.preinsert_success_pos_threshold,
        success_axis_threshold=args.success_axis_threshold,
        rew_axis=args.rew_axis,
        terminal_hold_bonus=args.terminal_hold_bonus,
        success_hold_steps=args.hold_success_steps,
    )
    if args.proxy_arm_radius is not None:
        env_kwargs["proxy_arm_radius"] = args.proxy_arm_radius
    if args.proxy_ee_radius is not None:
        env_kwargs["proxy_ee_radius"] = args.proxy_ee_radius
    print(f"[DIAG ENV] {env_kwargs}")
    print(f"[DIAG] agent_path: {args.agent_path}")
    print(f"[DIAG] 全 {args.num_envs} env 聚合采样 (不再只看单 env)")

    mdp = DualArmPegHoleEnv(**env_kwargs)

    # 第一版用 XFormPrim(usd=False) 读, 但 cloned articulation 子 prim 的 Fabric
    # 没自动 publish, 所有帧都读到 init pose (min_link_dist 全等于肩对肩静态值,
    # ee_dist 全等于 2.14m). 改走 articulation physics_view (mushroom BODY_POS
    # 用的同一路径), 这是 live 的.
    robots = mdp._task.robots
    body_names = list(robots.body_names)

    def _resolve(path):
        name = path.lstrip("/")
        if name not in body_names:
            raise RuntimeError(
                f"link 名 '{name}' 不在 robots.body_names 里. available: {body_names}"
            )
        return body_names.index(name)

    left_idx = [_resolve(p) for p in _LEFT_LINK_PATHS]
    right_idx = [_resolve(p) for p in _RIGHT_LINK_PATHS]
    # ee_dist 取 hande_link (不是 finger). 用 path string 找对应位置, 避免硬编码下标.
    left_ee_local = _LEFT_LINK_PATHS.index("/left_hande_robotiq_hande_link")
    right_ee_local = _RIGHT_LINK_PATHS.index("/right_hande_robotiq_hande_link")
    print(f"[DIAG] body_names ({len(body_names)} bodies): {body_names}")
    print(f"[DIAG] left link col idx:  {left_idx}")
    print(f"[DIAG] right link col idx: {right_idx}")
    print(f"[DIAG] hande_link local idx (ee_dist 使用): left={left_ee_local} right={right_ee_local}")

    # 探测 physics_view API: get_link_transforms() 返回的形状版本之间不一定一致.
    # 先 dry-run 一次看 shape, 写好 reshape 逻辑.
    physics_view = robots._physics_view
    test = physics_view.get_link_transforms()
    print(f"[DIAG] physics_view.get_link_transforms() shape: {tuple(test.shape)}  dtype: {test.dtype}")
    n_envs = mdp._n_envs
    n_bodies = len(body_names)
    if test.dim() == 3 and test.shape[0] == n_envs and test.shape[1] == n_bodies:
        link_layout = "env_body"
    elif test.dim() == 2 and test.shape[0] == n_envs * n_bodies:
        link_layout = "flat"
    else:
        raise RuntimeError(
            f"未知 link_transforms 形状 {tuple(test.shape)} "
            f"(n_envs={n_envs}, n_bodies={n_bodies})"
        )
    print(f"[DIAG] link_transforms layout: {link_layout}")

    # 采样存储 — 全 env 聚合, 每条记录是 [n_envs] 维度, 最后 concatenate.
    samples = {
        "pos_err": [],
        "axis_err": [],
        "success": [],
        "min_link_dist": [],     # 旧版: link origin 两两 min 距离 (无半径)
        "min_link_pair_l": [],   # 哪个 left link 离 right 最近 (per env)
        "min_link_pair_r": [],
        "ee_dist": [],           # hande_link ↔ hande_link 距离
        # M2c sphere-proxy clearance (commit 1 验证: 与 min_link_dist 对比)
        "min_clearance": [],     # 19 球 proxy: dist - r_L - r_R, 减半径后的真 clearance
        "min_clearance_pair_l": [],
        "min_clearance_pair_r": [],
        "physx_coll_10": [],
        "physx_coll_low": [],
    }
    sample_failures = {"count": 0, "first_error": None}
    original_create_obs = mdp._create_observation
    low_thresh = args.collision_force_thresh_low

    def patched_create_obs(raw_obs):
        agent_obs = original_create_obs(raw_obs)
        try:
            xforms = physics_view.get_link_transforms()  # [...] xyz+quat
            xforms_t = torch.as_tensor(xforms, device=mdp._device, dtype=torch.float32)
            if link_layout == "env_body":
                pos_all = xforms_t[..., :3]                  # [n_envs, n_bodies, 3]
            else:
                pos_all = xforms_t[..., :3].view(n_envs, n_bodies, 3)
            left_pos = pos_all[:, left_idx, :]    # [n_envs, nL, 3]
            right_pos = pos_all[:, right_idx, :]  # [n_envs, nR, 3]
            # 两两距离: [n_envs, nL, nR]
            diff = left_pos.unsqueeze(2) - right_pos.unsqueeze(1)
            dist = diff.norm(dim=-1)
            nR = dist.shape[2]
            min_flat = dist.view(n_envs, -1).argmin(dim=1)   # [n_envs]
            min_vals = dist.view(n_envs, -1).gather(1, min_flat.unsqueeze(1)).squeeze(1)
            li = (min_flat // nR).cpu().numpy()              # [n_envs]
            ri = (min_flat % nR).cpu().numpy()

            ee_dist = (left_pos[:, left_ee_local, :] - right_pos[:, right_ee_local, :]).norm(dim=-1)

            # M2c sphere-proxy clearance — env 内置, 与上面 11 link origin 距离对比.
            min_clear, clear_info = mdp._compute_min_clearance()

            pos_err, axis_err, success_mask = mdp._compute_task_errors(agent_obs)
            samples["pos_err"].append(pos_err.detach().cpu().numpy().copy())
            samples["axis_err"].append(axis_err.detach().cpu().numpy().copy())
            samples["success"].append(success_mask.detach().cpu().numpy().astype(bool).copy())
            samples["min_link_dist"].append(min_vals.detach().cpu().numpy().copy())
            samples["min_link_pair_l"].append(li.copy())
            samples["min_link_pair_r"].append(ri.copy())
            samples["ee_dist"].append(ee_dist.detach().cpu().numpy().copy())
            samples["min_clearance"].append(min_clear.detach().cpu().numpy().copy())
            samples["min_clearance_pair_l"].append(
                clear_info["min_pair_left_idx"].detach().cpu().numpy().copy()
            )
            samples["min_clearance_pair_r"].append(
                clear_info["min_pair_right_idx"].detach().cpu().numpy().copy()
            )

            # PhysX collision @ 两个阈值. _check_collision 用显式 threshold 入参,
            # 不读 self._collision_threshold, 所以不需要 saved/restore.
            c10 = mdp._check_collision("arm_L", "arm_R", 10.0, dt=mdp._timestep)
            cl = mdp._check_collision("arm_L", "arm_R", low_thresh, dt=mdp._timestep)
            samples["physx_coll_10"].append(c10.detach().cpu().numpy().astype(bool).copy())
            samples["physx_coll_low"].append(cl.detach().cpu().numpy().astype(bool).copy())
        except Exception as e:
            sample_failures["count"] += 1
            if sample_failures["first_error"] is None:
                sample_failures["first_error"] = repr(e)
        return agent_obs

    mdp._create_observation = patched_create_obs

    from mushroom_rl.core import Agent, VectorCore
    agent = Agent.load(args.agent_path)
    core = VectorCore(agent, mdp)

    print(f"[DIAG] 跑 {args.n_episodes} episode (deterministic policy)...")
    with deterministic_policy(agent):
        core.evaluate(n_episodes=args.n_episodes,
                      render=not args.headless, quiet=False)

    mdp._create_observation = original_create_obs
    mdp.stop()

    if sample_failures["count"] > 0:
        raise RuntimeError(
            f"采集失败 {sample_failures['count']} 次. first error: {sample_failures['first_error']}"
        )
    if not samples["pos_err"]:
        raise RuntimeError("采集为空 — patched _create_observation 没被调用.")

    pos = np.concatenate(samples["pos_err"])
    ax = np.concatenate(samples["axis_err"])
    suc = np.concatenate(samples["success"]).astype(bool)
    md = np.concatenate(samples["min_link_dist"])
    lpair = np.concatenate(samples["min_link_pair_l"])
    rpair = np.concatenate(samples["min_link_pair_r"])
    ee_dist = np.concatenate(samples["ee_dist"])
    mc = np.concatenate(samples["min_clearance"])
    mc_lpair = np.concatenate(samples["min_clearance_pair_l"])
    mc_rpair = np.concatenate(samples["min_clearance_pair_r"])
    c10 = np.concatenate(samples["physx_coll_10"]).astype(bool)
    clo = np.concatenate(samples["physx_coll_low"]).astype(bool)

    n_total = pos.size
    n_suc = int(suc.sum())
    print()
    print(f"[DIAG] 采样 step 数 (全 env 聚合) = {n_total}")
    print(f"[DIAG] success step 数 = {n_suc} ({100.0 * n_suc / n_total:.2f}%) "
          f"(pos<{args.preinsert_success_pos_threshold:.3f} ∧ axis<{args.success_axis_threshold:.3f})")
    print(f"[DIAG] PhysX collision @ 10N rate = {100.0 * c10.mean():.2f}%")
    print(f"[DIAG] PhysX collision @ {low_thresh}N rate = {100.0 * clo.mean():.2f}%")

    print()
    print("[DIAG 整体分布 — 全部 step]")
    _print_dist("pos_err",        pos, "m")
    _print_dist("axis_err",       ax)
    _print_dist("min_link_dist",  md, "m")
    _print_dist("min_clearance",  mc, "m")
    _print_dist("ee_dist",        ee_dist, "m")

    if n_suc > 0:
        print()
        print("[DIAG 子集 — success step]  ← M2c 决策的关键")
        _print_dist("pos_err",       pos[suc], "m")
        _print_dist("axis_err",      ax[suc])
        _print_dist("min_link_dist", md[suc], "m")
        _print_dist("min_clearance", mc[suc], "m")
        _print_dist("ee_dist",       ee_dist[suc], "m")

        # 判定: 几个 clearance 阈值下的 success-而且-clear 帧数
        print()
        print("[DIAG] success 帧的 min_link_dist 分桶 (link origin, 不减半径):")
        for th in (0.005, 0.01, 0.03, 0.05, 0.10):
            n = int((md[suc] >= th).sum())
            pct = 100.0 * n / n_suc
            print(f"  min_link_dist ≥ {th*100:5.1f}cm   {n:5d}/{n_suc}  ({pct:5.1f}%)")

        # M2c sphere-proxy clearance 分桶 (减了半径, 是 M2c reward / hard absorbing
        # 的真实信号)
        print()
        print("[DIAG] success 帧的 min_clearance 分桶 (sphere proxy, 已减半径):")
        for th in (-0.02, 0.0, 0.01, 0.02, 0.05, 0.10):
            n = int((mc[suc] >= th).sum())
            pct = 100.0 * n / n_suc
            print(f"  min_clearance ≥ {th*100:+5.1f}cm   {n:5d}/{n_suc}  ({pct:5.1f}%)")

        # 一致性 sanity: clearance ≈ link_dist - r_L - r_R; trend 应一致.
        # 不要求逐点相等 (link_dist 是 11-link origin, clearance 是 19 球 proxy,
        # 选中的 pair 不一定一样), 但相关性应明显.
        corr_all = float(np.corrcoef(md, mc)[0, 1]) if md.size > 1 else float("nan")
        corr_suc = float(np.corrcoef(md[suc], mc[suc])[0, 1]) if n_suc > 1 else float("nan")
        print()
        print(f"[DIAG] link_dist vs clearance 相关性 (sanity check):  "
              f"all={corr_all:+.3f}  success={corr_suc:+.3f}  (期望 > 0.7)")

        # 哪几对 link 最常贴在一起 (link origin 11×11)
        print()
        print("[DIAG] success 帧里 min_link_pair 频次 top-5 (link origin):")
        from collections import Counter
        pairs = Counter(zip(lpair[suc].tolist(), rpair[suc].tolist()))
        names_l = [p.lstrip("/") for p in _LEFT_LINK_PATHS]
        names_r = [p.lstrip("/") for p in _RIGHT_LINK_PATHS]
        for (l, r), c in pairs.most_common(5):
            print(f"  {names_l[l]:<32s} ↔ {names_r[r]:<32s}  count={c}")

        # sphere proxy 选中的 pair 频次 top-5 (19 球: 8 关节 + 7 中点 + 4 EE).
        # idx 顺序与 _gather_side_proxies 保持一致.
        proxy_names = (
            [f"left_arm_link_{i}" for i in range(8)]                  # 0..7  关节
            + [f"left_arm_link_{i}-{i+1}_mid" for i in range(7)]      # 8..14 中点
            + ["left_coupler", "left_hande_link",
               "left_finger_L", "left_finger_R"]                      # 15..18
        )
        proxy_names_r = [n.replace("left_", "right_") for n in proxy_names]
        print()
        print("[DIAG] success 帧里 min_clearance_pair 频次 top-5 (sphere proxy):")
        proxy_pairs = Counter(zip(mc_lpair[suc].tolist(), mc_rpair[suc].tolist()))
        for (l, r), c in proxy_pairs.most_common(5):
            print(f"  {proxy_names[l]:<28s} ↔ {proxy_names_r[r]:<28s}  count={c}")

        # PhysX collision 在 success 时是否触发
        c10_s = c10[suc].mean()
        clo_s = clo[suc].mean()
        print()
        print(f"[DIAG] success 帧 PhysX coll @ 10N: {100.0 * c10_s:.2f}%")
        print(f"[DIAG] success 帧 PhysX coll @ {low_thresh}N: {100.0 * clo_s:.2f}%")

        # verdict — 用 sphere proxy clearance (mc) 而不是 link origin (md).
        # 老版 verdict 用 md ≥ 5cm 会被半径欺骗 (5cm origin 距离 + 6cm 总半径
        # = -1cm 真 clearance), 误判 "存在 40% 非交叉解".
        ge_5cm = float((mc[suc] >= 0.05).mean())
        ge_2cm = float((mc[suc] >= 0.02).mean())
        ge_0cm = float((mc[suc] >= 0.0).mean())
        ge_neg2 = float((mc[suc] >= -0.02).mean())
        print()
        print("[DIAG VERDICT] success 帧 min_clearance (sphere proxy 真 clearance) 累计:")
        print(f"  ≥ +5cm: {ge_5cm:.3f}    ≥ +2cm: {ge_2cm:.3f}    "
              f"≥  0cm: {ge_0cm:.3f}    ≥ -2cm: {ge_neg2:.3f}")
        print()
        if ge_5cm >= 0.20:
            print("→ checkpoint 已有 ≥20% 完全安全 (≥5cm) success. M2c 可直接上目标参数:")
            print("    --rew_clearance 2.0 --clearance_soft 0.05 --clearance_hard 0.02")
        elif ge_0cm >= 0.20:
            print("→ checkpoint 有 ≥20% sphere 不穿插 (≥0cm) success, 但完全安全比例不足.")
            print("  M2c curriculum 建议:")
            print("    M2c.1: --clearance_soft 0.02 --clearance_hard 0.00")
            print("    M2c.2: --clearance_soft 0.05 --clearance_hard 0.02   (--load_agent M2c.1)")
        elif ge_neg2 >= 0.50:
            print("→ checkpoint 多数 success 处于 sphere 浅穿插 (≥50% @ ≥-2cm).")
            print("  M2c curriculum 必须从 soft<0 起步:")
            print("    M2c.1: --clearance_soft 0.00 --clearance_hard -0.03  (推到 just touching)")
            print("    M2c.2: --clearance_soft 0.02 --clearance_hard 0.00   (--load_agent M2c.1)")
            print("    M2c.3: --clearance_soft 0.05 --clearance_hard 0.02   (--load_agent M2c.2)")
        else:
            print("→ checkpoint 大部分 success 严重穿插 (<50% @ ≥-2cm). M2c 难直接续训.")
            print("  建议从 M2a checkpoint (cross-over 还没固化) warm-start, 不要从这个续.")
    else:
        print()
        print("[DIAG] 警告: 没有 success step. M2b checkpoint 在这次 eval 里没达标 — "
              "确认 --preinsert_success_pos_threshold / --success_axis_threshold 与训练时一致.")


if __name__ == "__main__":
    main()
