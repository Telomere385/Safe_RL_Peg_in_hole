"""生成 dual_arm_iiwa_with_peghole.usda — 机器人 + peg/hole 几何的 composed 资产.

最初的 Phase 1.5 版本只加 visual peg/hole 与定位帧。Stage 3 起, 本脚本同时
生成 peg/hole collision proxy。

-- 关键设计: peg/hole 是 EE link 的附属几何 -----------------------------------
本文件通过 USD references 引用原始 ``dual_arm_iiwa.usd`` (robot 0 修改),
然后用 ``over`` 在左/右 EE link 下挂 Peg / Hole prim. Peg 和 Hole 本身:
  * 没有 RigidBodyAPI  -> 不是动力学刚体, 不进入 articulation 计算
  * 没有 MassAPI        -> 对 G(q) 重力补偿 0 贡献
但 Stage 3 起, 它们会带 invisible collision 子 prim, 作为对应 EE link 的
附属 collision shapes, 用来产生 peg-hole 真实接触。

注意: 这不改变 articulation DoF / 质量矩阵, 但会改变 collider 集合。Stage 3
环境需要把 peg-hole 正常接触从 "双臂自碰撞 hard absorbing" 中排除。

-- 稳定的 prim 路径 (env 代码在后续 commit 引用) ----------------------------
  /bh_robot/left_hande_robotiq_hande_link/Peg
  /bh_robot/left_hande_robotiq_hande_link/Peg/peg_tip
  /bh_robot/right_hande_robotiq_hande_link/Hole
  /bh_robot/right_hande_robotiq_hande_link/Hole/hole_entry

-- 几何与挂载参数 -----------------------------------------------------------
  PEG:  实心圆柱, radius=0.016, height=0.070, 轴 = local Z.
  HOLE: 空心圆柱 mesh, outer_r=0.024, inner_r=0.020, height=0.060,
        底封顶开 (方便 peg 从 +Z 插入).
        Collision: peg 用 Cylinder; hole 内壁用多个 invisible Cube 拼环,
        避免用移动 concave mesh 做碰撞。
  挂载 (Step 2 新约定): translate = (PART_X, 0, PART_Z), orient = identity.
        结果: peg/hole 的轴 (local Z) 直接对齐 EE 的 +Z 方向 = 夹爪正前方.
        reward 中的理想插入对齐定义为两侧轴在 world 中反平行
        (axis_dot=-1, axis_err=0).

-- 运行 ---------------------------------------------------------------------
  python assets/usd/dual_arm_iiwa/build_peghole_usd.py
输出 dual_arm_iiwa_with_peghole.usda 会放到本脚本同目录.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# 几何 ----------------------------------------------------------------------
# Step 2 重设计: 等比 2× 放大. hole_outer 24mm 半径 = 48mm 直径,
# 给 Robotiq Hande 50mm 最大开度留 2mm 安全余量.
PEG_RADIUS   = 0.016
PEG_HEIGHT   = 0.070
HOLE_OUTER_R = 0.024
HOLE_INNER_R = 0.020
HOLE_HEIGHT  = 0.060

# 挂载偏移 (EE 本地帧, 应用 orient 之前) --------------------------------------
PART_X = -0.0055   # 补偿夹爪 mimic 不对称的横向微偏
PART_Z =  0.155    # 沿 EE +Z 的偏移; 让零件后端坐在 finger 末端,
                   # finger 只夹住最后 ~3cm, 前段全部伸出.

# identity orient: quat (w, x, y, z) = (1, 0, 0, 0). 不再做 R_x(+90°), 让
# peg/hole 的 local +Z 直接对齐 EE +Z (夹爪正前方).
_C = 1.0
_S = 0.0

# 颜色 ----------------------------------------------------------------------
PEG_COLOR  = (0.85, 0.12, 0.10)   # red
HOLE_COLOR = (0.20, 0.75, 0.20)   # green

# hollow cylinder mesh 分段数 --------------------------------------------------
NUM_SEGMENTS = 48

# PhysX collision proxy -------------------------------------------------------
# Peg 是简单 cylinder collider; Hole 不直接用 concave Mesh collider, 而是用
# 盒子拼成内壁环。这样 collision shapes 都是 convex/simple, 对移动 articulation
# 更稳定。contactOffset 必须小于 peg/hole 径向余量 (4mm), 否则会把有效孔径吃掉。
HOLE_COLLISION_SEGMENTS = 16
HOLE_BOTTOM_COLLIDER_HEIGHT = 0.003
PHYSX_CONTACT_OFFSET = 0.001
PHYSX_REST_OFFSET = 0.0

HERE = Path(__file__).resolve().parent
ROBOT_USD_REL = "./dual_arm_iiwa.usd"      # 被引用的原始 robot USD (同目录)
OUTPUT = HERE / "dual_arm_iiwa_with_peghole.usda"


def hollow_cylinder_mesh(outer_r: float, inner_r: float, height: float,
                         n_seg: int = NUM_SEGMENTS):
    """底封顶开的空心圆柱 mesh. 镜像 init_peg_hole_env.create_hollow_cylinder().

    顶点索引约定:
        [0..n_seg)          底面外环
        [n_seg..2n_seg)     底面内环
        [2n_seg..3n_seg)    顶面外环
        [3n_seg..4n_seg)    顶面内环
        4n_seg              底部中心 (用于 triangle fan 封底)
    """
    half_h = 0.5 * height
    angles = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False)
    ca, sa = np.cos(angles), np.sin(angles)

    pts: list[tuple[float, float, float]] = []
    for r, z in [(outer_r, -half_h), (inner_r, -half_h),
                 (outer_r,  half_h), (inner_r,  half_h)]:
        for i in range(n_seg):
            pts.append((float(r * ca[i]), float(r * sa[i]), float(z)))
    pts.append((0.0, 0.0, float(-half_h)))  # 底部中心

    center = 4 * n_seg
    fvc: list[int] = []
    fvi: list[int] = []
    for i in range(n_seg):
        j = (i + 1) % n_seg
        # 外壁 quad
        fvc.append(4); fvi.extend([i, j, 2 * n_seg + j, 2 * n_seg + i])
        # 内壁 quad
        fvc.append(4); fvi.extend([n_seg + i, 3 * n_seg + i, 3 * n_seg + j, n_seg + j])
        # 顶面环 quad
        fvc.append(4); fvi.extend([2 * n_seg + i, 2 * n_seg + j,
                                    3 * n_seg + j, 3 * n_seg + i])
        # 底面外环 quad
        fvc.append(4); fvi.extend([i, n_seg + i, n_seg + j, j])
        # 底面中心 triangle (封底)
        fvc.append(3); fvi.extend([center, n_seg + j, n_seg + i])
    return pts, fvc, fvi


def _fmt_vec3_array(pts) -> str:
    return "[" + ", ".join(f"({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})" for p in pts) + "]"


def _fmt_int_array(arr, per_line: int = 20) -> str:
    lines: list[str] = []
    for i in range(0, len(arr), per_line):
        lines.append(", ".join(str(x) for x in arr[i:i + per_line]))
    return "[" + ",\n                    ".join(lines) + "]"


def _fmt_collision_api_attrs(indent: str) -> str:
    return (
        f"{indent}float physxCollision:contactOffset = {PHYSX_CONTACT_OFFSET:.6f}\n"
        f"{indent}float physxCollision:restOffset = {PHYSX_REST_OFFSET:.6f}"
    )


def _peg_collision_usda(indent: str = "            ") -> str:
    i = indent
    attrs = _fmt_collision_api_attrs(i + "    ")
    return f"""\
{i}def Cylinder "Collision" (
{i}    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]
{i})
{i}{{
{i}    double radius = {PEG_RADIUS:.6f}
{i}    double height = {PEG_HEIGHT:.6f}
{i}    uniform token axis = "Z"
{i}    token visibility = "invisible"
{attrs}
{i}}}
"""


def _hole_collision_usda(indent: str = "            ") -> str:
    i = indent
    child = i + "    "
    wall_thickness = HOLE_OUTER_R - HOLE_INNER_R
    r_mid = 0.5 * (HOLE_OUTER_R + HOLE_INNER_R)
    tangent_width = (
        2.0 * r_mid * np.tan(np.pi / HOLE_COLLISION_SEGMENTS) * 1.05
    )
    attrs = _fmt_collision_api_attrs(child + "    ")
    lines = [
        f'{i}def Xform "Collision"',
        f"{i}{{",
    ]
    for k in range(HOLE_COLLISION_SEGMENTS):
        theta = 2.0 * np.pi * k / HOLE_COLLISION_SEGMENTS
        x = r_mid * np.cos(theta)
        y = r_mid * np.sin(theta)
        qw = np.cos(0.5 * theta)
        qz = np.sin(0.5 * theta)
        lines.extend([
            f'{child}def Cube "Wall_{k:02d}" (',
            f'{child}    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]',
            f"{child})",
            f"{child}{{",
            f"{child}    double size = 1.0",
            f"{child}    float3 xformOp:translate = ({x:.6f}, {y:.6f}, 0.000000)",
            f"{child}    quatf xformOp:orient = ({qw:.8f}, 0.0, 0.0, {qz:.8f})",
            f"{child}    float3 xformOp:scale = ({wall_thickness:.6f}, {tangent_width:.6f}, {HOLE_HEIGHT:.6f})",
            f'{child}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]',
            f'{child}    token visibility = "invisible"',
            attrs,
            f"{child}}}",
            "",
        ])

    bottom_z = -0.5 * HOLE_HEIGHT - 0.5 * HOLE_BOTTOM_COLLIDER_HEIGHT
    lines.extend([
        f'{child}def Cylinder "Bottom" (',
        f'{child}    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]',
        f"{child})",
        f"{child}{{",
        f"{child}    double radius = {HOLE_INNER_R:.6f}",
        f"{child}    double height = {HOLE_BOTTOM_COLLIDER_HEIGHT:.6f}",
        f'{child}    uniform token axis = "Z"',
        f"{child}    float3 xformOp:translate = (0.000000, 0.000000, {bottom_z:.6f})",
        f'{child}    uniform token[] xformOpOrder = ["xformOp:translate"]',
        f'{child}    token visibility = "invisible"',
        attrs,
        f"{child}}}",
        f"{i}}}",
    ])
    return "\n".join(lines)


def build() -> None:
    hole_pts, hole_fvc, hole_fvi = hollow_cylinder_mesh(
        HOLE_OUTER_R, HOLE_INNER_R, HOLE_HEIGHT
    )
    peg_collision = _peg_collision_usda()
    hole_collision = _hole_collision_usda()
    peg_tip_z    = 0.5 * PEG_HEIGHT     # tip 在 peg 本地 +Z 面中心
    hole_entry_z = 0.5 * HOLE_HEIGHT    # entry 在 hole 本地 +Z 面 (开口)中心

    usda = f"""#usda 1.0
(
    defaultPrim = "bh_robot"
    upAxis = "Z"
    metersPerUnit = 1.0
    doc = \"\"\"Composed stage: robot + peg/hole visual geometry and collision proxies.
Generated by build_peghole_usd.py.

Peg 和 Hole 本身没有 RigidBodyAPI / MassAPI, 不改变 articulation DoF 或质量矩阵
(G(q)). Stage 3 起, 它们带 invisible collision 子 prim: peg = Cylinder,
hole = {HOLE_COLLISION_SEGMENTS} 个 Cube wall + Bottom stopper, 作为 EE link 的
附属 collision shapes。

稳定 prim 路径 (env 代码引用):
    /bh_robot/left_hande_robotiq_hande_link/Peg
    /bh_robot/left_hande_robotiq_hande_link/Peg/peg_tip
    /bh_robot/right_hande_robotiq_hande_link/Hole
    /bh_robot/right_hande_robotiq_hande_link/Hole/hole_entry

挂载: translate=({PART_X}, 0, {PART_Z}), orient=identity (轴沿 EE +Z = 夹爪前方).
几何: PEG radius={PEG_RADIUS}, height={PEG_HEIGHT};
      HOLE outer_r={HOLE_OUTER_R}, inner_r={HOLE_INNER_R}, height={HOLE_HEIGHT}.
碰撞: contactOffset={PHYSX_CONTACT_OFFSET}, restOffset={PHYSX_REST_OFFSET}.
\"\"\"
)

def Xform "bh_robot" (
    prepend references = @{ROBOT_USD_REL}@</bh_robot>
)
{{
    over "left_hande_robotiq_hande_link"
    {{
        def Xform "Peg"
        {{
            float3 xformOp:translate = ({PART_X:.6f}, 0.000000, {PART_Z:.6f})
            quatf xformOp:orient = ({_C:.8f}, {_S:.8f}, 0.0, 0.0)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]

            def Cylinder "Geom"
            {{
                double radius = {PEG_RADIUS:.6f}
                double height = {PEG_HEIGHT:.6f}
                uniform token axis = "Z"
                color3f[] primvars:displayColor = [({PEG_COLOR[0]}, {PEG_COLOR[1]}, {PEG_COLOR[2]})]
            }}

{peg_collision}
            def Xform "peg_tip"
            {{
                float3 xformOp:translate = (0.000000, 0.000000, {peg_tip_z:.6f})
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }}
        }}
    }}

    over "right_hande_robotiq_hande_link"
    {{
        def Xform "Hole"
        {{
            float3 xformOp:translate = ({PART_X:.6f}, 0.000000, {PART_Z:.6f})
            quatf xformOp:orient = ({_C:.8f}, {_S:.8f}, 0.0, 0.0)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]

            def Mesh "Geom"
            {{
                point3f[] points = {_fmt_vec3_array(hole_pts)}
                int[] faceVertexCounts = {_fmt_int_array(hole_fvc)}
                int[] faceVertexIndices = {_fmt_int_array(hole_fvi)}
                uniform token subdivisionScheme = "none"
                bool doubleSided = true
                color3f[] primvars:displayColor = [({HOLE_COLOR[0]}, {HOLE_COLOR[1]}, {HOLE_COLOR[2]})]
            }}

{hole_collision}
            def Xform "hole_entry"
            {{
                float3 xformOp:translate = (0.000000, 0.000000, {hole_entry_z:.6f})
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }}
        }}
    }}
}}
"""
    OUTPUT.write_text(usda)
    print(f"wrote {OUTPUT}  ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
