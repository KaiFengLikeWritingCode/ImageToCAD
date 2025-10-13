# -*- coding: utf-8 -*-
# 从标注(图1) -> 还原轮廓(图2)
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ==============================
# 一、常量区（把这些数值按你的图纸核对/修改）
# 单位：mm，角度用度数更直观，代码中会转弧度
# ==============================
W_total_top = 420.0        # 顶部总宽（图1上边那条 420）
W_total_bottom = 380.0     # 底部总宽（图1下边那条 380）
W_between_blocks = 370.0   # 两大块外边缘水平距（图1 370）
W_left_block = 132.0       # 左大块宽（从左外到竖边）
W_right_block = 132.0      # 右大块宽（右边同理，若不同请改）
H_left_block = 178.0       # 左大块高度
H_right_block = 110.0      # 右大块高度（图1右边标 110）

W_left_inset = 93.0        # 左底部向右缩进（图1左下 93）
R_corner_big = 5.0         # 左大块四角倒角 R5
R_corner_small = 3.0       # 右块小圆角 R3（示意）

# 中间过渡的三段半径（图1标注）
R_mid_1 = 195.0
R_mid_2 = 130.0
R_mid_3 = 170.0
# 右端过渡处两个 R40（图1多处 R40）
R_right_fillet = 40.0

# 中间竖向标注（图1 68 / 71 / 25? 取决于你的图）
DY_mid_left = 68.0
DY_mid_right = 71.0

ANGLE_right = 15.0         # 右端倾角 15°
# ==============================


# ========== 几何元素 ==========
@dataclass
class Line:
    params: np.ndarray  # [x1,y1,x2,y2]
    def __init__(self, x1, y1, x2, y2):
        self.params = np.array([x1, y1, x2, y2], dtype=float)

@dataclass
class Arc:
    # (cx, cy, r, t1, t2) 逆时针
    params: np.ndarray
    def __init__(self, cx, cy, r, t1, t2):
        self.params = np.array([cx, cy, r, t1, t2], dtype=float)

# ========== 工具函数 ==========
def unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def cross2(u, v):
    return u[0]*v[1] - u[1]*v[0]

def angle_between(u, v):
    u = unit(u); v = unit(v)
    s = cross2(u, v)
    c = np.clip(np.dot(u, v), -1.0, 1.0)
    return np.arctan2(s, c)

def arc_point(A: Arc, use_t2=False):
    cx, cy, r, t1, t2 = A.params
    t = t2 if use_t2 else t1
    return np.array([cx + r*np.cos(t), cy + r*np.sin(t)])

def arc_tangent(A: Arc, use_t2=False):
    cx, cy, r, t1, t2 = A.params
    t = t2 if use_t2 else t1
    return np.array([-np.sin(t), np.cos(t)])

def line_dir(L: Line):
    x1, y1, x2, y2 = L.params
    return np.array([x2-x1, y2-y1])

def deg(a):  # 度->弧度
    return np.deg2rad(a)

# 统一“点选择器”：("L", idx, "p1"/"p2") 或 ("A", idx, "t1"/"t2")
def get_point(geom, p_spec):
    typ, idx, tag = p_spec
    if isinstance(idx, str):
        idx = int(idx)
    if typ == "L":
        L = geom[idx]
        return L.params[:2].copy() if tag == "p1" else L.params[2:4].copy()
    elif typ == "A":
        A = geom[idx]
        return arc_point(A, use_t2=(tag == "t2"))
    else:
        raise ValueError("bad p_spec")

# ========== 残差构建 ==========
def build_residuals(geom, constraints, weights):
    rs = []
    def w(key, default=1.0): return weights.get(key, default)

    for c in constraints:
        t = c["type"]

        if t == "length":
            L = geom[c["L"]]
            rs.append(w("length")*(np.linalg.norm(line_dir(L)) - c["L0"]))

        elif t == "radius":
            A = geom[c["A"]]
            rs.append(w("radius")*(A.params[2] - c["R"]))

        elif t == "point_distance_x":
            pa = get_point(geom, c["pa"]); pb = get_point(geom, c["pb"])
            rs.append(w("dist_xy")*((pb[0]-pa[0]) - c["dx"]))

        elif t == "point_distance_y":
            pa = get_point(geom, c["pa"]); pb = get_point(geom, c["pb"])
            rs.append(w("dist_xy")*((pb[1]-pa[1]) - c["dy"]))

        elif t == "coincident":
            pa = get_point(geom, c["pa"]); pb = get_point(geom, c["pb"])
            d = pa - pb
            rs.extend(w("coincident")*d)

        elif t == "parallel":
            u = unit(line_dir(geom[c["L1"]]))
            v = unit(line_dir(geom[c["L2"]]))
            rs.append(w("parallel")*cross2(u, v))
        elif t == "prefer_minor_arc":
            A = geom[c["A"]]
            t1, t2 = A.params[3], A.params[4]
            rs.append(weights.get("minor_arc", 1.0) * minor_arc_penalty(t1, t2, c.get("margin", 0.0)))

        elif t == "perpendicular":
            u = unit(line_dir(geom[c["L1"]]))
            v = unit(line_dir(geom[c["L2"]]))
            rs.append(w("perpendicular")*np.dot(u, v))

        elif t == "tangent_line_arc":
            L = geom[c["L"]]; A = geom[c["A"]]
            u = unit(line_dir(L)); tau = unit(arc_tangent(A, c.get("use_t2", False)))
            rs.append(w("tangent")*cross2(u, tau))

        elif t == "tangent_arc_arc":
            A1 = geom[c["A1"]]; A2 = geom[c["A2"]]
            t1 = unit(arc_tangent(A1, c.get("A1_use_t2", False)))
            t2 = unit(arc_tangent(A2, c.get("A2_use_t2", False)))
            rs.append(w("tangent")*cross2(t1, t2))

        elif t == "fix_point":
            pa = get_point(geom, c["pa"])
            rs.extend(w("fix")*(pa - np.array([c["x"], c["y"]])))

        elif t == "angle":  # sin(Δθ) 更稳
            # u: ("L",i,"dir") 或 ("A",j,"t1"/"t2")
            if c["u"][0] == "L":
                u = line_dir(geom[c["u"][1]])
            else:
                A = geom[c["u"][1]]; u = arc_tangent(A, c["u"][2]=="t2")
            if c["v"][0] == "L":
                v = line_dir(geom[c["v"][1]])
            else:
                A = geom[c["v"][1]]; v = arc_tangent(A, c["v"][2]=="t2")
            rs.append(w("angle")*np.sin(angle_between(u, v) - c["theta"]))

        else:
            raise ValueError(f"Unknown constraint {t}")
    return np.array(rs, dtype=float)

def pack_params(geom):
    return np.concatenate([g.params for g in geom])

def unpack_params(x, geom):
    i = 0
    for g in geom:
        n = len(g.params); g.params[:] = x[i:i+n]; i += n

def index_map(geom):
    idxs = []; i = 0
    for g in geom:
        idxs.append(i); i += len(g.params)
    return idxs

# ========== 二、拓扑定义（按图2的顺序，尽量与标注吻合） ==========
geom: List[Any] = []

# 左大块外轮廓（用四条边 + 圆角R5 简化；上、下两条与中间过渡通过圆弧 b11/b15 接）
# a0: 左外竖线（作为全局锚）
a0 = Line(0, 0, 0, H_left_block)
# a1: 左上水平短边（宽度未知，用与 a0 顶点及 R5 圆角衔接）
a1 = Line(0, H_left_block, W_left_block, H_left_block)
# a2: 左下水平短边（到 W_left_inset 位置）
a2 = Line(0, 0, W_left_inset, 0)
# a3: 左上斜边（与中间过渡连接）
a3 = Line(W_left_block, H_left_block, W_left_block*0.8, H_left_block*0.75)
# a4: 左下斜边（与中间过渡连接）
a4 = Line(W_left_inset, 0, W_left_block*0.8, H_left_block*0.15)

# 中部通道（用三段大圆弧近似：R195 -> R130 -> R170）
b12 = Arc(W_left_block+50, H_left_block*0.45, R_mid_1, deg(200), deg(330))  # 左段
b13 = Arc(W_left_block+140, H_left_block*0.55, R_mid_2, deg(160), deg(340)) # 中段
b16 = Arc(W_left_block+230, H_left_block*0.40, R_mid_3, deg(200), deg(330)) # 右段

# 右大块
# a8: 右外竖线（放在 W_total_bottom 处）
a8 = Line(W_total_bottom, 0, W_total_bottom, H_right_block)
# a7: 右上短边
a7 = Line(W_total_bottom - W_right_block, H_right_block, W_total_bottom, H_right_block)
# a6: 右下短边
a6 = Line(W_total_bottom - W_right_block, 0, W_total_bottom, 0)
# a5: 右上斜边（与中部通道连接，含 15°）
a5 = Line(W_total_bottom - W_right_block, H_right_block, W_total_bottom - W_right_block*0.8, H_right_block*0.7)

# 两端/转角圆角与过渡（示意：左块四角R5，右块小圆角R3；中右端与块连接用 R40）
b9  = Arc(W_left_block- R_corner_big, H_left_block- R_corner_big, R_corner_big, deg(90), deg(180))   # 左上外R5
b10 = Arc(W_left_inset+ R_corner_big, R_corner_big, R_corner_big, deg(0), deg(270))                  # 左下外R5
b18 = Arc(W_total_bottom - W_right_block + 30, H_right_block - 10, R_right_fillet, deg(180), deg(300)) # 右上R40
b19 = Arc(W_total_bottom - W_right_block + 20, 10, R_right_fillet, deg(60), deg(180))                # 右下R40
b20 = Arc(W_total_bottom - R_corner_small, R_corner_small, R_corner_small, deg(270), deg(0))         # 右下外R3

# 左块过渡到中部的两小圆弧（R40、R63 参照图1）
b11 = Arc(W_left_block-20, H_left_block*0.6, 40.0, deg(180), deg(330))   # 左上过渡 R40
b15 = Arc(W_left_block-10, H_left_block*0.1, 63.0, deg(30), deg(210))    # 左下过渡 R63

geom.extend([a0,a1,a2,a3,a4,b12,b13,b16,a8,a7,a6,a5,b9,b10,b18,b19,b20,b11,b15])

# ========== 三、约束（把图1所有标注逐条体现在这里） ==========
cons: List[Dict[str,Any]] = []

# 1) 基准/姿态锁定：固定左下角在 (0,0)，并保持 a0 竖直
cons += [
    {"type":"fix_point", "pa":("L",0,"p1"), "x":0.0, "y":0.0},                 # a0.p1 = (0,0)
    {"type":"point_distance_x", "pa":("L",0,"p1"), "pb":("L",0,"p2"), "dx":0.0} # a0 垂直
]

# 2) 左大块尺寸：高度 178，顶部/底部水平尺寸
cons += [
    {"type":"length", "L":0, "L0":H_left_block},                               # a0 高度
    {"type":"point_distance_x", "pa":("L",0,"p1"), "pb":("L",1,"p2"), "dx":W_left_block},  # 左上水平终点 x
    {"type":"point_distance_y", "pa":("L",1,"p1"), "pb":("L",1,"p2"), "dy":0.0},           # a1 水平
    {"type":"point_distance_x", "pa":("L",0,"p1"), "pb":("L",2,"p2"), "dx":W_left_inset},  # 左下短边长度
    {"type":"point_distance_y", "pa":("L",2,"p1"), "pb":("L",2,"p2"), "dy":0.0}            # a2 水平
]

# 3) 右大块尺寸：右外竖线 x = W_total_bottom，高度 110，上下短边水平
cons += [
    {"type":"point_distance_x", "pa":("L",0,"p1"), "pb":("L",8,"p1"), "dx":W_total_bottom}, # a8.x = W_total_bottom
    {"type":"length", "L":8, "L0":H_right_block},
    {"type":"point_distance_y", "pa":("L",9,"p1"), "pb":("L",9,"p2"), "dy":0.0},            # a7 水平
    {"type":"point_distance_y", "pa":("L",10,"p1"), "pb":("L",10,"p2"), "dy":0.0}           # a6 水平
]

# 例：b12/b13/b16 这三段通道大圆弧
cons += [
    {"type":"prefer_minor_arc", "A":5, "margin":0.0},
    {"type":"prefer_minor_arc", "A":6, "margin":0.0},
    {"type":"prefer_minor_arc", "A":7, "margin":0.0},
    # 若右端 R40 两个小圆角也有绕圈风险，同样加上
    {"type":"prefer_minor_arc", "A":14, "margin":0.0},
    {"type":"prefer_minor_arc", "A":15, "margin":0.0},
]


# 4) 顶/底总宽（420 / 380）
cons += [
    {"type":"point_distance_x", "pa":("L",1,"p2"), "pb":("L",9,"p2"), "dx":W_total_top - (W_total_bottom - W_left_block)},  # 顶部总宽约束(等价表达)
    {"type":"point_distance_x", "pa":("L",2,"p1"), "pb":("L",10,"p1"), "dx":W_total_bottom}                                  # 底部总宽
]

# 5) 左块四角和右块小圆角半径
cons += [
    {"type":"radius", "A":12, "R":R_corner_big},   # b9 R5
    {"type":"radius", "A":13, "R":R_corner_big},   # b10 R5
    {"type":"radius", "A":16, "R":R_corner_small}  # b20 R3
]

# 圆角与直线端点重合（只示范两处；其余可按需要补充）
cons += [
    {"type":"coincident", "pa":("L",1,"p1"), "pb":("L",0,"p2")},  # a1.p1 与 a0.p2
    {"type":"coincident", "pa":("A",12,"t2"), "pb":("L",1,"p1")}, # b9.t2 与 a1.p1
    {"type":"coincident", "pa":("A",13,"t1"), "pb":("L",2,"p2")}, # b10.t1 与 a2.p2
]

# 6) 中部三段大圆弧半径
cons += [
    {"type":"radius", "A":5, "R":R_mid_1},   # b12
    {"type":"radius", "A":6, "R":R_mid_2},   # b13
    {"type":"radius", "A":7, "R":R_mid_3},   # b16
]

# 7) 过渡关系：线-弧、弧-弧相切 + 端点重合（把链条串起来）
# 左上：a3 -> b11 -> b12
cons += [
    {"type":"coincident", "pa":("L",3,"p1"), "pb":("L",1,"p2")},                 # a3 起点接 a1 末端
    {"type":"coincident", "pa":("L",3,"p2"), "pb":("A",17,"t1")},                # a3 终点接 b11.t1
    {"type":"tangent_line_arc", "L":3, "A":17, "use_t2":False},
    {"type":"coincident", "pa":("A",17,"t2"), "pb":("A",5,"t1")},                # b11.t2 -> b12.t1
    {"type":"tangent_arc_arc", "A1":17, "A2":5}
]

# 左下：a4 -> b15 -> b12
cons += [
    {"type":"coincident", "pa":("L",4,"p1"), "pb":("L",2,"p2")},                 # a4 起点接 a2
    {"type":"coincident", "pa":("L",4,"p2"), "pb":("A",18,"t1")},                # a4 终点接 b15.t1
    {"type":"tangent_line_arc", "L":4, "A":18, "use_t2":False},
    {"type":"coincident", "pa":("A",18,"t2"), "pb":("A",5,"t1")},                # b15.t2 -> b12.t1
    {"type":"tangent_arc_arc", "A1":18, "A2":5}
]

# 中段连接：b12 -> b13 -> b16
cons += [
    {"type":"coincident", "pa":("A",5,"t2"), "pb":("A",6,"t1")},
    {"type":"coincident", "pa":("A",6,"t2"), "pb":("A",7,"t1")},
    {"type":"tangent_arc_arc", "A1":5, "A2":6},
    {"type":"tangent_arc_arc", "A1":6, "A2":7}
]

# 右端：b16 -> a5 -> R40(b18/b19) -> 右块
cons += [
    {"type":"coincident", "pa":("A",7,"t2"), "pb":("L",11,"p1")},                # b16.t2 -> a5.p1
    {"type":"tangent_line_arc", "L":11, "A":7, "use_t2":True},                   # a5 与 b16 相切
    {"type":"coincident", "pa":("L",11,"p2"), "pb":("A",14,"t1")},               # a5.p2 -> b18.t1
    {"type":"tangent_line_arc", "L":11, "A":14, "use_t2":False},
    {"type":"coincident", "pa":("A",14,"t2"), "pb":("L",9,"p1")},                # b18.t2 -> a7.p1
    {"type":"coincident", "pa":("A",15,"t1"), "pb":("L",10,"p1")},               # b19.t1 -> a6.p1
    {"type":"coincident", "pa":("A",15,"t2"), "pb":("L",11,"p2")},               # b19.t2 -> a5.p2
    {"type":"radius", "A":14, "R":R_right_fillet},
    {"type":"radius", "A":15, "R":R_right_fillet},
]

# 右上倾角 15°：用 a5 与水平的夹角
cons += [
    {"type":"angle",
     "u":("L",11,"dir"),
     "v":("L",9,"dir"),         # a7 水平线方向
     "theta":deg(ANGLE_right)}  # 15°
]

# 中间 68、71 的竖距（示意：约束 b12/b16 接触处相对底边 y 的高度差）
cons += [
    {"type":"point_distance_y", "pa":("L",2,"p1"), "pb":("A",5,"t1"), "dy":DY_mid_left},
    {"type":"point_distance_y", "pa":("L",10,"p1"), "pb":("A",7,"t2"), "dy":DY_mid_right},
]

# ========== 四、求解 ==========
weights = dict(length=1.0, radius=1.0, dist_xy=1.0, coincident=20.0,
               tangent=10.0, perpendicular=5.0, parallel=5.0,
               angle=3.0, fix=20.0)


x0 = pack_params(geom)
lb = -np.inf*np.ones_like(x0); ub = np.inf*np.ones_like(x0)

# 给若干圆心/变量设“范围”（避免跳到另一侧，增强稳定）
starts = index_map(geom)

param_names = []
for gi, g in enumerate(geom):
    if isinstance(g, Line):
        param_names += [f"L{gi}.x1", f"L{gi}.y1", f"L{gi}.x2", f"L{gi}.y2"]
    else:
        param_names += [f"A{gi}.cx", f"A{gi}.cy", f"A{gi}.r", f"A{gi}.t1", f"A{gi}.t2"]

# === 初始化 bounds（先给全量 -inf/+inf） ===
lb = -np.inf*np.ones_like(x0)
ub =  np.inf*np.ones_like(x0)

def bound_arc_center(i, x_min=None, x_max=None, y_min=None, y_max=None):
    if x_min is not None: lb[starts[i]+0] = x_min
    if x_max is not None: ub[starts[i]+0] = x_max
    if y_min is not None: lb[starts[i]+1] = y_min
    if y_max is not None: ub[starts[i]+1] = y_max

def set_arc_center_bounds(arc_idx, x_min=None, x_max=None, y_min=None, y_max=None):
    """更安全的圆心边界设置：容错 + 只在提供数值时才约束"""
    base = starts[arc_idx]
    cx_i, cy_i = base+0, base+1
    if x_min is not None: lb[cx_i] = x_min
    if x_max is not None: ub[cx_i] = x_max
    if y_min is not None: lb[cy_i] = y_min
    if y_max is not None: ub[cy_i] = y_max

# === （你原来的范围）按需设置，但先给得更宽松，确保包含初值 ===
# 查看 A5/A6/A7 初值（也就是 geom[5], geom[6], geom[7] 三段大圆弧）的 cx,cy
A5_cx, A5_cy = geom[5].params[0], geom[5].params[1]
A6_cx, A6_cy = geom[6].params[0], geom[6].params[1]
A7_cx, A7_cy = geom[7].params[0], geom[7].params[1]

# 以初值为中心给一个比较宽的盒子（比如 ±200mm），保证初值一定在范围内
set_arc_center_bounds(5, x_min=A5_cx-200, x_max=A5_cx+200, y_min=A5_cy-200, y_max=A5_cy+200)
set_arc_center_bounds(6, x_min=A6_cx-200, x_max=A6_cx+200, y_min=A6_cy-200, y_max=A6_cy+200)
set_arc_center_bounds(7, x_min=A7_cx-200, x_max=A7_cx+200, y_min=A7_cy-200, y_max=A7_cy+200)

# === 辅助：修正无效范围（lb > ub），并打印诊断 ===
def reconcile_and_report_bounds(x0, lb, ub, names):
    eps = 1e-12
    # 修正反转
    bad = lb > ub
    if np.any(bad):
        for i in np.where(bad)[0]:
            lbi, ubi = lb[i], ub[i]
            # 交换并留个极小间隙
            lb[i], ub[i] = min(lbi, ubi), max(lbi, ubi)
        print("[bounds] Some lb>ub found and fixed.")

    # 报告初值越界
    below = np.where(x0 < lb - eps)[0]
    above = np.where(x0 > ub + eps)[0]
    if below.size or above.size:
        print("=== Initial guess out of bounds ===")
        for i in below:
            print(f"  {names[i]}: x0={x0[i]:.6g} < lb={lb[i]:.6g}  (shift {lb[i]-x0[i]:.6g})")
        for i in above:
            print(f"  {names[i]}: x0={x0[i]:.6g} > ub={ub[i]:.6g}  (shift {x0[i]-ub[i]:.6g})")
    return below, above

below, above = reconcile_and_report_bounds(x0, lb, ub, param_names)

# 策略1：把 x0 clip 到可行域内（最简单直接）
if below.size or above.size:
    x0 = np.clip(x0, lb, ub)
    # 如果你更希望保留初值而放宽范围，用“策略2”：
    # ub[above] = np.maximum(ub[above], x0[above])
    # lb[below] = np.minimum(lb[below], x0[below])

def wrap_to_pi(a):
    """把角度差规约到 (-pi, pi]，避免 2π 多圈歧义"""
    return (a + np.pi) % (2*np.pi) - np.pi

def minor_arc_penalty(t1, t2, margin=0.0):
    """
    当 |Δ| 超过 π（非短弧）时产生罚项；否则为 0。
    用 softplus 保持可导：softplus(x)=log(1+exp(x))
    """
    Delta = np.abs(wrap_to_pi(t2 - t1))
    # 想“强制短弧”就把 margin 设为 0；想稍放宽就给 margin>0
    excess = Delta - (np.pi - margin)
    return np.log1p(np.exp(excess))  # softplus(excess)
# === 现在再走优化 ===
def fun(x):
    unpack_params(x, geom)
    return build_residuals(geom, cons, weights)

res = least_squares(fun, x0, bounds=(lb,ub), loss='soft_l1', ftol=1e-10, xtol=1e-10, gtol=1e-10, verbose=2)
unpack_params(res.x, geom)

print("\nSolve success:", res.success, "  message:", res.message)
print("final cost:", 0.5*np.sum(res.fun**2))

# ========== 五、绘制 ==========
fig, ax = plt.subplots(figsize=(9,4))
for idx, g in enumerate(geom):
    if isinstance(g, Line):
        x1,y1,x2,y2 = g.params
        ax.plot([x1,x2],[y1,y2], lw=2)
        ax.text((x1+x2)/2, (y1+y2)/2, f"a{idx}" if idx<=4 else (f"a{idx-3}" if 8<=idx<=11 else ""), fontsize=8)
    else:
        cx,cy,r,t1,t2 = g.params
        ts = np.linspace(t1, t2, 120)
        xs = cx + r*np.cos(ts); ys = cy + r*np.sin(ts)
        ax.plot(xs, ys, lw=2)
        ax.text(cx, cy, f"b{idx}", fontsize=8)
ax.set_aspect('equal','box'); ax.grid(True, alpha=0.3)
ax.set_title("Reconstructed profile")
plt.show()
