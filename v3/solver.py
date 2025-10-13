# -*- coding: utf-8 -*-
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# ========== 几何元素 ==========
@dataclass
class Line:
    # 端点 (x1,y1) -> (x2,y2)
    params: np.ndarray  # [x1,y1,x2,y2]
    def __init__(self, x1, y1, x2, y2):
        self.params = np.array([x1, y1, x2, y2], dtype=float)

@dataclass
class Arc:
    # 圆弧 (cx, cy, r, t1, t2) 角度弧度，逆时针
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
    return np.arctan2(s, c)  # (-pi, pi]

def arc_point(A: Arc, use_t2=False):
    cx, cy, r, t1, t2 = A.params
    t = t2 if use_t2 else t1
    return np.array([cx + r*np.cos(t), cy + r*np.sin(t)])

def arc_tangent(A: Arc, use_t2=False):
    # 切向（未归一化）：(-sin t, cos t)
    cx, cy, r, t1, t2 = A.params
    t = t2 if use_t2 else t1
    return np.array([-np.sin(t), np.cos(t)])

def line_dir(L: Line):
    x1, y1, x2, y2 = L.params
    return np.array([x2-x1, y2-y1])

# 统一的“点选择器”，便于书写约束
#  p_spec: ("L", idx, "p1"/"p2") 或 ("A", idx, "t1"/"t2")
def get_point(geom, p_spec):
    # 兼容三种写法：
    # 1) ("L", 0, "p2") / ("A", 3, "t1")
    # 2) ("L", "0", "p2") / ("A", "3", "t1")   ← 关键：把字符串转 int
    # 3) {"typ":"L","idx":0,"tag":"p2"}       ← 可选的字典写法
    if isinstance(p_spec, dict):
        typ = p_spec.get("typ") or p_spec.get("type")
        idx = p_spec.get("idx")
        tag = p_spec.get("tag")
    else:
        typ, idx, tag = p_spec

    # 把 "0" 这样的字符串编号转为整数
    if isinstance(idx, str):
        try:
            idx = int(idx)
        except ValueError:
            raise TypeError(f"Index must be int or numeric str, got {idx!r}")

    if typ == "L":
        L = geom[idx]
        if tag == "p1":
            return L.params[:2].copy()
        elif tag == "p2":
            return L.params[2:4].copy()
        else:
            raise ValueError(f"Unknown line point tag: {tag!r}")

    elif typ == "A":
        A = geom[idx]
        return arc_point(A, use_t2=(tag == "t2"))

    else:
        raise ValueError(f"Unknown geometry type in p_spec: {typ!r}")


# ========== 残差构建 ==========
# 每条约束是一个 dict: {"type": "length"/"...", 其他键...}
def build_residuals(geom: List[Any], constraints: List[Dict], weights: Dict[str, float]):
    rs = []

    def w(key, default=1.0):
        return weights.get(key, default)

    for c in constraints:
        t = c["type"]

        # --- 尺寸类 ---
        if t == "length":  # 直线长度
            L: Line = geom[c["L"]]
            Llen = np.linalg.norm(line_dir(L))
            rs.append(w("length") * (Llen - c["L0"]))

        elif t == "radius":
            A: Arc = geom[c["A"]]
            rs.append(w("radius") * (A.params[2] - c["R"]))

        elif t == "point_distance_x":
            pa = get_point(geom, c["pa"]); pb = get_point(geom, c["pb"])
            rs.append(w("dist_xy") * ((pb[0] - pa[0]) - c["dx"]))

        elif t == "point_distance_y":
            pa = get_point(geom, c["pa"]); pb = get_point(geom, c["pb"])
            rs.append(w("dist_xy") * ((pb[1] - pa[1]) - c["dy"]))

        elif t == "angle":  # 两方向夹角 = theta（弧度）
            # u: (typ,idx,"dir") where typ in {"L","A"}; for A 用切向
            if c["u"][0] == "L":
                u = line_dir(geom[c["u"][1]])
            else:  # ("A", idx, "t1"/"t2")
                A: Arc = geom[c["u"][1]]
                u = arc_tangent(A, use_t2=(c["u"][2] == "t2"))
            if c["v"][0] == "L":
                v = line_dir(geom[c["v"][1]])
            else:
                A: Arc = geom[c["v"][1]]
                v = arc_tangent(A, use_t2=(c["v"][2] == "t2"))
            rs.append(w("angle") * np.sin(angle_between(u, v) - c["theta"]))

        # --- 拓扑/接触类 ---
        elif t == "coincident":
            pa = get_point(geom, c["pa"]); pb = get_point(geom, c["pb"])
            d = pa - pb
            rs.extend(w("coincident") * d)  # 两个分量

        elif t == "parallel":
            u = line_dir(geom[c["L1"]]); v = line_dir(geom[c["L2"]])
            rs.append(w("parallel") * cross2(unit(u), unit(v)))

        elif t == "perpendicular":
            u = line_dir(geom[c["L1"]]); v = line_dir(geom[c["L2"]])
            rs.append(w("perpendicular") * np.dot(unit(u), unit(v)))

        elif t == "tangent_line_arc":
            L: Line = geom[c["L"]]; A: Arc = geom[c["A"]]
            u = unit(line_dir(L))
            tau = unit(arc_tangent(A, use_t2=c.get("use_t2", False)))
            rs.append(w("tangent") * cross2(u, tau))

        elif t == "tangent_arc_arc":
            A1: Arc = geom[c["A1"]]; A2: Arc = geom[c["A2"]]
            tau1 = unit(arc_tangent(A1, use_t2=c.get("A1_use_t2", False)))
            tau2 = unit(arc_tangent(A2, use_t2=c.get("A2_use_t2", False)))
            rs.append(w("tangent") * cross2(tau1, tau2))

        elif t == "point_on_line":
            pa = get_point(geom, c["pa"])
            L: Line = geom[c["L"]]
            p1 = L.params[:2]; p2 = L.params[2:4]
            r = cross2(pa - p1, p2 - p1) / (np.linalg.norm(p2 - p1) + 1e-12)
            rs.append(w("point_on_line") * r)

        elif t == "fix_point":  # 把某点锚定在坐标上（去刚体自由度）
            pa = get_point(geom, c["pa"])
            rs.extend(w("fix") * (pa - np.array([c["x"], c["y"]])))

        else:
            raise ValueError(f"Unknown constraint type: {t}")

    return np.array(rs, dtype=float)

# ========== 参数打包/边界 ==========
def pack_params(geom):
    arrs = [g.params for g in geom]
    return np.concatenate(arrs)

def unpack_params(x, geom):
    i = 0
    for g in geom:
        n = len(g.params)
        g.params[:] = x[i:i+n]
        i += n

def index_map(geom):
    """返回每个元素在 x 向量里的起始下标，便于设置 bounds"""
    idxs = []
    i = 0
    for g in geom:
        idxs.append(i)
        i += len(g.params)
    return idxs

# ========== 演示：简单拓扑（直线-弧-直线-弧），含相切/尺寸 ==========
if __name__ == "__main__":
    geom: List[Any] = []
    # 左竖线 L0（基本锚）
    L0 = Line(0.0, 0.0, 0.0, 100.0)     # 初值
    # 中部斜线 L1
    L1 = Line(80.0, 60.0, 200.0, 90.0)
    # 过渡弧 A0（与 L0 末端、L1 起端相切）
    A0 = Arc(40.0, 50.0, 30.0, np.deg2rad(90), np.deg2rad(20))
    # 右端弧 A1（与 L1 末端相切，给定半径）
    A1 = Arc(240.0, 70.0, 40.0, np.deg2rad(200), np.deg2rad(330))

    geom.extend([L0, L1, A0, A1])

    # 约束列表（把图纸标注换成这些字典即可）
    cons: List[Dict] = [
        # 1) 拓扑：端点与弧端点重合（把“连起来”）
        {"type": "coincident", "pa": ("L","0","p2"), "pb": ("A","2","t1")},  # L0.p2 = A0.t1
        {"type": "coincident", "pa": ("L","1","p1"), "pb": ("A","2","t2")},  # L1.p1 = A0.t2
        {"type": "coincident", "pa": ("L","1","p2"), "pb": ("A","3","t1")},  # L1.p2 = A1.t1

        # 2) 相切：线-弧、弧-弧
        {"type": "tangent_line_arc", "L": 0, "A": 2, "use_t2": False},       # L0 ~ A0@t1
        {"type": "tangent_line_arc", "L": 1, "A": 2, "use_t2": True},        # L1 ~ A0@t2
        {"type": "tangent_line_arc", "L": 1, "A": 3, "use_t2": False},       # L1 ~ A1@t1

        # 3) 尺寸/方向：左竖线垂直；给长度与半径
        {"type": "perpendicular", "L1": 0, "L2": 1},                         # L0 ⟂ L1（示例：也可换成“L0 竖直”）
        {"type": "length", "L": 0, "L0": 100.0},                             # L0 长度=100
        {"type": "radius", "A": 3, "R": 40.0},                               # A1 半径=40

        # 4) X/Y 标注：控制整体尺寸（示例）
        {"type": "point_distance_x", "pa": ("L","0","p1"), "pb": ("L","1","p2"), "dx": 200.0},
        {"type": "point_distance_y", "pa": ("L","0","p1"), "pb": ("L","1","p2"), "dy": 90.0},

        # 5) 锚固：去刚体自由度（固定一个点+一个方向/坐标）
        {"type": "fix_point", "pa": ("L","0","p1"), "x": 0.0, "y": 0.0},     # 左下角锚定
        # 再固定 L0 顶点的 x（消旋转/平移自由度）
        {"type": "point_distance_x", "pa": ("L","0","p1"), "pb": ("L","0","p2"), "dx": 0.0},
    ]

    # 权重（可按精度/公差调）
    weights = dict(
        length=1.0, radius=1.0, dist_xy=1.0, coincident=10.0,
        tangent=10.0, perpendicular=5.0, parallel=5.0, angle=3.0,
        point_on_line=5.0, fix=10.0
    )

    # 变量打包与边界（不等式/范围）
    x0 = pack_params(geom)
    lb = -np.inf*np.ones_like(x0)
    ub =  np.inf*np.ones_like(x0)

    # 例：给 A0 圆心一个“可行范围”（x >= 10, y ∈ [10, 120]）
    starts = index_map(geom)
    A0_cx = starts[2] + 0   # A0.cx
    A0_cy = starts[2] + 1   # A0.cy
    lb[A0_cx] = 10.0
    lb[A0_cy] = 10.0
    ub[A0_cy] = 120.0
    # 例：A1 圆心 x ≤ 260
    A1_cx = starts[3] + 0
    ub[A1_cx] = 260.0

    # 残差函数
    def fun(x):
        unpack_params(x, geom)
        return build_residuals(geom, cons, weights)

    # 求解
    res = least_squares(fun, x0, bounds=(lb, ub), loss='soft_l1',
                        ftol=1e-10, xtol=1e-10, gtol=1e-10, verbose=2)
    unpack_params(res.x, geom)

    print("\n=== Solve status ===")
    print("success:", res.success, " message:", res.message)
    print("final cost:", 0.5*np.sum(res.fun**2), "  ||grad||_inf:", np.max(np.abs(res.jac.T @ res.fun)))

    # ========== 可视化 ==========
    fig, ax = plt.subplots(figsize=(6, 3))
    # 画线
    for idx, g in enumerate(geom):
        if isinstance(g, Line):
            x1, y1, x2, y2 = g.params
            ax.plot([x1, x2], [y1, y2], '-', lw=2)
            ax.text(x2, y2, f"L{idx}", fontsize=9)
        else:
            cx, cy, r, t1, t2 = g.params
            ts = np.linspace(t1, t2, 80)
            xs = cx + r*np.cos(ts); ys = cy + r*np.sin(ts)
            ax.plot(xs, ys, '-', lw=2)
            ax.text(cx, cy, f"A{idx}", fontsize=9)
    ax.set_aspect('equal', 'box')
    ax.grid(True, alpha=0.3)
    ax.set_title("Solved sketch (demo)")
    plt.show()
