# =======================
# 约束型几何求解器（全局化策略）
# - 基于 SciPy trust-constr + 多初值重启
# - 严格等式/不等式建模，避免把硬约束当软惩罚
# =======================
import numpy as np
from math import cos, sin, atan2, pi, isfinite
from scipy.optimize import minimize, NonlinearConstraint, Bounds

# ---------- 工具：角度归一化 ----------
def angle_diff(t2, t1):
    """返回 t2-t1 归一化到 [-pi, pi]"""
    d = (t2 - t1 + pi) % (2*pi) - pi
    return d

# ---------- 从对象取端点 ----------
def _get_line_endpoints(L: Line):
    (x1,y1), (x2,y2) = L.points()
    return np.array([x1,y1]), np.array([x2,y2])

def _get_arc_endpoints(A: Arc):
    (x1,y1), (x2,y2) = A.points()
    return np.array([x1,y1]), np.array([x2,y2])

def _get_endpoint(geom_objects, idx, which):
    obj = geom_objects[idx]
    if isinstance(obj, Line):
        p1, p2 = _get_line_endpoints(obj)
        return p1 if which == 0 else p2
    elif isinstance(obj, Arc):
        p1, p2 = _get_arc_endpoints(obj)
        return p1 if which == 0 else p2
    else:
        raise ValueError('unknown object')

# ---------- 切向/方向 ----------
def _line_dir(L: Line):
    p1, p2 = _get_line_endpoints(L)
    v = p2 - p1
    n = np.linalg.norm(v) + 1e-12
    return v / n

def _arc_tangent_at(A: Arc, use_end: bool):
    cx, cy, r, t1, t2 = A.params
    t = t2 if use_end else t1
    # 圆弧参数为逆时针，切向向量 = (-sin t, cos t)
    tx, ty = -np.sin(t), np.cos(t)
    v = np.array([tx, ty])
    n = np.linalg.norm(v) + 1e-12
    return v / n

def _arc_tangent_at_angle(A: Arc, t):
    tx, ty = -np.sin(t), np.cos(t)
    v = np.array([tx, ty])
    return v / (np.linalg.norm(v) + 1e-12)

# ---------- 线的有向符号距离（决定左右侧） ----------
def _line_signed(La: Line, P: np.ndarray):
    p1, p2 = _get_line_endpoints(La)
    v = p2 - p1
    w = P - p1
    # 2D 叉积 z 分量
    cross = v[0]*w[1] - v[1]*w[0]
    # 用段长当尺度，返回有符号“面积/长度”=> 与左右侧一致
    scale = np.linalg.norm(v) + 1e-12
    return cross / scale

# ---------- 打包/解包优化变量 ----------
def _pack_vars(objs):
    blocks = []
    idxs = []
    for oi, obj in enumerate(objs):
        mask = ~obj.fixed_mask
        blocks.append(obj.params[mask])
        idxs.append((oi, mask))
    if blocks:
        x0 = np.concatenate(blocks)
    else:
        x0 = np.zeros(0)
    return x0, idxs

def _unpack_to_objs(x, objs, idxs):
    k = 0
    for oi, mask in idxs:
        cnt = mask.sum()
        if cnt:
            objs[oi].params[mask] = x[k:k+cnt]
            k += cnt

# ---------- 采样点（用于 arc_side / arc_side_of_line） ----------
def _arc_sample_points(A: Arc, samples: int, anchor='mid'):
    cx, cy, r, t1, t2 = A.params
    if not isfinite(t1) or not isfinite(t2):
        # 给个保底
        ts = np.linspace(0, 1, samples)
        ang = ts * 2*np.pi
    else:
        ts = np.linspace(0, 1, samples)
        ang = t1 + ts * angle_diff(t2, t1)
    xs = cx + r * np.cos(ang)
    ys = cy + r * np.sin(ang)
    return np.stack([xs, ys], axis=1)

# ---------- 单个约束构造器：返回 NonlinearConstraint ----------
def _make_constraint(geom_objects, c):
    t = c['type']

    # ---- 等式：端点重合 ----
    if t == 'coincident':
        p1 = c['p1']; w1 = c['which1']
        p2 = c['p2']; w2 = c['which2']
        def fun(x):
            _unpack_to_objs(x, geom_objects, x._idxs)
            P = _get_endpoint(geom_objects, p1, w1)
            Q = _get_endpoint(geom_objects, p2, w2)
            return np.array([P[0]-Q[0], P[1]-Q[1]])
        return NonlinearConstraint(fun, lb=[0,0], ub=[0,0])

    # ---- 等式：线长、半径、点间竖距 ----
    if t == 'length':
        L = geom_objects[c['line']]
        value = c['value']
        def fun(x):
            _unpack_to_objs(x, geom_objects, x._idxs)
            p1, p2 = _get_line_endpoints(L)
            return np.array([np.linalg.norm(p2-p1) - value])
        return NonlinearConstraint(fun, lb=[0], ub=[0])

    if t == 'radius':
        A = geom_objects[c['arc']]
        value = c['value']
        def fun(x):
            _unpack_to_objs(x, geom_objects, x._idxs)
            return np.array([A.params[2] - value])
        return NonlinearConstraint(fun, lb=[0], ub=[0])

    if t == 'point_distance_y':
        p1 = c['p1']; w1 = c['which1']
        p2 = c['p2']; w2 = c['which2']
        value = c['value']
        def fun(x):
            _unpack_to_objs(x, geom_objects, x._idxs)
            P = _get_endpoint(geom_objects, p1, w1)
            Q = _get_endpoint(geom_objects, p2, w2)
            return np.array([(Q[1]-P[1]) - value])
        return NonlinearConstraint(fun, lb=[0], ub=[0])

    # ---- 相切：弧端点-直线（等式：平行；可选不等式：侧别）----
    if t in ('tangent_at_arc_end_to_line', 'tangent_at_arc_start_to_line'):
        L = geom_objects[c['line']]
        A = geom_objects[c['arc']]
        use_end = (t == 'tangent_at_arc_end_to_line')
        side = c.get('side', None)  # 'left'/'right'/None

        def fun_parallel(x):
            _unpack_to_objs(x, geom_objects, x._idxs)
            tl = _line_dir(L)
            ta = _arc_tangent_at(A, use_end)
            # 平行性：2D 叉积 = 0
            cross = tl[0]*ta[1] - tl[1]*ta[0]
            return np.array([cross])
        cons = [NonlinearConstraint(fun_parallel, lb=[0], ub=[0])]

        if side is not None:
            # 侧别：用线段方向的左侧为正，右侧为负
            which = 4 if use_end else 3
            def fun_side(x):
                _unpack_to_objs(x, geom_objects, x._idxs)
                # 取相切点
                P = _get_arc_endpoints(A)[1] if use_end else _get_arc_endpoints(A)[0]
                s = _line_signed(L, P)
                return np.array([s])
            if side == 'left':
                cons.append(NonlinearConstraint(fun_side, lb=[+1e-6], ub=[np.inf]))
            else:
                cons.append(NonlinearConstraint(fun_side, lb=[-np.inf], ub=[-1e-6]))
        return cons

    # ---- 相切：弧-弧（等式：平行；不等式：同/反向）----
    if t == 'tangent_at_arc_to_arc':
        A = geom_objects[c['Aarc']]
        B = geom_objects[c['Barc']]
        ta_end  = c.get('a_end', True)
        tb_end  = c.get('b_end', False)
        same_dir = bool(c.get('same_direction', True))

        def fun_parallel(x):
            _unpack_to_objs(x, geom_objects, x._idxs)
            va = _arc_tangent_at(A, ta_end)
            vb = _arc_tangent_at(B, tb_end)
            cross = va[0]*vb[1] - va[1]*vb[0]
            return np.array([cross])
        cons = [NonlinearConstraint(fun_parallel, lb=[0], ub=[0])]

        def fun_dot(x):
            _unpack_to_objs(x, geom_objects, x._idxs)
            va = _arc_tangent_at(A, ta_end)
            vb = _arc_tangent_at(B, tb_end)
            dot = va @ vb  # 同向 ~ +1；反向 ~ -1
            return np.array([dot])
        if same_dir:
            cons.append(NonlinearConstraint(fun_dot, lb=[0], ub=[1.0]))
        else:
            cons.append(NonlinearConstraint(fun_dot, lb=[-1.0], ub=[0]))
        return cons

    # ---- 不等式：扫角 ≤ 上限 ----
    if t == 'arc_sweep_leq':
        A = geom_objects[c['arc']]
        max_deg = c['max_deg']
        max_rad = np.deg2rad(max_deg)
        def fun(x):
            _unpack_to_objs(x, geom_objects, x._idxs)
            d = abs(angle_diff(A.params[4], A.params[3])) - max_rad
            return np.array([d])
        return NonlinearConstraint(fun, lb=[-np.inf], ub=[0])

    # ---- 不等式：端点 y/x 次序 ----
    if t == 'arc_endpoint_relation':
        A = geom_objects[c['arc']]
        mode   = c['mode']   # 'y_order' / 'x_order'
        order  = c['order']  # 'asc' / 'desc'
        margin = c.get('margin', 0.0)
        def fun(x):
            _unpack_to_objs(x, geom_objects, x._idxs)
            P1, P2 = _get_arc_endpoints(A)
            if mode == 'y_order':
                val = (P2[1]-P1[1]) if order == 'asc' else (P1[1]-P2[1])
            else:
                val = (P2[0]-P1[0]) if order == 'asc' else (P1[0]-P2[0])
            return np.array([val - margin])
        return NonlinearConstraint(fun, lb=[0], ub=[np.inf])

    # ---- 不等式：arc 在 chord 上/下（简化版：按 y 与弦的线性插值比较）----
    if t == 'arc_side':
        A = geom_objects[c['arc']]
        side = c['side']     # 'upper'/'lower'
        samples = int(c.get('samples', 3))
        margin  = float(c.get('margin', 0.0))

        def fun(x):
            _unpack_to_objs(x, geom_objects, x._idxs)
            P1, P2 = _get_arc_endpoints(A)
            S = _arc_sample_points(A, samples=samples)
            # 对每个采样点，与弦的线性插值 y_chord 比较
            v = P2 - P1
            L2 = np.dot(v,v) + 1e-12
            tproj = np.clip(((S - P1) @ v)/L2, 0.0, 1.0)
            chord = P1[None,:] + tproj[:,None]*v[None,:]
            dy = S[:,1] - chord[:,1]
            if side == 'upper':
                return dy - margin  # >= 0
            else:
                return -(dy + margin)  # >= 0  <=> dy <= -margin
        if c['side'] == 'upper':
            return NonlinearConstraint(fun, lb=np.zeros( max(1, samples) ), ub=np.full(max(1, samples), np.inf))
        else:
            return NonlinearConstraint(fun, lb=np.zeros( max(1, samples) ), ub=np.full(max(1, samples), np.inf))

    # ---- 不等式：arc 在 line 的左/右侧（按线方向）----
    if t == 'arc_side_of_line':
        L = geom_objects[c['line']]
        A = geom_objects[c['arc']]
        side = c['side']   # 'left' / 'right'
        samples = int(c.get('samples', 3))
        margin  = float(c.get('margin', 0.0))

        def fun(x):
            _unpack_to_objs(x, geom_objects, x._idxs)
            S = _arc_sample_points(A, samples=samples)
            vals = np.array([_line_signed(L, p) for p in S])
            if side == 'left':
                return vals - margin  # >= 0
            else:
                return -(vals + margin)  # >= 0  <=> vals <= -margin
        m = max(1, samples)
        return NonlinearConstraint(fun, lb=np.zeros(m), ub=np.full(m, np.inf))

    # 未实现类型：安全忽略/报错
    raise NotImplementedError(f"Constraint type not implemented for trust-constr: {t}")

# ---------- 主求解函数 ----------
def solve_geometry_constrained(geom_objects, constraint_list,
                               max_restarts=12, jitter_scale=0.03,
                               ftol=1e-10, xtol=1e-10, ctol=1e-8,
                               verbose=True):
    """
    以等式/不等式的形式精确建模几何约束，使用 trust-constr + 多初值重启。
    - max_restarts: 多起点次数（含原始起点在内）
    - jitter_scale: 对自由变量的相对扰动幅度
    - ctol: 可行性容忍度（所有约束在此阈值内视为满足）
    """
    # 打包初值
    x0, idxs = _pack_vars(geom_objects)
    if x0.size == 0:
        return {'x': np.array([]), 'status': 0, 'message':'no free vars'}

    # 构建 NonlinearConstraint 列表
    cons_all = []
    for c in constraint_list:
        built = _make_constraint(geom_objects, c)
        if isinstance(built, list):
            cons_all.extend(built)
        else:
            cons_all.append(built)

    # 变量边界（可放宽很大，也可按需设置）
    lower = []
    upper = []
    # 使用对象自身 fixed_mask，未固定的变量给出大范围边界；半径限定正值
    k = 0
    for oi, mask in idxs:
        obj = geom_objects[oi]
        for j, free in enumerate(mask):
            if not free: continue
            # 对半径（Arc.params[2]）给正界
            if isinstance(obj, Arc) and j == 2 - np.sum(obj.fixed_mask[:2]):  # 映射到当前自由参数下的索引有点绕，用下面更稳的判断
                pass
            # 更稳妥：直接根据全参数位置判断
            full_pos = np.where(mask)[0][np.where(mask)[0]==j][0]  # not used
            # 简化：直接取对象全参数索引
        # 简单策略：全部给(-1e5, 1e5)
    lower = np.full(x0.shape, -1e5)
    upper = np.full(x0.shape, +1e5)

    # 但要把“对应半径”的自由变量强制 > 0
    # 找出哪些自由变量是半径位置
    rad_positions = []
    cursor = 0
    for oi, mask in idxs:
        obj = geom_objects[oi]
        free_idx = np.where(mask)[0]
        for jj, j in enumerate(free_idx):
            if isinstance(obj, Arc) and j == 2:  # Arc.params[2] == r
                rad_positions.append(cursor + jj)
        cursor += len(free_idx)
    if rad_positions:
        lower[rad_positions] = np.maximum(lower[rad_positions], 1e-6)

    bounds = Bounds(lower, upper)

    # 目标函数：极小正则（把解温和拉回起点），可避免纯等式问题无目标导致退化
    def obj(x):
        return 1e-6 * np.sum((x - x0)**2)

    # 为了在约束函数里能访问 idxs，把它挂到 x 上（scipy不会保留这个属性，但我们在 fun 内部会每次挂上）
    def attach_idxs(x):
        if not hasattr(x, "_idxs"):
            x._idxs = idxs
        return x

    # 单次求解包装
    def run_one(start):
        start = attach_idxs(start.copy())
        res = minimize(
            lambda z: obj(attach_idxs(z)),
            start,
            method='trust-constr',
            bounds=bounds,
            constraints=cons_all,
            options=dict(verbose=3 if verbose else 0, xtol=xtol, gtol=ftol, barrier_tol=ctol,
                         maxiter=1000)
        )
        # 计算最大约束违反度
        max_viol = 0.0
        for nc in cons_all:
            v = nc.fun(res.x)
            lb = np.array(nc.lb, dtype=float)
            ub = np.array(nc.ub, dtype=float)
            # 对等式：lb==ub==0；对不等式：lb<=v<=ub
            viol_low  = np.maximum(0.0, lb - v)
            viol_high = np.maximum(0.0, v - ub)
            max_viol = max(max_viol, float(np.max(viol_low)), float(np.max(viol_high)))
        return res, max_viol

    best = None
    best_viol = np.inf
    best_obj = np.inf

    # 第一次：原始初值
    res0, viol0 = run_one(x0)
    if verbose:
        print(f"[seed 0] feas_vio={viol0:.2e}, obj={res0.fun:.3e}, success={res0.success}")
    if (viol0 < best_viol - 1e-9) or (abs(viol0 - best_viol) < 1e-9 and res0.fun < best_obj):
        best, best_viol, best_obj = res0, viol0, res0.fun

    # 其余多次随机扰动
    rng = np.random.default_rng(42)
    scale = np.maximum(1.0, np.abs(x0))
    for s in range(1, max_restarts):
        jitter = rng.normal(0.0, jitter_scale, size=x0.shape) * scale
        xs = x0 + jitter
        res, viol = run_one(xs)
        if verbose:
            print(f"[seed {s}] feas_vio={viol:.2e}, obj={res.fun:.3e}, success={res.success}")
        if (viol < best_viol - 1e-9) or (abs(viol - best_viol) < 1e-9 and res.fun < best_obj):
            best, best_viol, best_obj = res, viol, res.fun

    # 写回几何对象
    _unpack_to_objs(best.x, geom_objects, idxs)
    if verbose:
        print(f"[best] feas_vio={best_viol:.2e}, obj={best_obj:.3e}, success={best.success}, msg={best.message}")
    return {
        'x': best.x, 'feas_violation': best_viol, 'objective': best_obj,
        'success': best.success, 'message': best.message, 'nit': best.nit
    }
