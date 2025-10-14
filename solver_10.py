import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


# -----------------------------
# 定义几何对象
# -----------------------------
class Line:
    def __init__(self, x1=None, y1=None, x2=None, y2=None):
        # Store all parameters, None means unknown/optimizable
        self.params = np.array([x1, y1, x2, y2], dtype=float)
        self.fixed_mask = np.array([x1 is not None, y1 is not None, x2 is not None, y2 is not None])

    def points(self):
        return self.params[:2], self.params[2:]

    def get_optimizable_params(self):
        """Return only the parameters that need to be optimized"""
        return self.params[~self.fixed_mask]

    def set_optimizable_params(self, values):
        """Set the optimizable parameters with solved values"""
        self.params[~self.fixed_mask] = values


class Arc:
    def __init__(self, cx=None, cy=None, r=None, theta1=None, theta2=None):
        # Store all parameters, None means unknown/optimizable
        self.params = np.array([cx, cy, r, theta1, theta2], dtype=float)
        self.fixed_mask = np.array([cx is not None, cy is not None, r is not None,
                                    theta1 is not None, theta2 is not None])

    def points(self):
        cx, cy, r, t1, t2 = self.params
        x1 = cx + r * np.cos(t1)
        y1 = cy + r * np.sin(t1)
        x2 = cx + r * np.cos(t2)
        y2 = cy + r * np.sin(t2)
        return (x1, y1), (x2, y2)

    def get_optimizable_params(self):
        """Return only the parameters that need to be optimized"""
        return self.params[~self.fixed_mask]

    def set_optimizable_params(self, values):
        """Set the optimizable parameters with solved values"""
        self.params[~self.fixed_mask] = values




def constraints(vars, geom_objects, constraint_list):
    eqs = []
    idx = 0
    # 将 vars 拆分到各个对象的可优化参数
    for obj in geom_objects:
        # 这里的 obj.params 会随着 vars 的变化而变化
        obj.set_optimizable_params(vars[idx:idx + len(obj.get_optimizable_params())])
        idx += len(obj.get_optimizable_params())

    # 遍历约束
    for c in constraint_list:
        type_ = c['type']
        if type_ == 'length':
            line = geom_objects[c['line']]
            (x1, y1), (x2, y2) = line.points()
            eqs.append((x2 - x1) ** 2 + (y2 - y1) ** 2 - c['value'] ** 2)
        elif type_ == 'radius':
            arc = geom_objects[c['arc']]
            eqs.append(arc.params[2] - c['value'])
        elif type_ == 'point_distance_x':
            p1 = geom_objects[c['p1']].points()[c['which1']]
            p2 = geom_objects[c['p2']].points()[c['which2']]
            eqs.append(p2[0] - p1[0] - c['value'])
        elif type_ == 'point_distance_y':
            p1 = geom_objects[c['p1']].points()[c['which1']]
            p2 = geom_objects[c['p2']].points()[c['which2']]
            eqs.append(p2[1] - p1[1] - c['value'])
        elif type_ == 'tangent_line_arc':
            line = geom_objects[c['line']]
            arc = geom_objects[c['arc']]
            # 点到直线距离 = r
            x1, y1 = arc.params[0], arc.params[1]
            r = arc.params[2]
            lx1, ly1 = line.points()[0]
            lx2, ly2 = line.points()[1]
            A = ly1 - ly2
            B = lx2 - lx1
            C = lx1 * ly2 - lx2 * ly1
            # dist = np.abs(A*x1 + B*y1 + C)/np.sqrt(A**2 + B**2)
            # eqs.append(dist - r)
            den = A * A + B * B
            if den < 1e-12:
                # 线段退化（x1,y1==x2,y2），给一个温和的惩罚以拉开端点
                eqs.append(1.0)  # 或者 eqs.append((lx2-lx1)**2 + (ly2-ly1)**2 - 1.0)
            else:
                signed = (A * x1 + B * y1 + C)
                # (距离^2 - r^2) 作为残差，更平滑
                eqs.append(signed * signed / den - r * r)

        elif type_ == 'coincident':
            p1 = geom_objects[c['p1']].points()[c['which1']]
            p2 = geom_objects[c['p2']].points()[c['which2']]
            eqs.append(p1[0] - p2[0])
            eqs.append(p1[1] - p2[1])

        # ---------- 限制弧扫角：|t2 - t1| <= max_rad ----------
        elif type_ == 'arc_sweep_leq':
            arc = geom_objects[c['arc']]
            t1 = arc.params[3]
            t2 = arc.params[4]
            # 允许传入角度或弧度；默认 360°
            if 'max_rad' in c:
                max_rad = float(c['max_rad'])
            elif 'max_deg' in c:
                max_rad = np.deg2rad(float(c['max_deg']))
            else:
                max_rad = 2.0 * np.pi  # 默认 360°

            # 直接按当前参数化的差值衡量扫角（与 points() 的用法一致）
            sweep = t2 - t1
            # 惩罚：超过上限的部分（双边）
            over = np.maximum(0.0, abs(sweep) - max_rad)
            # 用一次残差即可（也可以 over 自己平方）
            eqs.append(over)
        # ---------- 线段端点 与 弧端点 重合 ----------
        # 用法：
        # {'type':'intersect_line_arc_at',
        #  'line':i, 'which':'start'/'end' 或 0/1,
        #  'arc':k,  'arc_end':'start'/'end'}
        elif type_ == 'intersect_line_arc_at':
            L   = geom_objects[c['line']]
            arc = geom_objects[c['arc']]

            # 线段端点
            which = c.get('which', 'end')
            if isinstance(which, str):
                li = 0 if which.lower() == 'start' else 1
            else:
                li = int(which)
            (lx, ly) = L.points()[li]

            # 弧端点
            arc_end = c.get('arc_end', 'start')
            use_t1  = (isinstance(arc_end, str) and arc_end.lower() == 'start') \
                      or (not isinstance(arc_end, str) and not arc_end)
            cx, cy, r, t1, t2 = arc.params
            t  = t1 if use_t1 else t2
            ax = cx + r * np.cos(t)
            ay = cy + r * np.sin(t)

            # 同点约束
            eqs.append(lx - ax)
            eqs.append(ly - ay)

        # ---------- 弧端点 与 弧端点 重合 ----------
        # 用法：
        # {'type':'intersect_arc_arc_at',
        #  'Aarc':i, 'a_end':'start'/'end',
        #  'Barc':j, 'b_end':'start'/'end'}
        elif type_ == 'intersect_arc_arc_at':
            A = geom_objects[c['Aarc']]
            B = geom_objects[c['Barc']]

            # A 的端点
            a_end = c.get('a_end', 'end')
            a_use_t1 = (isinstance(a_end, str) and a_end.lower() == 'start') \
                       or (not isinstance(a_end, str) and not a_end)
            acx, acy, ar, at1, at2 = A.params
            ta = at1 if a_use_t1 else at2
            ax = acx + ar * np.cos(ta)
            ay = acy + ar * np.sin(ta)

            # B 的端点
            b_end = c.get('b_end', 'start')
            b_use_t1 = (isinstance(b_end, str) and b_end.lower() == 'start') \
                       or (not isinstance(b_end, str) and not b_end)
            bcx, bcy, br, bt1, bt2 = B.params
            tb = bt1 if b_use_t1 else bt2
            bx = bcx + br * np.cos(tb)
            by = bcy + br * np.sin(tb)

            # 同点约束
            eqs.append(ax - bx)
            eqs.append(ay - by)

        elif type_ == 'tangent_at_arc_to_arc':
            """
            端点处弧-弧相切（G1）
            参数：
              Aarc, Barc: 参与的两段弧的索引
              a_end: True 表示用 A 的 t2（end），False 表示用 t1（start）
              b_end: True 表示用 B 的 t2（end），False 表示用 t1（start）
              same_direction (可选):
                  None/缺省 -> 只要求平行（允许同/反向）
                  True      -> 同向（切向夹角接近 0°）
                  False     -> 反向（切向夹角接近 180°）
            备注：
              需要配合 'coincident' 把这两个端点重合起来（几何上是同一点）。
            """
            Aarc = geom_objects[c['Aarc']]
            Barc = geom_objects[c['Barc']]

            a_end = c.get('a_end', True)  # True→A.t2；False→A.t1
            b_end = c.get('b_end', False)  # True→B.t2；False→B.t1
            same_dir = c.get('same_direction', None)

            # 选取两个弧的端点角
            ta = Aarc.params[4] if a_end else Aarc.params[3]
            tb = Barc.params[4] if b_end else Barc.params[3]

            # 端点处的单位切向量（圆参数化：(-sin t, cos t)）
            txa, tya = -np.sin(ta), np.cos(ta)
            txb, tyb = -np.sin(tb), np.cos(tb)

            # 1) 平行性（允许同向或反向）：叉积 = 0
            eqs.append(txa * tyb - tya * txb)

            # 2) 可选：控制“同向/反向”
            # 因为 (-sin t, cos t) 本身是单位向量，dot ∈ [-1,1]。
            # same_direction=True  -> dot 应接近 +1
            # same_direction=False -> dot 应接近 -1
            if same_dir is True:
                eqs.append((txa * txb + tya * tyb) - 1.0)
            elif same_dir is False:
                eqs.append((txa * txb + tya * tyb) + 1.0)





        # elif type_ == 'tangent_at_arc_to_arc':
        #     """
        #     端点处弧-弧相切（G1）
        #     参数：
        #       Aarc, Barc: 弧索引
        #       a_end: True→A用t2（end），False→A用t1（start）
        #       b_end: True→B用t2（end），False→B用t1（start）
        #       same_direction: None/True/False（控制切向同向/反向，可选）
        #       tangent_type: 'internal' 或 'external'（控制内切/外切，可选）
        #       curv_margin: 法向侧的微小裕量（默认0）
        #     说明：
        #       需要配合 'coincident' 把对应两个端点重合。
        #     """
        #     Aarc = geom_objects[c['Aarc']]
        #     Barc = geom_objects[c['Barc']]
        #
        #     a_end = c.get('a_end', True)
        #     b_end = c.get('b_end', False)
        #     same_dir = c.get('same_direction', None)
        #     tt = c.get('tangent_type', None)  # 'internal' or 'external'
        #     margin = float(c.get('curv_margin', 0.0))
        #
        #     ta = Aarc.params[4] if a_end else Aarc.params[3]
        #     tb = Barc.params[4] if b_end else Barc.params[3]
        #
        #     # 端点处切向（单位）
        #     txa, tya = -np.sin(ta), np.cos(ta)
        #     txb, tyb = -np.sin(tb), np.cos(tb)
        #
        #     # G1：切向平行（叉积=0）
        #     eqs.append(txa * tyb - tya * txb)
        #
        #     # 可选：同向/反向
        #     if same_dir is True:
        #         eqs.append((txa * txb + tya * tyb) - 1.0)
        #     elif same_dir is False:
        #         eqs.append((txa * txb + tya * tyb) + 1.0)
        #
        #     # 可选：内切/外切（看两圆心在公共法向的同侧/异侧）
        #     if tt is not None:
        #         # 公共接触点（由A算；因coincident，B相同）
        #         cxA, cyA, rA = Aarc.params[0], Aarc.params[1], Aarc.params[2]
        #         Px = cxA + rA * np.cos(ta)
        #         Py = cyA + rA * np.sin(ta)
        #
        #         # A 的法向（单位），指向弧的“凸侧”
        #         nx, ny = np.cos(ta), np.sin(ta)
        #
        #         # 两圆心到接触点在法向上的投影（有符号）
        #         da = (Aarc.params[0] - Px) * nx + (Aarc.params[1] - Py) * ny
        #         db = (Barc.params[0] - Px) * nx + (Barc.params[1] - Py) * ny
        #         prod = da * db
        #
        #         if tt == 'internal':
        #             # 内切：同侧 → prod >= margin^2
        #             eqs.append(np.maximum(0.0, (margin * margin) - prod))
        #         elif tt == 'external':
        #             # 外切：异侧 → prod <= -margin^2
        #             eqs.append(np.maximum(0.0, prod + (margin * margin)))
        #         else:
        #             raise ValueError("tangent_type must be 'internal' or 'external'")



        elif type_ == 'tangent_at_arc_start_to_line':
            line = geom_objects[c['line']]
            arc = geom_objects[c['arc']]
            (x1, y1), (x2, y2) = line.points()
            A, B = (y1 - y2), (x2 - x1)
            # 直线方向向量（单位化可不必，叉积只需方向）
            lx, ly = (x2 - x1), (y2 - y1)
            t = arc.params[3]  # theta1
            tx, ty = -np.sin(t), np.cos(t)  # 弧在 start 的切向
            eqs.append(lx * ty - ly * tx)

        elif type_ == 'tangent_at_arc_end_to_line':
            line = geom_objects[c['line']]
            arc = geom_objects[c['arc']]
            lx, ly = (line.params[2] - line.params[0]), (line.params[3] - line.params[1])
            t = arc.params[4]  # theta2
            tx, ty = -np.sin(t), np.cos(t)
            eqs.append(lx * ty - ly * tx)


        elif type_ == 'arc_side':
            """
            控制弧相对圆心“上半弧/下半弧”（默认按 y 轴判定）。
            用法：
              {'type':'arc_side','arc':k, 'side':'upper'/'lower',
               'samples':1 或 3, 'weight':1.0}
            解释：
              - side='upper' → 采样点的 (y - cy) >= 0
              - side='lower' → 采样点的 (y - cy) <= 0
              - samples: 取 1 表示用中点角；取 3 表示用 t1, mid, t2 三点都约束（更稳）
            """
            arc = geom_objects[c['arc']]
            cx, cy, r, t1, t2 = arc.params
            side = c.get('side', 'upper')  # 'upper' or 'lower'
            w = float(c.get('weight', 1.0))
            samples = int(c.get('samples', 1))

            # 采样角
            if samples <= 1:
                ts = [0.5 * (t1 + t2)]
            else:
                ts = [t1, 0.5 * (t1 + t2), t2]

            for t in ts:
                y_rel = cy + r * np.sin(t) - cy  # = r*sin(t)
                if side == 'upper':
                    # 需要 y_rel >= 0 → 违反量为 -y_rel
                    eqs.append(w * np.maximum(0.0, -y_rel))
                elif side == 'lower':
                    # 需要 y_rel <= 0 → 违反量为 +y_rel
                    eqs.append(w * np.maximum(0.0, y_rel))
                else:
                    raise ValueError("arc_side.side must be 'upper' or 'lower'")
        elif type_ == 'line_second_side':
            """
            约束线段的第二个点 (x2,y2) 相对第一个点 (x1,y1) 的上下关系。
            用法：
              {'type':'line_second_side','line':i,
               'side':'above'/'below', 'margin':0.0, 'weight':1.0}
            解释：
              side='above' → y2 >= y1 (+ margin)
              side='below' → y2 <= y1 (- margin)
              margin：给一点安全间隙，避免正好贴边抖动
            """
            L = geom_objects[c['line']]
            (x1, y1), (x2, y2) = L.points()

            side = c.get('side', 'above')  # 'above' or 'below'
            margin = float(c.get('margin', 0.0))
            w = float(c.get('weight', 1.0))

            dy = y2 - y1

            if side == 'above':
                # 需要 dy >= margin → 违反量 = margin - dy
                eqs.append(w * np.maximum(0.0, margin - dy))
            elif side == 'below':
                # 需要 dy <= -margin → 违反量 = dy + margin
                eqs.append(w * np.maximum(0.0, dy + margin))
            else:
                raise ValueError("line_second_side.side must be 'above' or 'below'")

        elif type_ == 'arc_side_of_line':
            """
            让整段弧（除相切端点附近）都在指定直线的左/右侧（相对 line 的方向 p0->p1）。
            line, arc, anchor('start'/'end'), side('left'/'right'), samples, margin, weight
            """
            L = geom_objects[c['line']]
            arc = geom_objects[c['arc']]
            (x0, y0), (x1, y1) = L.points()
            vx, vy = (x1 - x0), (y1 - y0)  # 线段方向

            cx, cy, r, t1, t2 = arc.params
            side = c.get('side', 'left')
            anchor = c.get('anchor', 'start')
            samples = int(c.get('samples', 3))
            margin = float(c.get('margin', 0.0))
            w = float(c.get('weight', 1.0))

            ts = []
            if samples <= 1:
                ts = [0.5 * (t1 + t2)]
            else:
                eps = 1e-3
                for tt in [t1, 0.5 * (t1 + t2), t2]:
                    if anchor == 'start' and abs(tt - t1) < eps: continue
                    if anchor == 'end' and abs(tt - t2) < eps: continue
                    ts.append(tt)
                if not ts: ts = [0.5 * (t1 + t2)]

            for t in ts:
                px = cx + r * np.cos(t);
                py = cy + r * np.sin(t)
                wx, wy = (px - x0), (py - y0)
                z = vx * wy - vy * wx  # z>0 左侧；z<0 右侧
                if side == 'left':
                    eqs.append(w * np.maximum(0.0, margin - z))
                elif side == 'right':
                    eqs.append(w * np.maximum(0.0, z + margin))
                else:
                    raise ValueError("arc_side_of_line.side must be 'left' or 'right'")
        elif type_ == 'tangent_at_endpoint_arc_arc':
            # 必须配合 'coincident' 让两弧端点重合
            A = geom_objects[c['Aarc']]
            B = geom_objects[c['Barc']]
            a_end = c.get('a_end', True)  # True->t2, False->t1
            b_end = c.get('b_end', False)

            ta = A.params[4] if a_end else A.params[3]
            tb = B.params[4] if b_end else B.params[3]

            # 端点处单位切向量（逆时针参数化的圆：t切向=(-sin t, cos t)）
            tAx, tAy = -np.sin(ta), np.cos(ta)
            tBx, tBy = -np.sin(tb), np.cos(tb)

            # 端点处单位法向（指向圆心）：n=(cos t, sin t)
            nAx, nAy = np.cos(ta), np.sin(ta)
            nBx, nBy = np.cos(tb), np.sin(tb)

            # 1) 切向平行（避免尖点）
            eqs.append(tAx * tBy - tAy * tBx)

            # 可选：切向同/反向
            if 'same_direction' in c:
                eqs.append((tAx * tBx + tAy * tBy) - (1.0 if c['same_direction'] else -1.0))

            # 2) 半径方向（法向）共线 + 内/外切判别
            # 共线
            eqs.append(nAx * nBy - nAy * nBx)
            # 方向：internal(+1) / external(-1)
            tang_type = c.get('tangent_type', 'external')  # 'internal' or 'external'
            s = +1.0 if tang_type == 'internal' else -1.0
            eqs.append((nAx * nBx + nAy * nBy) - s)

            # 3) 圆心距 = r1±r2
            cxA, cyA, rA = A.params[0], A.params[1], A.params[2]
            cxB, cyB, rB = B.params[0], B.params[1], B.params[2]
            dx, dy = (cxA - cxB), (cyA - cyB)
            if tang_type == 'internal':
                target = (rA - rB)
            else:
                target = (rA + rB)
            eqs.append(dx * dx + dy * dy - target * target)

    return eqs


def get_initial_params(geom_objects):
    """Extract initial parameters for optimization"""
    params = []
    for obj in geom_objects:
        optimizable = obj.get_optimizable_params()
        if len(optimizable) > 0:
            # Use current values if available, otherwise use reasonable defaults
            default_values = []
            # Use string-based type checking to avoid module import issues
            type_name = type(obj).__name__
            if type_name == 'Line':
                defaults = [0.0, 0.0, 1.0, 0.0]  # x1, y1, x2, y2
            elif type_name == 'Arc':
                defaults = [0.0, 0.0, 1.0, 0.0, np.pi / 2]  # cx, cy, r, theta1, theta2
            else:
                print(type(obj))
                raise ValueError("Invalid object type")

            for i, val in enumerate(optimizable):
                if np.isnan(val):
                    default_values.append(defaults[i])
                else:
                    default_values.append(val)
            params.extend(default_values)
    return np.array(params)


def solve_geometry(geom_objects, constraint_list, verbose=True):
    """
    Solve geometry with partial parameters

    Args:
        geom_objects: List of Line and Arc objects with partial parameters
        constraint_list: List of constraint dictionaries
        verbose: Whether to print solving progress

    Returns:
        Result object from scipy.optimize.least_squares
    """
    if verbose:
        print("Initial parameters:")
        for i, obj in enumerate(geom_objects):
            print(f"  Object {i}: {obj.params} (fixed: {obj.fixed_mask})")

    # Get initial parameters for optimization
    x0 = get_initial_params(geom_objects)
    if verbose:
        print(f"Initial optimization params: {x0}")

    # Solve
    res = least_squares(constraints, x0, args=(geom_objects, constraint_list))

    if verbose:
        print(f"Solver result: {res.success}")
        if res.success:
            print(f"Final parameters:")
            for i, obj in enumerate(geom_objects):
                print(f"  Object {i}: {obj.params}")
        else:
            print("Solver failed!")

    return res


# Helper函数：给出来一系列的几何对象（直线和圆弧），自动生成一系列的前后相连，最后一个对象的终点连回第一个的起点的约束
def generate_coincident_constraint_list(geom_objects):
    constraint_list = []
    for i in range(len(geom_objects)):
        constraint_list.append(
            {'type': 'coincident', 'p1': i, 'which1': 1, 'p2': (i + 1) % len(geom_objects), 'which2': 0})
    return constraint_list


if __name__ == "__main__":
    # Example 1: Line with only start point fixed, Arc with only radius fixed
    print("Example 1: 车轮生成示例")
    # extract the objects and constrainst from wheel.py
    from wheelobject import geom_objects, constraint_list

    print("Geom objects:")
    for obj in geom_objects:
        print(obj.params)
    print("Constraint list:")
    for c in constraint_list:
        print(c)

    # -----------------------------
    # 求解
    # -----------------------------
    res = solve_geometry(geom_objects, constraint_list)

    print("\n" + "=" * 50 + "\n")

    # -----------------------------
    # 可视化
    # -----------------------------
    plt.figure(figsize=(15, 10))
    plt.title("Wheel Profile - CAD Constraint Solution")

    # Plot the wheel profile
    for obj in geom_objects:
        if type(obj).__name__ == 'Line':
            (x1, y1), (x2, y2) = obj.points()
            plt.plot([x1, x2], [y1, y2], 'b-', linewidth=2)
        elif type(obj).__name__ == 'Arc':
            cx, cy, r, t1, t2 = obj.params
            ts = np.linspace(t1, t2, 50)
            xs = cx + r * np.cos(ts)
            ys = cy + r * np.sin(ts)
            plt.plot(xs, ys, 'r-', linewidth=2)
            # Mark center points for arcs
            plt.plot(cx, cy, 'ro', markersize=3, alpha=0.7)

    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title('Wheel Profile - Solved Geometry')

    # Add some annotations for key dimensions
    plt.text(0.02, 0.98, f'Total Width: 420mm\nHeight: 178mm',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()
