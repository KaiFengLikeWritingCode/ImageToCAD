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

def soft_hinge(z, k=10.0):
    """
    稳定版：soft_hinge(z,k) = softplus(k*z)/k
    softplus(x) = log1p(exp(x)) 的数值稳定实现：
      x>0: x + log1p(exp(-x))
      x<=0: log1p(exp(x))
    """
    x = k * z
    # 逐元素稳定计算
    pos = x > 0
    out = np.empty_like(x, dtype=float) if isinstance(x, np.ndarray) else float(x)
    # x>0 分支
    out = np.where(pos, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))
    return out / k

def softplus_stable(x):
    # 数值稳定 softplus
    return np.where(x>0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))
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



        elif type_ == 'tangent_at_arc_start_to_line':
            """
            弧在起点 t1 处与直线相切，并可选控制走向和曲率侧别
            keys:
              line, arc
              same_direction: True/False/None  (同向/反向/只要求平行)
              side: 'left'/'right'/None        (圆心在切线左/右侧)
              margin: float=0.0                (侧别的最小有符号距离，默认0)
            说明：
              直线方向按 p0->p1；左侧=叉积>0。
            """
            line = geom_objects[c['line']]
            arc = geom_objects[c['arc']]

            (x0, y0), (x1, y1) = line.points()
            vx, vy = (x1 - x0), (y1 - y0)
            vn = np.hypot(vx, vy) + 1e-12  # 防止除零

            t = arc.params[3]  # theta1
            tx, ty = -np.sin(t), np.cos(t)  # 弧在 t1 的单位切向

            # 1) 相切（平行）：叉积=0
            eqs.append(vx * ty - vy * tx)

            # 2) 可选：同向/反向（点积→ ±|v|）
            sd = c.get('same_direction', True)  # 默认同向更符合“走向”
            if sd is True:
                eqs.append((vx * tx + vy * ty) / vn - 1.0)
            elif sd is False:
                eqs.append((vx * tx + vy * ty) / vn + 1.0)

            # 3) 可选：控制圆心在切线的左/右侧（决定弧的“鼓起”方向）
            side = c.get('side', None)  # 'left' 或 'right'
            margin = float(c.get('margin', 0.0))
            if side is not None:
                cx, cy, r = arc.params[0], arc.params[1], arc.params[2]
                # 接触点（弧端点）
                px = cx + r * np.cos(t)
                py = cy + r * np.sin(t)
                # 切线的左法向（单位）
                nx, ny = -vy / vn, vx / vn
                # 圆心相对接触点在法向上的有符号距离：>0 表示在“左侧”
                s = (cx - px) * nx + (cy - py) * ny
                if side == 'left':
                    # 需要 s >= margin
                    eqs.append(soft_hinge(margin - s))
                elif side == 'right':
                    # 需要 s <= -margin
                    eqs.append(soft_hinge(s + margin))
                else:
                    raise ValueError("tangent_at_arc_start_to_line.side must be 'left' or 'right'")

        elif type_ == 'tangent_at_arc_end_to_line':
            """
            弧在终点 t2 处与直线相切，并可选控制走向和曲率侧别
            keys 同上
            """
            line = geom_objects[c['line']]
            arc = geom_objects[c['arc']]

            (x0, y0), (x1, y1) = line.points()
            vx, vy = (x1 - x0), (y1 - y0)
            vn = np.hypot(vx, vy) + 1e-12

            t = arc.params[4]  # theta2
            tx, ty = -np.sin(t), np.cos(t)

            # 1) 相切（平行）
            eqs.append(vx * ty - vy * tx)

            # 2) 同/反向
            sd = c.get('same_direction', True)
            if sd is True:
                eqs.append((vx * tx + vy * ty) / vn - 1.0)
            elif sd is False:
                eqs.append((vx * tx + vy * ty) / vn + 1.0)

            # 3) 侧别（圆心在切线左/右）
            side = c.get('side', None)
            margin = float(c.get('margin', 0.0))
            if side is not None:
                cx, cy, r = arc.params[0], arc.params[1], arc.params[2]
                px = cx + r * np.cos(t)
                py = cy + r * np.sin(t)
                nx, ny = -vy / vn, vx / vn
                s = (cx - px) * nx + (cy - py) * ny
                if side == 'left':
                    eqs.append(soft_hinge(margin - s))
                elif side == 'right':
                    eqs.append(soft_hinge(s + margin))
                else:
                    raise ValueError("tangent_at_arc_end_to_line.side must be 'left' or 'right'")






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

        elif type_ == 'arc_endpoint_relation':
            """
            约束圆弧两端点 P1(t1)、P2(t2) 的相对关系。
            用法示例：
              # 1) X 顺序：要求 x2 >= x1 (+margin)
              {'type':'arc_endpoint_relation','arc':k,'mode':'x_order','order':'asc','margin':0.0}
        
              # 2) Y 顺序：要求 y2 <= y1 (-margin)
              {'type':'arc_endpoint_relation','arc':k,'mode':'y_order','order':'desc','margin':0.1}
        
              # 3) X 差值等式：x2 - x1 = dx
              {'type':'arc_endpoint_relation','arc':k,'mode':'x_diff_eq','value':10.0}
        
              # 4) 端点距离等/不等式：
              {'type':'arc_endpoint_relation','arc':k,'mode':'dist_eq','value':25.0}
              {'type':'arc_endpoint_relation','arc':k,'mode':'dist_ge','value':20.0}
              {'type':'arc_endpoint_relation','arc':k,'mode':'dist_le','value':30.0}
        
              # 5) 方向锥：P1→P2 指向与角度 alpha（弧度）相差不超过 phi（半角）
              {'type':'arc_endpoint_relation','arc':k,'mode':'dir_cone','alpha':0.0,'phi':np.deg2rad(15)}
            """


            arc = geom_objects[c['arc']]
            cx, cy, r, t1, t2 = arc.params

            # 端点
            x1, y1 = cx + r * np.cos(t1), cy + r * np.sin(t1)
            x2, y2 = cx + r * np.cos(t2), cy + r * np.sin(t2)

            mode = c.get('mode', 'x_order')

            # ---- 1) X / Y 顺序：asc(≥) 或 desc(≤) ----
            if mode == 'x_order':
                order = c.get('order', 'asc')  # 'asc' or 'desc'
                margin = float(c.get('margin', 0.0))
                if order == 'asc':  # x2 >= x1 + margin
                    eqs.append(soft_hinge((x1 + margin) - x2))
                elif order == 'desc':  # x2 <= x1 - margin
                    eqs.append(soft_hinge(x2 - (x1 - margin)))
                else:
                    raise ValueError("x_order.order must be 'asc' or 'desc'")

            elif mode == 'y_order':
                order = c.get('order', 'asc')  # 'asc' or 'desc'
                margin = float(c.get('margin', 0.0))
                if order == 'asc':  # y2 >= y1 + margin
                    eqs.append(soft_hinge((y1 + margin) - y2))
                elif order == 'desc':  # y2 <= y1 - margin
                    eqs.append(soft_hinge(y2 - (y1 - margin)))
                else:
                    raise ValueError("y_order.order must be 'asc' or 'desc'")

            # ---- 2) X/Y 差值等式 ----
            elif mode == 'x_diff_eq':
                dx = float(c['value'])
                eqs.append((x2 - x1) - dx)
            elif mode == 'y_diff_eq':
                dy = float(c['value'])
                eqs.append((y2 - y1) - dy)

            # ---- 3) 端点距离：等式 / 不等式 ----
            elif mode == 'dist_eq':
                L = float(c['value'])
                dist = np.hypot(x2 - x1, y2 - y1)
                eqs.append(dist - L)  # 等式

            elif mode == 'dist_ge':
                L = float(c['value'])
                dist = np.hypot(x2 - x1, y2 - y1)
                eqs.append(soft_hinge(L - dist))  # dist >= L

            elif mode == 'dist_le':
                L = float(c['value'])
                dist = np.hypot(x2 - x1, y2 - y1)
                eqs.append(soft_hinge(dist - L))  # dist <= L

            # ---- 4) 方向锥：P1→P2 与目标方向 alpha 的夹角 ≤ phi ----
            elif mode == 'dir_cone':
                alpha = float(c.get('alpha', 0.0))  # 目标方向（弧度）
                phi = float(c.get('phi', np.deg2rad(10)))  # 半角
                ux, uy = np.cos(alpha), np.sin(alpha)  # 目标方向的单位向量

                vx, vy = (x2 - x1), (y2 - y1)
                vn = np.hypot(vx, vy) + 1e-12
                cosang = (vx * ux + vy * uy) / vn  # cos(angle(P1->P2, alpha))
                cos_bound = np.cos(phi)  # 需要 cos(angle) ≥ cos(phi)

                eqs.append(soft_hinge(cos_bound - cosang))

            else:
                raise ValueError("arc_endpoint_relation.mode not supported")

        # ---------- 线段端点左右关系（确定方向） ----------
        elif t == 'line_point_order_x':
            li = c['line']
            order = c.get('order', 'x1_left')  # x1_left 或 x2_left
            margin = float(c.get('margin', 0.0))
            sgn = -1.0 if order == 'x1_left' else +1.0
            # x1_left: 需要 x2 - x1 >= margin → -(x2 - x1 - margin) <= 0
            # x2_left: 需要 x1 - x2 >= margin → -(x1 - x2 - margin) <= 0
            def f_ineq(x, li=li, sgn=sgn, margin=margin):
                L: Line = self.G[li]
                x1, x2 = L.p(0)[0], L.p(1)[0]
                if sgn < 0:
                    return np.array([-(x2 - x1 - margin)])  # x1_left
                else:
                    return np.array([-(x1 - x2 - margin)])  # x2_left
            self.ineq_funcs.append(f_ineq)





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
    # res = least_squares(constraints, x0, args=(geom_objects, constraint_list))
    res = least_squares(
        constraints, x0,
        args=(geom_objects, constraint_list),
        loss='linear',  # 线性损失，不做鲁棒近似
        # loss='soft_l1',  # 尝试使用鲁棒损失函数
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
        # max_nfev=5000,  # 允许更多迭代
        verbose=2  # 调试期观察收敛
    )

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
    from wheelobject_4 import geom_objects, constraint_list

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
