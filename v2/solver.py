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
        x1 = cx + r*np.cos(t1)
        y1 = cy + r*np.sin(t1)
        x2 = cx + r*np.cos(t2)
        y2 = cy + r*np.sin(t2)
        return (x1,y1), (x2,y2)
    
    def get_optimizable_params(self):
        """Return only the parameters that need to be optimized"""
        return self.params[~self.fixed_mask]
    
    def set_optimizable_params(self, values):
        """Set the optimizable parameters with solved values"""
        self.params[~self.fixed_mask] = values

# -----------------------------
# 构建方程
# -----------------------------
# def constraints(vars, geom_objects, constraint_list):
#     eqs = []
#     idx = 0
#     # 将 vars 拆分到各个对象的可优化参数
#     for obj in geom_objects:
#         # 这里的 obj.params 会随着 vars 的变化而变化
#         obj.set_optimizable_params(vars[idx:idx+len(obj.get_optimizable_params())])
#         idx += len(obj.get_optimizable_params())
#
#     # 遍历约束
#     for c in constraint_list:
#         type_ = c['type']
#         if type_ == 'length':
#             line = geom_objects[c['line']]
#             (x1,y1),(x2,y2) = line.points()
#             eqs.append((x2-x1)**2 + (y2-y1)**2 - c['value']**2)
#         elif type_ == 'radius':
#             arc = geom_objects[c['arc']]
#             eqs.append(arc.params[2] - c['value'])
#         elif type_ == 'point_distance_x':
#             p1 = geom_objects[c['p1']].points()[c['which1']]
#             p2 = geom_objects[c['p2']].points()[c['which2']]
#             eqs.append(p2[0] - p1[0] - c['value'])
#         elif type_ == 'point_distance_y':
#             p1 = geom_objects[c['p1']].points()[c['which1']]
#             p2 = geom_objects[c['p2']].points()[c['which2']]
#             eqs.append(p2[1] - p1[1] - c['value'])
#         elif type_ == 'tangent_line_arc':
#             line = geom_objects[c['line']]
#             arc = geom_objects[c['arc']]
#             # 点到直线距离 = r
#             x1,y1 = arc.params[0], arc.params[1]
#             r = arc.params[2]
#             lx1,ly1 = line.points()[0]
#             lx2,ly2 = line.points()[1]
#             A = ly1 - ly2
#             B = lx2 - lx1
#             C = lx1*ly2 - lx2*ly1
#             # dist = np.abs(A*x1 + B*y1 + C)/np.sqrt(A**2 + B**2)
#             # eqs.append(dist - r)
#             den = A * A + B * B
#             if den < 1e-12:
#                 # 线段退化（x1,y1==x2,y2），给一个温和的惩罚以拉开端点
#                 eqs.append(1.0)  # 或者 eqs.append((lx2-lx1)**2 + (ly2-ly1)**2 - 1.0)
#             else:
#                 signed = (A * x1 + B * y1 + C)
#                 # (距离^2 - r^2) 作为残差，更平滑
#                 eqs.append(signed * signed / den - r * r)
#
#         elif type_ == 'coincident':
#             p1 = geom_objects[c['p1']].points()[c['which1']]
#             p2 = geom_objects[c['p2']].points()[c['which2']]
#             eqs.append(p1[0]-p2[0])
#             eqs.append(p1[1]-p2[1])
#
#         # --- 在 constraints() 里追加三种新类型 ---
#         elif type_ == 'tangent_at_arc_start_to_line':
#             line = geom_objects[c['line']]
#             arc = geom_objects[c['arc']]
#             (x1, y1), (x2, y2) = line.points()
#             vx, vy = x2 - x1, y2 - y1  # 线段方向
#             t = arc.params[3]  # theta1
#             tx, ty = -np.sin(t), np.cos(t)  # 弧在 start 处的切向
#             # 平行性：叉积为 0
#             eqs.append(vx * ty - vy * tx)
#
#         elif type_ == 'tangent_at_arc_end_to_line':
#             line = geom_objects[c['line']]
#             arc = geom_objects[c['arc']]
#             (x1, y1), (x2, y2) = line.points()
#             vx, vy = x2 - x1, y2 - y1
#             t = arc.params[4]  # theta2
#             tx, ty = -np.sin(t), np.cos(t)
#             eqs.append(vx * ty - vy * tx)
#
#
#         elif type_ == 'tangent_arc_arc':
#             # Aarc 与 Barc 在各自端点相切（G1）。默认 A 用 end(t2)，B 用 start(t1)。
#             Aarc = geom_objects[c['Aarc']]
#             Barc = geom_objects[c['Barc']]
#             a_end = c.get('a_end', True)  # True→用 A.t2；False→用 A.t1
#             b_end = c.get('b_end', False)  # True→用 B.t2；False→用 B.t1
#
#             ta = Aarc.params[4] if a_end else Aarc.params[3]
#             tb = Barc.params[4] if b_end else Barc.params[3]
#
#             # 弧在角度 t 的切向向量（未归一化也可）
#             txa, tya = -np.sin(ta), np.cos(ta)
#             txb, tyb = -np.sin(tb), np.cos(tb)
#
#             # 平行性（允许同向或反向）：叉积=0
#             eqs.append(txa * tyb - tya * txb)
#         elif type_ == 'tangent_at_arc_start_to_line':
#             line = geom_objects[c['line']]
#             arc = geom_objects[c['arc']]
#             (x1, y1), (x2, y2) = line.points()
#             A, B = (y1 - y2), (x2 - x1)
#             # 直线方向向量（单位化可不必，叉积只需方向）
#             lx, ly = (x2 - x1), (y2 - y1)
#             t = arc.params[3]  # theta1
#             tx, ty = -np.sin(t), np.cos(t)  # 弧在 start 的切向
#             eqs.append(lx * ty - ly * tx)
#
#         elif type_ == 'tangent_at_arc_end_to_line':
#             line = geom_objects[c['line']]
#             arc = geom_objects[c['arc']]
#             lx, ly = (line.params[2] - line.params[0]), (line.params[3] - line.params[1])
#             t = arc.params[4]  # theta2
#             tx, ty = -np.sin(t), np.cos(t)
#             eqs.append(lx * ty - ly * tx)
#
#
#     return eqs


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



        # elif type_ == 'tangent_arc_arc':
        #     # Aarc 与 Barc 在各自端点相切（G1）。默认 A 用 end(t2)，B 用 start(t1)。
        #     Aarc = geom_objects[c['Aarc']]
        #     Barc = geom_objects[c['Barc']]
        #     a_end = c.get('a_end', True)  # True→用 A.t2；False→用 A.t1
        #     b_end = c.get('b_end', False)  # True→用 B.t2；False→用 B.t1
        #
        #     ta = Aarc.params[4] if a_end else Aarc.params[3]
        #     tb = Barc.params[4] if b_end else Barc.params[3]
        #
        #     # 弧在角度 t 的切向向量（未归一化也可）
        #     txa, tya = -np.sin(ta), np.cos(ta)
        #     txb, tyb = -np.sin(tb), np.cos(tb)
        #
        #     # 平行性（允许同向或反向）：叉积=0
        #     eqs.append(txa * tyb - tya * txb)
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
                defaults = [0.0, 0.0, 1.0, 0.0, np.pi/2]  # cx, cy, r, theta1, theta2
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
        constraint_list.append({'type':'coincident','p1':i,'which1':1,'p2':(i+1)%len(geom_objects),'which2':0})
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

    print("\n" + "="*50 + "\n")

    # -----------------------------
    # 可视化
    # -----------------------------
    plt.figure(figsize=(15, 10))
    plt.title("Wheel Profile - CAD Constraint Solution")
    
    # Plot the wheel profile
    for obj in geom_objects:
        if type(obj).__name__ == 'Line':
            (x1,y1),(x2,y2) = obj.points()
            plt.plot([x1,x2],[y1,y2],'b-', linewidth=2)
        elif type(obj).__name__ == 'Arc':
            cx,cy,r,t1,t2 = obj.params
            ts = np.linspace(t1,t2,50)
            xs = cx + r*np.cos(ts)
            ys = cy + r*np.sin(ts)
            plt.plot(xs,ys,'r-', linewidth=2)
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
