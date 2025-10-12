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

    #
    def set_optimizable_params(self, values):
        """Set the optimizable parameters with solved values"""
        self.params[~self.fixed_mask] = values

class Arc:
    def __init__(self, cx=None, cy=None, r=None, theta1=None, theta2=None):
        # Store all parameters, None means unknown/optimizable
        self.params = np.array([cx, cy, r, theta1, theta2], dtype=float)
        self.fixed_mask = np.array([cx is not None, cy is not None, r is not None, 
                                   theta1 is not None, theta2 is not None])

    # 由圆心半径角度求两端点
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
def constraints(vars, geom_objects, constraint_list):
    eqs = []
    idx = 0
    # 将 vars 拆分到各个对象的可优化参数
    for obj in geom_objects:
        # 这里的 obj.params 会随着 vars 的变化而变化
        obj.set_optimizable_params(vars[idx:idx+len(obj.get_optimizable_params())])
        idx += len(obj.get_optimizable_params())
    
    # 遍历约束
    for c in constraint_list:
        type_ = c['type']
        if type_ == 'length':
            line = geom_objects[c['line']]
            (x1,y1),(x2,y2) = line.points()
            eqs.append((x2-x1)**2 + (y2-y1)**2 - c['value']**2)
        elif type_ == 'radius':
            arc = geom_objects[c['arc']]
            # 仅约束半径值，不约束圆心/位置/角度。
            eqs.append(arc.params[2] - c['value'])
        elif type_ == 'point_distance_x':
            # Arc.points() 返回由 (cx,cy,r,θ1/θ2) 计算出的两端点，而不是圆心。所以不要用它表达“圆心高度/横坐标”，否则语义错（应单独做“圆心坐标”约束。
            p1 = geom_objects[c['p1']].points()[c['which1']]
            p2 = geom_objects[c['p2']].points()[c['which2']]
            eqs.append(p2[0] - p1[0] - c['value'])
        elif type_ == 'point_distance_y':
            p1 = geom_objects[c['p1']].points()[c['which1']]
            p2 = geom_objects[c['p2']].points()[c['which2']]
            eqs.append(p2[1] - p1[1] - c['value'])
        # 直线–圆弧相切
        # elif type_ == 'tangent_line_arc':
        #     line = geom_objects[c['line']]
        #     arc = geom_objects[c['arc']]
        #     # 点到直线距离 = r
        #     x1,y1 = arc.params[0], arc.params[1]
        #     r = arc.params[2]
        #     lx1,ly1 = line.points()[0]
        #     lx2,ly2 = line.points()[1]
        #     A = ly1 - ly2
        #     B = lx2 - lx1
        #     C = lx1*ly2 - lx2*ly1
        #     dist = np.abs(A*x1 + B*y1 + C)/np.sqrt(A**2 + B**2)
        #     eqs.append(dist - r)
        elif type_ == 'tangent_line_arc':
            line = geom_objects[c['line']]
            arc = geom_objects[c['arc']]
            cx, cy, r = arc.params[0], arc.params[1], arc.params[2]
            (lx1, ly1), (lx2, ly2) = line.points()
            A = ly1 - ly2
            B = lx2 - lx1
            C = lx1 * ly2 - lx2 * ly1
            num = A * cx + B * cy + C
            den = A * A + B * B
            eqs.append((num * num) / (den + 1e-12) - r * r)
        elif type_ == 'coincident':
            p1 = geom_objects[c['p1']].points()[c['which1']]
            p2 = geom_objects[c['p2']].points()[c['which2']]
            eqs.append(p1[0]-p2[0])
            eqs.append(p1[1]-p2[1])
    return eqs


'''
把每个几何对象（Line/Arc）里未固定的参数收集成一个一维向量 x0，作为 least_squares 的初始值

'''
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
                # 默认是一条水平线 (0,0) → (1,0)；
                defaults = [0.0, 0.0, 1.0, 0.0]  # x1, y1, x2, y2
            elif type_name == 'Arc':
                # 默认是四分之一圆（半径 1，t1=0, t2=π/2），圆心在原点。
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
    # 优化器在每次评估时等价于调用：constraints(x, geom_objects, constraint_list)
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
# def generate_coincident_constraint_list(geom_objects):
#     constraint_list = []
#     for i in range(len(geom_objects)):
#         constraint_list.append({'type':'coincident','p1':i,'which1':1,'p2':(i+1)%len(geom_objects),'which2':0})
#     return constraint_list



if __name__ == "__main__":
    # Example 1: Line with only start point fixed, Arc with only radius fixed
    print("Example 1: 车轮生成示例")
    # extract the objects and constrainst from wheel.py
    # from simpleobject import geom_objects, constraint_list
    from wheelobject_my import geom_objects, constraint_list
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
