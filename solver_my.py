import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# ============== 基本几何 ==============
class Line:
    def __init__(self, x1=None, y1=None, x2=None, y2=None):
        self.params = np.array([x1, y1, x2, y2], dtype=float)
        self.fixed_mask = np.array([x1 is not None, y1 is not None, x2 is not None, y2 is not None])
    def points(self):
        return self.params[:2], self.params[2:]
    def get_optimizable_params(self):
        return self.params[~self.fixed_mask]
    def set_optimizable_params(self, values):
        self.params[~self.fixed_mask] = values

class Arc:
    def __init__(self, cx=None, cy=None, r=None, theta1=None, theta2=None):
        self.params = np.array([cx, cy, r, theta1, theta2], dtype=float)
        self.fixed_mask = np.array([cx is not None, cy is not None, r is not None,
                                    theta1 is not None, theta2 is not None])
    def points(self):
        cx, cy, r, t1, t2 = self.params
        x1 = cx + r*np.cos(t1); y1 = cy + r*np.sin(t1)
        x2 = cx + r*np.cos(t2); y2 = cy + r*np.sin(t2)
        return (x1,y1),(x2,y2)
    def get_optimizable_params(self):
        return self.params[~self.fixed_mask]
    def set_optimizable_params(self, values):
        self.params[~self.fixed_mask] = values

# ============== 约束 ==============
def _unit(v):
    n = np.linalg.norm(v) + 1e-12
    return v / n

def constraints(vars, geom_objects, constraint_list):
    eqs = []
    idx = 0
    # 回填变量
    for obj in geom_objects:
        k = len(obj.get_optimizable_params())
        if k:
            obj.set_optimizable_params(vars[idx:idx+k])
            idx += k

    for c in constraint_list:
        t = c['type']

        if t == 'length':
            line = geom_objects[c['line']]
            (x1,y1),(x2,y2) = line.points()
            eqs.append((x2-x1)**2 + (y2-y1)**2 - c['value']**2)

        elif t == 'radius':
            arc = geom_objects[c['arc']]
            eqs.append(arc.params[2] - c['value'])

        elif t == 'point_distance_x':
            p1 = geom_objects[c['p1']].points()[c['which1']]
            p2 = geom_objects[c['p2']].points()[c['which2']]
            eqs.append(p2[0] - p1[0] - c['value'])

        elif t == 'point_distance_y':
            p1 = geom_objects[c['p1']].points()[c['which1']]
            p2 = geom_objects[c['p2']].points()[c['which2']]
            eqs.append(p2[1] - p1[1] - c['value'])

        elif t == 'horizontal':
            (x1,y1),(x2,y2) = geom_objects[c['line']].points()
            eqs.append(y2 - y1)

        elif t == 'vertical':
            (x1,y1),(x2,y2) = geom_objects[c['line']].points()
            eqs.append(x2 - x1)

        elif t == 'angle_deg':
            (x1,y1),(x2,y2) = geom_objects[c['line']].points()
            ang = np.arctan2(y2-y1, x2-x1)
            eqs.append(ang - np.deg2rad(c['value']))

        elif t == 'coincident':
            p1 = geom_objects[c['p1']].points()[c['which1']]
            p2 = geom_objects[c['p2']].points()[c['which2']]
            eqs.append(p1[0]-p2[0]); eqs.append(p1[1]-p2[1])

        # —— 关键：在连接端点处相切（G1） ——
        elif t == 'tangent_line_arc_at':
            line = geom_objects[c['line']]
            arc  = geom_objects[c['arc']]
            which = c['arc_end']  # 0->theta1, 1->theta2
            theta = arc.params[3 + which]
            (x1,y1),(x2,y2) = line.points()
            t_line = _unit(np.array([x2-x1, y2-y1]))
            t_arc  = np.array([-np.sin(theta), np.cos(theta)])
            eqs.append(t_line[0] - t_arc[0])
            eqs.append(t_line[1] - t_arc[1])

        elif t == 'tangent_arc_arc_at':
            arc1 = geom_objects[c['arc1']]
            arc2 = geom_objects[c['arc2']]
            th1  = arc1.params[3 + c['which1']]
            th2  = arc2.params[3 + c['which2']]
            t1 = np.array([-np.sin(th1), np.cos(th1)])
            t2 = np.array([-np.sin(th2), np.cos(th2)])
            eqs.append(t1[0] - t2[0]); eqs.append(t1[1] - t2[1])

    return np.array(eqs, dtype=float)

# ============== 初值 + 边界 ==============
def get_initial_params(geom_objects):
    params = []
    for obj in geom_objects:
        tn = type(obj).__name__
        defaults = [0.0, 0.0, 120.0, 0.0] if tn=='Line' else [0.0, 0.0, 50.0, 0.0, np.pi/2]
        if len(obj.get_optimizable_params()):
            picks = []
            for k, fixed in enumerate(obj.fixed_mask):
                if not fixed:
                    v = obj.params[k]
                    picks.append(defaults[k] if np.isnan(v) else v)
            params.extend(picks)
    return np.array(params, dtype=float)

def get_bounds(geom_objects):
    lb = []; ub = []
    for obj in geom_objects:
        tn = type(obj).__name__
        if tn=='Line':
            lbi = [-np.inf]*4; ubi = [np.inf]*4
        else:
            lbi = [-np.inf, -np.inf, 1e-3, -np.pi, -np.pi]
            ubi = [ np.inf,  np.inf, 1e6,   2*np.pi, 2*np.pi]
        for k, fixed in enumerate(obj.fixed_mask):
            if not fixed:
                lb.append(lbi[k]); ub.append(ubi[k])
    return np.array(lb,float), np.array(ub,float)

def solve_geometry(geom_objects, constraint_list, verbose=True):
    x0 = get_initial_params(geom_objects)
    lb, ub = get_bounds(geom_objects)
    res = least_squares(
        constraints, x0,
        args=(geom_objects, constraint_list),
        bounds=(lb, ub),
        method='trf',
        loss='soft_l1', f_scale=1.0,
        x_scale='jac',
        max_nfev=4000,
        verbose=2 if verbose else 0
    )
    return res

def plot_profile(objs):
    plt.figure(figsize=(12,6))
    for o in objs:
        if type(o).__name__=='Line':
            (x1,y1),(x2,y2) = o.points()
            plt.plot([x1,x2],[y1,y2],'b-',lw=2)
        else:
            cx,cy,r,t1,t2 = o.params
            ts = np.linspace(t1,t2,120)
            xs = cx + r*np.cos(ts); ys = cy + r*np.sin(ts)
            plt.plot(xs,ys,'r-',lw=2); plt.plot(cx,cy,'ro',ms=2)
    plt.axis('equal'); plt.grid(True,alpha=.3); plt.tight_layout(); plt.show()

if __name__ == '__main__':
    from wheelobject_my2 import geom_objects, constraint_list
    res = solve_geometry(geom_objects, constraint_list, verbose=True)
    plot_profile(geom_objects)
