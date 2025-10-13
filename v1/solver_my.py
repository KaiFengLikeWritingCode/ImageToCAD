# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# ---------------- Basic geometry ----------------
class Line:
    def __init__(self, x1=None, y1=None, x2=None, y2=None):
        self.params = np.array([x1, y1, x2, y2], dtype=float)
        self.fixed_mask = np.array([x1 is not None, y1 is not None,
                                    x2 is not None, y2 is not None], dtype=bool)

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
                                    theta1 is not None, theta2 is not None], dtype=bool)

    def points(self):
        cx, cy, r, t1, t2 = self.params
        x1 = cx + r*np.cos(t1); y1 = cy + r*np.sin(t1)
        x2 = cx + r*np.cos(t2); y2 = cy + r*np.sin(t2)
        return (x1, y1), (x2, y2)

    def get_optimizable_params(self):
        return self.params[~self.fixed_mask]

    def set_optimizable_params(self, values):
        self.params[~self.fixed_mask] = values


# ---------------- Helpers ----------------
def _line_ABC(line: Line):
    (x1, y1), (x2, y2) = line.points()
    A = y1 - y2
    B = x2 - x1
    C = x1*y2 - x2*y1
    return A, B, C

def _point_line_signed_distance(px, py, line: Line):
    A, B, C = _line_ABC(line)
    denom = np.sqrt(A*A + B*B) + 1e-12
    return (A * px + B * py + C) / denom

def _dot(u, v):  return u[0]*v[0] + u[1]*v[1]
def _cross(u, v): return u[0]*v[1] - u[1]*v[0]

# ---------------- Constraints ----------------
def constraints(vars, geom_objects, constraint_list):
    eqs = []
    idx = 0
    for obj in geom_objects:
        n = len(obj.get_optimizable_params())
        if n:
            obj.set_optimizable_params(vars[idx:idx+n]); idx += n

    for c in constraint_list:
        t = c['type']

        if t == 'length':
            line = geom_objects[c['line']]
            (x1,y1),(x2,y2) = line.points()
            eqs.append(((x2-x1)**2 + (y2-y1)**2) - c['value']**2)

        elif t == 'radius':
            arc = geom_objects[c['arc']]
            eqs.append(arc.params[2] - c['value'])

        elif t == 'point_distance_x':
            p1 = geom_objects[c['p1']].points()[c['which1']]
            p2 = geom_objects[c['p2']].points()[c['which2']]
            eqs.append((p2[0] - p1[0]) - c['value'])

        elif t == 'point_distance_y':
            p1 = geom_objects[c['p1']].points()[c['which1']]
            p2 = geom_objects[c['p2']].points()[c['which2']]
            eqs.append((p2[1] - p1[1]) - c['value'])

        elif t == 'center_distance_x':
            arc = geom_objects[c['arc']]
            eqs.append(arc.params[0] - c['value'])

        elif t == 'center_distance_y':
            arc = geom_objects[c['arc']]
            eqs.append(arc.params[1] - c['value'])

        elif t == 'coincident':
            p1 = geom_objects[c['p1']].points()[c['which1']]
            p2 = geom_objects[c['p2']].points()[c['which2']]
            eqs.extend([p1[0]-p2[0], p1[1]-p2[1]])

        # legacy (weak) tangency, kept for compatibility
        elif t == 'tangent_line_arc':
            line = geom_objects[c['line']]
            arc  = geom_objects[c['arc']]
            cx, cy, r, _, _ = arc.params
            eqs.append(_point_line_signed_distance(cx, cy, line) - r)

        # strong: line ↔ arc tangency at arc end (0:start, 1:end)
        elif t == 'tangent_line_arc_at':
            line = geom_objects[c['line']]
            arc  = geom_objects[c['arc']]
            end  = c.get('arc_end', 1)
            (px, py) = arc.points()[end]
            cx, cy, _, _, _ = arc.params
            (x1,y1),(x2,y2) = line.points()
            dx, dy = (x2-x1, y2-y1)
            rx, ry = (px-cx, py-cy)
            eqs.append(_point_line_signed_distance(px, py, line))   # point on line
            eqs.append(_dot((rx, ry), (dx, dy)))                    # radius ⟂ line

        # strong: arc ↔ arc tangency at their ends
        elif t == 'tangent_arc_arc_at':
            a1 = geom_objects[c['arc1']]
            a2 = geom_objects[c['arc2']]
            e1 = c.get('end1', 1)
            e2 = c.get('end2', 0)
            p1 = a1.points()[e1]; p2 = a2.points()[e2]
            cx1, cy1, _, _, _ = a1.params
            cx2, cy2, _, _, _ = a2.params
            r1 = (p1[0]-cx1, p1[1]-cy1)
            r2 = (p2[0]-cx2, p2[1]-cy2)
            eqs.extend([p1[0]-p2[0], p1[1]-p2[1]])   # same contact point
            eqs.append(_cross(r1, r2))               # radii are colinear

    return np.array(eqs, dtype=float)

# ---------------- Init / solve / plot ----------------
def get_initial_params(geom_objects):
    params = []
    for obj in geom_objects:
        opt = obj.get_optimizable_params()
        if len(opt) == 0: continue
        if type(obj).__name__ == 'Line':
            defaults = [0.0, 0.0, 50.0, 0.0]
        else:
            defaults = [0.0, 0.0, 50.0, 0.0, np.pi/2]
        full = obj.params; mask = ~obj.fixed_mask; j = 0
        for m in range(len(full)):
            if mask[m]:
                params.append(defaults[j] if np.isnan(full[m]) else full[m]); j += 1
    return np.array(params, dtype=float)

def solve_geometry(geom_objects, constraint_list, verbose=True):
    if verbose:
        print("Initial objects:")
        for i, obj in enumerate(geom_objects):
            print(i, type(obj).__name__, obj.params, "fixed=", obj.fixed_mask)
    x0 = get_initial_params(geom_objects)
    if verbose: print("x0:", x0)

    res = least_squares(
        constraints, x0,
        args=(geom_objects, constraint_list),
        method='trf', loss='soft_l1', f_scale=1.0,
        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=4000
    )
    if verbose:
        print("\nSuccess:", res.success, "status:", res.status)
        print("Cost:", res.cost, "nfev:", res.nfev, "message:", res.message)
        print("Solved objects:")
        for i, obj in enumerate(geom_objects):
            print(i, type(obj).__name__, obj.params)
    return res

def plot_solution(geom_objects, title="Wheel Profile - Solved"):
    plt.figure(figsize=(7,7))
    ax = plt.gca()
    for obj in geom_objects:
        if type(obj).__name__ == 'Line':
            (x1,y1),(x2,y2) = obj.points()
            plt.plot([x1,x2],[y1,y2],'b-',lw=2)
        else:
            cx,cy,r,t1,t2 = obj.params
            ts = np.linspace(t1,t2,200)
            xs = cx + r*np.cos(ts); ys = cy + r*np.sin(ts)
            plt.plot(xs,ys,'r-',lw=2)
            plt.plot([cx],[cy],'r.',ms=4,alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3); plt.title(title)
    plt.xlabel("X (mm)"); plt.ylabel("Y (mm)")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    from wheelobject_my2 import geom_objects, constraint_list
    solve_geometry(geom_objects, constraint_list, verbose=True)
    plot_solution(geom_objects)
