import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, CheckButtons
import copy
from solver import Line, Arc, solve_geometry
import math

# =============================
# Helpers
# =============================
def deg2rad(x): return np.deg2rad(float(x))
def rad2deg(x): return np.rad2deg(float(x))

def near_display(p, q, ax, tol=8):
    tp = ax.transData.transform(p); tq = ax.transData.transform(q)
    return np.hypot(*(tp - tq)) <= tol

def snapshot_state(geom_objects, constraint_list):
    return (copy.deepcopy(geom_objects), copy.deepcopy(constraint_list))

# =============================
# Global model state
# =============================
geom_objects = []        # list[Line|Arc]
constraint_list = []     # list[dict]
undo_stack = []

state = {
    'mode': 'move',         # 'move'|'add_line'|'add_arc'|'connect'|'edit'
    'pending': None,
    'drag': None,
    'selected': None,
    'flip_ccw': True,       # arc preview (+sweep CCW)
    'seed_drag': True,      # True: 拖拽只改“初值”，不锁死 fixed_mask
}

defaults = {
    'line_len': 80.0,
    'arc_R': 40.0,
    'arc_sweep_deg': 60.0,
}

# =============================
# Matplotlib UI
# =============================
fig, ax = plt.subplots(figsize=(13, 8))
plt.subplots_adjust(left=0.30, right=0.98, top=0.96, bottom=0.08)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.3)

# Left control panel
ax_btn_load     = plt.axes([0.02, 0.92, 0.26, 0.06])
ax_btn_add_line = plt.axes([0.02, 0.84, 0.26, 0.06])
ax_btn_add_arc  = plt.axes([0.02, 0.76, 0.26, 0.06])
ax_btn_connect  = plt.axes([0.02, 0.68, 0.26, 0.06])
ax_btn_move     = plt.axes([0.02, 0.60, 0.26, 0.06])
ax_btn_solve    = plt.axes([0.02, 0.50, 0.26, 0.07])
ax_btn_undo     = plt.axes([0.02, 0.42, 0.26, 0.06])

ax_tb_len       = plt.axes([0.02, 0.32, 0.12, 0.05])
ax_tb_radius    = plt.axes([0.16, 0.32, 0.12, 0.05])
ax_tb_sweep     = plt.axes([0.02, 0.24, 0.26, 0.05])
ax_chk_arc_dir  = plt.axes([0.02, 0.17, 0.26, 0.06])
ax_chk_seeddrag = plt.axes([0.02, 0.10, 0.26, 0.06])

b_load      = Button(ax_btn_load,     'Load Wheel Template', color="#dddddd")
b_add_line  = Button(ax_btn_add_line, 'Add Line')
b_add_arc   = Button(ax_btn_add_arc,  'Add Arc')
b_connect   = Button(ax_btn_connect,  'Connect')
b_move      = Button(ax_btn_move,     'Move')
b_solve     = Button(ax_btn_solve,    'Solve (optimize)', color="#4477aa", hovercolor="#336699")
b_undo      = Button(ax_btn_undo,     'Undo')

tb_len    = TextBox(ax_tb_len,    'Len',    initial=str(defaults['line_len']))
tb_radius = TextBox(ax_tb_radius, 'R',      initial=str(defaults['arc_R']))
tb_sweep  = TextBox(ax_tb_sweep,  'Sweep°', initial=str(defaults['arc_sweep_deg']))
chk_dir   = CheckButtons(ax_chk_arc_dir,  ['CCW +sweep'], [state['flip_ccw']])
chk_seed  = CheckButtons(ax_chk_seeddrag, ['Seed-drag (do not lock)'], [state['seed_drag']])

# Right edit panel
ax_edit_title  = plt.axes([0.72, 0.86, 0.25, 0.05])
ax_edit_t1     = plt.axes([0.72, 0.79, 0.25, 0.05])
ax_edit_t2     = plt.axes([0.72, 0.72, 0.25, 0.05])
ax_edit_t3     = plt.axes([0.72, 0.65, 0.25, 0.05])
ax_btn_apply   = plt.axes([0.72, 0.57, 0.25, 0.06])

edit_title = TextBox(ax_edit_title, '', initial='(select a line/arc to edit)')
edit_1 = TextBox(ax_edit_t1, '', initial='')
edit_2 = TextBox(ax_edit_t2, '', initial='')
edit_3 = TextBox(ax_edit_t3, '', initial='')
btn_apply = Button(ax_btn_apply, 'Apply Changes')

for w in [edit_1, edit_2, edit_3]: w.set_val('')

# =============================
# Rendering + View freeze
# =============================
VIEW = {'xlim': (-50.0, 500.0), 'ylim': (-100.0, 260.0), 'freeze': True}

def draw(preview=None):
    cur_xlim, cur_ylim = ax.get_xlim(), ax.get_ylim()
    ax.clear(); ax.set_aspect('equal', adjustable='box'); ax.grid(True, alpha=0.3)

    # objects
    for i, obj in enumerate(geom_objects):
        selected = (state['selected'] == i)
        if isinstance(obj, Line):
            (x1,y1),(x2,y2) = obj.points()
            ax.plot([x1,x2],[y1,y2], 'c-' if selected else 'b-', lw=2)
            ax.plot([x1,x2],[y1,y2], 'bo', ms=6, alpha=0.85)
            ax.text((x1+x2)/2, (y1+y2)/2, f"L{i}", color='blue', fontsize=9)
        else:
            cx,cy,r,t1,t2 = obj.params
            tt = np.linspace(t1, t2, 100)
            xs = cx + r*np.cos(tt); ys = cy + r*np.sin(tt)
            ax.plot(xs, ys, 'm-' if selected else 'r-', lw=2)
            ax.plot(cx, cy, 'ro', ms=6, alpha=0.85)
            (sx,sy),(ex,ey) = obj.points()
            ax.plot([sx,ex],[sy,ey], 'rs', ms=5, alpha=0.85)
            ax.text(cx, cy, f"A{i}", color='red', fontsize=9)

    # preview
    if preview is not None:
        if preview['kind'] == 'line':
            (x1,y1),(x2,y2) = preview['p']
            ax.plot([x1,x2],[y1,y2], 'b--', lw=1.5); ax.plot([x1,x2],[y1,y2], 'bo', ms=5, alpha=0.7)
        else:
            cx,cy,r,t1,t2 = preview['p']
            tt = np.linspace(t1,t2,80)
            ax.plot(cx + r*np.cos(tt), cy + r*np.sin(tt), 'r--', lw=1.5); ax.plot(cx, cy, 'ro', ms=5, alpha=0.7)

    # view freeze
    if VIEW['freeze']:
        ax.set_xlim(VIEW['xlim'] if VIEW['xlim'] is not None else cur_xlim)
        ax.set_ylim(VIEW['ylim'] if VIEW['ylim'] is not None else cur_ylim)
    else:
        ax.set_xlim(cur_xlim); ax.set_ylim(cur_ylim)

    ax.set_title(f"Mode: {state['mode']} | Seed-drag: {state['seed_drag']} | Objects: {len(geom_objects)} | Constraints: {len(constraint_list)}")
    fig.canvas.draw_idle()

def set_mode(m): state.update(mode=m, pending=None, drag=None); draw()

# =============================
# Picking / connecting
# =============================
def pick_handle(x, y):
    best, bestd = None, 1e9
    for i, obj in enumerate(geom_objects):
        if isinstance(obj, Line):
            for idx, p in enumerate(obj.points()):
                if near_display(p,(x,y),ax):
                    d = np.hypot(p[0]-x, p[1]-y)
                    if d < bestd: best, bestd = {'obj':i,'kind':'line_p','idx':idx}, d
        else:
            cx, cy = obj.params[0], obj.params[1]
            if near_display((cx,cy),(x,y),ax):
                d = np.hypot(cx-x, cy-y)
                if d < bestd: best, bestd = {'obj':i,'kind':'arc_c','idx':None}, d
            (p0, p1) = obj.points()
            for idx, p in enumerate([p0, p1]):
                if near_display(p,(x,y),ax):
                    d = np.hypot(p[0]-x, p[1]-y)
                    if d < bestd: best, bestd = {'obj':i,'kind':'arc_ep','idx':idx}, d
    return best

def pick_object_body(x, y):
    best, bestd = None, 1e9
    for i, obj in enumerate(geom_objects):
        if isinstance(obj, Line):
            (x1,y1),(x2,y2) = obj.points()
            vx,vy = x2-x1, y2-y1; wx,wy = x-x1, y-y1
            L2 = vx*vx+vy*vy
            if L2 == 0: continue
            t = np.clip((vx*wx+vy*wy)/L2, 0, 1)
            px,py = x1+t*vx, y1+t*vy
            d = np.hypot(px-x,py-y)
            if d < bestd and near_display((px,py),(x,y),ax,8): best, bestd = i, d
        else:
            cx,cy,r,t1,t2 = obj.params
            tt = np.linspace(t1,t2,60)
            xs,ys = cx+r*np.cos(tt), cy+r*np.sin(tt)
            d = np.min(np.hypot(xs-x,ys-y))
            if d < bestd: best, bestd = i, d
    return best

def connect_endpoints(a, b):
    if a['kind'] == 'arc_c' or b['kind'] == 'arc_c': return
    p1_obj, p1_idx = a['obj'], a.get('idx',0)
    p2_obj, p2_idx = b['obj'], b.get('idx',0)
    constraint_list.append({'type':'coincident','p1':p1_obj,'which1':p1_idx,'p2':p2_obj,'which2':p2_idx})

    def add_line_arc(line_obj, arc_obj, arc_idx):
        key = 'tangent_at_arc_start_to_line' if arc_idx==0 else 'tangent_at_arc_end_to_line'
        constraint_list.append({'type':key,'line':line_obj,'arc':arc_obj})

    if isinstance(geom_objects[p1_obj], Line) and isinstance(geom_objects[p2_obj], Arc):
        add_line_arc(p1_obj, p2_obj, p2_idx)
    elif isinstance(geom_objects[p2_obj], Line) and isinstance(geom_objects[p1_obj], Arc):
        add_line_arc(p2_obj, p1_obj, p1_idx)
    elif isinstance(geom_objects[p1_obj], Arc) and isinstance(geom_objects[p2_obj], Arc):
        constraint_list.append({'type':'tangent_at_arc_to_arc','Aarc':p1_obj,'Barc':p2_obj,
                                'a_end':(p1_idx==1),'b_end':(p2_idx==1),'same_direction':True})

# =============================
# Selection + Edit
# =============================
def select_object(i):
    state['selected'] = i
    if i is None:
        edit_title.set_val('(select a line/arc to edit)')
        for w in [edit_1,edit_2,edit_3]: w.set_val('')
        draw(); return
    obj = geom_objects[i]
    if isinstance(obj, Line):
        (x1,y1),(x2,y2) = obj.points(); L = np.hypot(x2-x1,y2-y1)
        edit_title.set_val(f'Line L{i}')
        edit_1.label.set_text('x1,y1'); edit_1.set_val(f"{x1:.3f},{y1:.3f}")
        edit_2.label.set_text('x2,y2'); edit_2.set_val(f"{x2:.3f},{y2:.3f}")
        edit_3.label.set_text('Length'); edit_3.set_val(f"{L:.3f}")
    else:
        cx,cy,r,t1,t2 = obj.params
        edit_title.set_val(f'Arc A{i}')
        edit_1.label.set_text('cx,cy'); edit_1.set_val(f"{cx:.3f},{cy:.3f}")
        edit_2.label.set_text('R, sweep°'); edit_2.set_val(f"{r:.3f},{rad2deg(t2-t1):.3f}")
        edit_3.label.set_text('t1° (start)'); edit_3.set_val(f"{rad2deg(t1):.3f}")
    draw()

def apply_changes(event=None):
    i = state['selected']
    if i is None: return
    push_undo()
    obj = geom_objects[i]
    try:
        if isinstance(obj, Line):
            x1,y1 = map(float, edit_1.text.split(','))
            x2,y2 = map(float, edit_2.text.split(','))
            L = float(edit_3.text)
            obj.params[:] = [x1,y1,x2,y2]
            # 不锁定，作为初值
            obj.fixed_mask[:] = [False,False,False,False]
            constraint_list.append({'type':'length','line':i,'value':L})
        else:
            cx,cy = map(float, edit_1.text.split(','))
            r, sweep_deg = map(float, edit_2.text.split(','))
            t1 = deg2rad(edit_3.text); t2 = t1 + np.deg2rad(sweep_deg)
            obj.params[:] = [cx,cy,r,t1,t2]
            obj.fixed_mask[:] = [False,False,False,False,False]
            constraint_list.append({'type':'radius','arc':i,'value':r})
            constraint_list.append({'type':'arc_sweep_leq','arc':i,'max_deg':360.0})
    except Exception as e:
        print('[Apply error]', e)
    draw()

btn_apply.on_clicked(apply_changes)

# =============================
# Undo
# =============================
def push_undo(): undo_stack.append(snapshot_state(geom_objects, constraint_list))
def undo(event=None):
    if not undo_stack: return
    global geom_objects, constraint_list
    geom_objects, constraint_list = undo_stack.pop()
    if state['selected'] is not None and state['selected'] >= len(geom_objects):
        state['selected'] = None
    draw()
b_undo.on_clicked(undo)

# =============================
# Mouse / keyboard
# =============================
def on_click(event):
    if event.inaxes != ax: return
    x, y = event.xdata, event.ydata

    if state['mode'] == 'add_line':
        L = float(tb_len.text)
        if state['pending'] is None:
            state['pending'] = {'kind':'line','x1':x,'y1':y}
        else:
            x1,y1 = state['pending']['x1'], state['pending']['y1']
            vx,vy = x-x1, y-y1; d = np.hypot(vx,vy)
            vx,vy = (1.0,0.0) if d<1e-9 else (vx/d,vy/d)
            x2,y2 = x1 + L*vx, y1 + L*vy
            obj = Line(x1=x1,y1=y1,x2=x2,y2=y2)
            # 作为初值参与优化：不固定
            obj.fixed_mask[:] = [False,False,False,False]
            geom_objects.append(obj)
            constraint_list.append({'type':'length','line':len(geom_objects)-1,'value':L})
            push_undo(); state['pending']=None; set_mode('move')

    elif state['mode'] == 'add_arc':
        R = float(tb_radius.text); sweep = deg2rad(tb_sweep.text)
        if state['pending'] is None:
            state['pending'] = {'kind':'arc','cx':x,'cy':y}
        else:
            cx,cy = state['pending']['cx'], state['pending']['cy']
            ang = np.arctan2(y-cy, x-cx)
            sgn = +1.0 if state['flip_ccw'] else -1.0
            t1, t2 = ang, ang + sgn*sweep
            arc = Arc(cx=cx, cy=cy, r=R, theta1=t1, theta2=t2)
            arc.fixed_mask[:] = [False,False,False,False,False]
            geom_objects.append(arc)
            constraint_list.append({'type':'radius','arc':len(geom_objects)-1,'value':R})
            constraint_list.append({'type':'arc_sweep_leq','arc':len(geom_objects)-1,'max_deg':360.0})
            push_undo(); state['pending']=None; set_mode('move')

    elif state['mode'] == 'connect':
        h = pick_handle(x,y)
        if h is None: return
        if state['pending'] is None:
            state['pending'] = {'pick':h}
        else:
            a,b = state['pending']['pick'], h
            if a['obj']==b['obj'] and a['kind']==b['kind'] and a.get('idx')==b.get('idx'):
                state['pending']=None; return
            connect_endpoints(a,b); push_undo(); state['pending']=None; draw()

    elif state['mode'] == 'move':
        h = pick_handle(x,y)
        if h is not None:
            state['drag'] = h; select_object(h['obj']); return
        i = pick_object_body(x,y); select_object(i)

def on_release(event): state['drag']=None

def on_motion(event):
    if event.inaxes != ax: return
    x, y = event.xdata, event.ydata

    # previews
    if state['mode']=='add_line' and state['pending'] and state['pending'].get('kind')=='line':
        L = float(tb_len.text)
        x1,y1 = state['pending']['x1'], state['pending']['y1']
        vx,vy = x-x1, y-y1; d = np.hypot(vx,vy); vx,vy = (1,0) if d<1e-9 else (vx/d,vy/d)
        draw(preview={'kind':'line','p':((x1,y1),(x1+L*vx,y1+L*vy))}); return

    if state['mode']=='add_arc' and state['pending'] and state['pending'].get('kind')=='arc':
        R = float(tb_radius.text); sweep = deg2rad(tb_sweep.text)
        cx,cy = state['pending']['cx'], state['pending']['cy']
        ang = np.arctan2(y-cy, x-cx); sgn = +1.0 if state['flip_ccw'] else -1.0
        draw(preview={'kind':'arc','p':(cx,cy,R,ang,ang+sgn*sweep)}); return

    # dragging (move/seed)
    if state['drag'] is not None and state['mode']=='move':
        h = state['drag']; obj = geom_objects[h['obj']]
        lock = not state['seed_drag']  # lock when seed_drag is False
        if h['kind']=='line_p':
            if h['idx']==0:
                obj.params[0],obj.params[1] = x,y
                if lock: obj.fixed_mask[0]=obj.fixed_mask[1]=True
            else:
                obj.params[2],obj.params[3] = x,y
                if lock: obj.fixed_mask[2]=obj.fixed_mask[3]=True
        elif h['kind']=='arc_c':
            obj.params[0],obj.params[1] = x,y
            if lock: obj.fixed_mask[0]=obj.fixed_mask[1]=True
        elif h['kind']=='arc_ep':
            cx,cy,r,t1,t2 = obj.params
            ang = np.arctan2(y-cy, x-cx)
            if h['idx']==0: obj.params[3]=ang
            else: obj.params[4]=ang
            if lock: obj.fixed_mask[3]=obj.fixed_mask[4]=True
        draw()

def on_key(event):
    if event.key=='q': plt.close(fig)
    elif event.key=='r':
        VIEW['freeze']=False; ax.relim(); ax.autoscale_view()
        VIEW['xlim'], VIEW['ylim'] = ax.get_xlim(), ax.get_ylim()
        VIEW['freeze']=True; draw()
    elif event.key=='escape':
        state['pending']=None; state['drag']=None; draw()
    elif event.key=='f':
        state['flip_ccw'] = not state['flip_ccw']; chk_dir.set_active(0)

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('key_press_event', on_key)

# =============================
# Button handlers
# =============================
def on_add_line(event=None): set_mode('add_line')
def on_add_arc(event=None):  set_mode('add_arc')
def on_connect(event=None):  state['pending']=None; set_mode('connect')
def on_move(event=None):     set_mode('move')

def on_solve(event=None):
    if not geom_objects: return
    print(f"[Solve] objects={len(geom_objects)} constraints={len(constraint_list)}")
    res = solve_geometry(geom_objects, constraint_list, verbose=True)
    if not res.success: print("WARNING: solver did not converge.")
    draw()

b_add_line.on_clicked(on_add_line)
b_add_arc.on_clicked(on_add_arc)
b_connect.on_clicked(on_connect)
b_move.on_clicked(on_move)
b_solve.on_clicked(on_solve)

chk_dir.on_clicked(lambda _: setattr(state, 'flip_ccw', not state['flip_ccw']))
chk_seed.on_clicked(lambda _: setattr(state, 'seed_drag', not state['seed_drag']))

def update_defaults_len(text):  defaults['line_len']=float(text) if text else defaults['line_len']
def update_defaults_R(text):    defaults['arc_R']=float(text) if text else defaults['arc_R']
def update_defaults_sweep(text): defaults['arc_sweep_deg']=float(text) if text else defaults['arc_sweep_deg']
tb_len.on_submit(update_defaults_len); tb_radius.on_submit(update_defaults_R); tb_sweep.on_submit(update_defaults_sweep)

# =============================
# Wheel template (your second script) — same geometry & constraints
# =============================
def load_wheel_template(event=None):
    global geom_objects, constraint_list
    push_undo()
    geom_objects = []; constraint_list = []

    # ---- Lines ----
    L_left_vert = Line(x1=93.0, y1=0.0, x2=93.0, y2=178.0);                L_left_vert.fixed_mask[:] = True
    L_left_top_h = Line(x1=93.0, y1=178.0, x2=132.0, y2=178.0);             L_left_top_h.fixed_mask[:] = True
    L_left_bottom_h = Line(x1=93.0, y1=0.0, x2=132.0, y2=0.0);              L_left_bottom_h.fixed_mask[:] = True
    L_left_top_xie = Line(x1=None, y1=None, x2=None, y2=None)               # unknowns as seeds (not fixed)
    L_left_bottom_xie = Line(x1=None, y1=None, x2=None, y2=None)
    L_right_top_xie = Line(x1=None, y1=None, x2=None, y2=None)
    L_right_bottom_xie = Line(x1=None, y1=None, x2=None, y2=None)
    L_right_top_h = Line(x1=None, y1=68+135, x2=None, y2=68+135);           L_right_top_h.fixed_mask[1]=L_right_top_h.fixed_mask[3]=True
    L_right_bottom_h = Line(x1=None, y1=68.0,   x2=None, y2=68.0);          L_right_bottom_h.fixed_mask[1]=L_right_bottom_h.fixed_mask[3]=True
    L_right_vert = Line(x1=400.0, y1=68.0, x2=400.0, y2=68.0+135);          L_right_vert.fixed_mask[:] = True

    for L in [L_left_vert,L_left_top_h,L_left_bottom_h,L_left_top_xie,L_left_bottom_xie,
              L_right_top_xie,L_right_bottom_xie,L_right_top_h,L_right_bottom_h,L_right_vert]:
        geom_objects.append(L)

    # Index map to stay identical to your script
    # 0..8 as above, 21 is index 9 in current list → keep original indices by padding list:
    # We want: 0..8 are lines; then arcs 9..20; then line 21 (right vert)
    # Our order already matches: 0..8 lines, then we will append arcs, then append right_vert already appended as index 9,
    # So reorder to match original indices:
    # Actually simpler: just remember new indices:
    # 0:L_left_vert,1:L_left_top_h,2:L_left_bottom_h,3:L_left_top_xie,4:L_left_bottom_xie,
    # 5:L_right_top_xie,6:L_right_bottom_xie,7:L_right_top_h,8:L_right_bottom_h,9:L_right_vert

    # ---- Arcs ----
    def A(cx,cy,r,t1,t2,fixed=(False,False,False,False,False)):
        a = Arc(cx=cx, cy=cy, r=r, theta1=t1, theta2=t2); a.fixed_mask[:] = list(fixed); return a

    A_left_top_cham    = A(None,None,5.0, math.radians(12.0),        math.radians(90.0))
    A_left_bottom_cham = A(None,None,5.0, math.radians(270.0),       math.radians(360-12.0))
    l_top_hu1          = A(None,None,40.0,  None, None)
    l_top_hu2          = A(None,None,130.0, None, None)
    l_top_hu3          = A(None,None,195.0, None, None)
    l_top_hu4          = A(None,None,40.0,  None, None)
    l_bottom_hu1       = A(None,-3.0,63.0,  None, None, fixed=(False,True,False,False,False))
    l_bottom_hu2       = A(None,None,170.0, None, None)
    l_bottom_hu3       = A(None,None,176.0, None, None)
    l_bottom_hu4       = A(None,None,40.0,  None, None)
    A_right_top_cham   = A(None,None,3.0,   math.radians(90.0),      math.radians(180.0-15.0))
    A_right_bot_cham   = A(None,None,3.0,   math.radians(180.0+15.0),math.radians(270.0))

    arcs = [A_left_top_cham,A_left_bottom_cham,l_top_hu1,l_top_hu2,l_top_hu3,l_top_hu4,
            l_bottom_hu1,l_bottom_hu2,l_bottom_hu3,l_bottom_hu4,A_right_top_cham,A_right_bot_cham]
    geom_objects[9:9] = arcs  # insert after first 9 lines so indices与原版一致

    # 索引对齐（完全复用你原来的约束）
    # lines: 0..8 (与原脚本一致), right_vert 是 21→此处 index 21 仍成立
    # arcs: 9..20

    # ---- Constraints (完全照你的第二段) ----
    constraint_list.clear()
    # 拓扑
    constraint_list += [
        {'type':'coincident','p1':0,'which1':1,'p2':1,'which2':0},
        {'type':'coincident','p1':0,'which1':0,'p2':2,'which2':0},
        {'type':'coincident','p1':1,'which1':1,'p2':9,'which2':1},
        {'type':'coincident','p1':9,'which1':0,'p2':3,'which2':0},
        {'type':'coincident','p1':3,'which1':1,'p2':11,'which2':0},
        {'type':'coincident','p1':11,'which1':1,'p2':12,'which2':0},
        {'type':'coincident','p1':12,'which1':1,'p2':13,'which2':1},
        {'type':'coincident','p1':13,'which1':0,'p2':14,'which2':0},
        {'type':'coincident','p1':14,'which1':1,'p2':5,'which2':0},
        {'type':'coincident','p1':5, 'which1':1,'p2':19,'which2':1},
        {'type':'coincident','p1':19,'which1':0,'p2':7,'which2':0},
        {'type':'coincident','p1':7, 'which1':1,'p2':21,'which2':1},
        {'type':'coincident','p1':0,'which1':0,'p2':2,'which2':0},
        {'type':'coincident','p1':2,'which1':1,'p2':10,'which2':0},
        {'type':'coincident','p1':10,'which1':1,'p2':4,'which2':0},
        {'type':'coincident','p1':4,'which1':1,'p2':15,'which2':1},
        {'type':'coincident','p1':15,'which1':0,'p2':16,'which2':0},
        {'type':'coincident','p1':16,'which1':1,'p2':17,'which2':1},
        {'type':'coincident','p1':17,'which1':0,'p2':18,'which2':1},
        {'type':'coincident','p1':18,'which1':0,'p2':6, 'which2':0},
        {'type':'coincident','p1':6, 'which1':1,'p2':20,'which2':0},
        {'type':'coincident','p1':20,'which1':1,'p2':8, 'which2':0},
        {'type':'coincident','p1':8, 'which1':1,'p2':21,'which2':0},
    ]
    # 端点相切（线↔弧）
    constraint_list += [
        {'type':'tangent_at_arc_end_to_line', 'line':1, 'arc':9},
        {'type':'tangent_at_arc_start_to_line','line':3, 'arc':9},
        {'type':'tangent_at_arc_start_to_line','line':3, 'arc':11},
        {'type':'tangent_at_arc_start_to_line','line':2, 'arc':10},
        {'type':'tangent_at_arc_end_to_line',  'line':4, 'arc':10},
        {'type':'tangent_at_arc_end_to_line',  'line':4, 'arc':15},
        {'type':'tangent_at_arc_end_to_line',  'line':5, 'arc':14},
        {'type':'tangent_at_arc_end_to_line',  'line':5, 'arc':19},
        {'type':'tangent_at_arc_start_to_line','line':6, 'arc':18},
        {'type':'tangent_at_arc_start_to_line','line':6, 'arc':20},
        {'type':'tangent_at_arc_start_to_line','line':7, 'arc':19},
        {'type':'tangent_at_arc_end_to_line',  'line':8, 'arc':20},
    ]
    # 线段方向（上下）
    constraint_list += [
        {'type':'line_second_side','line':3,'side':'below'},
        {'type':'line_second_side','line':4,'side':'above'},
        {'type':'line_second_side','line':5,'side':'above'},
        {'type':'line_second_side','line':6,'side':'below'},
    ]
    # 弧↔弧端点相切
    constraint_list += [
        {'type':'tangent_at_arc_to_arc','Aarc':11,'Barc':12,'a_end':True,'b_end':False,'same_direction':True},
        {'type':'tangent_at_arc_to_arc','Aarc':12,'Barc':13,'a_end':True,'b_end':True,'same_direction':True},
        {'type':'tangent_at_arc_to_arc','Aarc':13,'Barc':14,'a_end':False,'b_end':False,'same_direction':True},
        {'type':'tangent_at_arc_to_arc','Aarc':15,'Barc':16,'a_end':False,'b_end':False,'same_direction':True},
        {'type':'tangent_at_arc_to_arc','Aarc':16,'Barc':17,'a_end':True,'b_end':True,'same_direction':True},
        {'type':'tangent_at_arc_to_arc','Aarc':17,'Barc':18,'a_end':False,'b_end':True,'same_direction':True},
    ]
    # 辅助“上下/左右”关系
    constraint_list += [
        {'type':'arc_side','arc':11,'side':'lower'},
        {'type':'arc_side','arc':12,'side':'lower'},
        {'type':'arc_side','arc':13,'side':'upper'},
        {'type':'arc_side','arc':14,'side':'lower'},
        {'type':'arc_side','arc':15,'side':'upper'},
        {'type':'arc_side','arc':16,'side':'lower'},
        {'type':'arc_side','arc':17,'side':'upper'},
        {'type':'arc_side','arc':18,'side':'upper'},
        {'type':'arc_end_x_side','arc':11,'side':'right'},
        {'type':'arc_end_x_side','arc':12,'side':'right'},
        {'type':'arc_end_x_side','arc':13,'side':'left'},
        {'type':'arc_end_x_side','arc':14,'side':'right'},
        {'type':'arc_end_x_side','arc':15,'side':'left'},
        {'type':'arc_end_x_side','arc':16,'side':'right'},
        {'type':'arc_end_x_side','arc':17,'side':'left'},
        {'type':'arc_end_x_side','arc':18,'side':'left'},
        {'type':'arc_end_y_side','arc':11,'side':'below','margin':1.0},
        {'type':'arc_end_y_side','arc':12,'side':'above','margin':1.0},
        {'type':'arc_end_y_side','arc':13,'side':'below','margin':1.0},
        {'type':'arc_end_y_side','arc':14,'side':'above','margin':1.0},
        {'type':'arc_end_y_side','arc':15,'side':'below','margin':1.0},
        {'type':'arc_end_y_side','arc':16,'side':'above','margin':1.0},
        {'type':'arc_end_y_side','arc':17,'side':'below','margin':1.0},
        {'type':'arc_end_y_side','arc':18,'side':'above','margin':1.0},
    ]
    # 尺寸
    constraint_list.append({'type':'point_distance_x','p1':0,'which1':1,'p2':1,'which2':1,'value':132.0})
    constraint_list.append({'type':'length','line':8,'value':24.0})
    constraint_list.append({'type':'length','line':7,'value':24.0})
    # 半径硬约束
    for arc_idx,R in [(9,5.0),(10,5.0),(11,40.0),(12,130.0),(13,195.0),(14,40.0),
                      (15,63.0),(16,170.0),(17,176.0),(18,40.0),(19,3.0),(20,3.0)]:
        constraint_list.append({'type':'radius','arc':arc_idx,'value':R})
    # 扫角≤180°
    for arc_idx in [9,10,11,12,13,14,15,16,17,18,19,20]:
        constraint_list.append({'type':'arc_sweep_leq','arc':arc_idx,'max_deg':180.0})

    # 给一些合理 theta 初值（非硬约束，仅利于收敛）
    # 上链
    for idx, t1,t2 in [(11,20,50),(12,50,110),(13,110,160),(14,160,185)]:
        a = geom_objects[idx]
        if np.isnan(a.params[3]): a.params[3]=math.radians(t1)
        if np.isnan(a.params[4]): a.params[4]=math.radians(t2)
    # 下链
    for idx, t1,t2 in [(15,-20,10),(16,10,40),(17,40,80),(18,80,100)]:
        a = geom_objects[idx]
        if np.isnan(a.params[3]): a.params[3]=math.radians(t1)
        if np.isnan(a.params[4]): a.params[4]=math.radians(t2)
    # 右侧小圆角角度
    a19 = geom_objects[19]
    if np.isnan(a19.params[3]): a19.params[3]=math.radians(90.0)
    if np.isnan(a19.params[4]): a19.params[4]=math.radians(180.0-15.0)
    a20 = geom_objects[20]
    if np.isnan(a20.params[3]): a20.params[3]=math.radians(180.0+15.0)
    if np.isnan(a20.params[4]): a20.params[4]=math.radians(270.0)

    # 视图重置为模板范围
    VIEW['xlim']=(-20, 460); VIEW['ylim']=(-40, 220)
    draw()

b_load.on_clicked(load_wheel_template)

# =============================
# Init
# =============================
select_object(None)
draw()
plt.show()
