import math
from solver import Line, Arc

geom_objects = []
constraint_list = []

# ========== 左基准 ==========
L_left_vert = Line(x1=93.0, y1=0.0, x2=93.0, y2=178.0); geom_objects.append(L_left_vert)  # 0
L_left_top  = Line(x1=93.0, y1=178.0, x2=None, y2=178.0); geom_objects.append(L_left_top)  # 1
L_left_bot  = Line(x1=93.0, y1=0.0,   x2=None, y2=0.0);   geom_objects.append(L_left_bot)  # 2
L_left_x1   = Line(x1=None, y1=None, x2=None, y2=None);   geom_objects.append(L_left_x1)   # 3 (180-12)
L_left_x2   = Line(x1=None, y1=None, x2=None, y2=None);   geom_objects.append(L_left_x2)   # 4 (12)

# ========== 中部链 ==========
A_u40  = Arc(cx=None, cy=None, r=40.0,  theta1=None, theta2=None);  geom_objects.append(A_u40)  # 5
A_u130 = Arc(cx=None, cy=None, r=130.0, theta1=None, theta2=None);  geom_objects.append(A_u130) # 6
A_u195 = Arc(cx=None, cy=None, r=195.0, theta1=None, theta2=None);  geom_objects.append(A_u195) # 7

A_d63  = Arc(cx=None, cy=None, r=63.0,  theta1=None, theta2=None);  geom_objects.append(A_d63)  # 8
A_d170 = Arc(cx=None, cy=None, r=170.0, theta1=None, theta2=None);  geom_objects.append(A_d170) # 9

# ========== 右端 ==========
L_right_x1  = Line(x1=None, y1=None, x2=None, y2=None);   geom_objects.append(L_right_x1)  # 10 (180-15)
L_right_x2  = Line(x1=None, y1=None, x2=None, y2=None);   geom_objects.append(L_right_x2)  # 11 (-75)
L_right_top = Line(x1=None, y1=135.0, x2=None, y2=135.0); geom_objects.append(L_right_top) # 12
L_right_bot = Line(x1=None, y1=70.0,  x2=None, y2=70.0);  geom_objects.append(L_right_bot) # 13

# ----------------- 拓扑（端点重合） -----------------
# 左：竖 → 顶/底水平
constraint_list += [
    {'type':'coincident','p1':0,'which1':1,'p2':1,'which2':0},
    {'type':'coincident','p1':0,'which1':0,'p2':2,'which2':0},

    # 顶/底水平 → 左两条12°斜边
    {'type':'coincident','p1':1,'which1':1,'p2':3,'which2':0},
    {'type':'coincident','p1':2,'which1':1,'p2':4,'which2':0},

    # 左斜 → 上/下首段弧
    {'type':'coincident','p1':3,'which1':1,'p2':5,'which2':0},
    {'type':'coincident','p1':4,'which1':1,'p2':8,'which2':0},

    # 上链：40 → 130 → 195
    {'type':'coincident','p1':5,'which1':1,'p2':6,'which2':0},
    {'type':'coincident','p1':6,'which1':1,'p2':7,'which2':0},

    # 下链：63 → 170
    {'type':'coincident','p1':8,'which1':1,'p2':9,'which2':0},

    # 右收尾：195 → 右上斜；170 → 右下斜
    {'type':'coincident','p1':7,'which1':1,'p2':10,'which2':0},
    {'type':'coincident','p1':9,'which1':1,'p2':11,'which2':0},

    # 右上/下斜 → 右上/下水平
    {'type':'coincident','p1':10,'which1':1,'p2':12,'which2':0},
    {'type':'coincident','p1':11,'which1':1,'p2':13,'which2':0},
]

# ----------------- 尺寸/基准 -----------------
constraint_list += [
    {'type':'vertical','line':0},
    {'type':'horizontal','line':1},
    {'type':'horizontal','line':2},
    {'type':'point_distance_y','p1':0,'which1':0,'p2':0,'which2':1,'value':178.0},

    # 顶/底水平右端与左竖的水平距离 = 132
    {'type':'point_distance_x','p1':0,'which1':1,'p2':1,'which2':1,'value':132.0},
    {'type':'point_distance_x','p1':0,'which1':0,'p2':2,'which2':1,'value':132.0},

    # 右两条水平
    {'type':'horizontal','line':12},
    {'type':'horizontal','line':13},

    # 右两条斜边角度（图 2：上 15°，下 75°；以全局 X 轴为参考，逆时针为正）
    {'type':'angle_deg','line':10,'value':180-15},
    {'type':'angle_deg','line':11,'value':-75},

    # 左两条 12° 斜边
    {'type':'angle_deg','line':3,'value':180-12},
    {'type':'angle_deg','line':4,'value':12},

    # 总宽：用右下水平的右端近似“最右点”
    {'type':'point_distance_x','p1':0,'which1':0,'p2':13,'which2':1,'value':420.0},
]

# 半径锁定
for idx, R in [(5,40.0),(6,130.0),(7,195.0),(8,63.0),(9,170.0)]:
    constraint_list.append({'type':'radius','arc':idx,'value':R})

# ----------------- 关键：在端点处相切（G1） -----------------
constraint_list += [
    # 左斜 ↔ 首段上/下弧（在弧的 theta1 端相切）
    {'type':'tangent_line_arc_at','line':3,'arc':5,'arc_end':0},
    {'type':'tangent_line_arc_at','line':4,'arc':8,'arc_end':0},

    # 上链弧-弧
    {'type':'tangent_arc_arc_at','arc1':5,'which1':1,'arc2':6,'which2':0},
    {'type':'tangent_arc_arc_at','arc1':6,'which1':1,'arc2':7,'which2':0},

    # 下链弧-弧
    {'type':'tangent_arc_arc_at','arc1':8,'which1':1,'arc2':9,'which2':0},

    # 右侧：195 ↔ 右上斜；170 ↔ 右下斜（在两条斜线的起点处相切）
    {'type':'tangent_line_arc_at','line':10,'arc':7,'arc_end':1},
    {'type':'tangent_line_arc_at','line':11,'arc':9,'arc_end':1},
]
