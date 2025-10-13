# wheelobject_min.py
import math
from solver_my import Line, Arc

geom_objects = []
constraint_list = []

# 左竖固定
L_left = Line(x1=0.0, y1=0.0, x2=0.0, y2=178.0)
geom_objects.append(L_left)

# 左上水平、左下水平：起点固定，终点未知
L_top = Line(x1=0.0, y1=178.0, x2=None, y2=178.0); geom_objects.append(L_top)
L_bot = Line(x1=0.0, y1=0.0,   x2=None, y2=0.0);   geom_objects.append(L_bot)

# 左上 R5、左下 R63：半径固定，其余未知
A_R5  = Arc(cx=None, cy=None, r=5.0,  theta1=None, theta2=None);  geom_objects.append(A_R5)
A_R63 = Arc(cx=None, cy=None, r=63.0, theta1=None, theta2=None);  geom_objects.append(A_R63)

# 两条斜线：让优化器决定
L_s1 = Line(x1=None, y1=None, x2=None, y2=None); geom_objects.append(L_s1)
L_s2 = Line(x1=None, y1=None, x2=None, y2=None); geom_objects.append(L_s2)

# 中段 R170、R130：半径固定，圆心/角度未知；R170 圆心标高 68
A_R170 = Arc(cx=None, cy=None, r=170.0, theta1=None, theta2=None); geom_objects.append(A_R170)
A_R130 = Arc(cx=None, cy=None, r=130.0, theta1=None, theta2=None); geom_objects.append(A_R130)

# 约束
constraint_list += [
    {'type':'radius','arc':3,'value':5.0},
    {'type':'radius','arc':4,'value':63.0},
    {'type':'radius','arc':7,'value':170.0},
    {'type':'radius','arc':8,'value':130.0},
    {'type':'center_distance_y','arc':7,'value':68.0},               # 驱动
    {'type':'point_distance_y','p1':0,'which1':0,'p2':0,'which2':1,'value':178.0},
    # 连接 + 端点相切
    {'type':'coincident','p1':0,'which1':1,'p2':3,'which2':0},       # 左竖顶 ↔ R5.start
    {'type':'tangent_line_arc_at','line':1,'arc':3,'arc_end':1},     # 上水平 ↔ R5.end 切
    {'type':'coincident','p1':0,'which1':0,'p2':4,'which2':1},       # 左竖底 ↔ R63.end
    {'type':'tangent_line_arc_at','line':2,'arc':4,'arc_end':0},     # 下水平 ↔ R63.start 切
    {'type':'coincident','p1':5,'which1':0,'p2':3,'which2':0},       # 斜线1起 ↔ R5.start
    {'type':'coincident','p1':6,'which1':0,'p2':4,'which2':0},       # 斜线2起 ↔ R63.start
    {'type':'coincident','p1':5,'which1':1,'p2':8,'which2':0},       # 斜线1终 ↔ R130.start
    {'type':'coincident','p1':6,'which1':1,'p2':7,'which2':0},       # 斜线2终 ↔ R170.start
    {'type':'tangent_arc_arc_at','arc1':7,'end1':1,'arc2':8,'end2':0}# R170.end ↔ R130.start 切
]

