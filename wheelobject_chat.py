import math
from solver import Line, Arc
# 变量定义
'''
线段（用起点、终点坐标表示）
圆弧 （圆心坐标、半径，两起始边界的弧度_逆时针计算）
'''
geom_objects = []
# 约束定义 - 尽量覆盖图上所有标注的尺寸与相对位置
'''
length
radius 
point_distance_x
point_distance_y 
tangent_line_arc 
coincident
'''
constraint_list = []

# ===============================================
# 线段定义
# =========

# 0 最左边的竖线
L_left_vert = Line(x1=93.0, y1=0.0, x2=93.0, y2=178.0)   # 我把左右两端都先固定为便于约束链
geom_objects.append(L_left_vert)

# 1: 左上水平短边 L_left_top_h - 顶部与倒角连接 (x2 未定)
L_left_top_h = Line(x1=93.0, y1=178.0, x2=132, y2=178.0)
geom_objects.append(L_left_top_h)

# 2： 左下水平短边
L_left_bottom_h = Line(x1=93.0, y1=0, x2=132, y2=0)
geom_objects.append(L_left_bottom_h)

# 3 左上斜线（连接左上小倒角 和 左上第一个大圆弧）
L_left_top_xie = Line(x1=None, y1=None, x2=None, y2=None)
geom_objects.append(L_left_top_xie)

# 4 左下斜线（连接左下小倒角 和 左下第一个大圆弧）
L_left_bottom_xie = Line(x1=None, y1=None, x2=None, y2=None)
geom_objects.append(L_left_bottom_xie)


# 5 右上斜线
L_right_top_xie = Line(x1=None, y1=None, x2=None, y2=None)
geom_objects.append(L_right_top_xie)

# 6 右下斜线
L_right_bottom_xie = Line(x1=None, y1=None, x2=None, y2=None)
geom_objects.append(L_right_bottom_xie)

# 7 右上水平短边
L_right_top_h = Line(x1=None, y1=68+135, x2=None, y2=68+135)
geom_objects.append(L_right_top_h)

# 8 右下水平短边
L_right_bottom_h = Line(x1=None, y1=68.0, x2=None, y2=68.0)
geom_objects.append(L_right_bottom_h)





# ===============================================
# 弧线、倒角定义
# =========
'''
左侧倒角
'''
# 9: 左上小倒角 A_topleft_chamfer R5 (设置 r=5, 用 theta1/theta2 给出大致切点角度  -12°偏移)
A_left_top_cham = Arc(cx=None, cy=None, r=5.0,
                        theta1=math.radians(12.0), theta2=math.radians(90.0))  # small chamfer
geom_objects.append(A_left_top_cham)

# 10: 左下小倒角 A_left_bottom_cham R5 (另一个 12° 标注)
A_left_bottom_cham = Arc(cx=None, cy=None, r=5.0,
                         theta1=math.radians(270), theta2=math.radians(360-12.0))
geom_objects.append(A_left_bottom_cham)



'''
中间部分的圆弧
'''
# === 上 ====
# 11 ：
l_top_hu1 = Arc(cx=None, cy=None, r=40.0, theta1=None, theta2=None)
geom_objects.append(l_top_hu1)
# 12 ：
l_top_hu2 = Arc(cx=None, cy=None, r=130.0, theta1=None, theta2=None)
geom_objects.append(l_top_hu2)

# 13：
l_top_hu3 = Arc(cx=None, cy=None, r=195.0, theta1=None, theta2=None)
geom_objects.append(l_top_hu3)
# 14 ：
l_top_hu4 = Arc(cx=None, cy=None, r=40.0, theta1=None, theta2=None)
geom_objects.append(l_top_hu4)

#==== 下 ===
# 15 ：
l_bottom_hu1 = Arc(cx=None, cy=-3, r=63.0, theta1=None, theta2=None)
geom_objects.append(l_bottom_hu1)
# 16：
l_bottom_hu2 = Arc(cx=None, cy=None, r=170.0, theta1=None, theta2=None)
geom_objects.append(l_bottom_hu2)
# 17 ：
l_bottom_hu3 = Arc(cx=None, cy=None, r=176.0, theta1=None, theta2=None)
geom_objects.append(l_bottom_hu3)
# 18：
l_bottom_hu4 = Arc(cx=None, cy=None, r=40.0, theta1=None, theta2=None)
geom_objects.append(l_bottom_hu4)



'''
右侧倒角
'''
# 19: 右上小倒角1
A_right_top_cham = Arc(cx=None, cy=None, r=3.0,
                        theta1=math.radians(90.0), theta2=math.radians(180.0-15))  # small chamfer
geom_objects.append(A_right_top_cham)

# 20: 右下小倒角2
A_right_bottom_cham = Arc(cx=None, cy=None, r=3.0,
                         theta1=math.radians(180+15), theta2=math.radians(270))
geom_objects.append(A_right_bottom_cham)


'''
简化版本
'''
# 21： 右侧基线
L_right_vert = Line(x1=400.0, y1=68, x2=400.0, y2=68.0 + 135)   # 我把左右两端都先固定为便于约束链
geom_objects.append(L_right_vert)


# 顶部左立柱 → 左上水平
constraint_list += [
    {'type':'coincident','p1':0,'which1':1,'p2':1,'which2':0},
    {'type':'coincident','p1':0,'which1':0,'p2':2,'which2':0},
]

# a1.end == b9[0]；b9[1] == a3.start
constraint_list += [
    {'type':'coincident_arc_idx','arc':9,'which':0, 'p2':1,'which2':1},
    {'type':'coincident_arc_idx','arc':9,'which':1, 'p2':3,'which2':0},
]

# a3.end == b11[0]
constraint_list += [
    {'type':'coincident_arc_idx','arc':11,'which':0, 'p2':3,'which2':1},
]

# 上链： b11[1] = b12[0] = b13[0] = b14[0] 逐段连接
constraint_list += [
    {'type':'coincident_arc_idx','arc':12,'which':0, 'p2':11,'which2':1},
    {'type':'coincident_arc_idx','arc':13,'which':0, 'p2':12,'which2':1},
    {'type':'coincident_arc_idx','arc':14,'which':0, 'p2':13,'which2':1},
]

# b14[1] == a5.start；a5.end == b19[0]；b19[1] == a7.start；a7.end == a21.top
constraint_list += [
    {'type':'coincident_arc_idx','arc':14,'which':1, 'p2':5,'which2':0},
    {'type':'coincident_arc_idx','arc':19,'which':0, 'p2':5,'which2':1},
    {'type':'coincident_arc_idx','arc':19,'which':1, 'p2':7,'which2':0},
    {'type':'coincident','p1':7,'which1':1,'p2':21,'which2':1},
]

# 底部链
constraint_list += [
    {'type':'coincident','p1':2,'which1':1,'p2':10,'which2':0},             # a2.end == b10[0]
    {'type':'coincident_arc_idx','arc':10,'which':1, 'p2':4,'which2':0},     # b10[1] == a4.start
    {'type':'coincident_arc_idx','arc':15,'which':0, 'p2':4,'which2':1},     # a4.end == b15[0]

    {'type':'coincident_arc_idx','arc':16,'which':0, 'p2':15,'which2':1},    # b15[1] == b16[0]
    {'type':'coincident_arc_idx','arc':17,'which':0, 'p2':16,'which2':1},    # b16[1] == b17[0]
    {'type':'coincident_arc_idx','arc':18,'which':0, 'p2':17,'which2':1},    # b17[1] == b18[0]

    {'type':'coincident_arc_idx','arc':18,'which':1, 'p2':6,'which2':0},     # b18[1] == a6.start
    {'type':'coincident_arc_idx','arc':20,'which':0, 'p2':6,'which2':1},     # a6.end  == b20[0]
    {'type':'coincident_arc_idx','arc':20,'which':1, 'p2':8,'which2':0},     # b20[1] == a8.start
    {'type':'coincident','p1':8,'which1':1,'p2':21,'which2':0},              # a8.end  == a21.bottom
]

constraint_list += [
    {'type':'tangent_line_to_arc_idx','line':1, 'arc':9,  'which':0},
    {'type':'tangent_line_to_arc_idx','line':3, 'arc':9,  'which':1},
    {'type':'tangent_line_to_arc_idx','line':3, 'arc':11, 'which':0},

    {'type':'tangent_line_to_arc_idx','line':2, 'arc':10, 'which':0},
    {'type':'tangent_line_to_arc_idx','line':4, 'arc':10, 'which':1},
    {'type':'tangent_line_to_arc_idx','line':4, 'arc':15, 'which':0},

    {'type':'tangent_line_to_arc_idx','line':5, 'arc':14, 'which':1},
    {'type':'tangent_line_to_arc_idx','line':5, 'arc':19, 'which':0},
    {'type':'tangent_line_to_arc_idx','line':7, 'arc':19, 'which':1},

    {'type':'tangent_line_to_arc_idx','line':6, 'arc':18, 'which':1},
    {'type':'tangent_line_to_arc_idx','line':6, 'arc':20, 'which':0},
    {'type':'tangent_line_to_arc_idx','line':8, 'arc':20, 'which':1},
]

constraint_list += [
    {'type':'tangent_arc_idx_to_arc_idx','Aarc':11,'wa':1,'Barc':12,'wb':0,'same_direction':True},
    {'type':'tangent_arc_idx_to_arc_idx','Aarc':12,'wa':1,'Barc':13,'wb':0,'same_direction':True},
    {'type':'tangent_arc_idx_to_arc_idx','Aarc':13,'wa':1,'Barc':14,'wb':0,'same_direction':True},

    {'type':'tangent_arc_idx_to_arc_idx','Aarc':15,'wa':1,'Barc':16,'wb':0,'same_direction':True},
    {'type':'tangent_arc_idx_to_arc_idx','Aarc':16,'wa':1,'Barc':17,'wb':0,'same_direction':True},
    {'type':'tangent_arc_idx_to_arc_idx','Aarc':17,'wa':1,'Barc':18,'wb':0,'same_direction':True},
]

for arc_idx in [9,10,11,12,13,14,15,16,17,18,19,20]:
    constraint_list.append({'type':'arc_sweep_leq', 'arc':arc_idx, 'max_deg': 360.0})




import numpy as np

# ========== 初值种子（非硬约束，只是利于收敛） ==========
# 右上/右下斜线给一个大致方向
for idx, seed in [(5, (260.0, 180.0, 300.0, 200.0)),   # a5  右上斜线
                  (6, (300.0,  80.0, 330.0,  70.0))]:  # a6  右下斜线
    L = geom_objects[idx]
    for j, v in enumerate(seed):
        if np.isnan(L.params[j]): L.params[j] = v

# 右上/右下小圆角给个 theta 初值（图上 15° 为参考）
a19 = geom_objects[19];
if np.isnan(a19.params[3]): a19.params[3] = math.radians(90.0)
if np.isnan(a19.params[4]): a19.params[4] = math.radians(180.0-15.0)

a20 = geom_objects[20];
if np.isnan(a20.params[3]): a20.params[3] = math.radians(180.0+15.0)
if np.isnan(a20.params[4]): a20.params[4] = math.radians(270.0)

# 中央“上链”四段的 theta 初值（上拱）
for idx, t1, t2 in [(11,  20,  50), (12, 50, 110), (13,110,160), (14,160,185)]:
    arc = geom_objects[idx]
    if np.isnan(arc.params[3]): arc.params[3] = math.radians(t1)
    if np.isnan(arc.params[4]): arc.params[4] = math.radians(t2)

# 中央“下链”四段的 theta 初值（下拱）
for idx, t1, t2 in [(15, -20,  10), (16, 10,  40), (17, 40,  80), (18, 80, 100)]:
    arc = geom_objects[idx]
    if np.isnan(arc.params[3]): arc.params[3] = math.radians(t1)
    if np.isnan(arc.params[4]): arc.params[4] = math.radians(t2)

# ========== 可选：R170 圆心到基线的垂距 68 ==========
# constraint_list.append({'type':'center_distance_y','arc':16,'baseline_y':0.0,'value':68.0})


