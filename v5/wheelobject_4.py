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





# ===============================
# 约束（coincident / tangent / size / radius）
# ===============================



# ---- 1) 拓扑连通：按外轮廓一圈连接 ----
# 左上柱顶 → 左上小 R5.start → 左上水平.start（已绑定在对象里是同点，这里明确一下端点链）
constraint_list += [
    {'type':'coincident','p1':0,'which1':1,'p2':1,'which2':0},   # a0.top == a1.start
    {'type':'coincident','p1':0,'which1':0,'p2':2,'which2':0},
    {'type':'coincident','p1':1,'which1':1,'p2':9,'which2':1},   # a1.end == b9.start
    {'type':'coincident','p1':9,'which1':0,'p2':3,'which2':0},   # b9.end == a3.start
    {'type':'coincident','p1':3,'which1':1,'p2':11,'which2':0},  # a3.end == b11.start (上链第一弧)
]

# 上链：b11 → b12 → b13 → b14
constraint_list += [
    {'type':'coincident','p1':11,'which1':1,'p2':12,'which2':0},
    {'type':'coincident','p1':12,'which1':1,'p2':13,'which2':1},
    {'type':'coincident','p1':13,'which1':0,'p2':14,'which2':0},
]

# 上链末 → 右上斜线 → 右上 R3 → 右上水平 → 右侧基准竖线 top
constraint_list += [
    {'type':'coincident','p1':14,'which1':1,'p2':5,'which2':0},   # b14.end == a5.start
    {'type':'coincident','p1':5, 'which1':1,'p2':19,'which2':1},  # a5.end  == b19.start
    {'type':'coincident','p1':19,'which1':0,'p2':7,'which2':0},   # b19.end == a7.start
    {'type':'coincident','p1':7, 'which1':1,'p2':21,'which2':1},  # a7.end  == a21.top
]

# 左下柱底 → 左下小 R5.start → 左下水平.end → 左下斜线 → 下链第一弧
constraint_list += [
    {'type':'coincident','p1':0,'which1':0,'p2':2,'which2':0},    # a0.bottom == a2.start
    {'type':'coincident','p1':2,'which1':1,'p2':10,'which2':0},   # a2.end   == b10.start
    {'type':'coincident','p1':10,'which1':1,'p2':4,'which2':0},   # b10.end  == a4.start
    {'type':'coincident','p1':4,'which1':1,'p2':15,'which2':1},   # a4.end   == b15.start (下链第一弧)
]

# 下链：b15 → b16 → b17 → b18
constraint_list += [
    {'type':'coincident','p1':15,'which1':0,'p2':16,'which2':0},
    {'type':'coincident','p1':16,'which1':1,'p2':17,'which2':1},
    {'type':'coincident','p1':17,'which1':0,'p2':18,'which2':1},
]

# 下链末 → 右下斜线 → 右下 R3 → 右下水平 → 右侧基准竖线 bottom
constraint_list += [
    {'type':'coincident','p1':18,'which1':0,'p2':6, 'which2':0},  # b18.end == a6.start
    {'type':'coincident','p1':6, 'which1':1,'p2':20,'which2':0},  # a6.end  == b20.start
    {'type':'coincident','p1':20,'which1':1,'p2':8, 'which2':0},  # b20.end == a8.start
    {'type':'coincident','p1':8, 'which1':1,'p2':21,'which2':0},  # a8.end  == a21.bottom
]

# ---- 2) 相切（G1）----
# 线 ↔ 弧：两侧小圆角与相邻直线相切
# constraint_list += [
#     {'type':'tangent_line_arc','line':1, 'arc':9},   # a1 ↔ b9
#     {'type':'tangent_line_arc','line':3, 'arc':9},   # a3 ↔ b9
#     {'type':'tangent_line_arc','line':3, 'arc':11},   # a3 ↔ b11
#     {'type':'tangent_line_arc','line':2, 'arc':10},  # a2 ↔ b10
#     {'type':'tangent_line_arc','line':4, 'arc':10},  # a4 ↔ b10
#     {'type':'tangent_line_arc','line':4, 'arc':15},  # a4 ↔ b15
#     {'type':'tangent_line_arc','line':5, 'arc':14},  # a5 ↔ b14
#     {'type':'tangent_line_arc','line':5, 'arc':19},  # a5 ↔ b19
#     {'type':'tangent_line_arc','line':6, 'arc':18},  # a6 ↔ b18
#     {'type':'tangent_line_arc','line':6, 'arc':20},  # a6 ↔ b20
#     {'type':'tangent_line_arc','line':7, 'arc':19},  # a7 ↔ b19
#     {'type':'tangent_line_arc','line':8, 'arc':20},  # a8 ↔ b20
# ]

'''
tangent_at_arc_start_to_line：用弧的 起点（theta1）的切向；
tangent_at_arc_end_to_line：用弧的 终点（theta2）的切向；
'''
# ---- 2) 相切（G1）---- 端点处相切
constraint_list += [
    # 左上：a1.end 与 b9.start 相连 → 在 b9.start 处与 a1 相切
    {'type':'tangent_at_arc_end_to_line', 'line':1, 'arc':9},

    # b9.end 与 a3.start 相连 → 在 b9.end 处与 a3 相切
    {'type':'tangent_at_arc_start_to_line',   'line':3, 'arc':9},

    # a3.end 与 b11.start 相连 → 在 b11.start 处与 a3 相切
    {'type':'tangent_at_arc_start_to_line', 'line':3, 'arc':11},

    # a2.end 与 b10.start 相连 → 在 b10.start 处与 a2 相切
    {'type':'tangent_at_arc_start_to_line', 'line':2, 'arc':10},

    # b10.end 与 a4.start 相连 → 在 b10.end 处与 a4 相切
    {'type':'tangent_at_arc_end_to_line', 'line':4, 'arc':10},

    # a4.end 与 b15.start 相连 → 在 b15.start 处与 a4 相切
    {'type':'tangent_at_arc_end_to_line', 'line':4, 'arc':15},

    # b14.end 与 a5.start 相连 → 在 b14.end 处与 a5 相切
    {'type':'tangent_at_arc_end_to_line',   'line':5, 'arc':14},

    # a5.end 与 b19.start 相连 → 在 b19.start 处与 a5 相切
    {'type':'tangent_at_arc_end_to_line', 'line':5, 'arc':19},

    # b18.end 与 a6.start 相连 → 在 b18.end 处与 a6 相切
    {'type':'tangent_at_arc_start_to_line',   'line':6, 'arc':18},

    # a6.end 与 b20.start 相连 → 在 b20.start 处与 a6 相切
    {'type':'tangent_at_arc_start_to_line', 'line':6, 'arc':20},

    # b19.end 与 a7.start 相连 → 在 b19.end 处与 a7 相切
    {'type':'tangent_at_arc_start_to_line',   'line':7, 'arc':19},

    # b20.end 与 a8.start 相连 → 在 b20.end 处与 a8 相切
    {'type':'tangent_at_arc_end_to_line',   'line':8, 'arc':20},
]

constraint_list.append({'type':'point_distance_y', 'p1':11, 'which1':1, 'p2':15, 'which2':0, 'value':25.0})


# 弧 ↔ 弧：中央蛇形上/下两条链，逐段相切
constraint_list += [
    {'type':'tangent_at_arc_to_arc','Aarc':11,'Barc':12,'a_end':True,'b_end':False,'same_direction':True},  # b11 ↔ b12
    {'type':'tangent_at_arc_to_arc','Aarc':12,'Barc':13,'a_end':True,'b_end':True,'same_direction':True},  # b12 ↔ b13
    {'type':'tangent_at_arc_to_arc','Aarc':13,'Barc':14,'a_end':False,'b_end':False,'same_direction':True},  # b13 ↔ b14

    {'type':'tangent_at_arc_to_arc','Aarc':15,'Barc':16,'a_end':False,'b_end':False,'same_direction':True},  # b15 ↔ b16
    {'type':'tangent_at_arc_to_arc','Aarc':16,'Barc':17,'a_end':True,'b_end':True,'same_direction':True},  # b16 ↔ b17
    {'type':'tangent_at_arc_to_arc','Aarc':17,'Barc':18,'a_end':False,'b_end':True,'same_direction':True},  # b17 ↔ b18
]


# 每个圆弧的扫角不超过 360°
for arc_idx in [9,10,11,12,13,14,15,16,17,18,19,20]:
    constraint_list.append({'type': 'arc_sweep_leq', 'arc': arc_idx, 'max_deg': 180.0})
# 上链在圆心上侧（上拱）
for k in [11,12,16,14]:
    constraint_list.append({'type':'arc_side','arc':k,'side':'lower','samples':3,'weight':1.0})

# 下链在圆心下侧（下拱）
for k in [15,13,17,18]:
    constraint_list.append({'type':'arc_side','arc':k,'side':'upper','samples':3,'weight':1.0})

# 上链在两条相邻切线形成的走廊内
for k in [11,12,13,14]:
    constraint_list += [
        {'type':'arc_side_of_line','line':3,'arc':k,'anchor':'start','side':'left','samples':3,'margin':0.0,'weight':1.0},
        {'type':'arc_side_of_line','line':5,'arc':k,'anchor':'end'  ,'side':'right','samples':3,'margin':0.0,'weight':1.0},
    ]

# 下链在两条相邻切线形成的走廊内
for k in [15,16,17,18]:
    constraint_list += [
        {'type':'arc_side_of_line','line':4,'arc':k,'anchor':'start','side':'right','samples':3,'margin':0.0,'weight':1.0},
        {'type':'arc_side_of_line','line':6,'arc':k,'anchor':'end'  ,'side':'left' ,'samples':3,'margin':0.0,'weight':1.0},
    ]


# ---- 3) 尺寸（来自图上的关键尺寸；避免与已固定坐标冲突）----
# 左上水平终点相对左竖顶的水平偏移 = 132
constraint_list.append({'type':'point_distance_x','p1':0,'which1':1,'p2':1,'which2':1,'value':132.0})

# 右下水平短边长度 = 24
constraint_list.append({'type':'length','line':8,'value':24.0})
constraint_list.append({'type':'length','line':7,'value':24.0})
# 如需总宽/右侧高度等，可按需要再加；a21 已用固定坐标表达了 135 与 110 的高度关系

# ---- 4) 半径（硬约束）----
for arc_idx, R in [
    (9,5.0),(10,5.0),            # 左侧倒角
    (11,40.0),(12,130.0),(13,195.0),(14,40.0),   # 上链
    (15,63.0),(16,170.0),(17,176.0),(18,40.0),   # 下链
    (19,3.0),(20,3.0),           # 右上/右下小圆角
]:
    constraint_list.append({'type':'radius','arc':arc_idx,'value':R})

# 每个圆弧的扫角不超过 360°
for arc_idx in [9,10,11,12,13,14,15,16,17,18,19,20]:
    constraint_list.append({'type': 'arc_sweep_leq', 'arc': arc_idx, 'max_deg': 180.0})



# constraint_list.append({'type':'center_bound', 'arc':12, 'axis':'y', 'op':'ge', 'value':178.0, 'margin':2.0, 'weight':2.0})
# constraint_list.append({'type':'center_bound', 'arc':16, 'axis':'y', 'op':'ge', 'value':178.0, 'margin':2.0, 'weight':2.0})
# # constraint_list.append({'type':'center_bound', 'arc':14, 'axis':'y', 'op':'ge', 'value':178.0, 'margin':2.0, 'weight':2.0})
#
# # b11 圆心 x 落在 [200, 340] 之间
# constraint_list.append({'type':'center_bound', 'arc':11, 'axis':'y', 'op':'between', 'lo':79.0, 'hi':178.0})
# constraint_list.append({'type':'center_bound', 'arc':18, 'axis':'y', 'op':'between', 'lo':68.0, 'hi':178.0})
# constraint_list.append({'type':'center_bound', 'arc':14, 'axis':'y', 'op':'between', 'lo':178.0, 'hi':68.0+135})
#




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


