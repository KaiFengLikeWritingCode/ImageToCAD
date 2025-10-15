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

# # 6 右下斜线
# L_right_bottom_xie = Line(x1=None, y1=None, x2=None, y2=None)
# geom_objects.append(L_right_bottom_xie)

# 7 右上水平短边
# 6
L_right_top_h = Line(x1=None, y1=68+135, x2=None, y2=68+135)
geom_objects.append(L_right_top_h)

# # 8 右下水平短边
# L_right_bottom_h = Line(x1=None, y1=68.0, x2=None, y2=68.0)
# geom_objects.append(L_right_bottom_h)





# ===============================================
# 弧线、倒角定义
# =========
'''
左侧倒角
'''
# 9: 左上小倒角 A_topleft_chamfer R5 (设置 r=5, 用 theta1/theta2 给出大致切点角度  -12°偏移)
# 7
A_left_top_cham = Arc(cx=None, cy=None, r=5.0,
                        theta1=math.radians(12.0), theta2=math.radians(90.0))  # small chamfer
geom_objects.append(A_left_top_cham)

# 10: 左下小倒角 A_left_bottom_cham R5 (另一个 12° 标注)
# 8
A_left_bottom_cham = Arc(cx=None, cy=None, r=5.0,
                         theta1=math.radians(270), theta2=math.radians(360-12.0))
geom_objects.append(A_left_bottom_cham)



'''
中间部分的圆弧
'''
# === 上 ====
# 11 ：
# 9
l_top_hu1 = Arc(cx=None, cy=None, r=40.0, theta1=None, theta2=None)
geom_objects.append(l_top_hu1)
# 12 ：
# 10
l_top_hu2 = Arc(cx=None, cy=None, r=130.0, theta1=None, theta2=None)
geom_objects.append(l_top_hu2)

# 13：
# 11
l_top_hu3 = Arc(cx=None, cy=None, r=195.0, theta1=None, theta2=None)
geom_objects.append(l_top_hu3)
# 12
l_top_hu4 = Arc(cx=None, cy=None, r=40.0, theta1=None, theta2=None)
geom_objects.append(l_top_hu4)

#==== 下 ===
# 15 ：
# 13
l_bottom_hu1 = Arc(cx=None, cy=-3, r=63.0, theta1=None, theta2=None)
geom_objects.append(l_bottom_hu1)
# # 16：
# l_bottom_hu2 = Arc(cx=None, cy=None, r=170.0, theta1=None, theta2=None)
# geom_objects.append(l_bottom_hu2)
# # 17 ：
# l_bottom_hu3 = Arc(cx=None, cy=None, r=176.0, theta1=None, theta2=None)
# geom_objects.append(l_bottom_hu3)
# # 18：
# l_bottom_hu4 = Arc(cx=None, cy=None, r=40.0, theta1=None, theta2=None)
# geom_objects.append(l_bottom_hu4)



'''
右侧倒角
'''
# 19: 右上小倒角1
# 14
A_right_top_cham = Arc(cx=None, cy=None, r=3.0,
                        theta1=math.radians(90.0), theta2=math.radians(180.0-15))  # small chamfer
geom_objects.append(A_right_top_cham)

# # 20: 右下小倒角2
# A_right_bottom_cham = Arc(cx=None, cy=None, r=3.0,
#                          theta1=math.radians(180+15), theta2=math.radians(270))
# geom_objects.append(A_right_bottom_cham)


'''
简化版本
'''
# 21： 右侧基线
# 15
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
    {'type':'coincident','p1':1,'which1':1,'p2':7,'which2':1},   # a1.end == b9.start
    {'type':'coincident','p1':7,'which1':0,'p2':3,'which2':0},   # b9.end == a3.start
    {'type':'coincident','p1':3,'which1':1,'p2':9,'which2':0},  # a3.end == b11.start (上链第一弧)
]

# 上链：b11 → b12 → b13 → b14
constraint_list += [
    {'type':'coincident','p1':9,'which1':1,'p2':10,'which2':0},
    {'type':'coincident','p1':10,'which1':1,'p2':11,'which2':1},
    {'type':'coincident','p1':11,'which1':0,'p2':12,'which2':0},
]

# 上链末 → 右上斜线 → 右上 R3 → 右上水平 → 右侧基准竖线 top
constraint_list += [
    {'type':'coincident','p1':12,'which1':1,'p2':5,'which2':0},   # b14.end == a5.start
    {'type':'coincident','p1':5, 'which1':1,'p2':14,'which2':1},  # a5.end  == b19.start
    {'type':'coincident','p1':14,'which1':0,'p2':6,'which2':0},   # b19.end == a7.start
    {'type':'coincident','p1':6, 'which1':1,'p2':15,'which2':1},  # a7.end  == a21.top
]

# 左下柱底 → 左下小 R5.start → 左下水平.end → 左下斜线 → 下链第一弧
constraint_list += [
    {'type':'coincident','p1':0,'which1':0,'p2':2,'which2':0},    # a0.bottom == a2.start
    {'type':'coincident','p1':2,'which1':1,'p2':8,'which2':0},   # a2.end   == b10.start
    {'type':'coincident','p1':8,'which1':1,'p2':4,'which2':0},   # b10.end  == a4.start
    {'type':'coincident','p1':4,'which1':1,'p2':13,'which2':1},   # a4.end   == b15.start (下链第一弧)
]





# ---- 2) 相切（G1）----
# 线 ↔ 弧：两侧小圆角与相邻直线相切



# ---- 2) 相切（G1）---- 端点处相切
constraint_list += [
    # 左上：a1.end 与 b9.start 相连 → 在 b9.start 处与 a1 相切
    {'type':'tangent_at_arc_end_to_line', 'line':1, 'arc':7},

    # b9.end 与 a3.start 相连 → 在 b9.end 处与 a3 相切
    {'type':'tangent_at_arc_start_to_line',   'line':3, 'arc':7},

    # a3.end 与 b11.start 相连 → 在 b11.start 处与 a3 相切
    {'type':'tangent_at_arc_start_to_line', 'line':3, 'arc':9, 'side':'right'},

    # a2.end 与 b10.start 相连 → 在 b10.start 处与 a2 相切
    {'type':'tangent_at_arc_start_to_line', 'line':2, 'arc':8},

    # b10.end 与 a4.start 相连 → 在 b10.end 处与 a4 相切
    {'type':'tangent_at_arc_end_to_line', 'line':4, 'arc':8},

    # a4.end 与 b15.start 相连 → 在 b15.start 处与 a4 相切
    {'type':'tangent_at_arc_end_to_line', 'line':4, 'arc':13, 'side':'right'},

    # b14.end 与 a5.start 相连 → 在 b14.end 处与 a5 相切
    {'type':'tangent_at_arc_end_to_line',   'line':5, 'arc':12, 'side':'left'},

    # a5.end 与 b19.start 相连 → 在 b19.start 处与 a5 相切
    {'type':'tangent_at_arc_end_to_line', 'line':5, 'arc':14},



    # b19.end 与 a7.start 相连 → 在 b19.end 处与 a7 相切
    {'type':'tangent_at_arc_start_to_line',   'line':6, 'arc':14},


]


# constraint_list.append({'type':'point_distance_y', 'p1':11, 'which1':1, 'p2':15, 'which2':0, 'value':25.0})
constraint_list.append({'type':'point_distance_y', 'p1':13, 'which1':0, 'p2':9, 'which2':1, 'value':25.0})



constraint_list += [
    {'type':'tangent_at_arc_to_arc','Aarc':9,'Barc':10,'a_end':True,'b_end':False,'same_direction':True},  # b11 ↔ b12
    {'type':'tangent_at_arc_to_arc','Aarc':10,'Barc':11,'a_end':True,'b_end':True,'same_direction':False},  # b12 ↔ b13
    {'type':'tangent_at_arc_to_arc','Aarc':12,'Barc':13,'a_end':False,'b_end':False,'same_direction':False},  # b13 ↔ b14


]


for k in [9,10,12]:
    constraint_list.append({'type':'arc_side','arc':k,'side':'lower','samples':3,'weight':1.0})
for k in [13]:
    constraint_list.append({'type':'arc_side','arc':k,'side':'upper','samples':3,'weight':1.0})

# # 每个圆弧的扫角不超过 360°
for arc_idx in [7,8,9,10,11,12,13,14]:
    constraint_list.append({'type': 'arc_sweep_leq', 'arc': arc_idx, 'max_deg': 180.0})




# for k in [11,12,14,16]:
#     constraint_list.append({'type':'arc_endpoint_relation','arc':k,'mode':'x_order','order':'asc','margin':0.0})
#
# for k in [13,15,17,18]:
#     constraint_list.append({'type':'arc_endpoint_relation','arc':k,'mode':'x_order','order':'desc','margin':0.0})

for k in [10,12]:
    constraint_list.append({'type':'arc_endpoint_relation','arc':k,'mode':'y_order','order':'asc','margin':0.0})

for k in [9,11,13]:
    constraint_list.append({'type':'arc_endpoint_relation','arc':k,'mode':'y_order','order':'desc','margin':0.0})



# 右下水平短边长度 = 24
constraint_list.append({'type':'length','line':8,'value':24.0})
constraint_list.append({'type':'length','line':7,'value':24.0})
# 如需总宽/右侧高度等，可按需要再加；a21 已用固定坐标表达了 135 与 110 的高度关系

# ---- 4) 半径（硬约束）----
for arc_idx, R in [
    (7,5.0),(8,5.0),            # 左侧倒角
    (9,40.0),(10,130.0),(11,195.0),(12,40.0),   # 上链

    (14,3.0),         # 右上/右下小圆角
]:
    constraint_list.append({'type':'radius','arc':arc_idx,'value':R})

# 每个圆弧的扫角不超过 360°
for arc_idx in [7,8,9,10,11,12,13,14]:
    constraint_list.append({'type': 'arc_sweep_leq', 'arc': arc_idx, 'max_deg': 180.0})

# 上链在两条相邻切线形成的走廊内
for k in [9,10,11,12]:
    constraint_list += [
        {'type':'arc_side_of_line','line':3,'arc':k,'anchor':'start','side':'left','samples':3,'margin':0.0,'weight':1.0},
        {'type':'arc_side_of_line','line':5,'arc':k,'anchor':'end'  ,'side':'right','samples':3,'margin':0.0,'weight':1.0},
    ]

# 下链在两条相邻切线形成的走廊内
for k in [13]:
    constraint_list += [
        {'type':'arc_side_of_line','line':4,'arc':k,'anchor':'start','side':'right','samples':3,'margin':0.0,'weight':1.0},
        {'type':'arc_side_of_line','line':6,'arc':k,'anchor':'end'  ,'side':'left' ,'samples':3,'margin':0.0,'weight':1.0},
    ]





constraint_list += [
    # 限制线段6的起点在左边
    {'type':'line_point_order_x', 'line':6, 'order':'x1_left', 'margin':1e-3},

]









import numpy as np

# ========== 初值种子（非硬约束，只是利于收敛） ==========
# 右上/右下斜线给一个大致方向
for idx, seed in [(5, (260.0, 180.0, 300.0, 200.0))]:
    L = geom_objects[idx]
    for j, v in enumerate(seed):
        if np.isnan(L.params[j]): L.params[j] = v

# 右上/右下小圆角给个 theta 初值（图上 15° 为参考）
a19 = geom_objects[14];
if np.isnan(a19.params[3]): a19.params[3] = math.radians(90.0)
if np.isnan(a19.params[4]): a19.params[4] = math.radians(180.0-15.0)




for idx, t1, t2 in [(9,  135,  225), (10, 150, 240), (11,45,135), (12,225,315)]:
    arc = geom_objects[idx]
    if np.isnan(arc.params[3]): arc.params[3] = math.radians(t1)
    if np.isnan(arc.params[4]): arc.params[4] = math.radians(t2)

for idx, t1, t2 in [(13, 60,  150)]:
    arc = geom_objects[idx]
    if np.isnan(arc.params[3]): arc.params[3] = math.radians(t1)
    if np.isnan(arc.params[4]): arc.params[4] = math.radians(t2)


# ========== 可选：R170 圆心到基线的垂距 68 ==========
# constraint_list.append({'type':'center_distance_y','arc':16,'baseline_y':0.0,'value':68.0})


