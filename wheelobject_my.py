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
L_left_top_h = Line(x1=93.0, y1=178.0, x2=None, y2=178.0)
geom_objects.append(L_left_top_h)

# 2： 左下水平短边
L_left_bottom_h = Line(x1=93.0, y1=0, x2=None, y2=0)
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
L_right_top_h = Line(x1=None, y1=71+135, x2=None, y2=71+135)
geom_objects.append(L_right_top_h)

# 8 右下水平短边
L_right_bottom_h = Line(x1=None, y1=71.0, x2=None, y2=71.0)
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
l_bottom_hu1 = Arc(cx=None, cy=None, r=63.0, theta1=None, theta2=None)
geom_objects.append(l_bottom_hu1)
# 16：
l_bottom_hu2 = Arc(cx=None, cy=None, r=170.0, theta1=None, theta2=None)
geom_objects.append(l_bottom_hu2)
# 17 ：
l_bottom_hu3 = Arc(cx=None, cy=None, r=716.0, theta1=None, theta2=None)
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



# 左侧顶端连接 L_left_vert(0) end -> A_topleft_chamfer(1) start
constraint_list.append({'type':'coincident','p1':1,'which1':1,'p2':9,'which2':0})
constraint_list.append({'type':'coincident','p1':2,'which1':1,'p2':10,'which2':0})

constraint_list.append({'type':'coincident','p1':9,'which1':1,'p2':3,'which2':0})
constraint_list.append({'type':'coincident','p1':10,'which1':1,'p2':4,'which2':0})


constraint_list.append({'type':'coincident','p1':3,'which1':1,'p2':11,'which2':0})
constraint_list.append({'type':'coincident','p1':4,'which1':1,'p2':15,'which2':0})

constraint_list.append({'type':'coincident','p1':11,'which1':1,'p2':12,'which2':0})
constraint_list.append({'type':'coincident','p1':12,'which1':1,'p2':13,'which2':0})
constraint_list.append({'type':'coincident','p1':13,'which1':1,'p2':14,'which2':0})

constraint_list.append({'type':'coincident','p1':15,'which1':1,'p2':16,'which2':0})
constraint_list.append({'type':'coincident','p1':16,'which1':1,'p2':17,'which2':0})
constraint_list.append({'type':'coincident','p1':17,'which1':1,'p2':18,'which2':0})

constraint_list.append({'type':'coincident','p1':14,'which1':1,'p2':5,'which2':0})
constraint_list.append({'type':'coincident','p1':18,'which1':1,'p2':6,'which2':0})

constraint_list.append({'type':'coincident','p1':5,'which1':1,'p2':19,'which2':0})
constraint_list.append({'type':'coincident','p1':6,'which1':1,'p2':20,'which2':0})

constraint_list.append({'type':'coincident','p1':19,'which1':1,'p2':7,'which2':0})
constraint_list.append({'type':'coincident','p1':20,'which1':1,'p2':8,'which2':0})



constraint_list.append({'type':'tangent_line_arc', 'line':1, 'arc':9})
constraint_list.append({'type':'tangent_line_arc', 'line':2, 'arc':10})
constraint_list.append({'type':'tangent_line_arc', 'line':5, 'arc':19})
constraint_list.append({'type':'tangent_line_arc', 'line':6, 'arc':20})


