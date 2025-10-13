import math
from solver import Line, Arc

# create a simple object(半圆形) with Line and Arc

geom_objects = []

# 0: 右侧竖直线 L_left_vert - 固定起点在右下角 (0,0)，高度 178.00 (图中左竖尺寸)
L_left_vert = Line(x1=0.0, y1=0.0)   # 我把左右两端都先固定为便于约束链
geom_objects.append(L_left_vert)

# 1: 半圆弧
A_half_circle = Arc(cx=None, cy=None, r=10.0, theta1=math.radians(90.0), theta2=math.radians(270.0))
geom_objects.append(A_half_circle)

# 创造约束：首尾连接
constraint_list = []
constraint_list.append({'type':'coincident','p1':0,'which1':1,'p2':1,'which2':0})
constraint_list.append({'type':'coincident','p1':1,'which1':1,'p2':0,'which2':0})
