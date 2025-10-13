# --- wheelobject.py (修正版) ---
import math, numpy as np
from solver import Line, Arc

geom_objects = []
constraint_list = []

# ================== 线段 ==================
# a0 最左竖
a0 = Line(x1=93.0, y1=0.0, x2=93.0, y2=178.0);          geom_objects.append(a0)
# a1 左上水平（放开 x2，避免与 132 约束冲突）
a1 = Line(x1=93.0, y1=178.0, x2=None, y2=178.0);        geom_objects.append(a1)
# a2 左下水平（同理放开）
a2 = Line(x1=93.0, y1=0.0,   x2=None, y2=0.0);          geom_objects.append(a2)
# a3 左上斜
a3 = Line(x1=None, y1=None, x2=None, y2=None);          geom_objects.append(a3)
# a4 左下斜
a4 = Line(x1=None, y1=None, x2=None, y2=None);          geom_objects.append(a4)
# a5 右上斜
a5 = Line(x1=None, y1=None, x2=None, y2=None);          geom_objects.append(a5)
# a6 右下斜
a6 = Line(x1=None, y1=None, x2=None, y2=None);          geom_objects.append(a6)
# a7 右上平台（y=68+135）
a7 = Line(x1=None, y1=68+135, x2=None, y2=68+135);      geom_objects.append(a7)
# a8 右下平台（y=68）
a8 = Line(x1=None, y1=68.0,   x2=None, y2=68.0);        geom_objects.append(a8)
# a21 右侧基准竖
a21= Line(x1=400.0, y1=68.0, x2=400.0, y2=68.0+135.0);  geom_objects.append(a21)

# ================== 圆弧 ==================
# 左侧倒角
b9  = Arc(cx=None, cy=None, r=5.0,   theta1=math.radians(12.0),   theta2=math.radians(90.0));   geom_objects.append(b9)
b10 = Arc(cx=None, cy=None, r=5.0,   theta1=math.radians(270.0),  theta2=math.radians(348.0));  geom_objects.append(b10)

# 中央上链：R40 → R130 → R195 → R40
b11 = Arc(cx=None, cy=None, r=40.0,  theta1=None, theta2=None);   geom_objects.append(b11)
b12 = Arc(cx=None, cy=None, r=130.0, theta1=None, theta2=None);   geom_objects.append(b12)
b13 = Arc(cx=None, cy=None, r=195.0, theta1=None, theta2=None);   geom_objects.append(b13)
b14 = Arc(cx=None, cy=None, r=40.0,  theta1=None, theta2=None);   geom_objects.append(b14)

# 中央下链：R63 → R170 → R176 → R40
b15 = Arc(cx=None, cy=-3.0, r=63.0,  theta1=None, theta2=None);   geom_objects.append(b15)
b16 = Arc(cx=None, cy=None, r=170.0, theta1=None, theta2=None);   geom_objects.append(b16)
b17 = Arc(cx=None, cy=None, r=176.0, theta1=None, theta2=None);   geom_objects.append(b17)
b18 = Arc(cx=None, cy=None, r=40.0,  theta1=None, theta2=None);   geom_objects.append(b18)

# 右侧倒角/圆角
b19 = Arc(cx=None, cy=None, r=3.0,   theta1=math.radians(90.0),   theta2=math.radians(165.0)); geom_objects.append(b19) # 15°
b20 = Arc(cx=None, cy=None, r=3.0,   theta1=math.radians(195.0),  theta2=math.radians(270.0)); geom_objects.append(b20)
# 右下 R12（与平台相连）
b21 = Arc(cx=None, cy=None, r=12.0,  theta1=None, theta2=None);   geom_objects.append(b21)

# ================== 连通（coincident） ==================
constraint_list += [
    # 左上：竖 → 上水平 → R5 → 左上斜 → 上链
    {'type':'coincident','p1':0,'which1':1,'p2':1,'which2':0},
    {'type':'coincident','p1':1,'which1':1,'p2':9,'which2':0},
    {'type':'coincident','p1':9,'which1':1,'p2':3,'which2':0},
    {'type':'coincident','p1':3,'which1':1,'p2':11,'which2':0},

    # 上链：b11→b12→b13→b14
    {'type':'coincident','p1':11,'which1':1,'p2':12,'which2':0},
    {'type':'coincident','p1':12,'which1':1,'p2':13,'which2':0},
    {'type':'coincident','p1':13,'which1':1,'p2':14,'which2':0},

    # 上链末 → 右上斜 → 右上 R3 → 右上平台 → 右侧竖 top
    {'type':'coincident','p1':14,'which1':1,'p2':5,'which2':0},
    {'type':'coincident','p1':5, 'which1':1,'p2':19,'which2':0},
    {'type':'coincident','p1':19,'which1':1,'p2':7,'which2':0},
    {'type':'coincident','p1':7, 'which1':1,'p2':21,'which2':1},

    # 左下：竖 → 下水平 → R5 → 左下斜 → 下链
    {'type':'coincident','p1':0,'which1':0,'p2':2,'which2':0},
    {'type':'coincident','p1':2,'which1':1,'p2':10,'which2':0},
    {'type':'coincident','p1':10,'which1':1,'p2':4,'which2':0},
    {'type':'coincident','p1':4,'which1':1,'p2':15,'which2':0},

    # 下链：b15→b16→b17→b18
    {'type':'coincident','p1':15,'which1':1,'p2':16,'which2':0},
    {'type':'coincident','p1':16,'which1':1,'p2':17,'which2':0},
    {'type':'coincident','p1':17,'which1':1,'p2':18,'which2':0},

    # 下链末 → 右下斜 → 右下 R3 → 右下平台 → R12 → 右侧竖 bottom
    {'type':'coincident','p1':18,'which1':1,'p2':6, 'which2':0},
    {'type':'coincident','p1':6, 'which1':1,'p2':20,'which2':0},
    {'type':'coincident','p1':20,'which1':1,'p2':8, 'which2':0},
    {'type':'coincident','p1':8, 'which1':1,'p2':21,'which2':0},
    {'type':'coincident','p1':8, 'which1':1,'p2':21,'which2':0},
    {'type':'coincident','p1':8, 'which1':1,'p2':21,'which2':0},
    {'type':'coincident','p1':8, 'which1':1,'p2':21,'which2':0},
    {'type':'coincident','p1':8, 'which1':1,'p2':21,'which2':0},
]
# （注：R12 b21 在 a8 末端：）
constraint_list += [
    {'type':'coincident','p1':8,'which1':1,'p2':21,'which2':0},  # a8.end==a21.bottom（已上）
    {'type':'coincident','p1':8,'which1':1,'p2':21,'which2':0},
]

# ================== 相切（G1） ==================
# ——需要在 solver.py 里增加这两类端点相切：
#    tangent_at_arc_start_to_line / tangent_at_arc_end_to_line
constraint_list += [
    # 左侧倒角两端
    {'type':'tangent_at_arc_end_to_line',  'line':1,'arc':9},  # a1 ↔ b9.end
    {'type':'tangent_at_arc_start_to_line','line':3,'arc':9},  # a3 ↔ b9.start
    {'type':'tangent_at_arc_end_to_line',  'line':2,'arc':10}, # a2 ↔ b10.end
    {'type':'tangent_at_arc_start_to_line','line':4,'arc':10}, # a4 ↔ b10.start

    # 上链入口/出口
    {'type':'tangent_at_arc_start_to_line','line':3,'arc':11},  # a3 ↔ b11.start
    {'type':'tangent_at_arc_end_to_line',  'line':5,'arc':14},  # a5 ↔ b14.end

    # 右侧小圆角与平台/斜线
    {'type':'tangent_at_arc_start_to_line','line':5,'arc':19},  # a5 ↔ b19.start
    {'type':'tangent_at_arc_end_to_line',  'line':7,'arc':19},  # a7 ↔ b19.end
    {'type':'tangent_at_arc_start_to_line','line':6,'arc':20},  # a6 ↔ b20.start
    {'type':'tangent_at_arc_end_to_line',  'line':8,'arc':20},  # a8 ↔ b20.end

    # 下链入口/出口
    {'type':'tangent_at_arc_start_to_line','line':4,'arc':15},  # a4 ↔ b15.start
    {'type':'tangent_at_arc_end_to_line',  'line':6,'arc':18},  # a6 ↔ b18.end
]

# 弧 ↔ 弧（中央蛇形）
constraint_list += [
    {'type':'tangent_arc_arc','Aarc':11,'Barc':12,'a_end':True,'b_end':False},
    {'type':'tangent_arc_arc','Aarc':12,'Barc':13,'a_end':True,'b_end':False},
    {'type':'tangent_arc_arc','Aarc':13,'Barc':14,'a_end':True,'b_end':False},
    {'type':'tangent_arc_arc','Aarc':15,'Barc':16,'a_end':True,'b_end':False},
    {'type':'tangent_arc_arc','Aarc':16,'Barc':17,'a_end':True,'b_end':False},
    {'type':'tangent_arc_arc','Aarc':17,'Barc':18,'a_end':True,'b_end':False},
]

# ================== 尺寸/半径 ==================
# 左上水平终点相对左竖顶的水平偏移 = 132
constraint_list.append({'type':'point_distance_x','p1':0,'which1':1,'p2':1,'which2':1,'value':132.0})
# 右下平台长度 = 24
constraint_list.append({'type':'length','line':8,'value':24.0})
# R170 圆心到基线的垂距 = 68（定位蛇形高度；可选但强烈建议）
constraint_list.append({'type':'center_distance_y','arc':16,'baseline_y':0.0,'value':68.0})
# 半径硬约束
for idx, R in [(9,5.0),(10,5.0),
               (11,40.0),(12,130.0),(13,195.0),(14,40.0),
               (15,63.0),(16,170.0),(17,176.0),(18,40.0),
               (19,3.0),(20,3.0),(21,12.0)]:
    constraint_list.append({'type':'radius','arc':idx,'value':R})

# ================== 初值种子（非硬约束） ==================
# 右上/右下斜线给大致方向，避免零长度
for idx, seed in [(5,(260,180,300,200)), (6,(300,80,330,70))]:
    L = geom_objects[idx]
    for j,v in enumerate(seed):
        if np.isnan(L.params[j]): L.params[j] = float(v)

# 上链“上拱”、下链“下拱”的角度初值
for (idx,t1,t2) in [(11,20,50),(12,50,110),(13,110,160),(14,160,185),
                    (15,-20,10),(16,10,40),(17,40,80),(18,80,100)]:
    A = geom_objects[idx]
    if np.isnan(A.params[3]): A.params[3] = math.radians(t1)
    if np.isnan(A.params[4]): A.params[4] = math.radians(t2)
