import numpy as np
import math


class Arc:
    def __init__(self, cx=None, cy=None, r=None, theta1=None, theta2=None):
        # Store all parameters, None means unknown/optimizable
        self.params = np.array([cx, cy, r, theta1, theta2], dtype=float)
        self.fixed_mask = np.array([cx is not None, cy is not None, r is not None,
                                    theta1 is not None, theta2 is not None])

    def points(self):
        cx, cy, r, t1, t2 = self.params
        x1 = cx + r * np.cos(t1)
        y1 = cy + r * np.sin(t1)
        x2 = cx + r * np.cos(t2)
        y2 = cy + r * np.sin(t2)
        return (x1, y1), (x2, y2)

    def get_optimizable_params(self):
        """Return only the parameters that need to be optimized"""
        return self.params[~self.fixed_mask]

    def set_optimizable_params(self, values):
        """Set the optimizable parameters with solved values"""
        self.params[~self.fixed_mask] = values


def initialize_geometry_from_drawing():
    """根据图2的标注信息初始化所有几何对象"""
    geom_objects = {}

    # 基于图2的半径信息初始化弧线
    # 根据图2中的半径标注 R40.00, R170.00, R195.00, R220.00, R3.00, R4.00, R5.00, R63.00, R71.00
    radius_mapping = {
        # 小半径圆弧 (R3-R5)
        9: 3.00, 10: 3.00,  # R3.00
        19: 4.00, 20: 4.00,  # R4.00
        5: 5.00, 6: 5.00,  # R5.00

        # 中等半径圆弧
        7: 40.00, 8: 40.00,  # R40.00
        15: 63.00, 16: 63.00,  # R63.00
        17: 71.00, 18: 71.00,  # R71.00

        # 大半径圆弧
        11: 170.00, 12: 170.00,  # R170.00
        13: 195.00, 14: 195.00,  # R195.00
        1: 220.00, 2: 220.00,  # R220.00
    }

    # 基于图2的尺寸信息估算圆心位置
    # 主要参考尺寸: 420.00, 132.00, 178.00, 1370.00, 93.00, 370.00, 380.00, 1130.00等
    base_x, base_y = 0, 0  # 假设左下角为原点

    # 初始化所有弧线对象
    for arc_id in range(1, 21):
        r = radius_mapping.get(arc_id, 50.0)  # 默认半径50.0

        # 根据弧线在图中的大致位置估算圆心
        if arc_id in [1, 2, 11, 12]:  # 左侧大圆弧
            cx, cy = base_x + 200, base_y + 700
        elif arc_id in [13, 14]:  # 中间偏右圆弧
            cx, cy = base_x + 800, base_y + 600
        elif arc_id in [15, 16]:  # 右侧中等圆弧
            cx, cy = base_x + 1000, base_y + 500
        elif arc_id in [17, 18]:  # 右下角圆弧
            cx, cy = base_x + 1100, base_y + 300
        elif arc_id in [7, 8]:  # 上方小圆弧
            cx, cy = base_x + 400, base_y + 900
        else:  # 其他小圆弧
            cx, cy = base_x + 600, base_y + 400

        geom_objects[arc_id] = Arc(cx=cx, cy=cy, r=r)

    return geom_objects


def set_arc_angles(geom_objects):
    """根据图1的弧线连接关系和图2的角度标注设置弧线的起始和终止角度"""

    # 基于图2中的角度标注 (12°, 15°, 50°等) 和弧线连接关系
    # 弧线11: 从180°到270° (左下象限)
    arc11 = geom_objects[11]
    arc11.params[3] = math.radians(180)  # theta1
    arc11.params[4] = math.radians(270)  # theta2

    # 弧线12: 从180°到300°
    arc12 = geom_objects[12]
    arc12.params[3] = math.radians(180)
    arc12.params[4] = math.radians(300)

    # 弧线13: 从90°到180° (左上象限)
    arc13 = geom_objects[13]
    arc13.params[3] = math.radians(90)
    arc13.params[4] = math.radians(180)

    # 弧线14: 从270°到360° (右下象限)
    arc14 = geom_objects[14]
    arc14.params[3] = math.radians(270)
    arc14.params[4] = math.radians(360)

    # 弧线15: 从90°到180°
    arc15 = geom_objects[15]
    arc15.params[3] = math.radians(90)
    arc15.params[4] = math.radians(180)

    # 弧线16: 从270°到360°
    arc16 = geom_objects[16]
    arc16.params[3] = math.radians(270)
    arc16.params[4] = math.radians(360)

    # 弧线17: 从90°到180°
    arc17 = geom_objects[17]
    arc17.params[3] = math.radians(90)
    arc17.params[4] = math.radians(180)

    # 弧线18: 从0°到180° (上半圆)
    arc18 = geom_objects[18]
    arc18.params[3] = math.radians(0)
    arc18.params[4] = math.radians(180)

    # 为其他弧线设置合理的默认角度范围
    for idx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20]:
        arc = geom_objects[idx]
        # 根据弧线在图形中的大致位置设置角度范围
        if idx in [1, 2, 7, 8]:  # 上方弧线
            arc.params[3] = math.radians(180)
            arc.params[4] = math.radians(270)
        elif idx in [3, 4, 9, 10]:  # 下方弧线
            arc.params[3] = math.radians(90)
            arc.params[4] = math.radians(180)
        else:  # 中间弧线
            arc.params[3] = math.radians(135)
            arc.params[4] = math.radians(225)


def create_initial_guess():
    """创建完整的初始猜测"""
    geom_objects = initialize_geometry_from_drawing()
    set_arc_angles(geom_objects)

    # 创建初始参数向量
    x0 = []
    for arc_id in sorted(geom_objects.keys()):
        arc = geom_objects[arc_id]
        optimizable_params = arc.get_optimizable_params()
        x0.extend(optimizable_params)

    return np.array(x0), geom_objects


# 使用示例
if __name__ == "__main__":
    x0, geom_objects = create_initial_guess()
    print(f"初始参数向量长度: {len(x0)}")
    print("几何对象初始化完成:")

    for arc_id, arc in geom_objects.items():
        print(f"弧线 {arc_id}: 圆心({arc.params[0]:.1f}, {arc.params[1]:.1f}), "
              f"半径{arc.params[2]:.1f}, 角度[{math.degrees(arc.params[3]):.0f}°, "
              f"{math.degrees(arc.params[4]):.0f}°]")