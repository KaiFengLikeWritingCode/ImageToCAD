# v5: 
添加arc_endpoint_relation 

![./images/v5.png](images%2Fv5.png)

# v7: 
检查了约束设置，调整了

#constraint_list.append({'type':'point_distance_y', 'p1':11, 'which1':1, 'p2':15, 'which2':0, 'value':25.0})

constraint_list.append({'type':'point_distance_y', 'p1':15, 'which1':0, 'p2':11, 'which2':1, 'value':25.0})
![./images/v7.png](images%2Fv7.png)


# v6
1. 新思路 
- 通过AI根据设计图，给你一套有关中间弧线的初始化方案 
- 不要采用随机初始化，根据原先的弧进行优化  

2. 新思路2 
- 在一定基础上将上半部分和加班部分分开优化求解

![./images/v6.png](images%2Fv6.png)


