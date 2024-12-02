import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy
import numpy as np

# Define the figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Define the colors for the lanes and lines
road_color = '#333333'
line_color = '#FFFFFF'

# Define lane width and road width
lane_width = 3.75 
road_width = 2 * lane_width * 2  # Two lanes in each direction

# Draw the vertical road
ax.add_patch(patches.Rectangle((-lane_width * 2 + 10, -30), 4 * lane_width, 60, color=road_color))
# Draw the horizontal road
ax.add_patch(patches.Rectangle((-40, -lane_width * 2), 80, 4 * lane_width, color=road_color))

# Draw dashed lane dividers for vertical lanes
for i in range(3):
    if i % 2 == 1:
        plt.plot([-40, -2 * lane_width + 10], 
                 [0, 0], color=line_color, linewidth=2)
        plt.plot([2 * lane_width + 10, 80],
                 [0, 0], color=line_color, linewidth=2)
    else:
        plt.plot([-40, -2 * lane_width + 10], 
                 [(i - 1) * lane_width, (i - 1) * lane_width], 
                 linestyle='--', color=line_color, linewidth=2)
        plt.plot([2 * lane_width + 10, 80], 
                 [(i - 1) * lane_width, (i - 1) * lane_width], 
                 linestyle='--', color=line_color, linewidth=2)
for i in range(3):
    if i % 2 == 1:
        plt.plot([10, 10], 
                 [-30, -2 * lane_width], color=line_color, linewidth=2)
        plt.plot([10, 10],
                 [2 * lane_width, 30], color=line_color, linewidth=2)
    else:
        plt.plot([10 + (i - 1) * lane_width, 10 + (i - 1) * lane_width], 
                 [-30, -2 * lane_width],
                 linestyle='--', color=line_color, linewidth=2)
        plt.plot([10 + (i - 1) * lane_width, 10 + (i - 1) * lane_width], 
                 [2 * lane_width, 30], 
                 linestyle='--', color=line_color, linewidth=2)


plt.plot([-2 * lane_width + 10, -2 * lane_width + 10], 
         [-2 * lane_width, 2 * lane_width], color=line_color, linewidth=2)
plt.plot([2 * lane_width + 10, 2 * lane_width + 10], 
         [-2 * lane_width, 2 * lane_width], color=line_color, linewidth=2)
plt.plot([-2 * lane_width + 10, 2 * lane_width + 10], 
         [-2 * lane_width, -2 * lane_width], color=line_color, linewidth=2)
plt.plot([-2 * lane_width + 10, 2 * lane_width + 10], 
         [2 * lane_width, 2 * lane_width], color=line_color, linewidth=2)

# Ego vehicle 
center_x, center_y, length, width = -20, -lane_width * 1.5, 5, 2.5
x_left, y_bottom = center_x - length / 2, center_y - width / 2
rect = patches.Rectangle((x_left, y_bottom), length, width, edgecolor='blue', facecolor='blue')
ax.add_patch(rect)

center = [center_x, center_y, 5, 0]
root = [center_x, center_y, 5, 0]
path, res = [center], []

# 深度限制和分支优化
def dfs(node, depth, path):
    if depth == 10:  # 将深度减少到10
        res.append(deepcopy(path))
        return
    
    x, y, vel, heading = node
    for delta in np.arange(-0.3, 0.6, 0.3):  # 降低分支数量
        new_x = x + vel * np.cos(heading) * 0.4
        new_y = y + vel * np.sin(heading) * 0.4
        new_heading = heading + vel * np.tan(delta) / 3.0 * 0.4
        new_vel = vel
        new_node = [new_x, new_y, new_vel, new_heading]
        dfs(new_node, depth + 1, path + [new_node])

dfs(root, 0, path)

# 仅绘制一部分路径
# 随机选择10000条路径
import random
random.shuffle(res)
for p in res[:50000]:
    x = [point[0] for point in p]
    y = [point[1] for point in p]
    plt.plot(x, y, color='gray', linewidth=0.5)

# plot some other vehicles
center_x_1, center_y_1, length_1, width_1 = -10, -lane_width * 0.5, 5, 2.5
x_left_1, y_bottom_1 = center_x_1 - length_1 / 2, center_y_1 - width_1 / 2
rect_1 = patches.Rectangle((x_left_1, y_bottom_1), length_1, width_1, edgecolor='red', facecolor='red')
ax.add_patch(rect_1)

center_x_2, center_y_2, length_2, width_2 = -30, -lane_width * 0.5, 5, 2.5
x_left_2, y_bottom_2 = center_x_2 - length_2 / 2, center_y_2 - width_2 / 2
rect_2 = patches.Rectangle((x_left_2, y_bottom_2), length_2, width_2, edgecolor='red', facecolor='red')
ax.add_patch(rect_2)



ax.set_aspect('equal', 'box')
ax.axis('off')  # Hide the axes
# Show the intersection plot
plt.savefig("sample_a_delta.png",dpi=600)