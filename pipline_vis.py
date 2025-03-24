import matplotlib.pyplot as plt
import numpy as np


def load_time(M):
    return 5.57 * M ** 0.44


def inference_time(M):
    return 0.15 * M ** 0.73


# 示例M值
M_values = [0.1641392,  0.1699896 , 0.19018713, 0.22362748 ,0.25205659]

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))

# 当前y坐标
current_y = 0

for i, M in enumerate(M_values):
    # 计算load、inference、通信时间
    l_time = load_time(M)
    i_time = inference_time(M)
    comm_time = 0.1  # 固定通信时间

    # 绘制load time的矩形块（蓝色）
    ax.add_patch(plt.Rectangle((0, current_y), l_time, 1, color='blue', label='Load Time' if i == 0 else ""))

    # 绘制inference time的矩形块（橙色）
    ax.add_patch(
        plt.Rectangle((l_time, current_y), i_time, 1, color='orange', label='Inference Time' if i == 0 else ""))

    # 绘制通信时间矩形块（绿色）
    ax.add_patch(
        plt.Rectangle((l_time + i_time, current_y), comm_time, 1, color='green', label='Comm Time' if i == 0 else ""))

    # 更新y坐标
    current_y += 1

# # 绘制后续加载的蓝色矩形块
# ax.add_patch(plt.Rectangle((load_time(M_values[0]) + inference_time(M_values[0]), 0),
#                            load_time(sum(M_values) - M_values[0]), 1, color='blue',
#                            label='Load Time' if i == 0 else ""))

# 设置坐标轴和标签
ax.set_xlim(0, 8)
ax.set_ylim(0, len(M_values) + 1)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Stages (D)')
ax.set_title('Pipeline Schedule')
ax.legend()
ax.grid(True)

# 显示图像
plt.show()
