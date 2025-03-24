import numpy as np
from scipy.optimize import fsolve


# 给定的拟合参数和公式
def load_time(M):
    return 5.57 * M ** 0.44


def inference_time(M):
    return 0.15 * M ** 0.73


# 通信时间常数
C = 0.5  # 假设通信时间为0.5秒


# 递推关系的方程
def recursive_equation(M_next, M_current):
    return load_time(M_current) + inference_time(M_current) + C - load_time(M_next)


# 计算所有 M_i
def calculate_model_partitions(total_model_size, num_devices):
    model_sizes = []

    # 假设第一个设备的模型大小初始为 total_model_size / num_devices (可以根据需要调整)
    M1_initial = total_model_size / num_devices
    model_sizes.append(M1_initial)

    # 递推计算每个 M_i
    for i in range(1, num_devices):
        # 使用 fsolve 解决递推方程
        M_next = fsolve(recursive_equation, model_sizes[-1], args=(model_sizes[-1]))[0]
        model_sizes.append(M_next)

    return model_sizes


# 总模型大小和设备数量
total_model_size = 1.0  # 假设总模型大小为 1B
num_devices = 2  # 假设有2个设备

# 计算各设备上的模型大小
model_sizes = calculate_model_partitions(total_model_size, num_devices)

# 输出结果
for i, Mi in enumerate(model_sizes):
    print(f"Device {i + 1}: Model Size = {Mi:.4f} B")
