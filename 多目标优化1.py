import numpy as np
import random


# 定义加载时间和推理时间的函数
def load_time(M):
    return 5.57 * M ** 0.44


def inference_time(M):
    return 0.15 * M ** 0.73


# 定义总模型大小和设备数量
M_total = 1  # 设备3上的完整模型大小
n =9 # n个设备

# 定义通信时间和阈值
comm_time = 0.1  # 假设通信时间为常数
threshold = 0.5  # 时间差异阈值


# 模拟退火优化算法
def simulated_annealing(M_total, n, iterations=5000, initial_temp=100, cooling_rate=0.99):
    # 随机初始化子模型分配，确保顺序性
    Mi = np.sort(np.random.dirichlet(np.ones(n), size=1)[0] * M_total)

    # 定义初始温度和最佳方案
    current_temp = initial_temp
    best_Mi = Mi
    best_time = calc_total_time(best_Mi)

    # 退火过程
    for iteration in range(iterations):
        # 随机生成新的分配方案，确保顺序性
        new_Mi = np.sort(np.random.dirichlet(np.ones(n), size=1)[0] * M_total)

        # 计算新的总时间
        new_time = calc_total_time(new_Mi)

        # 判断是否接受新方案
        if new_time < best_time or random.uniform(0, 1) < np.exp((best_time - new_time) / current_temp):
            best_Mi = new_Mi
            best_time = new_time

        # 降低温度
        current_temp *= cooling_rate

    return best_Mi, best_time


# 计算给定模型分配的总时间
def calc_total_time(Mi):
    total_times = []

    # 初始化第一个设备的时间（加载 + 推理 + 通信）
    time_i = load_time(Mi[0]) + inference_time(Mi[0]) + comm_time
    total_times.append(time_i)

    # 上一个设备的结束时间（推理 + 通信）
    previous_end_time = inference_time(Mi[0]) + comm_time

    # 依次计算其他设备的时间，并添加约束
    for i in range(1, n):
        load_diff = load_time(Mi[i]) - load_time(Mi[i - 1])
        inference_time_i = inference_time(Mi[i])
        time_diff = abs(load_diff - inference_time(Mi[i - 1]) - comm_time)

        # 如果违反约束，返回无穷大，表示该方案不可行
        if time_diff > threshold:
            return float('inf')

        # 当前设备的推理必须在前一个设备的推理和通信结束之后开始
        start_time = max(previous_end_time, load_time(Mi[i]))

        # 计算当前设备的总时间：加载 + 推理 + 通信
        time_i = start_time + inference_time_i + comm_time
        total_times.append(time_i)

        # 更新当前设备的结束时间
        previous_end_time = start_time + inference_time_i + comm_time

    # 返回所有设备中最大的时间，表示流水线的无感恢复总时间
    return max(total_times)


def print_device_info(Mi):
    # 打印表头
    print(f"{'设备':<5}{'模型大小':<15}{'加载时间 (s)':<15}{'推理时间 (s)':<15}")

    # 遍历每个设备上的模型大小，计算并打印加载时间和推理时间
    for i in range(len(Mi)):
        model_size = Mi[i]
        load_t = load_time(model_size)
        inference_t = inference_time(model_size)

        # 打印设备编号、模型大小、加载时间、推理时间
        print(f"{i:<5}{model_size:<15.4f}{load_t:<15.4f}{inference_t:<15.4f}")


# 运行优化
best_Mi, best_time = simulated_annealing(M_total, n)
print(f"最佳子模型分配: {best_Mi}")
print(f"最小无感恢复时间: {best_time:.2f} 秒")
print_device_info(best_Mi)
