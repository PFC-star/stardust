import numpy as np
import random

# 定义加载时间和推理时间的函数
def load_time(M):
    return 5.57 * M ** 0.44

def inference_time(M):
    return 0.15 * M ** 0.73

# 定义load_time函数的反函数
def inverse_load_time(t):
    # 解方程 load_time(M) = t
    return (t / 5.57) ** (1 / 0.44)

# 定义总模型大小和设备数量
M_total = 2  # 设备3上的完整模型大小
n = 5  # n个设备

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

# 计算给定模型分配的总时间（无感恢复时间）
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

# 计算完全恢复时间，并输出详细过程
def calc_full_recovery_time(Mi, total_weights):
    # 完全恢复设备是最后一个设备 (n-1)
    last_device_model = Mi[-1]
    remaining_weights = total_weights - last_device_model

    # 计算前面所有设备的推理和通信时间总和
    available_time = sum(inference_time(Mi[i]) + comm_time for i in range(n - 1))

    # 模拟每轮的推理和通信过程，并计算最后一个设备逐步加载权重的时间
    rounds = 0
    load_accumulated = 0

    # 输出详细信息
    print(f"{'轮次':<5}{'加载时间 (s)':<20}{'加载的权重大小':<20}")

    while load_accumulated < remaining_weights:
        rounds += 1
        # 使用前面设备的推理和通信时间窗口逐步加载权重
        load_this_round = inverse_load_time(available_time)
        load_accumulated += load_this_round

        # 确保不超过剩余的权重
        if load_accumulated > remaining_weights:
            load_this_round -= (load_accumulated - remaining_weights)
            load_accumulated = remaining_weights

        # 输出当前轮次的加载信息
        print(f"{rounds:<5}{available_time:<20.4f}{load_this_round:<20.4f}")

    # 返回完全恢复的时间和完成恢复的轮数
    total_recovery_time = rounds * available_time
    return total_recovery_time, rounds

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

# 计算无感恢复时间
print(f"最佳子模型分配: {best_Mi}")
print(f"最小无感恢复时间: {best_time:.2f} 秒")
print_device_info(best_Mi)

# 计算并详细输出完全恢复时间
total_weights = M_total  # 假设总权重为 M_total
print("\n完全恢复过程：")
full_recovery_time, recovery_rounds = calc_full_recovery_time(best_Mi, total_weights)
print(f"\n完全恢复时间: {full_recovery_time:.2f} 秒, 在第 {recovery_rounds} 轮完成")
