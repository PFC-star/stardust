import numpy as np
import random
import matplotlib.pyplot as plt

# 定义加载时间和推理时间的函数
def load_time(M):
    return 5.57 * M ** 0.44

def inference_time(M):
    return 0.15 * M ** 0.73

# 定义总模型大小和设备数量
M_total = 1  # 设备3上的完整模型大小
n = 5 # n个设备

# 定义通信时间和阈值
comm_time = 0.1  # 假设通信时间为常数
threshold = 0.1  # 时间差异阈值

# 模拟退火优化算法
def simulated_annealing(M_total, n, iterations=5000, initial_temp=100, cooling_rate=0.99):
    # 随机初始化子模型分配，确保顺序性
    Mi = np.sort(np.random.dirichlet(np.ones(n), size=1)[0] * M_total)

    # 定义初始温度和最佳方案
    current_temp = initial_temp
    best_Mi = Mi
    best_time, stages = calc_total_time(best_Mi)

    # 退火过程
    for iteration in range(iterations):
        # 随机生成新的分配方案，确保顺序性
        new_Mi = np.sort(np.random.dirichlet(np.ones(n), size=1)[0] * M_total)

        # 计算新的总时间
        new_time, new_stages = calc_total_time(new_Mi)

        # 判断是否接受新方案
        if new_time < best_time or random.uniform(0, 1) < np.exp((best_time - new_time) / current_temp):
            best_Mi = new_Mi
            best_time = new_time
            stages = new_stages

        # 降低温度
        current_temp *= cooling_rate

    return best_Mi, best_time, stages


# 计算给定模型分配的总时间，并记录各阶段时间
def calc_total_time(Mi):
    total_times = []
    stages = []  # 用于记录每个设备的时间段信息

    # 第一个设备的加载、推理和通信都从0时刻开始
    load_start = 0
    load_end = load_time(Mi[0])
    infer_start = load_end
    infer_end = infer_start + inference_time(Mi[0])
    comm_start = infer_end
    comm_end = comm_start + comm_time

    total_times.append(comm_end)

    # 记录阶段时间
    stages.append({
        'load_start': load_start, 'load_end': load_end,
        'infer_start': infer_start, 'infer_end': infer_end,
        'comm_start': comm_start, 'comm_end': comm_end
    })

    # 依次计算其他设备的时间
    for i in range(1, n):
        # 当前设备的加载从time 0开始
        load_start = 0
        load_end = load_time(Mi[i])

        # 当前设备的推理必须在前一个设备通信结束后开始
        infer_start = max(stages[i - 1]['comm_end'], load_end)
        infer_end = infer_start + inference_time(Mi[i])
        comm_start = infer_end
        comm_end = comm_start + comm_time

        total_times.append(comm_end)

        # 记录阶段时间
        stages.append({
            'load_start': load_start, 'load_end': load_end,
            'infer_start': infer_start, 'infer_end': infer_end,
            'comm_start': comm_start, 'comm_end': comm_end
        })

    # 返回所有设备中最大的时间，表示流水线的无感恢复总时间
    return max(total_times), stages


# 打印每个设备的阶段时间信息
def print_stage_info(stages):
    print(f"{'设备':<5}{'加载开始 (s)':<15}{'加载结束 (s)':<15}{'推理开始 (s)':<15}{'推理结束 (s)':<15}{'通信开始 (s)':<15}{'通信结束 (s)':<15}")
    for i, stage in enumerate(stages):
        print(f"{i:<5}{stage['load_start']:<15.2f}{stage['load_end']:<15.2f}{stage['infer_start']:<15.2f}{stage['infer_end']:<15.2f}{stage['comm_start']:<15.2f}{stage['comm_end']:<15.2f}")


# 画出流水线图
def plot_pipeline(stages):
    fig, ax = plt.subplots(figsize=(10, n))

    for i, stage in enumerate(stages):
        # 加载阶段
        ax.broken_barh([(stage['load_start'], stage['load_end'] - stage['load_start'])], (i - 0.4, 0.8),
                       facecolors='tab:blue', label="laod" if i == 0 else "")
        # 推理阶段
        ax.broken_barh([(stage['infer_start'], stage['infer_end'] - stage['infer_start'])], (i - 0.4, 0.8),
                       facecolors='tab:green', label="infer" if i == 0 else "")
        # 通信阶段
        ax.broken_barh([(stage['comm_start'], stage['comm_end'] - stage['comm_start'])], (i - 0.4, 0.8),
                       facecolors='tab:red', label="commu" if i == 0 else "")

    ax.set_xlabel('time (s)')
    ax.set_ylabel('device')
    ax.set_yticks(range(n))
    ax.set_yticklabels([f'device {i}' for i in range(n)])
    ax.grid(True)
    ax.legend()
    plt.show()


# 运行优化并记录时间
best_Mi, best_time, stages = simulated_annealing(M_total, n)
print(f"最佳子模型分配: {best_Mi}")
print(f"最小无感恢复时间: {best_time:.2f} 秒")

# 打印每个设备的时间段信息
print_stage_info(stages)

# 画出流水线图
plot_pipeline(stages)
