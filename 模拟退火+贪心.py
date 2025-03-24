import time

import numpy as np
import random
import matplotlib.pyplot as plt
from tabulate import tabulate
# import ace_tools as tools
import pandas as pd
import os
# 定义ping延迟和带宽数组
# 单位 秒 s

ping_latency_total = [
                        [0	,0.034866,0.017796	,0.076512,	0.030985],
                        [0.03257,	0,	0.027257	,0.079517,	0.018823],
                        [0.108798	,0.065485	,0,	0.064679	,0.043849],
                        [0.206233	,0.028255	,0.034938	,0,	0.02909],
                        [0.021318	,0.034842	,0.019245,	0.042098,	0]]

# MB ps


bandwidths_total = [
                        [float("inf")	,1.666069031,	1.81388855	,2.56729126	,4.906654358],
                        [0.312805176	,float("inf")	,1.200675964	,0.575065613,	0.590324402],
                        [0.994682312,	2.725601196	,float("inf")	,0.332832336,	0.495910645],
                        [0.772476196,	1.02519989	,1.168251038,	float("inf")	,1.210212708],
                        [2.743721008,	1.764297485	,1.543045044,	3.170013428,	float("inf")]]


# 单位 MB
tatal_memory_total = [5395.599999999999, 11161.5, 4134.2, 5332.599999999999, 5603.5]

available_memory_total = [2146.2, 0.4*7867.299999999999, 1964.1999999999998,2522.7999999999997, 0.8*3695.9999999999995]


flops_spped_total= np.array([17128238928,52008456659,22955103793,34432655689,18917814128])

# load_time_param = np.array([(3.5830231,0.99784825),
#                             (0.95286353,1.07230762),
#                             (1.44911418,1.6059872),
#                             (10.91643267,0.74477676),
#                              (1.14697389,1.59626346)])
load_time_param = np.array([(10.27,0.46),
                            (0.74,1.87),
                            (2.62,1.27),
                            (14.71,0.59),
                            (5.36,0.90)])
model_list= ["bloom560","bloom560-int8" ,"bloom1b7","bloom1b7-int8"]
model_dict = {"bloom560":{"FLOPs":12300000000,"memory":3072,"param":0.560},
              "bloom560-int8": {"FLOPs": 12300000000, "memory": 806,"param":0.560} ,
              "bloom7b1": {"FLOPs": 155510450560, "memory": 806,"param":7.1}}
model_name = "bloom7b1"

flop = model_dict[model_name]["FLOPs"]
memory = model_dict[model_name]["memory"]
param = model_dict[model_name]["param"]


# initial_module_arrangement=[
#                              [1 ,1 ,1 ,1, 1, 1 ,1 ,1, 1, 1 ,1 ,1 ,1 ,1 ,1 ,1, 1 ,1 ,1 ,1 ,1 ,0 ,0 ,0, 1],
#                             [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,1, 1 ,1, 0 ]]

#
# initial_module_arrangement=[
#                              [1 ,1 ,1 ,1, 1, 1 ,1 ,1, 1, 1 ,1 ,1 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 0],
#                             [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0, 0 ,1 ,1 ,1,1, 1 ,1 ,1 ,1 ,1 ,1, 1 ,1, 1 ]]









# 定义通信数据大小
data_size_kb = 20  # 20 KB = 160 千比特

# 定义加载时间和推理时间的函数
def load_time(M, device):
    param_ = load_time_param[device]
    a = param_[0]
    b = param_[1]
    return max(a* M ** b,0.5)

def inference_time(m,device):
    flop = m/M_total* FLOPs_allocation[faulty_device]
    inference_time_ = flop / flops_spped_total[device]
    # print("inference_time_:",inference_time_)
    return inference_time_
# 加入Memory_spped



# 计算设备之间的通信时间
def communication_time(i, j, data_size_kb=20):
    data_size_kilobits = data_size_kb/ 1024 # 转换为MB
    commu_time= ping_latency_total[i][j] + data_size_kilobits / bandwidths_total[i][j]
    return commu_time
# Simulated annealing function

def memory_constraint(Mi):
    ratio =np.array( [m / sum(Mi) for m in Mi])
    memory_occupy = np.array(memory) * ratio
    for index,device_index in enumerate(selected_device_index):
        if memory_occupy[index] > available_memory_total[device_index]:
            return False
    return True
def simulated_annealin_greedy_initial(M_total, n, iterations=1000, initial_temp=100, cooling_rate=0.95):
    # Initialize model allocation randomly
    # n = len(device_lst)
    while True:
        # 初始化分配，使用负载均衡策略生成初始解
        best_Mi = heuristic_initialization(
            M_total,
            n,
            np.array(flops_spped_total)[selected_device_index],  # 修复错误，确保传入的是整数索引
            np.array(available_memory_total)[selected_device_index],  # 修复错误
            memory
        )

        current_temp = initial_temp


        # print("M：",best_Mi)
        best_time, stages = calc_total_time(best_Mi)
        if memory_constraint(best_Mi):
            break
    # Annealing process
    for iteration in range(iterations):
        # Generate new allocation, sorted for fairness
        new_Mi = np.sort(np.random.dirichlet(np.ones(n), size=1)[0] * M_total)
        if memory_constraint(new_Mi):
            # Calculate total time
            new_time, new_stages = calc_total_time(new_Mi)

            # Accept the new allocation if it's better or by chance
            if new_time < best_time or random.uniform(0, 1) < np.exp((best_time - new_time) / current_temp):
                best_Mi = new_Mi
                best_time = new_time
                stages = new_stages

            # Cool down
            current_temp *= cooling_rate

    return best_Mi, best_time, stages
def simulated_annealing(M_total, n, iterations=1000, initial_temp=100, cooling_rate=0.95):
    # Initialize model allocation randomly
    # n = len(device_lst)
    while True:
        Mi = np.sort(np.random.dirichlet(np.ones(n), size=1)[0] * M_total)

        current_temp = initial_temp
        best_Mi = Mi

        # print("M：",best_Mi)
        best_time, stages = calc_total_time(best_Mi)
        if memory_constraint(best_Mi):
            break
    # Annealing process
    for iteration in range(iterations):
        # Generate new allocation, sorted for fairness
        new_Mi = np.sort(np.random.dirichlet(np.ones(n), size=1)[0] * M_total)
        if memory_constraint(new_Mi):
            # Calculate total time
            new_time, new_stages = calc_total_time(new_Mi)

            # Accept the new allocation if it's better or by chance
            if new_time < best_time or random.uniform(0, 1) < np.exp((best_time - new_time) / current_temp):
                best_Mi = new_Mi
                best_time = new_time
                stages = new_stages

            # Cool down
            current_temp *= cooling_rate

    return best_Mi, best_time, stages


def calc_total_time(Mi):

    total_times = []
    stages = []

    # First device timing (starts at t=0)
    load_start = 0
    load_end = load_time(Mi[0],selected_device_index[0])
    infer_start = load_end
    infer_end = infer_start + inference_time(Mi[0],selected_device_index[0])
    comm_start = infer_end
    comm_end = comm_start + communication_time(selected_device_index[0], selected_device_index[1])  # First device communicates with the next

    total_times.append(comm_end)

    # Record stage timing
    stages.append({
        'load_start': load_start, 'load_end': load_end,
        'infer_start': infer_start, 'infer_end': infer_end,
        'comm_start': comm_start, 'comm_end': comm_end
    })

    # Compute timing for all other devices
    for i in range(1, n):
        load_start = 0
        load_end = load_time(Mi[i],selected_device_index[i])
        infer_start = max(stages[i - 1]['comm_end'], load_end) # 上一轮通信结束或者本轮load结束之后才可以开启推理

        infer_end = infer_start + inference_time(Mi[i],selected_device_index[i])
        comm_start = infer_end
        if i==n-1:
            comm_end = comm_start + communication_time(selected_device_index[i], initial_device_index)  # Last device communicates with the first
        else:
            comm_end = comm_start + communication_time(selected_device_index[i], (selected_device_index[i] + 1) % n)  # Circular communication between devices

        total_times.append(comm_end)

        # Record stage timing
        stages.append({
            'load_start': load_start, 'load_end': load_end,
            'infer_start': infer_start, 'infer_end': infer_end,
            'comm_start': comm_start, 'comm_end': comm_end
        })

    # Return the maximum total time, representing the total recovery time
    return max(total_times), stages










def simulated_annealing_random(M_total, n):
    # Initialize model allocation randomly
    # n = len(device_lst)
    for i in range(100):
        Mi = np.sort(np.random.dirichlet(np.ones(n), size=1)[0] * M_total)


        best_Mi = Mi

        # print("M：",best_Mi)
        best_time, stages = calc_total_time(best_Mi)
        if memory_constraint(best_Mi):
            break




    return best_Mi, best_time, stages
def apply_greedy_adjustment(allocation, stages):
    """
    贪心调整策略：优化模型分配。
    目标是让后一个设备的加载阶段尽量利用前一个设备的推理和通信时间。
    """
    adjusted_allocation = allocation.copy()

    for i in range(1, len(allocation)):
        # 计算前一个设备的通信结束时间
        prev_comm_end = stages[i - 1]["comm_end"]

        # 当前设备的加载时间
        current_load_time = load_time(adjusted_allocation[i], selected_device_index[i])

        # 如果加载时间能够覆盖前一个设备的通信和推理时间，则微调分配
        if current_load_time >= prev_comm_end:
            adjusted_allocation[i] *= 1.1  # 当前设备增加分配
            adjusted_allocation[i - 1] *= 0.9  # 前一个设备减少分配

    # 确保总分配等于模型总大小
    adjusted_allocation = adjusted_allocation / np.sum(adjusted_allocation) * M_total
    return adjusted_allocation
def simulated_annealing_with_greedy(M_total, n, iterations=1000, initial_temp=100, cooling_rate=0.95):
    """
    模拟退火算法，结合贪心策略调整分配。
    """


    # 初始化随机分配
    # while True:
    #     Mi = np.sort(np.random.dirichlet(np.ones(n), size=1)[0] * M_total)
    #     current_temp = initial_temp
    #     best_Mi = Mi
    #
    #     # 计算当前方案的目标函数
    #     best_time, stages = calc_total_time(best_Mi)
    #     if memory_constraint(best_Mi):
    #         break

    # 初始化分配，使用负载均衡策略生成初始解
    best_Mi = heuristic_initialization(
        M_total,
        n,
        np.array(flops_spped_total)[selected_device_index],  # 修复错误，确保传入的是整数索引
        np.array(available_memory_total)[selected_device_index],  # 修复错误
        memory
    )
    current_temp = initial_temp

    # 计算当前方案的目标函数
    best_time, stages = calc_total_time(best_Mi)

    # 模拟退火过程
    for iteration in range(iterations):
        # 随机生成新解
        new_Mi = np.sort(np.random.dirichlet(np.ones(n), size=1)[0] * M_total)

        # 应用贪心调整策略
        new_Mi = apply_greedy_adjustment(new_Mi, stages)

        # 检查新解是否满足内存约束
        if memory_constraint(new_Mi):
            # 计算新解的目标函数值
            new_time, new_stages = calc_total_time(new_Mi)

            # 使用 Metropolis 准则决定是否接受新解
            if new_time < best_time or random.uniform(0, 1) < np.exp((best_time - new_time) / current_temp):
                best_Mi = new_Mi
                best_time = new_time
                stages = new_stages

        # 退火降温
        current_temp *= cooling_rate

    return best_Mi, best_time, stages
# Calculating the total time for all devices

def load_balanced_initialization(M_total, n, flops_speeds, available_memories, memory_per_unit):
    """
    基于负载均衡分配的快速初始解
    """
    sorted_indices = np.argsort(-flops_speeds)  # 按 FLOPs 性能降序排列设备
    allocation = np.zeros(n)

    for idx in sorted_indices:
        # 当前设备能分配的最大模型大小
        max_memory = available_memories[idx] / memory_per_unit
        allocation[idx] = min(max_memory, M_total)  # 分配尽可能大的模型
        M_total -= allocation[idx]  # 剩余模型大小
        if M_total <= 0:
            break  # 如果模型分配完成，退出

    return np.sort(allocation)

def heuristic_initialization(M_total, n, flops_speeds, available_memories, memory_per_unit):
    """
    基于 FLOPs 和内存限制的比例快速生成初始解
    """
    # 按 FLOPs 比例分配初步分配值
    flops_ratios = flops_speeds / np.sum(flops_speeds)
    initial_allocation = flops_ratios * M_total

    # 检查内存约束
    for i in range(n):
        max_memory = available_memories[i] / memory_per_unit
        if initial_allocation[i] > max_memory:  # 如果超出内存限制
            initial_allocation[i] = max_memory  # 限制分配
            # 重新调整剩余模型大小
            remaining = M_total - np.sum(initial_allocation)
            if remaining > 0:
                remaining_ratios = flops_speeds / np.sum(flops_speeds)
                initial_allocation += remaining_ratios * remaining

    return np.sort(initial_allocation)
def randomized_heuristic_initialization(M_total, n, flops_speeds, available_memories, memory_per_unit, perturbation=0.1):
    """
    基于启发式规则的随机扰动初始解
    """
    # 获取启发式初始解
    initial_allocation = heuristic_initialization(M_total, n, flops_speeds, available_memories, memory_per_unit)

    # 加入随机扰动
    noise = np.random.uniform(-perturbation, perturbation, size=n)
    initial_allocation += initial_allocation * noise

    # 确保分配合法性
    initial_allocation = np.maximum(initial_allocation, 0)  # 确保非负
    initial_allocation = initial_allocation / np.sum(initial_allocation) * M_total  # 归一化到总模型大小

    # 检查内存约束
    for i in range(n):
        max_memory = available_memories[i] / memory_per_unit
        if initial_allocation[i] > max_memory:
            initial_allocation[i] = max_memory

    return np.sort(initial_allocation)
def greedy_initialization(M_total, n, flops_speeds, available_memories, memory_per_unit):
    """
    局部贪心法生成初始解
    """
    allocation = np.zeros(n)

    for i in range(n):
        # 每次分配剩余模型大小中按 FLOPs 比例调整
        max_memory = available_memories[i] / memory_per_unit
        device_allocation = min(max_memory, M_total / (n - i))
        allocation[i] = device_allocation
        M_total -= device_allocation
        if M_total <= 0:
            break

    return np.sort(allocation)

def heuristic_init():
    # 基于FLOPs比例分配初始解
    flops_ratios = flops_spped_total[selected_device_index] / np.sum(flops_spped_total[selected_device_index])
    Mi = np.sort(flops_ratios * M_total)
    # 调整到满足内存约束
    for i in range(n):
        max_mem = available_memory_total[selected_device_index[i]] / memory
        Mi[i] = min(Mi[i], max_mem)
    return Mi / np.sum(Mi) * M_total

def print_stage_info(stages):
    # 创建数据表格
    table = []
    headers = ['设备', '加载开始 (s)', '加载结束 (s)', '推理开始 (s)', '推理结束 (s)', '通信开始 (s)', '通信结束 (s)']

    # 填充数据
    for i, stage in enumerate(stages):
        row = [
            i,
            stage['load_start'],
            stage['load_end'],
            stage['infer_start'],
            stage['infer_end'],
            stage['comm_start'],
            stage['comm_end']
        ]
        table.append(row)

    # 使用 tabulate 格式化输出表格
    print(tabulate(table, headers=headers, floatfmt=".2f", tablefmt="grid"))


def plot_pipeline(results_plot):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,4*n), sharex=True)  # 横向排列两个子图
    for index, ax in enumerate(axes):
        stages, (min_index,last_load_time), recovery_time_single_device_lst = results_plot[index]


        for i, stage in enumerate(stages):
            # 加载阶段
            ax.broken_barh([(stage['load_start'], stage['load_end'] - stage['load_start'])], (i - 0.4, 0.8),
                           facecolors='tab:blue', label="load" if i == 0 else "")
            # 推理阶段
            ax.broken_barh([(stage['infer_start'], stage['infer_end'] - stage['infer_start'])], (i - 0.4, 0.8),
                           facecolors='tab:green', label="infer" if i == 0 else "")
            # 通信阶段
            ax.broken_barh([(stage['comm_start'], stage['comm_end'] - stage['comm_start'])], (i - 0.4, 0.8),
                           facecolors='tab:red', label="commu" if i == 0 else "")
            if selected_device_index.index(min_index)== i:
                # 加载阶段
                ax.broken_barh([(stage['comm_end'],last_load_time)], (i - 0.4, 0.8),
                               facecolors='tab:cyan', label="load remain" )
        # 单设备恢复
        for index,single_device in enumerate(recovery_time_single_device_lst):
              ax.broken_barh([(0, single_device)], (i +(index+1) - 0.4, 0.8),
                       facecolors='tab:grey', label="single device" if i == 0 else "")
        ax.set_xlabel('time (s)')
        ax.set_ylabel('device')
        ax.set_yticks(range(2*n))
        ax.set_yticklabels([f'device {i}' for i in selected_device_index+selected_device_index])
        ax.grid(True)
        ax.legend()
    plt.tight_layout()  # 调整子图之间的间距
    plt.show()

# 单手机进行恢复

def recovery_time_single_device(selected_device_index):
    for i in  selected_device_index:
        load_time_single_device = load_time(M_total,i)
        inference_time_single_device = inference_time(M_total,i)
        commu_time_single_device = communication_time(i,initial_device_index)
        recovery_time_single_device = load_time_single_device + inference_time_single_device + commu_time_single_device
        # print(f"单设备:{i} 恢复时间: {recovery_time_single_device:.2f} 秒")
        print(recovery_time_single_device)

        recovery_time_single_device_lst.append(recovery_time_single_device)
def last_load(best_Mi,stages):
    recovery_time_temp_lst = []
    recovery_time_lst = []
    last_load_time_lst = []
    for index,Mi in enumerate(best_Mi):
         last_M = M_total - Mi
         last_load_time = load_time(last_M,selected_device_index[index])
         recovery_time = last_load_time + stages[index]['comm_end']
         recovery_time_lst.append(stages[index]['comm_end'])
         recovery_time_temp_lst.append(recovery_time)
         last_load_time_lst.append(last_load_time)
    min_recovery_time_temp = min(recovery_time_temp_lst)
    min_index = recovery_time_temp_lst.index(min_recovery_time_temp)
    recovery_time_lst.append(min_recovery_time_temp)
    min_recovery_time = max(recovery_time_lst)
    return min_recovery_time,selected_device_index[min_index], last_load_time_lst[min_index]


def genetic_initialization(M_total, n, population_size=50, generations=10):
    """
    使用遗传算法生成模拟退火的初始解。
    """
    # 初始化种群
    population = [np.sort(np.random.dirichlet(np.ones(n), size=1)[0] * M_total) for _ in range(population_size)]

    # 评估函数
    def fitness(Mi):
        time, _ = calc_total_time(Mi)
        return -time  # 最小化时间

    for _ in range(generations):
        # 选择：按适应度排序，保留前50%
        population = sorted(population, key=fitness)[:population_size // 2]

        # 交叉：随机选择两个父母，生成子代
        offspring = []
        for _ in range(population_size // 2):
            parents = random.sample(population, 2)
            cross_point = random.randint(1, n - 1)
            child = np.concatenate((parents[0][:cross_point], parents[1][cross_point:]))
            offspring.append(np.sort(child / np.sum(child) * M_total))  # 保证总分配为M_total
        population += offspring

        # 变异：随机调整部分解
        for i in range(len(population)):
            if random.random() < 0.1:  # 10%的变异概率
                mutation = np.random.dirichlet(np.ones(n), size=1)[0] * 0.1 * M_total
                population[i] = np.sort((population[i] + mutation) / np.sum(population[i] + mutation) * M_total)

    # 返回最优解
    return population[0]


def dynamic_simulated_annealing(M_total, n, initial_temp=1000, min_temp=1, max_iterations=100):
    """
    动态调整冷却速率的模拟退火算法。
    """
    # 初始化解
    # best_Mi = genetic_initialization(M_total, n)  # 使用遗传算法生成初始解
    best_Mi = heuristic_initialization(
        M_total,
        n,
        np.array(flops_spped_total)[selected_device_index],  # 修复错误，确保传入的是整数索引
        np.array(available_memory_total)[selected_device_index],  # 修复错误
        memory
    )
    current_temp = initial_temp
    cooling_rate = 0.95  # 初始冷却速率
    best_time, stages = calc_total_time(best_Mi)

    # 模拟退火过程
    for iteration in range(max_iterations):
        # 生成新解
        new_Mi = np.sort(np.random.dirichlet(np.ones(n), size=1)[0] * M_total)

        # 可选：在前10%迭代中加入贪心调整
        if iteration < max_iterations * 0.1:
            new_Mi = apply_greedy_adjustment(new_Mi, stages)

        # 检查约束
        if memory_constraint(new_Mi):
            new_time, new_stages = calc_total_time(new_Mi)
            if new_time < best_time or random.uniform(0, 1) < np.exp((best_time - new_time) / current_temp):
                best_Mi = new_Mi
                best_time = new_time
                stages = new_stages

        # 动态调整冷却速率
        current_temp = max(min_temp, current_temp * cooling_rate)
        cooling_rate = max(0.9, cooling_rate * 0.995)  # 冷却速率逐渐减小

    return best_Mi, best_time, stages

# 参数


#
# initial_module_arrangement=[ [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0, 0 ,0, 0 ],
#                              [1 ,1 ,1 ,1, 1, 1 ,1 ,1, 1, 1 ,1 ,1 ,1 ,1 ,1 ,1, 1 ,1 ,1 ,1 ,1 ,0 ,0 ,0, 1],
#                             [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,1, 1 ,1, 0 ],
#                              [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0, 0 ,0, 0 ],
#                              [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0, 0 ,0, 0 ]]


initial_module_arrangement=[ [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0, 0 ,0, 0 ],
                             [1 ,0 ,0 ,0, 0, 0 ,1 ,1, 1, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0],
                             [0 ,1 ,1 ,1 ,1 ,1, 0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0, 0 ,0, 0 ],
                             [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,1 ,1, 1 ,1 ,1 ,1 ,1, 1 ,1 ,1 ,1 ,1 ,1, 1 ,1, 0 ],
                             [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0, 0 ,0, 1 ]]

param_dict_1 = {"selected_device_index":[0,3],
              "selected_single_device_index":0,
              "initial_device_index":2,
              "faulty_device":1}
param_dict_2 = {"selected_device_index":[3,4],
              "selected_single_device_index":0,
              "initial_device_index":2,
              "faulty_device":1}
param_dict_3 = {"selected_device_index":[0,4],
              "selected_single_device_index":0,
              "initial_device_index":2,
              "faulty_device":1}

param_dict_4 = {"selected_device_index":[0,3,4],
              "selected_single_device_index":0,
              "initial_device_index":1,
              "faulty_device":1}
param_dict_5 = {"selected_device_index":[0,3],
              "selected_single_device_index":0,
              "initial_device_index":1,
              "faulty_device":2}
param_dict_6 = {"selected_device_index":[3,4],
              "selected_single_device_index":0,
              "initial_device_index":1,
              "faulty_device":2}
param_dict_7 = {"selected_device_index":[0,4],
              "selected_single_device_index":0,
              "initial_device_index":1,
              "faulty_device":2}


param_dict_8 = {"selected_device_index":[1,2,0,4,3],
              "selected_single_device_index":0,
              "initial_device_index":1,
              "faulty_device":1}
param_list = [param_dict_1,param_dict_2,param_dict_3,param_dict_4,param_dict_5,param_dict_6,param_dict_7,param_dict_8]

param_allocation = [ sum(i)/len(i)* param for i in initial_module_arrangement]
FLOPs_allocation = [ sum(i)/len(i)* flop for i in initial_module_arrangement]


results_out = []

# for param_dict in param_list[3]:
for i in range (1):
    param_dict = param_list[7]
    results = []
    results_plot = []
    recovery_time_single_device_lst = []
    selected_device_index = param_dict["selected_device_index"]
    selected_single_device_index = param_dict["selected_single_device_index"]
    initial_device_index = param_dict["initial_device_index"]
    faulty_device = param_dict["faulty_device"]
    print("selected_device_index:",selected_device_index)
    print("selected_single_device_index:",selected_single_device_index)
    print("initial_device_index:",initial_device_index)
    print("faulty_device:",faulty_device)

    # M_total = param_allocation[faulty_device]
    # M_total = sum(param_allocation)
    M_total = param_allocation[faulty_device]

    print("\ndevice_allocation:", param_allocation)
    print("faulty_device:", faulty_device)
    print("M_total:", M_total)

    # ping_latency =np.array(   [ping_latency_total[i].tolist() for i in selected_device_index])

    # # bandwidths =np.array( [bandwidths_total[i].tolist()  for i in selected_device_index])
    # tatal_memory = np.array( [tatal_memory_total[i]   for i in selected_device_index])
    # available_memory =  [available_memory_total[i]   for i in selected_device_index]
    # flops_spped =  [flops_spped_total[i]  for i in selected_device_index]
    device_lst = ["device" + str(i) for i in selected_device_index]
    print("device_lst:", device_lst)

    # 定义总模型大小和设备数量
    # M_total = 1 # 设备3上的完整模型大小
    n = len(selected_device_index)  # n个设备

    recovery_time_single_device(selected_device_index)

    # 运行优化并记录时间


    # # 打印每个设备的时间段信息
    # print_stage_info(stages)
    startime = time.time()

    best_Mi, best_time, stages =  simulated_annealin_greedy_initial(M_total, n)

    min_recovery_time, min_index, last_load_time = last_load(best_Mi, stages)
    last_load_time_tuple = (min_index, last_load_time)

    endtime = time.time()
    print("solve time greedy_initial:", endtime - startime, "   best time:", best_time)

    startime = time.time()
    best_Mi, best_time, stages = simulated_annealing_with_greedy(M_total, n)

    min_recovery_time, min_index, last_load_time = last_load(best_Mi, stages)
    last_load_time_tuple = (min_index, last_load_time)
    endtime = time.time()
    print("solve time greedy:", endtime - startime,"   best time:",best_time)

    startime = time.time()

    best_Mi, best_time, stages = dynamic_simulated_annealing(M_total, n)
    min_recovery_time, min_index, last_load_time = last_load(best_Mi, stages)
    last_load_time_tuple = (min_index, last_load_time)
    endtime = time.time()
    print("solve time dynamic:", endtime - startime,"   best time:",best_time)

    # 多手机最优恢复（Ours)
    startime = time.time()

    best_Mi, best_time, stages = simulated_annealing(M_total, n,iterations=1000)

    min_recovery_time, min_index, last_load_time = last_load(best_Mi, stages)
    last_load_time_tuple = (min_index, last_load_time)

    endtime = time.time()
    print("solve time initial:", endtime - startime, "   best time:", best_time)

    # print(f"最佳子模型分配: {best_Mi}")
    # print(f"最小无感恢复时间: {best_time:.2f} 秒")
    # print(f"最完全恢复时间: {min_recovery_time:.2f} 秒")
    results_out.append([best_Mi, best_time, min_recovery_time, min_index])
    results_plot.append([stages, last_load_time_tuple, recovery_time_single_device_lst])


    # startime = time.time()
    #
    # best_Mi, best_time, stages =  simulated_annealing_opi(M_total, n, iterations=1000)
    #
    # min_recovery_time, min_index, last_load_time = last_load(best_Mi, stages)
    # last_load_time_tuple = (min_index, last_load_time)
    #
    # endtime = time.time()
    # print("solve time opi:", endtime - startime, "   best time:", best_time)

    # 多手机随机恢复
    # 均分

    for i in range(30):
        best_Mi, best_time, stages = simulated_annealing_random(M_total, n)
        min_recovery_time, min_index, last_load_time = last_load(best_Mi, stages)
        last_load_time_tuple = (min_index, last_load_time)
        # 记录结果

        results_plot.append([stages, last_load_time_tuple, recovery_time_single_device_lst])
        results.append([best_Mi, best_time, min_recovery_time])

    # 画出流水线图
    plot_pipeline(results_plot)

    # 转换为 DataFrame
    df_results = pd.DataFrame(results, columns=["子模型分配", "无感恢复时间 (秒)", "完全恢复时间 (秒)"])
    # 计算均值并添加到表格
    mean_times = df_results[["无感恢复时间 (秒)", "完全恢复时间 (秒)"]].mean()
    mean_times["子模型分配"] = "均值"
    df_results = pd.concat([df_results, pd.DataFrame([mean_times], columns=df_results.columns)], ignore_index=True)
    # tools.display_dataframe_to_user(name="模拟退火结果", dataframe=df_results)
    # print(df_results)
    # print(f"最佳子模型分配: {best_Mi}")
    # print(f"无感恢复时间: {df_results['无感恢复时间 (秒)'].iloc[-1]:.2f} 秒")
    # print(f"完全恢复时间: {df_results['完全恢复时间 (秒)'].iloc[-1]:.2f} 秒")
    recovery_time_single_device([0])
    # break

    results_out.append(
        [[], df_results['无感恢复时间 (秒)'].iloc[-1], df_results['完全恢复时间 (秒)'].iloc[-1], min_index])

    df_results_out = pd.DataFrame(results_out,
                                  columns=["子模型分配", "无感恢复时间 (秒)", "完全恢复时间 (秒)", "二次加载选择设备"])



    # Output to an Excel file
    os.makedirs( "results",exist_ok=True)
    excel_file_path = 'results/df_results_out_transposed.csv'
    df_results_out.T.to_csv(excel_file_path, header=False)
    # print(df_results_out)