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



# 参数


#
# initial_module_arrangement=[ [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0, 0 ,0, 0 ],
#                              [1 ,1 ,1 ,1, 1, 1 ,1 ,1, 1, 1 ,1 ,1 ,1 ,1 ,1 ,1, 1 ,1 ,1 ,1 ,1 ,0 ,0 ,0, 1],
#                             [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,1, 1 ,1, 0 ],
#                              [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0, 0 ,0, 0 ],
#                              [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0, 0 ,0, 0 ]]


initial_module_arrangement=[ [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0, 0 ,0, 0 ],
                             [0 ,0 ,0 ,0, 0, 0 ,1 ,1, 1, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0],
                             [1 ,1 ,1 ,1 ,1 ,1, 0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0, 0 ,0, 0 ],
                             [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,1 ,1, 1 ,1 ,1 ,1 ,1, 1 ,1 ,1 ,1 ,1 ,1, 1 ,1, 1 ],
                             [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0, 0 ,0, 0 ]]



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