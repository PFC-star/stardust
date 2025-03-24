import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# 定义ping延迟和带宽数组（单位：秒、MB）
ping_latency_total = [
    [0, 0.034866, 0.017796, 0.076512, 0.030985],
    [0.03257, 0, 0.027257, 0.079517, 0.018823],
    [0.108798, 0.065485, 0, 0.064679, 0.043849],
    [0.206233, 0.028255, 0.034938, 0, 0.02909],
    [0.021318, 0.034842, 0.019245, 0.042098, 0]
]

# 带宽数组（单位：MB/s）
bandwidths_total = [
    [float("inf"), 1.666069031, 1.81388855, 2.56729126, 4.906654358],
    [0.312805176, float("inf"), 1.200675964, 0.575065613, 0.590324402],
    [0.994682312, 2.725601196, float("inf"), 0.332832336, 0.280007935],
    [0.734411507, 0.634801417, 0.433241694, float("inf"), 0.703819996],
    [1.302121697, 1.801237821, 0.502415113, 0.52142342, float("inf")]
]


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

def simulated_annealing(M_total, n, iterations=10000, initial_temp=100, cooling_rate=0.95):
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


# Calculating the total time for all devices
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


param_dict_8 = {"selected_device_index":[0,3,4],
              "selected_single_device_index":0,
              "initial_device_index":1,
              "faulty_device":2}
param_list = [param_dict_1,param_dict_2,param_dict_3,param_dict_4,param_dict_5,param_dict_6,param_dict_7,param_dict_8]

param_allocation = [ sum(i)/len(i)* param for i in initial_module_arrangement]
FLOPs_allocation = [ sum(i)/len(i)* flop for i in initial_module_arrangement]


results_out = []

for param_dict in param_list:
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
    M_total = param_allocation[4]

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

    # 多手机最优恢复（Ours)
    best_Mi, best_time, stages = simulated_annealing(M_total, n)

    min_recovery_time, min_index, last_load_time = last_load(best_Mi, stages)
    last_load_time_tuple = (min_index, last_load_time)
    print("last_load_time:",last_load_time)
    # print(f"最佳子模型分配: {best_Mi}")
    # print(f"最小无感恢复时间: {best_time:.2f} 秒")
    # print(f"最完全恢复时间: {min_recovery_time:.2f} 秒")
    results_out.append([best_Mi, best_time, min_recovery_time, min_index])
    results_plot.append([stages, last_load_time_tuple, recovery_time_single_device_lst])
    # # 打印每个设备的时间段信息
    # print_stage_info(stages)

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
    break

    results_out.append(
        [[], df_results['无感恢复时间 (秒)'].iloc[-1], df_results['完全恢复时间 (秒)'].iloc[-1], min_index])

    df_results_out = pd.DataFrame(results_out,
                                  columns=["子模型分配", "无感恢复时间 (秒)", "完全恢复时间 (秒)", "二次加载选择设备"])
print(df_results_out)


# Output to an Excel file
os.makedirs( "results",exist_ok=True)
excel_file_path = 'results/df_results_out_transposed.csv'
df_results_out.T.to_csv(excel_file_path, header=False)

# 模拟每个手机的状态，包括性能、积分等
class Phone:
    def __init__(self, id, performance, available_time):
        self.id = id
        self.performance = performance  # 性能（计算能力，越大越快）
        self.available_time = available_time  # 手机空闲时间，单位为秒
        self.is_free = True
        self.used_time = 0  # 用于记录手机使用的时间

    def use(self, task):
        if self.is_free:
            self.is_free = False
            processing_time = task.complexity / self.performance  # 根据任务复杂度与手机性能来计算处理时间
            self.used_time = processing_time
            return processing_time
        return None

    def free_up(self):
        self.is_free = True
        self.used_time = 0

# 用户类，表示每个用户的手机、积分等
class User:
    def __init__(self, id, phone, initial_points):
        self.id = id
        self.phone = phone
        self.points = initial_points  # 用户初始积分

    def spend_points(self, points):
        """消耗积分"""
        self.points -= points

    def earn_points(self, points):
        """获得积分"""
        self.points += points

# 任务类，表示推理任务，包含任务复杂度和所需的手机处理的时间
class Task:
    def __init__(self, token_count, model_complexity):
        self.token_count = token_count  # token数量
        self.model_complexity = model_complexity  # 模型复杂度
        self.complexity = self.token_count * self.model_complexity  # 总任务复杂度 = token数 * 模型复杂度

# 模拟器类，管理任务调度、积分计算等
class Simulator:
    def __init__(self, users):
        self.users = users  # 用户列表

    def execute_task(self, main_user, task):
        """
        主用户发起推理任务，协同用户参与推理。
        主用户消耗积分，协同用户获得积分。
        """
        # 计算任务的总时间和每个协同用户的任务分配情况
        total_processing_time = 0
        total_points_consumed = 0
        total_points_earned = 0

        # 任务拆分给多个协同用户处理
        for user in self.users:
            if user != main_user:
                phone = user.phone
                processing_time = phone.use(task)
                if processing_time is not None:
                    total_processing_time += processing_time
                    # 协同用户根据他们的处理时间和任务复杂度来赚取积分
                    points_earned = task.complexity * (processing_time / total_processing_time)
                    user.earn_points(points_earned)
                    total_points_earned += points_earned

        # 主用户消耗积分
        points_spent = task.complexity * (total_processing_time / task.complexity)*3
        main_user.spend_points(points_spent)
        total_points_consumed += points_spent

        # 输出任务执行情况
        print(f"Task executed by main user {main_user.id} with {len(self.users) - 1} collaborative users.")
        print(f"Total points consumed by main user: {total_points_consumed:.2f}")
        print(f"Total points earned by collaborative users: {total_points_earned:.2f}")
        print(f"Total processing time: {total_processing_time:.2f} seconds")
        print("\nUser points after task execution:")
        for user in self.users:
            print(f"User {user.id}: {user.points:.2f} points")

# 主程序
if __name__ == "__main__":
    # 创建用户和他们的手机
    users = [User(i, Phone(i, random.uniform(1, 5), random.randint(5, 10)), 100) for i in range(5)]
    main_user = users[0]  # 假设第一个用户是主用户

    # 创建任务
    task = Task(token_count=1000, model_complexity=0.5)

    # 创建并运行模拟器
    simulator = Simulator(users)
    simulator.execute_task(main_user, task)