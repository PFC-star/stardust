import numpy as np
from scipy.optimize import minimize


def load_time(M):
    return 5.57 * M ** 0.44


def inference_time(M):
    return 0.15 * M ** 0.73


# 目标函数
def objective(Mi):
    load_Mn = load_time(Mi[-1]) + inference_time(Mi[-1])
    load_M0 = load_time(Mi[0]) + inference_time(Mi[0])
    load_M0_sum = load_time(M_total - Mi[0])
    return max(load_Mn, load_M0 + load_M0_sum)


# 约束条件
def constraints(M_values):
    commu = 0.1  # 假设的通信时间
    constraints_eq = []

    for i in range(len(M_values) - 1):
        eq = load_time(M_values[i]) + inference_time(M_values[i]) + commu - load_time(M_values[i + 1])
        constraints_eq.append(eq)

    # 约束一：确保所有子模型之和大于等于整体模型
    constraints_eq.append(sum(M_values) - M_total)

    return constraints_eq


# 整体模型大小
M_total = 1  # 假设的整体模型大小

# 初始猜测
num_devices = 5  # 假设有5个设备
initial_guess = np.ones(num_devices) * (M_total / num_devices)  # 平均分配

# 约束定义
cons = [{'type': 'ineq', 'fun': constraints},  # 确保加载和推理时间的约束
        {'type': 'ineq', 'fun': lambda M_values: M_values}]  # 确保所有 M_i 大于等于 0

# 优化
result = minimize(objective, initial_guess, constraints=cons)

# 结果
print("优化结果：", result.x)
print("最小化的目标值：", result.fun)

print("模型之和：",sum(result.x))
def pipeline_time(Mi):
    load_Mn = load_time(Mi[-1]) + inference_time(Mi[-1])
    load_M0 = load_time(Mi[0]) + inference_time(Mi[0])
    load_M0_sum = load_time(M_total - Mi[0])
    return load_Mn, load_M0 + load_M0_sum
load_Mn, load_M0_load_M0_sum = pipeline_time(result.x)
for i,m in enumerate( result.x):
    print("流水线 ：",i+1,"model size:",m,"  load time:",load_time(m),"  infer time:",inference_time(m))

print("load_Mn ：",load_Mn)
print("load_M0 + load_M0_sum 恢复成功时间 ：",load_M0_load_M0_sum)

print("原始恢复成功时间 ：",load_time(M_total))