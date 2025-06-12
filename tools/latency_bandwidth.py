import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 给定的latency_data
latency_data_1 = {
    'iQOOZ5': {'iQOOZ5': 0.0, 'xiaomi 14': 0.034866, 'iQOOz6x': 0.017796, 'OnePlus 9': 0.076512, 'OnePlus 6': 0.030985},
    'xiaomi 14': {'iQOOZ5': 0.03257, 'xiaomi 14': 0.0, 'iQOOz6x': 0.027257, 'OnePlus 9': 0.079517, 'OnePlus 6': 0.018823},
    'iQOOz6x': {'iQOOZ5': 0.108798, 'xiaomi 14': 0.065485, 'iQOOz6x': 0.0, 'OnePlus 9': 0.064679, 'OnePlus 6': 0.043849},
    'OnePlus 9': {'iQOOZ5': 0.206233, 'xiaomi 14': 0.028255, 'iQOOz6x': 0.034938, 'OnePlus 9': 0.0, 'OnePlus 6': 0.02909},
    'OnePlus 6': {'iQOOZ5': 0.021318, 'xiaomi 14': 0.034842, 'iQOOz6x': 0.019245, 'OnePlus 9': 0.042098, 'OnePlus 6': 0.0}
}
latency_data_2 = {
    'Samsung S23 Ultra': {'Samsung S23 Ultra': 0.0, 'vivo X50': 0.013082, 'OPPO A72': 0.0172, 'HUAWEI Mate30': 0.025223, 'HUAWEI P40': 0.031922, 'HUAWEI nova7': 0.01801},
    'vivo X50': {'Samsung S23 Ultra': 0.027314, 'vivo X50': 0.0, 'OPPO A72': 0.008898, 'HUAWEI Mate30': 0.017059, 'HUAWEI P40': 0.03395, 'HUAWEI nova7': 0.03103},
    'OPPO A72': {'Samsung S23 Ultra': 0.079005, 'vivo X50': 0.04417, 'OPPO A72': 0.0, 'HUAWEI Mate30': 0.026379, 'HUAWEI P40': 0.011349, 'HUAWEI nova7': 0.00828},
    'HUAWEI Mate30': {'Samsung S23 Ultra': 0.021333, 'vivo X50': 0.013099, 'OPPO A72': 0.015191, 'HUAWEI Mate30': 0.0, 'HUAWEI P40': 0.011349, 'HUAWEI nova7': 0.01398},
    'HUAWEI P40': {'Samsung S23 Ultra': 0.019544, 'vivo X50': 0.016483, 'OPPO A72': 0.015265, 'HUAWEI Mate30': 0.01583, 'HUAWEI P40': 0.0, 'HUAWEI nova7': 0.01537},
    'HUAWEI nova7': {'Samsung S23 Ultra': 0.094404, 'vivo X50': 0.064853, 'OPPO A72': 0.019197, 'HUAWEI Mate30': 0.029847, 'HUAWEI P40': 0.024146, 'HUAWEI nova7': 0.0}
}


# 计算每部手机的发送能力和接收能力的分布（使用伽玛分布）
def calculate_send_distributions(latency_data, phone_name,target_latency=None):
    send_latencies = []


    # 发送延迟（行）来自该手机到其他手机的延迟
    for other_phone, latency in latency_data[phone_name].items():
        if phone_name != other_phone:  # 排除对角线（即与自身的延迟）
            send_latencies.append(latency)  # 视为发送延迟


    if target_latency is not None:
        send_latencies.append(target_latency)

    # 使用伽玛分布拟合发送和接收延迟
    send_shape, send_loc, send_scale = stats.gamma.fit(send_latencies, floc=0)  # 固定位置参数为0


    # 创建伽玛分布
    send_dist = stats.gamma(a=send_shape, loc=send_loc, scale=send_scale)


    return send_dist

def calculate_recv_distributions(latency_data, phone_name,target_latency=None):

    recv_latencies = []



    # 接收延迟（列）来自其他手机到该手机的延迟
    for other_phone, latency in latency_data.items():
        if phone_name in latency  :  # 确保当前手机为接收手机
            if latency[phone_name]:
                recv_latencies.append(latency[phone_name])  # 视为接收延迟
    if target_latency is not None:

        recv_latencies.append(target_latency)
    # 使用伽玛分布拟合发送和接收延迟

    recv_shape, recv_loc, recv_scale = stats.gamma.fit(recv_latencies, floc=0)

    # 创建伽玛分布

    recv_dist = stats.gamma(a=recv_shape, loc=recv_loc, scale=recv_scale)

    return recv_dist


# 生成新的手机池数据
def generate_new_latency_data(latency_data, num_samples=100000):  # 增加采样数量
    new_latency_data = {}

    # 遍历每部手机，计算其发送和接收延迟的分布
    for phone in latency_data.keys():
        new_latency_data[phone] = {}

        for other_phone in latency_data.keys():
            if phone == other_phone:
                new_latency_data[phone][other_phone] = 0.0
            else:
                # 计算每部手机的发送和接收延迟分布
                send_dist  = calculate_send_distributions(latency_data, phone, target_latency=latency_data[phone][other_phone])
                recv_dist = calculate_recv_distributions(latency_data, phone,
                                                         target_latency=latency_data[phone][other_phone])

                # 从分布中采样
                send_sample = send_dist.rvs(1)[0]
                recv_sample = recv_dist.rvs(1)[0]

                # 总延迟是发送延迟与接收延迟之和
                total_latency = (send_sample + recv_sample)/2

                # 存储模拟后的延迟数据
                new_latency_data[phone][other_phone] = total_latency

    return new_latency_data


# 生成新的手机池数据
new_latency_data = generate_new_latency_data(latency_data_1)

# 1. 比较均值和标准差
def compare_statistics(real_data, simulated_data):
    real_means = {}
    real_stds = {}
    simulated_means = {}
    simulated_stds = {}

    for phone, data in real_data.items():
        real_values = [latency for other_phone, latency in data.items() if phone != other_phone]
        simulated_values = [simulated_data[phone][other_phone] for other_phone in data.keys() if phone != other_phone]

        real_means[phone] = np.mean(real_values)
        real_stds[phone] = np.std(real_values)

        simulated_means[phone] = np.mean(simulated_values)
        simulated_stds[phone] = np.std(simulated_values)

    return real_means, real_stds, simulated_means, simulated_stds


real_means, real_stds, simulated_means, simulated_stds = compare_statistics(latency_data_1, new_latency_data)

# 2. 可视化分布比较
def visualize_distribution_comparison(real_data, simulated_data):
    plt.figure(figsize=(12, 6))

    # 对于每部手机，绘制真实数据和模拟数据的分布
    for phone, data in real_data.items():
        real_values = [latency for other_phone, latency in data.items() if phone != other_phone]
        simulated_values = [simulated_data[phone][other_phone] for other_phone in data.keys() if phone != other_phone]

        plt.subplot(2, 3, list(real_data.keys()).index(phone) + 1)
        plt.hist(real_values, bins=30, alpha=0.7, label=f'Real Latency ({phone})', color='blue')
        plt.hist(simulated_values, bins=30, alpha=0.5, label=f'Simulated Latency ({phone})', color='red')
        plt.title(f'Distribution Comparison for {phone}')
        plt.xlabel('Latency (seconds)')
        plt.ylabel('Frequency')
        plt.legend()

    plt.tight_layout()
    plt.show()


# 3. 检查特定值的合理性
def check_specific_values(real_data, simulated_data, phone_A, phone_B):
    real_value = real_data[phone_A][phone_B]
    simulated_value = simulated_data[phone_A][phone_B]
    print(f"Real latency for {phone_A} -> {phone_B}: {real_value}")
    print(f"Simulated latency for {phone_A} -> {phone_B}: {simulated_value}")
    print(f"Difference: {abs(real_value - simulated_value)}")

# 比较统计量
print("Real Means:", real_means)
print("Simulated Means:", simulated_means)
print("Real Standard Deviations:", real_stds)
print("Simulated Standard Deviations:", simulated_stds)

# 可视化分布比较
visualize_distribution_comparison(latency_data_1, new_latency_data)

# 检查特定值的合理性
check_specific_values(latency_data_1, new_latency_data, 'iQOOZ5', 'OnePlus 6')
