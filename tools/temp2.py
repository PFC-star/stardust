import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 给定的latency_data
latency_data = {
    'iQOOZ5': {'iQOOZ5': 0.0, 'xiaomi 14': 0.034866, 'iQOOz6x': 0.017796, 'OnePlus 9': 0.076512, 'OnePlus 6': 0.030985},
    'xiaomi 14': {'iQOOZ5': 0.03257, 'xiaomi 14': 0.0, 'iQOOz6x': 0.027257, 'OnePlus 9': 0.079517,
                  'OnePlus 6': 0.018823},
    'iQOOz6x': {'iQOOZ5': 0.108798, 'xiaomi 14': 0.065485, 'iQOOz6x': 0.0, 'OnePlus 9': 0.064679,
                'OnePlus 6': 0.043849},
    'OnePlus 9': {'iQOOZ5': 0.206233, 'xiaomi 14': 0.028255, 'iQOOz6x': 0.034938, 'OnePlus 9': 0.0,
                  'OnePlus 6': 0.02909},
    'OnePlus 6': {'iQOOZ5': 0.021318, 'xiaomi 14': 0.034842, 'iQOOz6x': 0.019245, 'OnePlus 9': 0.042098,
                  'OnePlus 6': 0.0}
}


# 计算每部手机的发送能力和接收能力的分布
def calculate_send_and_recv_distributions(latency_data, phone_name, target_latency=None):
    send_latencies = []
    recv_latencies = []

    # 收集该手机的发送延迟和接收延迟
    for other_phone, latency in latency_data[phone_name].items():
        if phone_name != other_phone:  # 排除对角线（即与自身的延迟）
            send_latencies.append(latency)  # 视为发送延迟
            recv_latencies.append(latency)  # 视为接收延迟

    # 如果给定了目标延迟（如真实延迟），则将其加入分布数据中
    if target_latency is not None:
        send_latencies.append(target_latency)
        recv_latencies.append(target_latency)

    # 拟合发送延迟和接收延迟的正态分布
    send_mean, send_std = np.mean(send_latencies), np.std(send_latencies)
    recv_mean, recv_std = np.mean(recv_latencies), np.std(recv_latencies)

    send_dist = stats.norm(loc=send_mean, scale=send_std)
    recv_dist = stats.norm(loc=recv_mean, scale=recv_std)

    return send_dist, recv_dist


# 模拟从手机A到手机B的延迟
def simulate_latency(phone_A, phone_B, latency_data, target_latency=None, num_samples=10000):
    send_dist_A, recv_dist_B = calculate_send_and_recv_distributions(latency_data, phone_A, target_latency)

    # 从手机A的发送分布和手机B的接收分布中采样
    send_samples_A = send_dist_A.rvs(num_samples)
    recv_samples_B = recv_dist_B.rvs(num_samples)

    # 计算总延迟
    total_latency = send_samples_A + recv_samples_B

    # 可视化模拟的延迟分布
    plt.figure(figsize=(6, 6))
    plt.hist(total_latency, bins=30, alpha=0.7, color='purple', label=f'Total Latency ({phone_A} -> {phone_B})')
    plt.title(f'Simulated Total Latency from {phone_A} to {phone_B}')
    plt.xlabel('Total Latency (seconds)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


# 示例：模拟iQOOZ5到OnePlus 6的延迟，确保0.030985在模拟分布内
simulate_latency('iQOOZ5', 'OnePlus 6', latency_data, target_latency=latency_data['iQOOZ5']['OnePlus 6'])
