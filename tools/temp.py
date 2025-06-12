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
def calculate_send_and_recv_distributions(latency_data, phone_name):
    send_latencies = []
    recv_latencies = []

    # 收集该手机的发送延迟和接收延迟
    for other_phone, latency in latency_data[phone_name].items():
        if phone_name != other_phone:  # 排除对角线（即与自身的延迟）
            send_latencies.append(latency)  # 视为发送延迟
            recv_latencies.append(latency)  # 视为接收延迟

    # 拟合发送延迟和接收延迟的正态分布
    send_mean, send_std = np.mean(send_latencies), np.std(send_latencies)
    recv_mean, recv_std = np.mean(recv_latencies), np.std(recv_latencies)

    send_dist = stats.norm(loc=send_mean, scale=send_std)
    recv_dist = stats.norm(loc=recv_mean, scale=recv_std)

    return send_dist, recv_dist


# 为整个手机池模拟发送和接收延迟
def simulate_phone_pool(latency_data, num_samples=1000):
    simulated_data = {}

    # 为每部手机计算发送和接收分布
    for phone_A in latency_data:
        send_dist_A, recv_dist_A = calculate_send_and_recv_distributions(latency_data, phone_A)

        # 对每部手机从发送和接收分布中采样
        send_samples_A = send_dist_A.rvs(num_samples)
        recv_samples_A = recv_dist_A.rvs(num_samples)

        # 保存模拟数据
        simulated_data[phone_A] = {
            'send_samples': send_samples_A,
            'recv_samples': recv_samples_A
        }

    # 返回模拟数据
    return simulated_data


# 可视化模拟结果（以iQOOZ5为例）
simulated_data = simulate_phone_pool(latency_data, num_samples=1000)

# 画出iQOOZ5的发送与接收延迟分布
phone_A = 'iQOOZ5'
plt.figure(figsize=(12, 6))

# 发送延迟分布
plt.subplot(1, 2, 1)
plt.hist(simulated_data[phone_A]['send_samples'], bins=30, alpha=0.7, color='blue', label=f'{phone_A} Send Latency')
plt.title(f'Simulated Send Latency for {phone_A}')
plt.xlabel('Latency (seconds)')
plt.ylabel('Frequency')
plt.legend()

# 接收延迟分布
plt.subplot(1, 2, 2)
plt.hist(simulated_data[phone_A]['recv_samples'], bins=30, alpha=0.7, color='green', label=f'{phone_A} Recv Latency')
plt.title(f'Simulated Recv Latency for {phone_A}')
plt.xlabel('Latency (seconds)')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()
