import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
# 给定的两个latency_data池
latency_data_1 = {
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
latency_data_2 = {
    'Samsung S23 Ultra': {'Samsung S23 Ultra': 0.0, 'vivo X50': 0.013082, 'OPPO A72': 0.0172, 'HUAWEI Mate30': 0.025223,
                          'HUAWEI P40': 0.031922, 'HUAWEI nova7': 0.01801},
    'vivo X50': {'Samsung S23 Ultra': 0.027314, 'vivo X50': 0.0, 'OPPO A72': 0.008898, 'HUAWEI Mate30': 0.017059,
                 'HUAWEI P40': 0.03395, 'HUAWEI nova7': 0.03103},
    'OPPO A72': {'Samsung S23 Ultra': 0.079005, 'vivo X50': 0.04417, 'OPPO A72': 0.0, 'HUAWEI Mate30': 0.026379,
                 'HUAWEI P40': 0.011349, 'HUAWEI nova7': 0.00828},
    'HUAWEI Mate30': {'Samsung S23 Ultra': 0.021333, 'vivo X50': 0.013099, 'OPPO A72': 0.015191, 'HUAWEI Mate30': 0.0,
                      'HUAWEI P40': 0.011349, 'HUAWEI nova7': 0.01398},
    'HUAWEI P40': {'Samsung S23 Ultra': 0.019544, 'vivo X50': 0.016483, 'OPPO A72': 0.015265, 'HUAWEI Mate30': 0.01583,
                   'HUAWEI P40': 0.0, 'HUAWEI nova7': 0.01537},
    'HUAWEI nova7': {'Samsung S23 Ultra': 0.094404, 'vivo X50': 0.064853, 'OPPO A72': 0.019197,
                     'HUAWEI Mate30': 0.029847, 'HUAWEI P40': 0.024146, 'HUAWEI nova7': 0.0}
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
# 合并两个数据池
def merge_data(latency_data_1, latency_data_2):
    combined_latency_data = {}

    # 合并两个数据池的手机
    combined_phones = list(latency_data_1.keys()) + list(latency_data_2.keys())

    # 为每部手机计算发送和接收延迟分布
    for send_phone in combined_phones:
        combined_latency_data[send_phone] = {}

        # 计算每对手机之间的延迟
        for recv_phone in combined_phones:
            if send_phone == recv_phone:
                combined_latency_data[send_phone][recv_phone] = 0.0  # 对角线为0
            else:
                # 判断手机是否来自不同的池
                if send_phone in latency_data_1 and recv_phone in latency_data_1:
                    send_dist = calculate_send_distributions(latency_data_1, send_phone,
                                                                                 target_latency=latency_data_1[
                                                                                     send_phone].get(recv_phone, None))
                    recv_dist = calculate_recv_distributions(latency_data_1, recv_phone,
                                                                                 target_latency=latency_data_1[
                                                                                     recv_phone].get(send_phone, None))

                    send_sample_1 = send_dist.rvs(1)[0]
                    recv_sample_1 = recv_dist.rvs(1)[0]
                    total_latency = (send_sample_1 + recv_sample_1) / 2
                elif send_phone in latency_data_2 and recv_phone in latency_data_2:
                    send_dist = calculate_send_distributions(latency_data_2, send_phone,
                                                             target_latency=latency_data_2[
                                                                 send_phone].get(recv_phone, None))
                    recv_dist = calculate_recv_distributions(latency_data_2, recv_phone,
                                                             target_latency=latency_data_2[
                                                                 recv_phone].get(send_phone, None))


                    # 计算合并后的延迟
                    send_sample_1 = send_dist.rvs(1)[0]
                    recv_sample_1 = recv_dist.rvs(1)[0]
                    total_latency = (send_sample_1 + recv_sample_1 ) / 2

                elif send_phone in latency_data_2 and recv_phone in latency_data_1:
                      # 如果手机来自不同的池，通过合并的方式生成

                      send_dist = calculate_send_distributions(latency_data_2, send_phone)
                      recv_dist = calculate_recv_distributions(latency_data_1, recv_phone)

                      send_sample_1 = send_dist.rvs(1)[0]
                      recv_sample_1 = recv_dist.rvs(1)[0]

                      total_latency = (send_sample_1 + recv_sample_1) / 2


                elif send_phone in latency_data_1 and recv_phone in latency_data_2:
                    # 如果手机来自不同的池，通过合并的方式生成

                    send_dist = calculate_send_distributions(latency_data_1, send_phone)
                    recv_dist = calculate_recv_distributions(latency_data_2, recv_phone)

                    send_sample_1 = send_dist.rvs(1)[0]
                    recv_sample_1 = recv_dist.rvs(1)[0]

                    total_latency = (send_sample_1 + recv_sample_1) / 2

                combined_latency_data[send_phone][recv_phone] = total_latency
                # print(combined_latency_data)

    return combined_latency_data


# 合并两个数据池
combined_latency_data = merge_data(latency_data_1, latency_data_2)

# 打印合并后的11x11数据池
for phone, data in combined_latency_data.items():
    print(f"{phone}: {data}")
# 将数据保存为Excel文件
df = pd.DataFrame(combined_latency_data)

# 保存到Excel
df.to_csv('combined_latency_data.csv', index=True)

print("数据已保存为 Excel 文件：'combined_latency_data.csv'")