import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd


# 给定的两个latency_data池

# 从表格数据中提取并转换为字典格式
bandwidth_data_1 = {
    'iQOOZ5': {'iQOOZ5': 0.0, 'xiaomi 14': 1.666069031, 'iQOOz6x': 1.81388855, 'OnePlus 9': 2.567291258, 'OnePlus 6': 4.906654358},
    'xiaomi 14': {'iQOOZ5': 0.312805176, 'xiaomi 14': 0.0, 'iQOOz6x': 1.200675964, 'OnePlus 9': 0.575065613, 'OnePlus 6': 0.590324402},
    'iQOOz6x': {'iQOOZ5': 0.994682312, 'xiaomi 14': 2.725601196, 'iQOOz6x': 0.0, 'OnePlus 9': 0.332832336, 'OnePlus 6': 0.495910645},
    'OnePlus 9': {'iQOOZ5': 0.772476196, 'xiaomi 14': 1.02519919, 'iQOOz6x': 1.168251038, 'OnePlus 9': 0.0, 'OnePlus 6': 1.210212708},
    'OnePlus 6': {'iQOOZ5': 2.743721008, 'xiaomi 14': 1.764297485, 'iQOOz6x': 1.543045044, 'OnePlus 9': 3.170013428, 'OnePlus 6': 0.0}
}

# 将第二个数据池也转换为字典格式
bandwidth_data_2 = {
    'Samsung S23 Ultra': {'Samsung S23 Ultra': 0.0, 'vivo X50': 0.493049622, 'OPPO A72': 0.238418579, 'HUAWEI Mate30': 0.210762024, 'HUAWEI P40': 0.324249268, 'HUAWEI nova7': 0.652313232},
    'vivo X50': {'Samsung S23 Ultra': 0.686645508, 'vivo X50': 0.0, 'OPPO A72': 0.731468201, 'HUAWEI Mate30': 1.775741577, 'HUAWEI P40': 0.592231575, 'HUAWEI nova7': 1.087188721},
    'OPPO A72': {'Samsung S23 Ultra': 0.968933105, 'vivo X50': 0.823974609, 'OPPO A72': 0.0, 'HUAWEI Mate30': 0.550270081, 'HUAWEI P40': 0.046730042, 'HUAWEI nova7': 0.242233276},
    'HUAWEI Mate30': {'Samsung S23 Ultra': 3.914833069, 'vivo X50': 0.361442566, 'OPPO A72': 0.710487366, 'HUAWEI Mate30': 0.0, 'HUAWEI P40': 1.114845276, 'HUAWEI nova7': 0.784873962},
    'HUAWEI P40': {'Samsung S23 Ultra': 1.071929932, 'vivo X50': 0.091552734, 'OPPO A72': 0.123023987, 'HUAWEI Mate30': 0.250267029, 'HUAWEI P40': 0.0, 'HUAWEI nova7': 0.135421753},
    'HUAWEI nova7': {'Samsung S23 Ultra': 1.264572144, 'vivo X50': 0.612258911, 'OPPO A72': 0.432014465, 'HUAWEI Mate30': 0.589370728, 'HUAWEI P40': 0.129699707, 'HUAWEI nova7': 0.0}
}


# 计算每部手机的发送能力和接收能力的分布（使用伽玛分布）
def calculate_send_distributions(bandwidth_data, phone_name,target_latency=None):
    send_latencies = []


    # 发送延迟（行）来自该手机到其他手机的延迟
    for other_phone, latency in bandwidth_data[phone_name].items():
        if phone_name != other_phone:  # 排除对角线（即与自身的延迟）
            send_latencies.append(latency)  # 视为发送延迟


    if target_latency is not None:
        send_latencies.append(target_latency)

    # 使用伽玛分布拟合发送和接收延迟
    send_shape, send_loc, send_scale = stats.gamma.fit(send_latencies, floc=0)  # 固定位置参数为0


    # 创建伽玛分布
    send_dist = stats.gamma(a=send_shape, loc=send_loc, scale=send_scale)


    return send_dist

def calculate_recv_distributions(bandwidth_data, phone_name,target_latency=None):

    recv_latencies = []



    # 接收延迟（列）来自其他手机到该手机的延迟
    for other_phone, latency in bandwidth_data.items():
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
def merge_data(bandwidth_data_1, bandwidth_data_2):
    combined_bandwidth_data = {}

    # 合并两个数据池的手机
    combined_phones = list(bandwidth_data_1.keys()) + list(bandwidth_data_2.keys())

    # 为每部手机计算发送和接收延迟分布
    for send_phone in combined_phones:
        combined_bandwidth_data[send_phone] = {}

        # 计算每对手机之间的延迟
        for recv_phone in combined_phones:
            if send_phone == recv_phone:
                combined_bandwidth_data[send_phone][recv_phone] = 0.0  # 对角线为0
            else:
                # 判断手机是否来自不同的池
                if send_phone in bandwidth_data_1 and recv_phone in bandwidth_data_1:
                    send_dist = calculate_send_distributions(bandwidth_data_1, send_phone,
                                                                                 target_latency=bandwidth_data_1[
                                                                                     send_phone].get(recv_phone, None))
                    recv_dist = calculate_recv_distributions(bandwidth_data_1, recv_phone,
                                                                                 target_latency=bandwidth_data_1[
                                                                                     recv_phone].get(send_phone, None))

                    send_sample_1 = send_dist.rvs(1)[0]
                    recv_sample_1 = recv_dist.rvs(1)[0]
                    total_latency = (send_sample_1 + recv_sample_1) / 2
                elif send_phone in bandwidth_data_2 and recv_phone in bandwidth_data_2:
                    send_dist = calculate_send_distributions(bandwidth_data_2, send_phone,
                                                             target_latency=bandwidth_data_2[
                                                                 send_phone].get(recv_phone, None))
                    recv_dist = calculate_recv_distributions(bandwidth_data_2, recv_phone,
                                                             target_latency=bandwidth_data_2[
                                                                 recv_phone].get(send_phone, None))


                    # 计算合并后的延迟
                    send_sample_1 = send_dist.rvs(1)[0]
                    recv_sample_1 = recv_dist.rvs(1)[0]
                    total_latency = (send_sample_1 + recv_sample_1 ) / 2

                elif send_phone in bandwidth_data_2 and recv_phone in bandwidth_data_1:
                      # 如果手机来自不同的池，通过合并的方式生成

                      send_dist = calculate_send_distributions(bandwidth_data_2, send_phone)
                      recv_dist = calculate_recv_distributions(bandwidth_data_1, recv_phone)

                      send_sample_1 = send_dist.rvs(1)[0]
                      recv_sample_1 = recv_dist.rvs(1)[0]

                      total_latency = (send_sample_1 + recv_sample_1) / 2


                elif send_phone in bandwidth_data_1 and recv_phone in bandwidth_data_2:
                    # 如果手机来自不同的池，通过合并的方式生成

                    send_dist = calculate_send_distributions(bandwidth_data_1, send_phone)
                    recv_dist = calculate_recv_distributions(bandwidth_data_2, recv_phone)

                    send_sample_1 = send_dist.rvs(1)[0]
                    recv_sample_1 = recv_dist.rvs(1)[0]

                    total_latency = (send_sample_1 + recv_sample_1) / 2

                combined_bandwidth_data[send_phone][recv_phone] = total_latency
                # print(combined_bandwidth_data)

    return combined_bandwidth_data


# 合并两个数据池
combined_bandwidth_data = merge_data(bandwidth_data_1, bandwidth_data_2)

# 打印合并后的11x11数据池
for phone, data in combined_bandwidth_data.items():
    print(f"{phone}: {data}")
# 将数据保存为Excel文件
df = pd.DataFrame(combined_bandwidth_data)

# 保存到Excel
df.to_csv('combined_bandwidth_data.csv', index=True)

print("数据已保存为 Excel 文件：'combined_bandwidth_data.csv'")