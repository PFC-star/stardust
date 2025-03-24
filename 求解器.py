def compute_submodels(M, n, C, alpha):
    # 计算 M_0
    M0 = (M - (C / alpha) * (n * (n + 1)) / 2) / (n + 1)

    # 计算每个子模型 Mi
    submodels = [M0 + i * (C / alpha) for i in range(n + 1)]
    return submodels


# 示例参数
M = 100  # 总模型大小
n = 4  # 设备数量
C = 1  # 通信时间
alpha = 0.5  # 加载时间系数

# 计算子模型大小
submodels = compute_submodels(M, n, C, alpha)
print(submodels)
