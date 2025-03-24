import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 给定的数据
model_sizes = np.array([0.56, 1.1, 3.0])  # 模型大小 (B)
load_times = np.array([4.82, 5.18, 9.19])  # 加载时间 (s)
inference_times = np.array([0.084, 0.1725, 0.33])  # 推理时间 (s/token)

# 假设加载时间与模型大小的关系为非线性：load(M) = a * M^b
def load_func(M, a, b):
    return a * M ** b

# 假设推理时间与模型大小的关系为非线性：inference(M) = c * M^d
def inference_func(M, c, d):
    return c * M ** d

# 对加载时间进行非线性拟合
popt_load, _ = curve_fit(load_func, model_sizes, load_times)
a, b = popt_load

# 对推理时间进行非线性拟合
popt_infer, _ = curve_fit(inference_func, model_sizes, inference_times)
c, d = popt_infer

# 打印拟合参数
popt_load, popt_infer

# 生成拟合曲线数据
model_sizes_fit = np.linspace(0.5, 3.5, 100)
load_times_fit = load_func(model_sizes_fit, *popt_load)
inference_times_fit = inference_func(model_sizes_fit, *popt_infer)

# 绘制拟合曲线和原始数据点
plt.figure(figsize=(10, 5))

# 加载时间
plt.subplot(1, 2, 1)
plt.scatter(model_sizes, load_times, color='blue', label='Data (Load Time)')
plt.plot(model_sizes_fit, load_times_fit, color='red', label=f'Fit: {a:.2f} * M^{b:.2f}')
plt.xlabel('Model Size (B)')
plt.ylabel('Load Time (s)')
plt.title('Model Size vs Load Time')
plt.legend()

# 推理时间
plt.subplot(1, 2, 2)
plt.scatter(model_sizes, inference_times, color='blue', label='Data (Inference Time)')
plt.plot(model_sizes_fit, inference_times_fit, color='red', label=f'Fit: {c:.2f} * M^{d:.2f}')
plt.xlabel('Model Size (B)')
plt.ylabel('Inference Time (s)')
plt.title('Model Size vs Inference Time')
plt.legend()

plt.tight_layout()
plt.show()
