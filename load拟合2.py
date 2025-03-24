import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Updated data
x_data = np.array([0.56, 1.1, 1.7, 3, 7.1])
# y_data_all = np.array([
#     [1.2708, 1.8516, 1.7346, 2.0908, 8.0254],  # Phone 1
#     [2.7764, 3.815, 4.7654, 11.6412, 25.2194],  # Phone 2
#     [1.6936, 1.9744, 2.8148, 8.5014, 33.7602],  # Phone 3
#     [10.2366, 10.7732, 14.6558, 24.7392, 47.2954],  # Phone 4
#     [1.8718, 2.2304, 4.1412, 4.9652, 26.4048]  # Phone 5
# ])

y_data_all = np.array([
    [5.836, 10.689, 14.404,18.226, 24.653],  # Phone 1
    [3.109, 3.006,3.175, 4.041, 29.216],  # Phone 2
    [2.399,3.299, 5.983, 9.302,31.82],  # Phone 3
    [10.684, 12.484, 23.901,27.205, 47.092],  # Phone 4
    [2.901,3.854,10.886, 14.128, 31.304]  # Phone 5
])

def log_model(x, a, b):
    return a * np.log(x) + b

def power_model(x, a, b):
    return a * x**b

def exp_model(x, a, b):
    return a * np.exp(b * x)
# Re-running the previous code for fitting and plotting

# Fit the models for each phone's loading time
params_log = []
params_power = []
params_exp = []

for y_data in y_data_all:
    # Fit the log model
    popt_log, _ = curve_fit(log_model, x_data, y_data)
    params_log.append(popt_log)

    # Fit the power model
    popt_power, _ = curve_fit(power_model, x_data, y_data)
    params_power.append(popt_power)

    # Fit the exponential model
    popt_exp, _ = curve_fit(exp_model, x_data, y_data)
    params_exp.append(popt_exp)

# Plot the results
x_fit = np.linspace(min(x_data), max(x_data), 100)

plt.figure(figsize=(12, 8))

for i, y_data in enumerate(y_data_all):
    # plt.subplot(3, 2, i + 1)
    plt.scatter(x_data, y_data, label="Data", color="red")

    # # Plot the log model
    # y_log_fit = log_model(x_fit, *params_log[i])
    # plt.plot(x_fit, y_log_fit, label=f"Log model: $y = {params_log[i][0]:.2f} \cdot \log(x) + {params_log[i][1]:.2f}$",
    #          color="blue")

    # Plot the power model
    y_power_fit = power_model(x_fit, *params_power[i])
    plt.plot(x_fit, y_power_fit,
             label=f"Phone{i} :Power model: $y = {params_power[i][0]:.2f} \cdot x^{{{params_power[i][1]:.2f}}}$")

    # # Plot the exponential model
    # y_exp_fit = exp_model(x_fit, *params_exp[i])
    # plt.plot(x_fit, y_exp_fit, label=f"Exp model: $y = {params_exp[i][0]:.2f} \cdot e^{{{params_exp[i][1]:.2f}x}}$",
    #          color="orange")

    plt.title(f"Phone {i + 1}")
    plt.xlabel('Model Parameter Count')
    plt.ylabel('Load Time')
    plt.legend()

plt.tight_layout()
plt.show()
