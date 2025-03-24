import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the data
x_data = np.array([0.56, 1.1, 1.7, 3, 7.1])
y_data = np.array([1.8718,
2.2304,
4.1412,
4.9652,
26.4048])

# Define a model function (e.g., a power law or exponential model)
def model(x, a, b):
    return a * x**b

# Fit the model to the data
params, covariance = curve_fit(model, x_data, y_data)

# Plot the data and the fitted curve
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = model(x_fit, *params)

plt.scatter(x_data, y_data, label="Data", color="red")
plt.plot(x_fit, y_fit, label=f"Fitted curve: $y = {params[0]:.2f}x^{params[1]:.2f}$", color="blue")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print(params)
