import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Batch Gradient Descent
def batchGradientDescent(x, y, theta, alpha, numIterations):
    m = len(y)  # Number of examples
    for iteration in range(numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        gradient = np.dot(x.T, loss) / m  # Compute gradient over all examples
        theta = theta - alpha * gradient  # Update theta
    return theta

# Stochastic Gradient Descent
def stochasticGradientDescent(x, y, theta, alpha, numIterations):
    m = len(y)  # Number of examples
    for iteration in range(numIterations):
        for i in range(m):
            xi = x[i].reshape(1, -1)  # Single example
            yi = y[i]
            hypothesis = np.dot(xi, theta)
            loss = hypothesis - yi
            gradient = xi.T * loss  # Gradient for this example
            theta = theta - alpha * gradient.flatten()  # Update theta
    return theta

# Load and preprocess dataset
dataset = pd.read_csv("dataset/car_data.csv")
dataset = dataset.dropna()

x = dataset.iloc[:, 14:15].values
y = dataset.iloc[:, -1].values

# Add bias term (intercept)
x = np.c_[np.ones(x.shape[0]), x]  # Add a column of ones to x

# Standardize features
x_mean = x[:, 1].mean()
x_std = x[:, 1].std()
x[:, 1] = (x[:, 1] - x_mean) / x_std

# Parameters
m, n = x.shape  # Number of examples and features
theta = np.zeros(n)  # Initialize theta
alpha = 0.01  # Learning rate
numIterations = 100  # Number of epochs for SGD

# Run Batch Gradient Descent
theta_batch = batchGradientDescent(x, y, theta.copy(), alpha, numIterations)

# Run Stochastic Gradient Descent
theta_sgd = stochasticGradientDescent(x, y, theta.copy(), alpha, numIterations)

# Output results
print(f"Batch Gradient Descent Parameters: {theta_batch}")
print(f"Stochastic Gradient Descent Parameters: {theta_sgd}")

# Plotting
plt.scatter(x[:, 1], y, color='red', label='Data points')  # Scaled feature
plt.plot(x[:, 1], np.dot(x, theta_batch), color='blue', label='Batch GD Line')
plt.plot(x[:, 1], np.dot(x, theta_sgd), color='green', linestyle='--', label='SGD Line')
plt.title('Car length vs Price')
plt.xlabel('Standardized Length')
plt.ylabel('Price')
plt.legend()
plt.show()
