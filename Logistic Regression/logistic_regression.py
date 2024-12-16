import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv("winequality-red.csv")

# Convert target to binary classification
dataset['quality_binary'] = (dataset['quality'] > 5).astype(int)

# Extract the "Alcohol" feature and target
X_alcohol = dataset[['alcohol']].values  # Single feature
y = dataset['quality_binary'].values  # Binary target

# Standardize the "Alcohol" feature
scaler = StandardScaler()
X_alcohol_scaled = scaler.fit_transform(X_alcohol)

# Add bias term
X = np.hstack((np.ones((X_alcohol_scaled.shape[0], 1)), X_alcohol_scaled))

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Batch Gradient Descent
def batchGradientDescent(X, y, theta, alpha, num_iterations):
    m = len(y)
    for _ in range(num_iterations):
        hypothesis = sigmoid(np.dot(X, theta))
        loss = hypothesis - y
        gradient = np.dot(X.T, loss) / m
        theta -= alpha * gradient
    return theta

# Initialize parameters
theta = np.zeros(X.shape[1])  # Initialize weights (bias + single feature)
alpha = 0.01
num_iterations = 1000

# Train the model
theta = batchGradientDescent(X, y, theta, alpha, num_iterations)
print(f"Trained Parameters: {theta}")

# Logistic regression curve
# Generate alcohol values for the plot
alcohol_values = np.linspace(X_alcohol.min(), X_alcohol.max(), 100).reshape(-1, 1)
alcohol_values_scaled = scaler.transform(alcohol_values)  # Scale alcohol values
X_plot = np.hstack((np.ones((alcohol_values_scaled.shape[0], 1)), alcohol_values_scaled))
hypothesis = sigmoid(np.dot(X_plot, theta))

# Plot the data and logistic regression curve
plt.figure(figsize=(10, 6))
plt.scatter(X_alcohol, y, color='blue', alpha=0.5, label='Data (Alcohol vs Quality)')
plt.plot(alcohol_values, hypothesis, color='red', label='Logistic Regression Curve')
plt.title('Alcohol vs Quality (Binary Classification)')
plt.xlabel('Alcohol')
plt.ylabel('Probability of Good Quality')
plt.legend()
plt.show()
