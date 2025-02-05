import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class NeuralNetwork:
    def __init__(self, layer_dims, activation_functions, learning_rate=0.01, epochs=200, batch_size=64, decay_rate=0.001):
        self.layer_dims = layer_dims
        self.activation_functions = activation_functions
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.params = self.initialize_parameters()

    def initialize_parameters(self):
        np.random.seed(42)
        params = {}
        for l in range(1, len(self.layer_dims)):
            params[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2 / self.layer_dims[l-1])
            params[f'b{l}'] = np.zeros((self.layer_dims[l], 1))
        return params

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        expZ = np.exp(z - np.max(z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def apply_activation(self, Z, activation_type):
        if activation_type == 'sigmoid':
            return self.sigmoid(Z)
        elif activation_type == 'relu':
            return self.relu(Z)
        elif activation_type == 'softmax':
            return self.softmax(Z)
        else:
            raise ValueError(f"Unknown activation function: {activation_type}")

    def forward_propagation(self, X):
        cache = {'A0': X}
        for l in range(1, len(self.layer_dims)):
            W, b = self.params[f'W{l}'], self.params[f'b{l}']
            Z = np.dot(W, cache[f'A{l-1}']) + b
            cache[f'A{l}'] = self.apply_activation(Z, self.activation_functions[l-1])
        return cache[f'A{len(self.layer_dims) - 1}'], cache

    def compute_cost(self, AL, y):
        m = y.shape[1]
        return -np.sum(y * np.log(AL + 1e-8)) / m

    def compute_accuracy(self, AL, y):
        return np.mean(np.argmax(AL, axis=0) == np.argmax(y, axis=0)) * 100

    def backward_propagation(self, y, cache):
        m = y.shape[1]
        gradients = {}
        dZ = cache[f'A{len(self.layer_dims) - 1}'] - y  # Output layer gradient
        
        for l in reversed(range(1, len(self.layer_dims))):
            A_prev = cache[f'A{l-1}']
            W = self.params[f'W{l}']
            
            gradients[f'dW{l}'] = np.dot(dZ, A_prev.T) / m
            gradients[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m

            if l > 1:
                dZ = np.dot(W.T, dZ) * (A_prev > 0)  # ReLU derivative
                
        return gradients

    def update_parameters(self, gradients, epoch):
        lr = self.learning_rate / (1 + self.decay_rate * epoch)  # Learning rate decay
        for key in self.params:
            self.params[key] -= lr * gradients[f'd{key}']

    def train(self, X, y):
        m = X.shape[1]
        costs, accuracies = [], []

        for epoch in range(self.epochs):
            shuffled_indices = np.random.permutation(m)
            X_shuffled, y_shuffled = X[:, shuffled_indices], y[:, shuffled_indices]

            batches = np.split(np.arange(m), np.arange(self.batch_size, m, self.batch_size))

            for batch_indices in batches:
                X_batch, y_batch = X_shuffled[:, batch_indices], y_shuffled[:, batch_indices]
                AL, cache = self.forward_propagation(X_batch)
                gradients = self.backward_propagation(y_batch, cache)
                self.update_parameters(gradients, epoch)

            if epoch % 10 == 0:
                cost = self.compute_cost(AL, y_batch)
                accuracy = self.compute_accuracy(AL, y_batch)
                costs.append(cost)
                accuracies.append(accuracy)
                print(f"Epoch {epoch}, Cost: {cost:.4f}, Accuracy: {accuracy:.2f}%")

        return costs, accuracies

    def predict(self, X):
        return np.argmax(self.forward_propagation(X)[0], axis=0)

    def plot_metrics(self, costs, accuracies):
        epochs = range(0, self.epochs, 10)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, costs, label="Loss", color='r')
        plt.plot(epochs, accuracies, label="Accuracy", color='b')
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.title("Training Metrics")
        plt.legend()
        plt.show()

    def summary(self):
        print("\nModel Summary")
        print("=" * 30)
        print(f"Total Layers: {len(self.layer_dims) - 1}")
        print("Layer Details:")
        total_params = 0
        for l in range(1, len(self.layer_dims)):
            input_dim = self.layer_dims[l-1]
            output_dim = self.layer_dims[l]
            activation = self.activation_functions[l-1]
            params = (input_dim * output_dim) + output_dim  # Weights + Biases
            total_params += params
            print(f" Layer {l}: {activation.upper()} | Input: {input_dim} | Output: {output_dim} | Params: {params}")
        print("=" * 30)
        print(f"Total Trainable Parameters: {total_params}\n")

# Load MNIST Dataset
data = pd.read_csv("mnist.csv")

# Extract Features (X) and Labels (y)
X = data.drop(columns=['label']).values.T.astype(np.float32)  
y = data['label'].values.reshape(1, -1)

# One-Hot Encode Labels
encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y.T).T.astype(np.float32)

# Normalize Pixel Values
X /= 255.0  # Normalize to [0,1]

# Split into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X.T, y_one_hot.T, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T

# Reduce Training Set Size (Keep Only 25% of Training Data)
X_train, y_train = X_train[:, :X_train.shape[1] // 4], y_train[:, :y_train.shape[1] // 4]

# Define Neural Network Architecture
layer_dims = [784, 128, 64, 10]
activation_functions = ['relu', 'relu', 'softmax']

# Train the Neural Network
nn = NeuralNetwork(layer_dims, activation_functions, learning_rate=0.01, epochs=200, batch_size=64, decay_rate=0.001)
costs, accuracies = nn.train(X_train, y_train)

# Plot Training Metrics
nn.plot_metrics(costs, accuracies)

# Predictions and Accuracy
y_pred = nn.predict(X_test)
y_test_labels = np.argmax(y_test, axis=0)
accuracy = np.mean(y_pred == y_test_labels) * 100
print(f"Test Accuracy: {accuracy:.2f}%")
nn.summary()