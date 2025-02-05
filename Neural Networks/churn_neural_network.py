import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class NeuralNetwork:
    def __init__(self, layer_dims, activation_functions, learning_rate=0.001, epochs=1000):
        self.layer_dims = layer_dims
        self.activation_functions = activation_functions
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.params = self.initialize_parameters()
    
    def initialize_parameters(self):
        np.random.seed(42)
        params = {}
        for l in range(1, len(self.layer_dims)):
            params[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            params[f'b{l}'] = np.zeros((self.layer_dims[l], 1))
        return params
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def apply_activation(self, Z, activation_type):
        if activation_type == 'sigmoid':
            return self.sigmoid(Z)
        elif activation_type == 'relu':
            return self.relu(Z)
        else:
            raise ValueError(f"Unknown activation function: {activation_type}")
    
    def forward_propagation(self, X):
        cache = {'A0': X}
        L = len(self.layer_dims) - 1
        
        for l in range(1, L + 1):
            W, b = self.params[f'W{l}'], self.params[f'b{l}']
            Z = np.dot(W, cache[f'A{l-1}']) + b
            cache[f'A{l}'] = self.apply_activation(Z, self.activation_functions[l-1])
        
        return cache[f'A{L}'], cache
    
    def compute_cost(self, A3, y):
        m = y.shape[1]
        cost = -np.sum(y * np.log(A3) + (1 - y) * np.log(1 - A3)) / m
        return np.squeeze(cost)
    
    def compute_accuracy(self, A3, y):
        predictions = (A3 > 0.5).astype(int)
        return np.mean(predictions == y) * 100
    
    def backward_propagation(self, X, y, cache):
        m = y.shape[1]
        gradients = {}
        L = len(self.layer_dims) - 1
        
        dZ = cache[f'A{L}'] - y  # Output layer gradient
        
        for l in reversed(range(1, L + 1)):
            A_prev = cache[f'A{l-1}']
            W = self.params[f'W{l}']
            
            gradients[f'dW{l}'] = (1 / m) * np.dot(dZ, A_prev.T)
            gradients[f'db{l}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            
            if l > 1:  # Compute gradients for hidden layers
                if self.activation_functions[l-2] == 'sigmoid':
                    dZ = np.dot(W.T, dZ) * A_prev * (1 - A_prev)
                elif self.activation_functions[l-2] == 'relu':
                    dZ = np.dot(W.T, dZ) * (A_prev > 0)
        
        return gradients
    
    def update_parameters(self, gradients):
        for key in self.params.keys():
            self.params[key] -= self.learning_rate * gradients[f'd{key}']
    
    def train(self, X, y):
        costs, accuracies = [], []
        
        for i in range(self.epochs):
            A3, cache = self.forward_propagation(X)
            cost = self.compute_cost(A3, y)
            accuracy = self.compute_accuracy(A3, y)
            
            costs.append(cost)
            accuracies.append(accuracy)
            
            gradients = self.backward_propagation(X, y, cache)
            self.update_parameters(gradients)
            
            if i % 100 == 0:
                print(f"Epoch {i}, Cost: {cost:.4f}, Accuracy: {accuracy:.2f}%")
        
        return costs, accuracies
    
    def predict(self, X):
        A3, _ = self.forward_propagation(X)
        return (A3 > 0.5).astype(int)
    
    def plot_metrics(self, costs, accuracies):
        epochs = range(len(costs))
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss", color='tab:red')
        ax1.plot(epochs, costs, color='tab:red', label="Loss")
        ax1.tick_params(axis='y', labelcolor='tab:red')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy (%)", color='tab:blue')
        ax2.plot(epochs, accuracies, color='tab:blue', label="Accuracy")
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        
        plt.title("Loss and Accuracy over Epochs")
        fig.tight_layout()
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

# Data Preprocessing
data = pd.read_csv("Churn_Modelling.csv")
X = data[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography', 'Gender']]
y = data['Exited'].values.reshape(1, -1)

X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).T

X_train, X_test, y_train, y_test = train_test_split(X_scaled.T, y.T, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T

# Training the Neural Network
nn = NeuralNetwork(layer_dims=[X_train.shape[0], 10, 5, 1], activation_functions=['relu', 'relu', 'sigmoid'], learning_rate=0.01, epochs=1000)
costs, accuracies = nn.train(X_train, y_train)

# Plot Metrics
nn.plot_metrics(costs, accuracies)

# Predictions and Accuracy
y_pred = nn.predict(X_test)
accuracy = np.mean(y_pred == y_test) * 100
print(f"Test Accuracy: {accuracy:.2f}%")
nn.summary()
