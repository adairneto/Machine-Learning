import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv(url, names=columns)

# Convert class labels to one-hot encoding
y = pd.get_dummies(df['class']).values
X = df.drop('class', axis=1).values.astype(float)

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Split into train and test sets
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
X_train, y_train = X[indices[:train_size]], y[indices[:train_size]]
X_test, y_test = X[indices[train_size:]], y[indices[train_size:]]

# Neural network parameters
input_size = 4
hidden_size = 8
output_size = 3
learning_rate = 0.01
epochs = 2000

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros(output_size)

# Store loss and accuracy for plotting
train_loss_history = []
test_loss_history = []
train_acc_history = []
test_acc_history = []

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, probs):
    m = y_true.shape[0]
    log_probs = -np.log(probs[range(m), y_true.argmax(axis=1)])
    return np.sum(log_probs) / m

def accuracy(y_true, probs):
    predictions = np.argmax(probs, axis=1)
    labels = np.argmax(y_true, axis=1)
    return np.mean(predictions == labels)

# Training loop
for epoch in range(epochs):
    # Forward propagation
    z1 = X_train.dot(W1) + b1
    a1 = relu(z1)
    z2 = a1.dot(W2) + b2
    probs = softmax(z2)
    
    # Compute loss and accuracy
    loss = cross_entropy_loss(y_train, probs)
    train_acc = accuracy(y_train, probs)
    train_loss_history.append(loss)
    train_acc_history.append(train_acc)
    
    # Backward propagation
    dz2 = probs - y_train
    dW2 = (a1.T).dot(dz2) / train_size
    db2 = np.sum(dz2, axis=0) / train_size
    
    dz1 = dz2.dot(W2.T) * (z1 > 0)
    dW1 = (X_train.T).dot(dz1) / train_size
    db1 = np.sum(dz1, axis=0) / train_size
    
    # Update parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    # Test evaluation
    z1_test = X_test.dot(W1) + b1
    a1_test = relu(z1_test)
    z2_test = a1_test.dot(W2) + b2
    probs_test = softmax(z2_test)
    
    test_loss = cross_entropy_loss(y_test, probs_test)
    test_acc = accuracy(y_test, probs_test)
    test_loss_history.append(test_loss)
    test_acc_history.append(test_acc)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Train Loss={loss:.4f}, Test Loss={test_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(test_loss_history, label='Test Loss')
plt.title('Training and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(test_acc_history, label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()