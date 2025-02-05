import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

# Load MNIST Dataset
data = pd.read_csv("mnist.csv")

# Extract Features (X) and Labels (y)
X = data.drop(columns=['label']).values.astype(np.float32)
y = data['label'].values.reshape(-1, 1)

# One-Hot Encode Labels
encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y).astype(np.float32)

# Normalize Pixel Values
X /= 255.0  # Normalize to [0,1]

# Split into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Reduce Training Set Size (Keep Only 25% of Training Data)
X_train, y_train = X_train[:len(X_train) // 4], y_train[:len(y_train) // 4]

# Define MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', learning_rate_init=0.01,
                    max_iter=200, batch_size=64, alpha=0.0001, random_state=42)

# Train the Model
mlp.fit(X_train, y_train)

# Get Training Loss and Accuracy
train_loss = log_loss(y_train, mlp.predict_proba(X_train))
y_train_pred = np.argmax(mlp.predict_proba(X_train), axis=1)
y_train_true = np.argmax(y_train, axis=1)
train_accuracy = accuracy_score(y_train_true, y_train_pred)

# Get Test Accuracy
y_test_pred = np.argmax(mlp.predict_proba(X_test), axis=1)
y_test_true = np.argmax(y_test, axis=1)
test_accuracy = accuracy_score(y_test_true, y_test_pred)

# Print Metrics
print(f"Training Loss: {train_loss:.4f}")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot Loss Curve
plt.plot(mlp.loss_curve_)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()
