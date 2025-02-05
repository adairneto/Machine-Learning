import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Data Preprocessing
data = pd.read_csv("Churn_Modelling.csv")
X = data[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography', 'Gender']]
y = data['Exited']

# One-Hot Encoding & Feature Scaling
X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define and Train the MLP Model
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Predictions & Accuracy
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot Loss Curve
plt.plot(mlp.loss_curve_)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve Over Epochs")
plt.show()
