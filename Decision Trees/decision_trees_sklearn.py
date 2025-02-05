import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Data preprocessing
data = pd.read_csv("weather_forecast_data.csv")
X = data[["Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Pressure"]].values
y = (data["Rain"] == "rain").astype(int).values  # Convert "rain"/"no rain" to 1/0

# Split the data into training and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree
clf = DecisionTreeClassifier(max_depth=3, min_samples_split=10, criterion="entropy", random_state=42)
clf.fit(X_train, y_train)

# Evaluate on training set
train_predictions = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# Evaluate on test set
test_predictions = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Visualize the decision tree structure
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=["Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Pressure"],
          class_names=["No Rain", "Rain"], filled=True, rounded=True)
plt.title("Decision Tree Structure")
plt.show()

# Use only the first two features for 2D visualization
X_2d = X_train[:, :2]  # Select the first two features
y_2d = y_train

# Train a new decision tree on these two features
clf_2d = DecisionTreeClassifier(max_depth=3, min_samples_split=10, criterion="entropy", random_state=42)
clf_2d.fit(X_2d, y_2d)

# Plot decision boundaries
def plot_decision_boundaries(X, y, model, feature_names):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, predictions, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(
        X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor="k"
    )
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Decision Boundaries")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()

# Plot decision boundaries for the new 2D decision tree
feature_names = ["Temperature", "Humidity"]  # Update with appropriate feature names
plot_decision_boundaries(X_2d, y_2d, clf_2d, feature_names)
