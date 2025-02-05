import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data preprocessing
data = pd.read_csv("weather_forecast_data.csv")
X = data[["Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Pressure"]].values
y = (data["Rain"] == "rain").astype(int).values  # Convert "rain"/"no rain" to 1/0

# Split the data into training and test sets (80-20 split)
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return {"type": "leaf", "class": np.argmax(np.bincount(y))}

        best_feature, best_threshold, best_score, splits = self._find_best_split(X, y)
        if splits is None:
            return {"type": "leaf", "class": np.argmax(np.bincount(y))}

        left_X, left_y, right_X, right_y = splits

        return {
            "type": "node",
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(left_X, left_y, depth + 1),
            "right": self._build_tree(right_X, right_y, depth + 1),
        }

    def _find_best_split(self, X, y):
        best_score = float("inf")
        best_feature, best_threshold, best_splits = None, None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if sum(left_mask) < self.min_samples_split or sum(right_mask) < self.min_samples_split:
                    continue

                score = self._calculate_loss(y[left_mask], y[right_mask])

                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
                    best_splits = (X[left_mask], y[left_mask], X[right_mask], y[right_mask])

        return best_feature, best_threshold, best_score, best_splits

    def _calculate_loss(self, left_y, right_y):
        if self.criterion == "gini":
            return self._gini_impurity(left_y, right_y)
        elif self.criterion == "entropy":
            return self._cross_entropy(left_y, right_y)

    def _gini_impurity(self, left_y, right_y):
        def gini(y):
            proportions = np.bincount(y) / len(y)
            return 1 - np.sum(proportions**2)

        n = len(left_y) + len(right_y)
        return (len(left_y) / n) * gini(left_y) + (len(right_y) / n) * gini(right_y)

    def _cross_entropy(self, left_y, right_y):
        def entropy(y):
            proportions = np.bincount(y) / len(y)
            return -np.sum(proportions * np.log2(proportions + 1e-9))

        n = len(left_y) + len(right_y)
        return (len(left_y) / n) * entropy(left_y) + (len(right_y) / n) * entropy(right_y)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, tree):
        if tree["type"] == "leaf":
            return tree["class"]

        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_one(x, tree["left"])
        else:
            return self._predict_one(x, tree["right"])

# Train the decision tree
dt = DecisionTree(max_depth=3, min_samples_split=10, criterion="entropy")
dt.fit(X_train, y_train)

# Evaluate on training set
train_predictions = dt.predict(X_train)
train_accuracy = np.mean(train_predictions == y_train)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# Evaluate on test set
test_predictions = dt.predict(X_test)
test_accuracy = np.mean(test_predictions == y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Use only the first two features for 2D visualization
X_2d = X_train[:, :2]  # Select the first two features
y_2d = y_train

# Train a new decision tree on these two features
dt_2d = DecisionTree(max_depth=3, min_samples_split=10, criterion="entropy")
dt_2d.fit(X_2d, y_2d)

def plot_decision_boundaries(X, y, tree, feature_names):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict for each point in the grid
    def predict_grid(grid):
        return np.array([tree._predict_one(point, tree.tree) for point in grid])

    predictions = predict_grid(grid).reshape(xx.shape)

    # Plot decision boundaries
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
plot_decision_boundaries(X_2d, y_2d, dt_2d, feature_names)
