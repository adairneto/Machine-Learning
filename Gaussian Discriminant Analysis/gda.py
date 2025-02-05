import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Data preprocessing
dataset = pd.read_csv("IRIS.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
np.random.seed(0)
indices = np.random.permutation(len(x))
train_size = int(0.75 * len(x))
train_indices = indices[:train_size]
test_indices = indices[train_size:]
X_train, X_test = x[train_indices], x[test_indices]
y_train, y_test = y[train_indices], y[test_indices]
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Number of classes and features
num_classes = len(np.unique(y_train))
num_features = X_train.shape[1]

# Compute class prior, class mean and covariance matrix
priors = np.zeros(num_classes)
means = np.zeros((num_classes, num_features))
cov_matrix = np.zeros((num_features, num_features))

for k in range(num_classes):
    X_k = X_train[y_train == k]
    priors[k] = X_k.shape[0] / X_train.shape[0]
    means[k] = np.mean(X_k, axis=0)

cov_matrix = np.cov(X_train, rowvar=False)

# Gaussian Discriminant
def gaussian_discriminant(x, priors, means, cov_matrix):
    cov_inv = np.linalg.inv(cov_matrix)
    cov_det = np.linalg.det(cov_matrix)
    discriminants = []
    for k in range(num_classes):
        diff = x - means[k]
        exponent = -0.5 * diff.T @ cov_inv @ diff
        log_likelihood = -0.5 * np.log(cov_det) + exponent
        log_prior = np.log(priors[k])
        discriminants.append(log_likelihood + log_prior)
    return np.argmax(discriminants)

# Make predictions
y_pred = np.array([gaussian_discriminant(x, priors, means, cov_matrix) for x in X_test])

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")

# Reduce to first two features for visualization
X_train_2d = X_train[:, :2]
X_test_2d = X_test[:, :2]
means_2d = means[:, :2]

# Create a mesh grid
xx, yy = np.meshgrid(np.arange(X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1, 0.02),
                     np.arange(X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1, 0.02))

# Classify each point in the grid
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_predictions = np.array([gaussian_discriminant(x, priors, means_2d, cov_matrix[:2, :2]) for x in grid_points])
grid_predictions = grid_predictions.reshape(xx.shape)

# Configure figure size and style
plt.rcParams['figure.figsize'] = [12, 4]

# Plot the decision regions
plt.pcolormesh(xx, yy, grid_predictions, cmap=plt.cm.Paired, shading='auto')

# Plot the training points
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Paired, s=50)

# Labels and title
plt.xlabel('Feature 1 (Sepal Length)')
plt.ylabel('Feature 2 (Sepal Width)')
plt.title('Gaussian Discriminant Analysis Decision Regions')
plt.show()