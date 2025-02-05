import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")
# Display basic information and the first few rows
# df.info(), df.head()

# Select numerical features
numerical_features = ["math score", "reading score", "writing score"]
X = df[numerical_features].values

# Standardize the data (Z-score normalization)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0, ddof=1)
X_standardized = (X - X_mean) / X_std

# Compute the covariance matrix
cov_matrix = np.cov(X_standardized, rowvar=False)

# Perform eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Compute factor loadings (sqrt of eigenvalues * eigenvectors)
factor_loadings = eigenvectors * np.sqrt(eigenvalues)

# Display factor loadings
factor_loadings

# Plot factor loadings (only first two factors for visualization)
fig, ax = plt.subplots(figsize=(8, 6))
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)

for i, feature in enumerate(["Math Score", "Reading Score", "Writing Score"]):
    ax.arrow(0, 0, factor_loadings[i, 0], factor_loadings[i, 1],
             head_width=0.05, head_length=0.05, color='b', alpha=0.75)
    ax.text(factor_loadings[i, 0] + 0.05, factor_loadings[i, 1] + 0.05, feature, fontsize=12)

ax.set_xlabel("Factor 1")
ax.set_ylabel("Factor 2")
ax.set_title("Factor Loadings Biplot")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
plt.grid(True)
plt.show()