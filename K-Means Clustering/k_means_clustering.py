import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# Load and preprocess data
data = pd.read_csv("credit_card.csv")
data = data.drop(columns=["CUST_ID"])  # Remove categorical column
data = data.fillna(data.mean())  # Fill missing values with column mean

# Normalize data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Apply PCA to reduce the data to 2 dimensions
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_normalized)

def kMeansClustering(data, k, num_iterations):
    m, n = data.shape  # Get dataset size (m samples, n features)
    
    # Initialize cluster centroids randomly from data points
    centroids = data[np.random.choice(m, k, replace=False)]
    
    cluster_assignments = np.zeros(m)  # Store cluster index for each point
    
    for _ in range(num_iterations):
        # Assign each point to the nearest centroid
        for i in range(m):
            distances = np.linalg.norm(data[i] - centroids, axis=1)  # Compute distances
            cluster_assignments[i] = np.argmin(distances)  # Assign closest centroid
        
        # Update centroids
        for j in range(k):
            points_in_cluster = data[cluster_assignments == j]
            if len(points_in_cluster) > 0:
                centroids[j] = np.mean(points_in_cluster, axis=0)
    
    return cluster_assignments, centroids

# Run the model
k = 3
num_iterations = 100
# cluster_assignments, centroids = kMeansClustering(data_normalized, k, num_iterations)

# Perform k-means clustering on the PCA-transformed data
cluster_assignments, centroids = kMeansClustering(data_pca, k, num_iterations)

# Visualize the clusters in the 2D PCA space
plt.figure(figsize=(10, 6))
for i in range(k):
    plt.scatter(data_pca[cluster_assignments == i, 0], 
                data_pca[cluster_assignments == i, 1], 
                label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='*', label='Centroids')
plt.title('K-Means Clustering with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Get the PCA components (loadings)
pca_components = pca.components_

# Create a DataFrame for the PCA components
pca_loadings = pd.DataFrame(pca_components, columns=data.columns, index=[f'PC{i+1}' for i in range(pca.n_components_)])
print(pca_loadings)

# Plot the PCA loadings
plt.figure(figsize=(10, 6))
sns.heatmap(pca_loadings, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('PCA Loadings')
plt.show()

# Get the explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)