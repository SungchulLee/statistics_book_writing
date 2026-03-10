# Unsupervised Learning

Unsupervised learning discovers hidden structure in data without labels. Common tasks include clustering, dimensionality reduction, and anomaly detection.

## Definition

Given only input features $\{\mathbf{x}_i\}_{i=1}^n$ (no target $y$), unsupervised learning finds patterns, groupings, or compressed representations. There is no loss function with ground-truth labels; evaluation relies on domain expertise and downstream utility.

## Explanation

**Clustering** groups similar observations: K-Means minimizes within-cluster variance, hierarchical clustering builds a dendrogram, and DBSCAN finds density-based clusters of arbitrary shape.

**Dimensionality reduction** projects data onto fewer dimensions while preserving structure: PCA finds orthogonal directions of maximum variance; t-SNE/UMAP provide nonlinear embeddings for visualization.

**Anomaly detection** identifies points that deviate from the expected pattern: Isolation Forest isolates anomalies by random partitioning.

Applications in finance: grouping stocks by return patterns for diversification, extracting principal component factors from correlated risk measures, and detecting fraudulent transactions.

## Examples

```python
import numpy as np

np.random.seed(42)

# Simulate 3 clusters
centers = np.array([[20, 5], [50, 30], [80, 15]])
data = np.vstack([
    np.random.normal(loc=c, scale=[8, 4], size=(100, 2))
    for c in centers
])

# Simple K-Means implementation
k = 3
centroids = data[np.random.choice(len(data), k, replace=False)]
for iteration in range(20):
    dists = np.linalg.norm(data[:, None] - centroids[None, :], axis=2)
    labels = dists.argmin(axis=1)
    new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(k)])
    if np.allclose(centroids, new_centroids):
        break
    centroids = new_centroids

for j in range(k):
    cluster = data[labels == j]
    print(f"Cluster {j}: n={len(cluster)}, "
          f"center=({cluster.mean(0)[0]:.1f}, {cluster.mean(0)[1]:.1f})")
print(f"Converged in {iteration + 1} iterations")
```
