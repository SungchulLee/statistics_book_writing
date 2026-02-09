# Unsupervised Learning (Pattern Discovery)

## Overview

In **unsupervised learning**, the model is given data **without explicit labels or outcomes**. The task is to uncover hidden patterns, structures, or relationships within the data. There is no "correct answer" to learn from—the algorithm must discover structure on its own.

## Key Characteristics

- **No labels**: The training data consists only of input features $X$; there is no target variable $Y$.
- **Exploratory**: The goal is discovery—finding groups, reducing complexity, or detecting anomalies.
- **Evaluation is harder**: Without labels, there is no straightforward way to measure "accuracy." Domain expertise and downstream utility guide evaluation.

## Common Tasks

### Clustering

Group similar observations together. Each cluster should contain data points that are more similar to each other than to points in other clusters.

- **K-Means**: Partitions data into $k$ clusters by minimizing within-cluster variance.
- **Hierarchical Clustering**: Builds a tree (dendrogram) of nested clusters.
- **DBSCAN**: Identifies clusters of arbitrary shape based on density.

**Finance application:** Grouping stocks with similar return patterns for portfolio diversification or regime detection.

### Dimensionality Reduction

Reduce the number of features while preserving the most important information.

- **Principal Component Analysis (PCA)**: Projects data onto orthogonal directions of maximum variance.
- **t-SNE / UMAP**: Non-linear methods for visualizing high-dimensional data in 2D or 3D.
- **Autoencoders**: Neural networks that learn compressed representations.

**Finance application:** Reducing hundreds of correlated risk factors to a handful of principal components for factor modeling.

### Anomaly Detection

Identify data points that deviate significantly from the expected pattern.

- **Isolation Forest**: Isolates anomalies by random partitioning.
- **One-Class SVM**: Learns a boundary around "normal" data.

**Finance application:** Detecting fraudulent transactions or unusual trading patterns.

## Example: Customer Segmentation

```python
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate customer data: annual spending vs. frequency of visits
cluster_centers = [(20, 5), (50, 30), (80, 15)]
data = np.vstack([
    np.random.normal(loc=c, scale=[8, 4], size=(100, 2))
    for c in cluster_centers
])

# K-Means clustering (k=3)
km = KMeans(n_clusters=3, random_state=0, n_init=10)
labels = km.fit_predict(data)

plt.figure(figsize=(7, 4))
for k in range(3):
    mask = labels == k
    plt.scatter(data[mask, 0], data[mask, 1], label=f"Cluster {k}", alpha=0.6)
plt.xlabel("Annual Spending ($k)")
plt.ylabel("Visit Frequency")
plt.title("Customer Segmentation via K-Means")
plt.legend()
plt.tight_layout()
plt.show()
```

No labels were provided—the algorithm discovered three natural groupings based solely on spending and visit patterns.

## Key Takeaways

- Unsupervised learning is about **discovering structure** in unlabeled data.
- Common tasks include clustering, dimensionality reduction, and anomaly detection.
- Evaluation requires domain knowledge because there is no ground-truth label to compare against.
- In finance, unsupervised methods are widely used for segmentation, factor extraction, and fraud detection.
