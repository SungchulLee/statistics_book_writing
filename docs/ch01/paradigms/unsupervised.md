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

## Exercises

**Exercise 1.**
Given 2D data points $\{(1,2), (1.5, 1.8), (5, 8), (8, 8), (1, 0.6), (9, 11)\}$, apply one iteration of the k-means algorithm with $k=2$ and initial centroids $\boldsymbol{\mu}_1 = (1, 2)$ and $\boldsymbol{\mu}_2 = (5, 8)$. Report the cluster assignments and new centroids.

??? success "Solution to Exercise 1"
    **Assignment step** (assign each point to the nearest centroid using Euclidean distance):

    | Point | Distance to $\boldsymbol{\mu}_1$ | Distance to $\boldsymbol{\mu}_2$ | Cluster |
    |---|---|---|---|
    | $(1, 2)$ | 0 | 7.21 | 1 |
    | $(1.5, 1.8)$ | 0.54 | 7.12 | 1 |
    | $(5, 8)$ | 7.21 | 0 | 2 |
    | $(8, 8)$ | 9.22 | 3.0 | 2 |
    | $(1, 0.6)$ | 1.4 | 8.46 | 1 |
    | $(9, 11)$ | 12.04 | 5.0 | 2 |

    **Update step** (compute new centroids):

    $$
    \boldsymbol{\mu}_1' = \frac{1}{3}\bigl((1,2) + (1.5, 1.8) + (1, 0.6)\bigr) = (1.167, 1.467)
    $$

    $$
    \boldsymbol{\mu}_2' = \frac{1}{3}\bigl((5,8) + (8,8) + (9,11)\bigr) = (7.333, 9.0)
    $$

---

**Exercise 2.**
Explain why PCA (Principal Component Analysis) finds the directions of maximum variance. What is the relationship between PCA and the eigendecomposition of the covariance matrix?

??? success "Solution to Exercise 2"
    PCA seeks the direction $\mathbf{w}$ (unit vector) that maximizes the variance of the projected data: $\max_{\lVert\mathbf{w}\rVert=1} \mathbf{w}^T\mathbf{S}\mathbf{w}$, where $\mathbf{S}$ is the sample covariance matrix. By the Rayleigh quotient theory, the maximum is achieved when $\mathbf{w}$ is the eigenvector corresponding to the largest eigenvalue of $\mathbf{S}$.

    The eigendecomposition $\mathbf{S} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$ directly provides all principal components: the columns of $\mathbf{Q}$ are the principal component directions, and the diagonal entries of $\boldsymbol{\Lambda}$ are the variances explained by each component. The first PC captures the most variance, the second PC captures the most remaining variance orthogonal to the first, and so on.

---

**Exercise 3.**
What is the key difference between clustering and dimensionality reduction as unsupervised learning tasks? Can they be used together?

??? success "Solution to Exercise 3"
    **Clustering** assigns each data point to a discrete group (cluster label), partitioning the data into subsets of similar observations. The output is a categorical assignment.

    **Dimensionality reduction** maps high-dimensional data to a lower-dimensional representation that preserves important structure (variance, distances, or neighborhood relationships). The output is a continuous embedding.

    They can be used together in several ways: (1) apply PCA first to reduce dimensions, then cluster in the reduced space (often more effective because the curse of dimensionality is mitigated); (2) use t-SNE or UMAP for visualization, then visually identify clusters; (3) cluster first, then use the cluster labels to color a dimensionality-reduction plot for interpretation.

---

**Exercise 4.**
A dataset has 1000 observations and 500 features. Explain why directly applying k-means might perform poorly, and describe how dimensionality reduction can help.

??? success "Solution to Exercise 4"
    In 500 dimensions, k-means suffers from the **curse of dimensionality**: Euclidean distances between points become nearly equal (all points are roughly equidistant), making it difficult to distinguish clusters. Many features may be noise, diluting the signal from the truly informative features.

    Dimensionality reduction helps by:

    1. **Removing noise:** PCA retains only the top components that capture the most variance, discarding noisy dimensions that obscure cluster structure.
    2. **Improving distance metrics:** In lower dimensions, Euclidean distance is more meaningful and clusters are more separable.
    3. **Computational efficiency:** k-means on 10-50 PCA components is much faster than on 500 raw features.

    A typical workflow: apply PCA to retain 90-95% of the variance (often reducing to 20-50 components), then run k-means on the reduced data.

---

**Exercise 5.**
The **elbow method** picks the number of clusters $k$ for k-means by plotting within-cluster sum of squares (WCSS) against $k$ and looking for a "kink." Why is the elbow always at some $k$, and what is a more principled alternative?

??? success "Solution to Exercise 5"
    WCSS is monotonically non-increasing in $k$: adding clusters can only reduce within-cluster variation. The plot looks like a decay curve, and any inflection point is a candidate "elbow." The choice is subjective — different analysts often pick different elbows on the same plot.

    Principled alternatives:

    - **Gap statistic** (Tibshirani, Walther, Hastie 2001): compare the WCSS of the observed data to the expected WCSS under a null reference distribution (uniform over the data's bounding box). Choose the smallest $k$ where the gap is approximately maximal.
    - **Silhouette coefficient**: for each point, compare its cohesion (mean distance to its own cluster) to its separation (mean distance to the nearest other cluster). The average silhouette across $k$ gives a clearer optimum.
    - **Information criteria** for model-based clustering (BIC for Gaussian mixture models): penalize $k$ explicitly.

    All these methods reduce — but do not eliminate — the inherent ambiguity in unsupervised tasks: "the correct number of clusters" is not a well-defined population parameter the way "the mean" is.

---

**Exercise 6.**
**Anomaly detection** can be framed as a one-class classification problem. Describe how the **isolation forest** algorithm differs in approach from a k-means-based anomaly detector, and one situation where each is preferred.

??? success "Solution to Exercise 6"
    **k-means-based anomaly detection:** fit k-means on the data; declare a point anomalous if its distance to its assigned cluster centroid exceeds some threshold. The model represents normal data as a small number of dense regions; anomalies are points far from any region.

    **Isolation forest** (Liu, Ting, Zhou 2008): build many random binary trees by repeatedly splitting on a random feature at a random threshold. Anomalies tend to be **isolated quickly** — they require few splits to be alone in a leaf. The anomaly score is the average path length to isolation; short paths signal anomalies. No notion of "normal cluster" is required.

    **k-means preferred:** when normal data forms a small number of clearly defined clusters (e.g., manufacturing yields concentrated around standard products). Anomalies are deviations from these clusters.

    **Isolation forest preferred:** when normal data has complex, possibly multi-modal structure that does not partition into a few centroids (e.g., financial transactions). Isolation forest scales well to high dimensions and does not require choosing $k$ — it discovers anomalies by the sparseness of their neighborhood rather than by distance to centroids.
