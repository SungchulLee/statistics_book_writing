# Correlation Visualization

## Overview

Visualizing a correlation matrix is essential for exploring multivariate datasets. This page demonstrates two standard techniques -- the correlation heatmap and the scatter matrix (pair plot) -- using Matplotlib. We simulate four variables with known correlation structure and show how these plots reveal linear dependencies at a glance.

---

## Simulating Correlated Variables

We construct four variables with a controlled correlation structure by mixing two independent standard normal sources $z_1$ and $z_2$:

$$
x_1 = z_1, \quad x_2 = 0.7\, z_1 + 0.3\, z_2, \quad x_3 = -0.5\, z_1 + 0.8\, \varepsilon, \quad x_4 = \varepsilon'
$$

where $\varepsilon$ and $\varepsilon'$ are independent standard normals. By construction, $x_1$ and $x_2$ have a strong positive correlation, $x_1$ and $x_3$ have a weak negative correlation, and $x_4$ is independent of the rest.

```python
import numpy as np

np.random.seed(7)
n = 200
z1 = np.random.randn(n)
z2 = np.random.randn(n)

x1 = z1
x2 = 0.7 * z1 + 0.3 * z2
x3 = -0.5 * z1 + np.random.randn(n) * 0.8
x4 = np.random.randn(n)

data = np.column_stack([x1, x2, x3, x4])
labels = ['X1', 'X2', 'X3', 'X4']
```

---

## The Correlation Matrix

The sample correlation matrix $\mathbf{R}$ is a symmetric $k \times k$ matrix whose $(i,j)$ entry is the Pearson correlation between variables $i$ and $j$:

$$
R_{ij} = \frac{\sum_{t=1}^{n}(x_{ti} - \bar{x}_i)(x_{tj} - \bar{x}_j)}{\sqrt{\sum_{t=1}^{n}(x_{ti} - \bar{x}_i)^2}\;\sqrt{\sum_{t=1}^{n}(x_{tj} - \bar{x}_j)^2}}
$$

The diagonal entries are always $R_{ii} = 1$, and the matrix is positive semi-definite.

```python
corr_matrix = np.corrcoef(data, rowvar=False)
```

---

## Correlation Heatmap

A heatmap encodes each entry of $\mathbf{R}$ as a color on a diverging scale. Blue typically indicates negative correlation and red indicates positive correlation, with white near zero.

```python
import matplotlib.pyplot as plt

k = data.shape[1]
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(k))
ax.set_yticks(range(k))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

for i in range(k):
    for j in range(k):
        ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                ha='center', va='center', fontsize=11,
                color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title('Correlation Heatmap')
plt.tight_layout()
plt.show()
```

The heatmap makes it immediately clear that $X_1$ and $X_2$ are strongly positively correlated (deep red), $X_1$ and $X_3$ are weakly negatively correlated (light blue), and $X_4$ is essentially uncorrelated with everything.

---

## Scatter Matrix (Pair Plot)

A scatter matrix displays all pairwise scatter plots in a grid, with univariate histograms on the diagonal. This provides a complete visual summary of both marginal distributions and bivariate relationships.

```python
fig, axes = plt.subplots(k, k, figsize=(10, 10))
for i in range(k):
    for j in range(k):
        ax = axes[i, j]
        if i == j:
            ax.hist(data[:, i], bins=20, edgecolor='k', alpha=0.7)
        else:
            ax.scatter(data[:, j], data[:, i], s=8, alpha=0.5)
        if j == 0:
            ax.set_ylabel(labels[i])
        if i == k - 1:
            ax.set_xlabel(labels[j])
        if j != 0:
            ax.set_yticklabels([])
        if i != k - 1:
            ax.set_xticklabels([])

fig.suptitle('Scatter Matrix (Pair Plot)', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
```

The scatter matrix reveals features that the heatmap cannot: nonlinear relationships, outliers, clusters, and the shape of marginal distributions.

---

## Interpretation

The heatmap and scatter matrix are complementary tools:

- **Heatmaps** excel at summarizing many pairwise correlations in a compact format. They are ideal for high-dimensional datasets where plotting all pairwise scatter plots is impractical.
- **Scatter matrices** provide richer information but become unwieldy when the number of variables exceeds approximately 8--10.

For a dataset with $k$ variables, there are $\binom{k}{2}$ distinct pairwise correlations. The correlation matrix is symmetric, so the heatmap is redundant across the diagonal. Some practitioners display only the lower triangle to save space.

A key caveat: the Pearson correlation heatmap captures only *linear* associations. A pair of variables can appear uncorrelated in the heatmap yet be strongly related through a nonlinear mapping. Always inspect scatter plots for important variable pairs.

---

## Exercises

**Exercise 1.**
Modify the simulation to create a fifth variable $x_5 = x_1^2 + \varepsilon$ with small noise. Compute the correlation matrix and note the Pearson correlation between $x_1$ and $x_5$. Explain why this value might be surprising given the deterministic relationship.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np

    np.random.seed(7)
    n = 200
    z1 = np.random.randn(n)
    x1 = z1
    x5 = x1 ** 2 + np.random.normal(0, 0.1, n)

    r = np.corrcoef(x1, x5)[0, 1]
    print(f"Pearson r(x1, x5) = {r:.4f}")
    ```

    The Pearson correlation between $x_1$ and $x_5 = x_1^2$ will be close to zero because the relationship is symmetric and nonlinear. Since $x_1 \sim \mathcal{N}(0, 1)$ is symmetric about zero, positive and negative deviations contribute equally, and the linear component of the relationship cancels out. The scatter plot would show a clear parabolic pattern that the heatmap completely misses. $\square$

---

**Exercise 2.**
Write a function that takes a correlation matrix and produces a heatmap showing only the lower triangle (masking the upper triangle and diagonal). This avoids redundant information.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    def lower_triangle_heatmap(corr, labels):
        k = corr.shape[0]
        mask = np.triu(np.ones_like(corr, dtype=bool))
        masked = np.ma.array(corr, mask=mask)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(masked, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)

        for i in range(k):
            for j in range(i):
                ax.text(j, i, f'{corr[i, j]:.2f}',
                        ha='center', va='center', fontsize=11)

        fig.colorbar(im, ax=ax)
        ax.set_title('Lower Triangle Correlation Heatmap')
        plt.tight_layout()
        plt.show()

    # Example usage
    np.random.seed(7)
    data = np.random.randn(200, 4)
    corr = np.corrcoef(data, rowvar=False)
    lower_triangle_heatmap(corr, ['X1', 'X2', 'X3', 'X4'])
    ```

    The upper triangle and diagonal are masked using `numpy.ma.array`. Since the correlation matrix is symmetric ($R_{ij} = R_{ji}$) and the diagonal is always 1, this eliminates redundant information and focuses attention on the $\binom{k}{2}$ unique correlations. $\square$

---

**Exercise 3.**
For $k$ variables, how many unique off-diagonal entries does the correlation matrix have? Prove that the correlation matrix $\mathbf{R}$ is always positive semi-definite.

??? success "Solution to Exercise 3"

    The correlation matrix $\mathbf{R}$ is a $k \times k$ symmetric matrix with ones on the diagonal. The number of unique off-diagonal entries is:

    $$
    \frac{k(k-1)}{2} = \binom{k}{2}
    $$

    To show $\mathbf{R}$ is positive semi-definite, let $\mathbf{Z}$ be the $n \times k$ matrix of standardized data (each column has mean zero and unit variance). Then the sample correlation matrix is:

    $$
    \mathbf{R} = \frac{1}{n-1}\mathbf{Z}^\top \mathbf{Z}
    $$

    For any vector $\mathbf{v} \in \mathbb{R}^k$:

    $$
    \mathbf{v}^\top \mathbf{R}\, \mathbf{v} = \frac{1}{n-1}\mathbf{v}^\top \mathbf{Z}^\top \mathbf{Z}\, \mathbf{v} = \frac{1}{n-1}\|\mathbf{Z}\mathbf{v}\|^2 \ge 0
    $$

    Since the quadratic form is non-negative for all $\mathbf{v}$, $\mathbf{R}$ is positive semi-definite. $\square$

---

**Exercise 4.**
Generate a dataset where the scatter matrix reveals a clear cluster structure (two clusters) but the correlation heatmap shows near-zero correlations. Explain why the heatmap fails here.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    n = 100
    # Cluster 1: centered at (2, 2)
    c1 = np.random.normal(loc=[2, 2], scale=0.5, size=(n, 2))
    # Cluster 2: centered at (-2, -2)
    c2 = np.random.normal(loc=[-2, -2], scale=0.5, size=(n, 2))
    data = np.vstack([c1, c2])

    r = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
    print(f"Pearson r = {r:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(data[:, 0], data[:, 1], s=10, alpha=0.6)
    ax1.set_title(f'Scatter Plot (r = {r:.3f})')
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')

    corr = np.corrcoef(data, rowvar=False)
    ax2.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_title('Heatmap')
    plt.tight_layout()
    plt.show()
    ```

    The overall Pearson $r$ is strongly positive because the two clusters align along the positive diagonal. However, if the clusters are arranged symmetrically (e.g., at $(2, -2)$ and $(-2, 2)$), the correlation becomes strongly negative. The heatmap shows a single number per pair and cannot represent the bimodal structure. The scatter plot immediately reveals the two clusters. This illustrates that correlation summaries can obscure important distributional features. $\square$

---

**Exercise 5.**
Prove that for standardized variables (mean zero, unit variance), the correlation matrix equals the covariance matrix. State the conditions under which this equivalence holds.

??? success "Solution to Exercise 5"

    Let $X_1, \ldots, X_k$ be random variables. Define the standardized versions:

    $$
    Z_i = \frac{X_i - \mu_i}{\sigma_i}
    $$

    The covariance of the standardized variables is:

    $$
    \text{Cov}(Z_i, Z_j) = \text{Cov}\!\left(\frac{X_i - \mu_i}{\sigma_i},\; \frac{X_j - \mu_j}{\sigma_j}\right) = \frac{\text{Cov}(X_i, X_j)}{\sigma_i \sigma_j} = \rho_{ij}
    $$

    Since $\text{Var}(Z_i) = 1$ for all $i$, the covariance matrix of the standardized variables is:

    $$
    \boldsymbol{\Sigma}_Z = \begin{pmatrix} 1 & \rho_{12} & \cdots & \rho_{1k} \\ \rho_{12} & 1 & \cdots & \rho_{2k} \\ \vdots & \vdots & \ddots & \vdots \\ \rho_{1k} & \rho_{2k} & \cdots & 1 \end{pmatrix} = \mathbf{R}
    $$

    This is exactly the correlation matrix of the original variables. The equivalence holds whenever each variable has unit variance. The variables need not have zero mean for the covariance matrix to equal the correlation matrix -- only unit variance is required, since covariance is invariant to location shifts. $\square$
