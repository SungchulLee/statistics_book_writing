# Covariance and Correlation

## Overview

**Covariance** and **correlation** quantify the linear relationship between two random variables. Covariance measures the direction and magnitude of co-movement (in original units), while correlation normalizes this to a dimensionless quantity between $-1$ and $+1$.

---

## Covariance

### Definition

$$
\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]
$$

### Proof of Equivalent Forms

$$
\begin{aligned}
\text{Cov}(X,Y) &= E[(X - \mu_X)(Y - \mu_Y)] \\
&= E[XY - X\mu_Y - \mu_X Y + \mu_X \mu_Y] \\
&= E[XY] - \mu_Y E[X] - \mu_X E[Y] + \mu_X \mu_Y \\
&= E[XY] - E[X]E[Y]
\end{aligned}
$$

### Properties

$$
\begin{aligned}
(1) &\quad \text{Cov}(X, X) = \text{Var}(X) \\[4pt]
(2) &\quad \text{Cov}(X, Y) = \text{Cov}(Y, X) \quad \text{(symmetry)} \\[4pt]
(3) &\quad \text{Cov}(aX + b, \, cY + d) = ac \cdot \text{Cov}(X, Y) \\[4pt]
(4) &\quad \text{Cov}\left(\sum_i X_i, \sum_j Y_j\right) = \sum_i \sum_j \text{Cov}(X_i, Y_j) \quad \text{(bilinearity)} \\[4pt]
(5) &\quad X \perp Y \implies \text{Cov}(X, Y) = 0
\end{aligned}
$$

**Warning:** The converse of (5) is **false** in general. Zero covariance does not imply independence.

### Variance of a Sum

The general formula for the variance of a sum follows from bilinearity:

$$
\text{Var}\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \text{Var}(X_i) + 2\sum_{i < j} \text{Cov}(X_i, X_j)
$$

For two variables:

$$
\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)
$$

$$
\text{Var}(X - Y) = \text{Var}(X) + \text{Var}(Y) - 2\text{Cov}(X, Y)
$$

---

## Correlation

### Definition

The **Pearson correlation coefficient** normalizes covariance by the standard deviations:

$$
\rho(X, Y) = \text{Corr}(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)\,\text{Var}(Y)}}
$$

### Properties

$$
\begin{aligned}
(1) &\quad -1 \leq \rho(X, Y) \leq 1 \\[4pt]
(2) &\quad \rho(X, Y) = \pm 1 \iff Y = aX + b \text{ for some } a \neq 0 \\[4pt]
(3) &\quad \rho(aX + b, \, cY + d) = \text{sign}(ac) \cdot \rho(X, Y) \\[4pt]
(4) &\quad \rho(X, Y) = 0 \text{ means } X, Y \text{ are **uncorrelated** (no linear relationship)}
\end{aligned}
$$

### Proof that $|\rho| \leq 1$ (Cauchy–Schwarz)

By the Cauchy–Schwarz inequality:

$$
|E[UV]|^2 \leq E[U^2] \cdot E[V^2]
$$

Setting $U = X - \mu_X$ and $V = Y - \mu_Y$:

$$
|\text{Cov}(X,Y)|^2 \leq \text{Var}(X) \cdot \text{Var}(Y) \implies |\rho(X,Y)| \leq 1
$$

---

## Interpreting Correlation

| $\rho$ | Interpretation |
|:---|:---|
| $\rho = +1$ | Perfect positive linear relationship |
| $0.7 \leq \rho < 1$ | Strong positive association |
| $0.3 \leq \rho < 0.7$ | Moderate positive association |
| $0 < \rho < 0.3$ | Weak positive association |
| $\rho = 0$ | No linear relationship |
| $\rho < 0$ | Negative association (analogous) |
| $\rho = -1$ | Perfect negative linear relationship |

**Caution:** Correlation measures only **linear** dependence. Variables can be strongly dependent yet have zero correlation if the relationship is nonlinear.

---

## Covariance Matrix

For a random vector $\mathbf{X} = (X_1, X_2, \ldots, X_n)^\top$, the **covariance matrix** is:

$$
\boldsymbol{\Sigma} = \text{Cov}(\mathbf{X}) = E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top]
$$

$$
\Sigma_{ij} = \text{Cov}(X_i, X_j), \qquad \Sigma_{ii} = \text{Var}(X_i)
$$

The covariance matrix is always symmetric and positive semi-definite.

### Correlation Matrix

$$
R_{ij} = \frac{\Sigma_{ij}}{\sqrt{\Sigma_{ii} \Sigma_{jj}}} = \rho(X_i, X_j)
$$

The diagonal entries of $\mathbf{R}$ are all 1.

---

## Worked Example

**Problem:** Compute the covariance and correlation from the joint PMF:

| | $Y=0$ | $Y=1$ |
|:---|:---:|:---:|
| $X=0$ | 0.2 | 0.1 |
| $X=1$ | 0.3 | 0.4 |

**Solution:**

$$
E[X] = 0(0.3) + 1(0.7) = 0.7, \quad E[Y] = 0(0.5) + 1(0.5) = 0.5
$$

$$
E[XY] = 0(0) \cdot 0.2 + 0(1) \cdot 0.1 + 1(0) \cdot 0.3 + 1(1) \cdot 0.4 = 0.4
$$

$$
\text{Cov}(X,Y) = E[XY] - E[X]E[Y] = 0.4 - 0.7 \cdot 0.5 = 0.05
$$

$$
\text{Var}(X) = E[X^2] - (E[X])^2 = 0.7 - 0.49 = 0.21
$$

$$
\text{Var}(Y) = E[Y^2] - (E[Y])^2 = 0.5 - 0.25 = 0.25
$$

$$
\rho(X,Y) = \frac{0.05}{\sqrt{0.21 \cdot 0.25}} = \frac{0.05}{0.2291} \approx 0.218
$$

---

## Python: Computing and Visualizing

### Covariance and Correlation from Data

```python
import numpy as np

np.random.seed(42)
n = 10_000
X = np.random.normal(0, 1, n)
Y = 0.7 * X + np.random.normal(0, 0.5, n)

cov_matrix = np.cov(X, Y)
corr_matrix = np.corrcoef(X, Y)

print(f"Cov(X,Y) = {cov_matrix[0,1]:.4f}")
print(f"Corr(X,Y) = {corr_matrix[0,1]:.4f}")
print(f"\nCovariance matrix:\n{cov_matrix}")
print(f"\nCorrelation matrix:\n{corr_matrix}")
```

### Visualizing Different Correlations

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 500
fig, axes = plt.subplots(1, 4, figsize=(14, 3))

for ax, rho in zip(axes, [-0.9, -0.3, 0.3, 0.9]):
    cov = [[1, rho], [rho, 1]]
    data = np.random.multivariate_normal([0, 0], cov, n)
    ax.scatter(data[:, 0], data[:, 1], s=5, alpha=0.5)
    ax.set_title(f'ρ = {rho}')
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()
```

### Correlation Heatmap

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 5000
X1 = np.random.normal(0, 1, n)
X2 = 0.5 * X1 + np.random.normal(0, 1, n)
X3 = -0.3 * X1 + 0.6 * X2 + np.random.normal(0, 1, n)
X4 = np.random.normal(0, 1, n)

data = np.column_stack([X1, X2, X3, X4])
corr = np.corrcoef(data, rowvar=False)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
labels = ['X1', 'X2', 'X3', 'X4']
ax.set_xticks(range(4))
ax.set_xticklabels(labels)
ax.set_yticks(range(4))
ax.set_yticklabels(labels)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', fontsize=10)
fig.colorbar(im, ax=ax)
plt.show()
```

### Sample Covariance from Joint PMF

```python
import numpy as np

# Joint PMF table
pmf = np.array([[0.2, 0.1],
                [0.3, 0.4]])
x_vals = np.array([0, 1])
y_vals = np.array([0, 1])

E_X = np.sum(x_vals[:, None] * pmf)
E_Y = np.sum(y_vals[None, :] * pmf)
E_XY = np.sum(x_vals[:, None] * y_vals[None, :] * pmf)

cov_XY = E_XY - E_X * E_Y
var_X = np.sum(x_vals[:, None]**2 * pmf) - E_X**2
var_Y = np.sum(y_vals[None, :]**2 * pmf) - E_Y**2
corr_XY = cov_XY / np.sqrt(var_X * var_Y)

print(f"E[X] = {E_X:.4f}, E[Y] = {E_Y:.4f}, E[XY] = {E_XY:.4f}")
print(f"Cov(X,Y) = {cov_XY:.4f}")
print(f"Corr(X,Y) = {corr_XY:.4f}")
```

---

## Key Takeaways

- Covariance measures the direction and magnitude of linear co-movement; correlation standardizes it to $[-1, 1]$.
- The shortcut formula $\text{Cov}(X,Y) = E[XY] - E[X]E[Y]$ is usually the most efficient for computation.
- Correlation captures only **linear** dependence; zero correlation does not imply independence.
- The covariance matrix generalizes pairwise covariances to vector-valued random variables and is fundamental to portfolio theory, PCA, and multivariate statistics.
- The variance of a sum formula $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$ simplifies to additive variances only under independence or zero correlation.
