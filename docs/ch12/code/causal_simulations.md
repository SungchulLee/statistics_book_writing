# Causal Inference Simulations

## Overview

This page demonstrates two classic pitfalls that make correlation different from causation: confounding variables and Simpson's paradox. Through simulation we show how a hidden common cause can create a misleading association between two variables, and how the direction of a correlation can reverse when data are aggregated across subgroups.

---

## Confounding Variables

A **confounding variable** $Z$ influences both $X$ and $Y$, creating a spurious association between them even when $X$ has no direct effect on $Y$. The directed acyclic graph (DAG) for this scenario is:

$$
X \leftarrow Z \rightarrow Y
$$

### Simulation

We generate data where $Z$ is the true common cause:

```python
import numpy as np
from scipy import stats

np.random.seed(21)
n = 300
Z = np.random.randn(n)
X = 0.6 * Z + np.random.randn(n) * 0.5
Y = 0.8 * Z + np.random.randn(n) * 0.5
```

Here $Y$ depends on $Z$ alone, not on $X$, yet $X$ and $Y$ will appear correlated because both are driven by $Z$.

### Partial Correlation

To remove the confounding effect, we compute the **partial correlation** of $X$ and $Y$ given $Z$:

$$
r_{XY \cdot Z} = \frac{r_{XY} - r_{XZ}\, r_{YZ}}{\sqrt{(1 - r_{XZ}^2)(1 - r_{YZ}^2)}}
$$

```python
r_xy, _ = stats.pearsonr(X, Y)
r_xz, _ = stats.pearsonr(X, Z)
r_yz, _ = stats.pearsonr(Y, Z)

r_partial = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

print(f"Pearson r(X, Y)       = {r_xy:.3f}")
print(f"Partial r(X, Y | Z)   = {r_partial:.3f}")
```

After controlling for $Z$, the association between $X$ and $Y$ nearly vanishes, confirming that the observed correlation was entirely due to the confounder.

---

## Simpson's Paradox

**Simpson's paradox** occurs when a trend that appears in several subgroups reverses or disappears when the subgroups are combined. Mathematically, it is possible that:

$$
r_{\text{subgroup } A} < 0, \quad r_{\text{subgroup } B} < 0, \quad \text{but} \quad r_{\text{aggregate}} > 0
$$

### Simulation

We create two subgroups with different baseline levels:

```python
rng = np.random.default_rng(42)

n_a, n_b = 100, 100
x_a = rng.uniform(10, 30, n_a)
y_a = -0.4 * x_a + 30 + rng.normal(0, 2, n_a)

x_b = rng.uniform(25, 50, n_b)
y_b = -0.4 * x_b + 45 + rng.normal(0, 2, n_b)
```

Within each subgroup, $Y$ decreases with $X$ (slope $= -0.4$). But Group B has a higher intercept and higher $X$ values, so when we pool the data the aggregate trend is positive:

```python
x_all = np.concatenate([x_a, x_b])
y_all = np.concatenate([y_a, y_b])

r_all, _ = stats.pearsonr(x_all, y_all)
r_a, _ = stats.pearsonr(x_a, y_a)
r_b, _ = stats.pearsonr(x_b, y_b)

print(f"Aggregate  r = {r_all:+.3f}")
print(f"Subgroup A r = {r_a:+.3f}")
print(f"Subgroup B r = {r_b:+.3f}")
```

### Visualization

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x_a, y_a, label='Group A', alpha=0.6)
ax.scatter(x_b, y_b, label='Group B', alpha=0.6, marker='s')

slope, intercept = np.polyfit(x_all, y_all, 1)
xs = np.linspace(x_all.min(), x_all.max(), 100)
ax.plot(xs, slope * xs + intercept, 'k--', linewidth=2,
        label=f'Aggregate OLS (r={r_all:+.2f})')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title("Simpson's Paradox")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Interpretation

These simulations illustrate two fundamental lessons for statistical practice:

1. **Confounding.** When a hidden variable drives both $X$ and $Y$, the marginal correlation $r_{XY}$ is misleading. The partial correlation $r_{XY \cdot Z}$ removes this confounding, and in our simulation it drops to near zero, correctly reflecting the absence of a direct $X \to Y$ effect.

2. **Simpson's paradox.** Aggregating heterogeneous subgroups can reverse the direction of an association. The positive aggregate correlation is an artifact of the different baseline levels of the groups, not a property of the within-group relationship. This is why stratified analysis and careful consideration of confounders are essential before drawing causal conclusions from observational data.

---

## Exercises

**Exercise 1.**
Simulate a confounding scenario with $n = 500$ where $Z \sim \mathcal{N}(0, 1)$, $X = 0.9Z + \varepsilon_X$, and $Y = 0.3Z + \varepsilon_Y$ with $\varepsilon_X, \varepsilon_Y \sim \mathcal{N}(0, 0.3^2)$. Compute both $r_{XY}$ and the partial correlation $r_{XY \cdot Z}$. How does reducing the noise variance affect the difference between the two?

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats

    np.random.seed(10)
    n = 500
    Z = np.random.randn(n)
    X = 0.9 * Z + np.random.normal(0, 0.3, n)
    Y = 0.3 * Z + np.random.normal(0, 0.3, n)

    r_xy, _ = stats.pearsonr(X, Y)
    r_xz, _ = stats.pearsonr(X, Z)
    r_yz, _ = stats.pearsonr(Y, Z)
    r_partial = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

    print(f"r(X, Y)     = {r_xy:.4f}")
    print(f"r(X, Y | Z) = {r_partial:.4f}")
    ```

    The marginal correlation $r_{XY}$ will be moderately positive because both $X$ and $Y$ share the common cause $Z$. The partial correlation $r_{XY \cdot Z}$ will be close to zero. Reducing the noise variance makes $r_{XZ}$ and $r_{YZ}$ closer to their theoretical values, which makes the confounding effect more pronounced (larger $r_{XY}$) while the partial correlation remains near zero. $\square$

---

**Exercise 2.**
Construct a Simpson's paradox example with three subgroups (not two). Within each subgroup, the slope of $Y$ on $X$ should be $+2$, but the aggregate slope should be negative. Plot the result.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    groups = [(100, 0, 50), (100, 10, 30), (100, 20, 10)]
    # (n, x_center, y_intercept) with positive slope within each group

    fig, ax = plt.subplots()
    all_x, all_y = [], []

    for n, xc, yb in groups:
        x = np.random.normal(xc, 1.5, n)
        y = yb + 2 * (x - xc) + np.random.normal(0, 1, n)
        ax.scatter(x, y, alpha=0.5, s=15)
        all_x.extend(x)
        all_y.extend(y)

    all_x, all_y = np.array(all_x), np.array(all_y)
    m, b = np.polyfit(all_x, all_y, 1)
    xs = np.linspace(all_x.min(), all_x.max(), 100)
    ax.plot(xs, m * xs + b, 'k--', lw=2, label=f'Aggregate slope = {m:.2f}')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.tight_layout()
    plt.show()
    ```

    Each group has a positive within-group slope of $+2$, but the group intercepts decrease as the group mean of $X$ increases. When pooled, the between-group trend dominates and the aggregate slope becomes negative. $\square$

---

**Exercise 3.**
Derive the formula for the partial correlation $r_{XY \cdot Z}$ starting from the residuals of the linear regressions of $X$ on $Z$ and $Y$ on $Z$.

??? success "Solution to Exercise 3"

    Let $e_X = X - \hat{\beta}_{XZ} Z$ be the residuals from regressing $X$ on $Z$, and similarly $e_Y = Y - \hat{\beta}_{YZ} Z$. By definition, the partial correlation is:

    $$
    r_{XY \cdot Z} = r(e_X, e_Y)
    $$

    Using the projection properties of OLS, $e_X$ is the component of $X$ orthogonal to $Z$, and $e_Y$ is the component of $Y$ orthogonal to $Z$. Writing $\hat{\beta}_{XZ} = r_{XZ} \cdot s_X / s_Z$ and expanding the Pearson formula on the residuals, one obtains after algebraic simplification:

    $$
    r_{XY \cdot Z} = \frac{r_{XY} - r_{XZ}\, r_{YZ}}{\sqrt{(1 - r_{XZ}^2)(1 - r_{YZ}^2)}}
    $$

    This is the standard partial correlation formula. The numerator removes the linear association of each variable with $Z$; the denominator rescales to maintain the $[-1, 1]$ range. $\square$

---

**Exercise 4.**
In the Simpson's paradox simulation, what happens to the aggregate correlation if you make the two subgroups have the same intercept but different slopes (one positive, one negative)? Simulate and explain.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from scipy import stats

    np.random.seed(42)
    n = 200
    x_a = np.random.uniform(0, 20, n)
    y_a = 10 + 0.5 * x_a + np.random.normal(0, 2, n)

    x_b = np.random.uniform(0, 20, n)
    y_b = 10 - 0.5 * x_b + np.random.normal(0, 2, n)

    x_all = np.concatenate([x_a, x_b])
    y_all = np.concatenate([y_a, y_b])

    r_a, _ = stats.pearsonr(x_a, y_a)
    r_b, _ = stats.pearsonr(x_b, y_b)
    r_all, _ = stats.pearsonr(x_all, y_all)

    print(f"Group A r = {r_a:+.3f}")
    print(f"Group B r = {r_b:+.3f}")
    print(f"Aggregate r = {r_all:+.3f}")
    ```

    When both subgroups share the same intercept and $X$ range but have opposite slopes, the aggregate correlation is approximately zero. The positive and negative relationships cancel each other out. This is not strictly Simpson's paradox (the sign does not reverse), but it demonstrates how mixing heterogeneous groups can mask real within-group effects entirely. $\square$

---

**Exercise 5.**
Prove that if $X \perp Y \mid Z$ (conditional independence given $Z$) and all three variables are jointly normally distributed, then the partial correlation $r_{XY \cdot Z} = 0$.

??? success "Solution to Exercise 5"

    For jointly normal random variables, the conditional distribution of $(X, Y) \mid Z$ is also bivariate normal. The conditional covariance is:

    $$
    \text{Cov}(X, Y \mid Z) = \sigma_{XY} - \frac{\sigma_{XZ}\, \sigma_{YZ}}{\sigma_{ZZ}}
    $$

    If $X \perp Y \mid Z$, then $\text{Cov}(X, Y \mid Z) = 0$, which gives:

    $$
    \sigma_{XY} = \frac{\sigma_{XZ}\, \sigma_{YZ}}{\sigma_{ZZ}}
    $$

    Converting to correlations by dividing by $\sigma_X \sigma_Y$:

    $$
    \rho_{XY} = \rho_{XZ}\, \rho_{YZ}
    $$

    Substituting into the partial correlation formula:

    $$
    \rho_{XY \cdot Z} = \frac{\rho_{XY} - \rho_{XZ}\, \rho_{YZ}}{\sqrt{(1 - \rho_{XZ}^2)(1 - \rho_{YZ}^2)}} = \frac{\rho_{XZ}\rho_{YZ} - \rho_{XZ}\rho_{YZ}}{\sqrt{(1 - \rho_{XZ}^2)(1 - \rho_{YZ}^2)}} = 0
    $$

    The converse also holds for jointly normal variables: $\rho_{XY \cdot Z} = 0$ implies $X \perp Y \mid Z$. This is a special property of the multivariate normal distribution. $\square$
