# Ordinary Least Squares Simulation (Monte Carlo)

## Overview

This page demonstrates OLS estimation from a linear-algebra perspective through Monte Carlo simulation. We verify key theoretical properties: the normal equation estimator, projection matrices and their idempotency, the ANOVA decomposition, unbiased variance estimation, and the sampling distribution of $\hat{\boldsymbol{\beta}}$. Repeated simulation confirms that $\hat{\boldsymbol{\beta}}$ is unbiased and its empirical standard deviation matches the theoretical standard error.

## Mathematical Background

### OLS Estimator

For the model $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{u}$ with $\mathbf{u} \sim N(\mathbf{0}, \sigma^2\mathbf{I})$:

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}.
$$

### Projection Matrices

The **projection matrix** $\mathbf{P} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$ projects onto the column space of $\mathbf{X}$:

$$
\hat{\mathbf{y}} = \mathbf{P}\mathbf{y}.
$$

The **annihilator matrix** $\mathbf{M} = \mathbf{I} - \mathbf{P}$ projects onto the orthogonal complement:

$$
\mathbf{e} = \mathbf{M}\mathbf{y}.
$$

Both are symmetric and idempotent ($\mathbf{P}^2 = \mathbf{P}$, $\mathbf{M}^2 = \mathbf{M}$), with $\operatorname{tr}(\mathbf{P}) = k$ and $\operatorname{tr}(\mathbf{M}) = n - k$.

### ANOVA Decomposition

$$
\underbrace{\sum(y_i - \bar{y})^2}_{\mathrm{TSS}} = \underbrace{\sum(\hat{y}_i - \bar{y})^2}_{\mathrm{ESS}} + \underbrace{\sum(y_i - \hat{y}_i)^2}_{\mathrm{RSS}}.
$$

### Unbiased Variance Estimator

$$
s^2 = \frac{\mathbf{e}^\top\mathbf{e}}{n - k}, \qquad E[s^2] = \sigma^2.
$$

### Covariance of the Estimator

$$
\mathrm{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}, \qquad \widehat{\mathrm{Var}}(\hat{\boldsymbol{\beta}}) = s^2(\mathbf{X}^\top\mathbf{X})^{-1}.
$$

## Code

### Core Functions

```python
import numpy as np

def gen_X(n, k):
    return np.hstack([np.ones((n, 1)), np.random.randn(n, k - 1)])

def ols(y, X):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def proj_P(X):
    return X @ np.linalg.inv(X.T @ X) @ X.T

def proj_M(X):
    return np.eye(X.shape[0]) - proj_P(X)

def anova_decomposition(y, X, beta_hat):
    y_bar = y.mean()
    y_hat = X @ beta_hat
    TSS = float(np.sum((y - y_bar) ** 2))
    ESS = float(np.sum((y_hat - y_bar) ** 2))
    RSS = float(np.sum((y - y_hat) ** 2))
    return TSS, ESS, RSS
```

### Monte Carlo Verification

```python
def monte_carlo(n=100, beta_true=[2, 3, -1], sigma=1.0, n_sim=5000):
    k = len(beta_true)
    estimates = np.empty((n_sim, k))
    for i in range(n_sim):
        X = gen_X(n, k)
        beta = np.array(beta_true).reshape(-1, 1)
        u = np.random.randn(n, 1) * sigma
        y = X @ beta + u
        bhat = ols(y, X)
        estimates[i] = bhat.flatten()
    return estimates

beta_true = [2, 3, -1]
estimates = monte_carlo(n=200, beta_true=beta_true, sigma=2.0, n_sim=5000)
mc_mean = estimates.mean(axis=0)
mc_std = estimates.std(axis=0, ddof=1)

for j in range(len(beta_true)):
    print(f"beta_{j}: true={beta_true[j]}, "
          f"MC mean={mc_mean[j]:.4f}, MC std={mc_std[j]:.4f}")
```

## Interpretation

- **Unbiasedness**: The Monte Carlo mean of $\hat{\beta}_j$ should be close to the true $\beta_j$. With 5000 replications, the MC mean typically agrees with the truth to within $\pm 0.05$.
- **Projection matrices**: $\mathbf{P}$ and $\mathbf{M}$ are the fundamental geometric objects of OLS. $\mathbf{P}$ projects onto the fitted-value subspace; $\mathbf{M}$ projects onto the residual subspace. Their idempotency and complementarity ($\mathbf{P} + \mathbf{M} = \mathbf{I}$) encode the orthogonal decomposition of $\mathbf{y}$.
- **ANOVA**: The identity TSS = ESS + RSS decomposes total variation into explained and unexplained parts. $R^2 = \mathrm{ESS}/\mathrm{TSS}$.
- **Variance estimation**: $s^2$ is unbiased for $\sigma^2$ because $\operatorname{tr}(\mathbf{M}) = n - k$ accounts for the degrees of freedom lost in estimation.

## Exercises

**Exercise 1.** Verify numerically that $\mathbf{P}$ and $\mathbf{M}$ are idempotent and symmetric for a specific realization of $\mathbf{X}$.

??? success "Solution to Exercise 1"

    ```python
    X = gen_X(50, 3)
    P = proj_P(X)
    M = proj_M(X)
    print("P idempotent:", np.allclose(P @ P, P))
    print("M idempotent:", np.allclose(M @ M, M))
    print("P symmetric:", np.allclose(P, P.T))
    print("M symmetric:", np.allclose(M, M.T))
    print("tr(P):", np.trace(P))  # should be 3
    print("tr(M):", np.trace(M))  # should be 47
    ```

    All checks pass: $\mathbf{P}^2 = \mathbf{P}$, $\mathbf{M}^2 = \mathbf{M}$, both symmetric, $\operatorname{tr}(\mathbf{P}) = k = 3$, $\operatorname{tr}(\mathbf{M}) = n - k = 47$. $\square$

---

**Exercise 2.** Modify the Monte Carlo to estimate the coverage probability of the 95% confidence interval $\hat{\beta}_j \pm t^*_{n-k,0.025} \cdot \mathrm{SE}(\hat{\beta}_j)$. Is it close to 95%?

??? success "Solution to Exercise 2"

    ```python
    from scipy import stats
    coverage = np.zeros(3)
    n, sigma = 200, 2.0
    beta_true_arr = np.array(beta_true)
    for i in range(5000):
        X = gen_X(n, 3)
        y = X @ beta_true_arr.reshape(-1, 1) + sigma * np.random.randn(n, 1)
        bhat = ols(y, X)
        e = y - X @ bhat
        s2 = np.sum(e ** 2) / (n - 3)
        se = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))
        t_star = stats.t(n - 3).ppf(0.975)
        for j in range(3):
            if bhat[j, 0] - t_star * se[j] <= beta_true[j] <= bhat[j, 0] + t_star * se[j]:
                coverage[j] += 1
    print("Coverage:", coverage / 5000)  # should be ~0.95
    ```

    The empirical coverage should be approximately 0.95 for each coefficient, confirming the theory. $\square$

---

**Exercise 3.** Show that $\mathbf{P}\mathbf{M} = \mathbf{0}$ and interpret this geometrically.

??? success "Solution to Exercise 3"

    Since $\mathbf{M} = \mathbf{I} - \mathbf{P}$:

    $$
    \mathbf{P}\mathbf{M} = \mathbf{P}(\mathbf{I} - \mathbf{P}) = \mathbf{P} - \mathbf{P}^2 = \mathbf{P} - \mathbf{P} = \mathbf{0}.
    $$

    Geometrically, $\mathbf{P}$ projects onto $\mathrm{col}(\mathbf{X})$ and $\mathbf{M}$ projects onto $\mathrm{col}(\mathbf{X})^\perp$. These subspaces are orthogonal, so projecting onto one and then the other yields the zero vector. This is why $\hat{\mathbf{y}}$ and $\mathbf{e}$ are orthogonal: $\hat{\mathbf{y}}^\top\mathbf{e} = (\mathbf{P}\mathbf{y})^\top(\mathbf{M}\mathbf{y}) = \mathbf{y}^\top\mathbf{P}\mathbf{M}\mathbf{y} = 0$. $\square$

---

**Exercise 4.** Increase $\sigma$ from 2 to 10 while keeping $n = 200$. How do the MC standard deviations and $R^2$ distribution change?

??? success "Solution to Exercise 4"

    With $\sigma = 10$, the noise-to-signal ratio increases 5-fold. The MC standard deviations of $\hat{\beta}_j$ increase proportionally (by a factor of 5), since $\mathrm{SE}(\hat{\beta}_j) \propto \sigma$. The $R^2$ values decrease dramatically because the explained sum of squares remains roughly the same while the total sum of squares grows by a factor of 25 ($\sigma^2$ ratio). Many simulations may produce $R^2$ near zero. $\square$

---

**Exercise 5.** Prove that $E[s^2] = \sigma^2$ using the trace of $\mathbf{M}$.

??? success "Solution to Exercise 5"

    The residual vector is $\mathbf{e} = \mathbf{M}\mathbf{y} = \mathbf{M}\mathbf{u}$ (since $\mathbf{M}\mathbf{X} = \mathbf{0}$). Then

    $$
    E[\mathbf{e}^\top\mathbf{e}] = E[\mathbf{u}^\top\mathbf{M}^\top\mathbf{M}\mathbf{u}] = E[\mathbf{u}^\top\mathbf{M}\mathbf{u}] = E[\operatorname{tr}(\mathbf{u}\mathbf{u}^\top\mathbf{M})],
    $$

    using the trace trick $\mathbf{u}^\top\mathbf{M}\mathbf{u} = \operatorname{tr}(\mathbf{M}\mathbf{u}\mathbf{u}^\top)$. Taking expectation:

    $$
    E[\operatorname{tr}(\mathbf{M}\mathbf{u}\mathbf{u}^\top)] = \operatorname{tr}(\mathbf{M}\,E[\mathbf{u}\mathbf{u}^\top]) = \operatorname{tr}(\mathbf{M}\sigma^2\mathbf{I}) = \sigma^2\operatorname{tr}(\mathbf{M}) = \sigma^2(n - k).
    $$

    Dividing by $n - k$: $E[s^2] = E[\mathbf{e}^\top\mathbf{e}/(n-k)] = \sigma^2$. $\square$
