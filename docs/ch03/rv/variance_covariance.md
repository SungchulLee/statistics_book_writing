# Variance and Covariance

## Overview

While the expected value summarizes the center of a distribution, **variance** measures how spread out the distribution is around the mean. **Covariance** and **correlation** capture the degree to which two random variables move together. These concepts are essential for risk measurement, portfolio theory, and statistical inference.

---

## Variance

### Definition

The **variance** of a random variable $X$ is the expected squared deviation from the mean:

$$
\text{Var}(X) = E\left[(X - \mu)^2\right] = E[X^2] - (E[X])^2
$$

where $\mu = E[X]$. The second form, $E[X^2] - (E[X])^2$, is often more convenient for computation.

### Standard Deviation

The **standard deviation** is the square root of variance, returning the spread to the original units:

$$
\sigma_X = \text{SD}(X) = \sqrt{\text{Var}(X)}
$$

---

## Properties of Variance

1. **Non-negativity:** $\text{Var}(X) \geq 0$, with equality if and only if $X$ is constant.
2. **Constant:** $\text{Var}(c) = 0$
3. **Scaling:** $\text{Var}(aX) = a^2 \text{Var}(X)$
4. **Shift invariance:** $\text{Var}(X + c) = \text{Var}(X)$
5. **Sum (independent):** If $X \perp\!\!\!\perp Y$, then $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

For general (possibly dependent) random variables:

$$
\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)
$$

---

## Covariance

### Definition

The **covariance** of $X$ and $Y$ measures how they co-vary:

$$
\text{Cov}(X, Y) = E\left[(X - \mu_X)(Y - \mu_Y)\right] = E[XY] - E[X] \cdot E[Y]
$$

- $\text{Cov}(X, Y) > 0$: $X$ and $Y$ tend to move in the same direction.
- $\text{Cov}(X, Y) < 0$: $X$ and $Y$ tend to move in opposite directions.
- $\text{Cov}(X, Y) = 0$: no linear relationship (but they may still be dependent).

### Properties of Covariance

1. **Self-covariance:** $\text{Cov}(X, X) = \text{Var}(X)$
2. **Symmetry:** $\text{Cov}(X, Y) = \text{Cov}(Y, X)$
3. **Bilinearity:** $\text{Cov}(aX + b, cY + d) = ac \cdot \text{Cov}(X, Y)$
4. **Independence implies zero:** If $X \perp\!\!\!\perp Y$, then $\text{Cov}(X, Y) = 0$ (but the converse is false)

---

## Correlation

The **Pearson correlation coefficient** normalizes covariance to lie in $[-1, 1]$:

$$
\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}
$$

- $\rho = 1$: perfect positive linear relationship
- $\rho = -1$: perfect negative linear relationship
- $\rho = 0$: no linear relationship (uncorrelated)

**Important:** Uncorrelated ($\rho = 0$) does not imply independent. For example, if $X \sim N(0,1)$ and $Y = X^2$, then $\text{Cov}(X, Y) = E[X^3] = 0$ but $X$ and $Y$ are clearly dependent.

---

## Variance of a Sum (General Case)

For any random variables $X_1, \ldots, X_n$:

$$
\text{Var}\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \text{Var}(X_i) + 2\sum_{i < j} \text{Cov}(X_i, X_j)
$$

If all $X_i$ are pairwise uncorrelated, the cross terms vanish and:

$$
\text{Var}\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \text{Var}(X_i)
$$

---

## Examples

### Example: Variance of a Fair Die

$$
E[X] = 3.5, \quad E[X^2] = \frac{1^2 + 2^2 + \cdots + 6^2}{6} = \frac{91}{6}
$$

$$
\text{Var}(X) = \frac{91}{6} - 3.5^2 = \frac{91}{6} - \frac{49}{4} = \frac{35}{12} \approx 2.917
$$

### Example: Bernoulli Random Variable

For $X \sim \text{Bernoulli}(p)$:

$$
E[X] = p, \quad E[X^2] = p, \quad \text{Var}(X) = p - p^2 = p(1-p)
$$

The variance is maximized at $p = 0.5$ (maximum uncertainty) and equals zero at $p = 0$ or $p = 1$ (certainty).

### Example: Portfolio Variance

Two assets with returns $R_1$ and $R_2$, weights $w_1$ and $w_2$ ($w_1 + w_2 = 1$). The portfolio return is $R_p = w_1 R_1 + w_2 R_2$:

$$
\text{Var}(R_p) = w_1^2 \sigma_1^2 + w_2^2 \sigma_2^2 + 2w_1 w_2 \text{Cov}(R_1, R_2)
$$

When $\rho < 1$, diversification reduces portfolio variance below the weighted average of individual variances.

---

## Python Exploration

```python
import numpy as np

# Variance of a fair die
values = np.arange(1, 7)
probs = np.ones(6) / 6
E_X = np.sum(values * probs)
E_X2 = np.sum(values**2 * probs)
var_X = E_X2 - E_X**2
print(f"E[X] = {E_X:.4f}")
print(f"E[X²] = {E_X2:.4f}")
print(f"Var(X) = {var_X:.4f}")
print(f"SD(X) = {np.sqrt(var_X):.4f}")
```

```python
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_correlation():
    """Show uncorrelated does not imply independent."""
    np.random.seed(42)
    n = 10_000

    X = np.random.randn(n)
    Y = X ** 2  # deterministically dependent on X

    cov_XY = np.cov(X, Y)[0, 1]
    corr_XY = np.corrcoef(X, Y)[0, 1]

    print(f"Cov(X, X²) = {cov_XY:.4f} (theoretically 0)")
    print(f"Corr(X, X²) = {corr_XY:.4f}")
    print(f"Yet X and X² are clearly dependent!")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(X[:500], Y[:500], alpha=0.3, s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y = X²')
    ax.set_title(f'Uncorrelated but Dependent (ρ = {corr_XY:.3f})')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()

demonstrate_correlation()
```

```python
import numpy as np
import matplotlib.pyplot as plt

def portfolio_variance_demo():
    """Demonstrate diversification benefit."""
    sigma1, sigma2 = 0.20, 0.30
    correlations = [-0.5, 0.0, 0.5, 1.0]

    fig, ax = plt.subplots(figsize=(12, 4))
    weights = np.linspace(0, 1, 100)

    for rho in correlations:
        cov_12 = rho * sigma1 * sigma2
        port_var = (weights**2 * sigma1**2
                    + (1 - weights)**2 * sigma2**2
                    + 2 * weights * (1 - weights) * cov_12)
        port_sd = np.sqrt(port_var)
        ax.plot(weights, port_sd, label=f'ρ = {rho}')

    ax.set_xlabel('Weight in Asset 1')
    ax.set_ylabel('Portfolio Std Dev')
    ax.set_title('Diversification: Portfolio Risk vs. Allocation')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()

portfolio_variance_demo()
```

---

## Key Takeaways

- **Variance** measures dispersion: $\text{Var}(X) = E[X^2] - (E[X])^2$.
- **Covariance** measures linear co-movement; **correlation** normalizes it to $[-1, 1]$.
- Uncorrelated ($\rho = 0$) does **not** imply independence.
- For independent variables, the variance of a sum equals the sum of variances; for dependent variables, covariance terms must be included.
- In finance, the covariance structure of asset returns determines the diversification benefit of portfolios.
