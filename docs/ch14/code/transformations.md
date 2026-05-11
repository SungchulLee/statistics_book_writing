# Transformation Demonstrations

## Overview

When data are non-normal, applying a mathematical transformation can often produce a distribution closer to normality, enabling the use of standard parametric procedures. This page demonstrates three common families of transformations -- logarithmic, square-root, and Box-Cox -- explains when each is appropriate, and illustrates how to assess improvement with graphical and formal checks.

## Why Transform

Many statistical methods (e.g., $t$-tests, ANOVA, linear regression) assume normally distributed errors. When the data are right-skewed or have non-constant variance, a suitable transformation can

1. symmetrise the distribution,
2. stabilise the variance, and
3. improve the approximation to normality.

## Logarithmic Transformation

For strictly positive data with right skew, the log transform is the first tool to try. If $X > 0$, define

$$
Y = \ln X.
$$

If $X \sim \text{Lognormal}(\mu, \sigma^2)$, then $Y \sim \mathcal{N}(\mu, \sigma^2)$ exactly. Even when the distribution is not exactly lognormal, the log transform often reduces skewness substantially.

### Code

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
x = rng.lognormal(mean=0.0, sigma=0.8, size=300)
y = np.log(x)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(x, bins=30, density=True, alpha=0.6, edgecolor="black")
axes[0].set_title("Original (Lognormal)")
axes[1].hist(y, bins=30, density=True, alpha=0.6, edgecolor="black")
axes[1].set_title("After log transform")
plt.tight_layout()
plt.show()

print(f"Before: skewness = {stats.skew(x, bias=False):.4f}")
print(f"After:  skewness = {stats.skew(y, bias=False):.4f}")
```

## Square-Root Transformation

For count data or data bounded below by zero, the square-root transform

$$
Y = \sqrt{X}
$$

is a milder correction than the logarithm. It is frequently used for Poisson-distributed counts, where the variance equals the mean and the square root approximately stabilises the variance.

## Box-Cox Transformation

The Box-Cox family generalises the log and power transforms through a single parameter $\lambda$:

$$
Y^{(\lambda)} =
\begin{cases}
\dfrac{X^\lambda - 1}{\lambda}, & \lambda \neq 0, \\[6pt]
\ln X, & \lambda = 0.
\end{cases}
$$

The optimal $\lambda$ is chosen by maximum likelihood. SciPy provides `stats.boxcox`, which returns the transformed data and the fitted $\lambda$.

### Code

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
x = rng.lognormal(mean=0.0, sigma=0.8, size=300)

y_bc, lam = stats.boxcox(x)
print(f"Optimal lambda: {lam:.4f}")

# Compare skewness
print(f"Before Box-Cox: skewness = {stats.skew(x, bias=False):.4f}")
print(f"After  Box-Cox: skewness = {stats.skew(y_bc, bias=False):.4f}")
```

When $\hat{\lambda} \approx 0$ the Box-Cox transform reduces to the log; when $\hat{\lambda} \approx 0.5$ it approximates the square root.

## Interpretation

After transforming, always re-check normality using both graphical methods (histogram, Q-Q plot) and formal tests (Shapiro-Wilk, Anderson-Darling). A transformation that removes skewness but introduces bimodality, for instance, has not improved the situation. Additionally, remember that inference conducted on the transformed scale must be back-transformed for interpretation on the original scale.

## Exercises

**Exercise 1.** Generate $n = 400$ observations from a $\text{Lognormal}(0, 0.6)$ distribution. Apply the log transformation and run the Shapiro-Wilk test on both the original and transformed data. Compare the $p$-values.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = rng.lognormal(0, 0.6, size=400)

    _, p_orig = stats.shapiro(x)
    _, p_log = stats.shapiro(np.log(x))

    print(f"Original:    p = {p_orig:.4g}")
    print(f"Log-transformed: p = {p_log:.4g}")
    ```

    The original data yield $p \approx 0$ (strong rejection), while the log-transformed data yield $p$ close to 1 (no evidence against normality). This is expected because $\ln X \sim \mathcal{N}(0, 0.36)$ exactly. $\square$

---

**Exercise 2.** Apply the Box-Cox transformation to $n = 300$ observations from a $\text{Gamma}(2, 1)$ distribution. Report the optimal $\hat{\lambda}$ and the Shapiro-Wilk $p$-value of the transformed data.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(1)
    x = rng.gamma(shape=2.0, scale=1.0, size=300)

    y_bc, lam = stats.boxcox(x)
    _, p_bc = stats.shapiro(y_bc)

    print(f"Optimal lambda: {lam:.4f}")
    print(f"Shapiro-Wilk p-value after Box-Cox: {p_bc:.4g}")
    ```

    The Gamma(2,1) distribution is moderately right-skewed. The optimal $\hat{\lambda}$ is typically around 0.4--0.5, and the Shapiro-Wilk $p$-value after transformation is generally well above 0.05, indicating that the Box-Cox transformation successfully normalises the data. $\square$

---

**Exercise 3.** Explain why the Box-Cox transformation requires $X > 0$. What modification can be used when the data contain zeros or negative values?

??? success "Solution to Exercise 3"

    The Box-Cox formula $Y^{(\lambda)} = (X^\lambda - 1)/\lambda$ involves raising $X$ to an arbitrary real power $\lambda$. If $X \leq 0$, then $X^\lambda$ is undefined (for non-integer $\lambda$) or can produce complex numbers. When data contain zeros or negative values, a common modification is the *shifted* Box-Cox transform: apply Box-Cox to $X + c$ where $c > 0$ is a constant chosen so that $X + c > 0$ for all observations. Alternatively, the Yeo-Johnson transformation extends Box-Cox to handle non-positive data natively by using different formulas for $X \geq 0$ and $X < 0$. $\square$

---

**Exercise 4.** Prove that the Box-Cox transformation with $\lambda = 0$ reduces to $Y = \ln X$ by taking the limit as $\lambda \to 0$.

??? success "Solution to Exercise 4"

    For $\lambda \neq 0$,

    $$
    Y^{(\lambda)} = \frac{X^\lambda - 1}{\lambda} = \frac{e^{\lambda \ln X} - 1}{\lambda}.
    $$

    Apply L'Hopital's rule (or expand $e^{\lambda \ln X} = 1 + \lambda \ln X + O(\lambda^2)$):

    $$
    \lim_{\lambda \to 0} \frac{e^{\lambda \ln X} - 1}{\lambda} = \lim_{\lambda \to 0} \frac{(\ln X)\, e^{\lambda \ln X}}{1} = \ln X.
    $$

    Hence $Y^{(0)} = \ln X$. $\square$

---

**Exercise 5.** Generate $n = 500$ Poisson($\lambda = 4$) observations. Apply the square-root transformation and compare the sample skewness before and after. Create side-by-side histograms.

??? success "Solution to Exercise 5"

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(2)
    x = rng.poisson(lam=4, size=500)
    y = np.sqrt(x)

    print(f"Original skewness:  {stats.skew(x, bias=False):.4f}")
    print(f"Sqrt skewness:      {stats.skew(y, bias=False):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(x, bins=range(15), density=True, alpha=0.6, edgecolor="black")
    axes[0].set_title("Poisson(4) — original")
    axes[1].hist(y, bins=20, density=True, alpha=0.6, edgecolor="black")
    axes[1].set_title("After sqrt transform")
    plt.tight_layout()
    plt.show()
    ```

    The Poisson(4) distribution has skewness $1/\sqrt{4} = 0.5$. After the square-root transformation, the skewness drops substantially (typically below 0.15), and the histogram appears much more symmetric. The square-root transform is a variance-stabilising transformation for the Poisson family because $\text{Var}(\sqrt{X}) \approx 1/4$ regardless of $\lambda$. $\square$
