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

## Exercises

**Exercise 1.**
PMF: $P(X = 1, 2, 3, 4) = 0.1, 0.3, 0.4, 0.2$. (a) $\mathbb{E}[X]$. (b) $\mathbb{E}[X^2]$, $\mathrm{Var}(X)$. (c) $Y = 3X + 5$: $\mathbb{E}[Y]$, $\mathrm{Var}(Y)$.

??? success "Solution to Exercise 1"
    (a) $\mathbb{E}[X] = 1(0.1) + 2(0.3) + 3(0.4) + 4(0.2) = 2.7$.

    (b) $\mathbb{E}[X^2] = 1(0.1) + 4(0.3) + 9(0.4) + 16(0.2) = 8.1$. $\mathrm{Var}(X) = 8.1 - 7.29 = 0.81$.

    (c) $\mathbb{E}[Y] = 3 \cdot 2.7 + 5 = 13.1$. $\mathrm{Var}(Y) = 9 \cdot 0.81 = 7.29$. Constant shift does not affect variance; multiplication scales variance by the square.

---

**Exercise 2.**
**Prove the variance-of-sum formula:** $\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\mathrm{Cov}(X, Y)$.

??? success "Solution to Exercise 2"
    Let $\mu_X = \mathbb{E}[X]$ and $\mu_Y = \mathbb{E}[Y]$. Then $\mathbb{E}[X + Y] = \mu_X + \mu_Y$, and

    $$
    \mathrm{Var}(X + Y) = \mathbb{E}[(X + Y - \mu_X - \mu_Y)^2] = \mathbb{E}[((X - \mu_X) + (Y - \mu_Y))^2]
    $$

    Expand the square:

    $$
    = \mathbb{E}[(X - \mu_X)^2] + 2\mathbb{E}[(X - \mu_X)(Y - \mu_Y)] + \mathbb{E}[(Y - \mu_Y)^2]
    $$

    $$
    = \mathrm{Var}(X) + 2\mathrm{Cov}(X, Y) + \mathrm{Var}(Y)
    $$

    $\square$

    **Generalization:** $\mathrm{Var}(\sum_i X_i) = \sum_i \mathrm{Var}(X_i) + 2\sum_{i < j} \mathrm{Cov}(X_i, X_j)$. The double-sum structure is why correlations affect portfolio variance in finance: $N$ assets contribute $N(N-1)/2$ correlation terms in addition to $N$ variance terms.

---

**Exercise 3.**
**Uncorrelated $\ne$ independent.** Let $X \sim \mathrm{Uniform}(-1, 1)$ and $Y = X^2$. Show $\mathrm{Cov}(X, Y) = 0$ but $X$ and $Y$ are dependent.

??? success "Solution to Exercise 3"
    By symmetry of the uniform around 0: $\mathbb{E}[X] = 0$. Also $\mathbb{E}[X^3] = 0$ (odd function over symmetric domain).

    $\mathrm{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] = \mathbb{E}[X \cdot X^2] - 0 = \mathbb{E}[X^3] = 0$.

    So $X$ and $Y$ are *uncorrelated*. But knowing $X = 0.5$ tells you exactly that $Y = 0.25$ — they are *deterministically dependent*. The correlation coefficient measures only the linear association, missing the quadratic structure.

    **Lesson:** zero correlation is a *necessary* but not *sufficient* condition for independence. For multivariate normals (and a few special distributions), zero correlation does imply independence — but this is an exception, not the rule. Always plot the data; don't rely on correlation alone.

---

**Exercise 4.**
**Portfolio variance with two assets.** Two assets have $\sigma_1 = 0.20$, $\sigma_2 = 0.30$, $\rho = 0.30$. Find the portfolio weights $w_1, w_2$ ($w_1 + w_2 = 1$, both non-negative) minimizing portfolio variance.

??? success "Solution to Exercise 4"
    Portfolio variance:

    $$
    \sigma_p^2(w_1) = w_1^2 \sigma_1^2 + (1 - w_1)^2 \sigma_2^2 + 2 w_1(1 - w_1)\rho \sigma_1 \sigma_2
    $$

    Take the derivative with respect to $w_1$ and set to zero:

    $$
    \frac{d\sigma_p^2}{dw_1} = 2 w_1 \sigma_1^2 - 2(1 - w_1)\sigma_2^2 + 2(1 - 2w_1)\rho \sigma_1 \sigma_2 = 0
    $$

    Solve: $w_1^* = (\sigma_2^2 - \rho\sigma_1\sigma_2)/(\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2)$.

    With $\sigma_1 = 0.20$, $\sigma_2 = 0.30$, $\rho = 0.30$:

    Numerator: $0.09 - 0.30 \cdot 0.20 \cdot 0.30 = 0.09 - 0.018 = 0.072$.

    Denominator: $0.04 + 0.09 - 2 \cdot 0.30 \cdot 0.20 \cdot 0.30 = 0.13 - 0.036 = 0.094$.

    $w_1^* = 0.072/0.094 \approx 0.766$, $w_2^* \approx 0.234$.

    The minimum-variance portfolio puts more weight on the lower-volatility asset, as expected. The minimum variance is $\sigma_p^2 = 0.766^2 \cdot 0.04 + 0.234^2 \cdot 0.09 + 2 \cdot 0.766 \cdot 0.234 \cdot 0.018 \approx 0.0354$ — smaller than either individual variance alone.

---

**Exercise 5.**
**Covariance matrix.** Compute the $2 \times 2$ covariance matrix of $(X, Y)$ where $X \sim N(0, 1)$ and $Y = aX + Z$ with $Z \sim N(0, \sigma_Z^2)$ independent of $X$. What is $\rho(X, Y)$?

??? success "Solution to Exercise 5"
    Marginal variances:

    $\mathrm{Var}(X) = 1$, $\mathrm{Var}(Y) = a^2 \cdot 1 + \sigma_Z^2 = a^2 + \sigma_Z^2$.

    Covariance:

    $\mathrm{Cov}(X, Y) = \mathrm{Cov}(X, aX + Z) = a \mathrm{Var}(X) + \mathrm{Cov}(X, Z) = a + 0 = a$.

    Covariance matrix:

    $$
    \boldsymbol{\Sigma} = \begin{pmatrix} 1 & a \\ a & a^2 + \sigma_Z^2 \end{pmatrix}
    $$

    Correlation:

    $$
    \rho(X, Y) = \frac{a}{\sqrt{1 \cdot (a^2 + \sigma_Z^2)}} = \frac{a}{\sqrt{a^2 + \sigma_Z^2}}
    $$

    Special cases:

    - $\sigma_Z = 0$: $\rho = a/|a| = \pm 1$ — $Y$ is a deterministic function of $X$.
    - $\sigma_Z \to \infty$: $\rho \to 0$ — noise dominates, $X$ and $Y$ become essentially independent.

    This factorization $Y = aX + Z$ underlies linear regression. The coefficient $a$ is the regression slope; the correlation $\rho$ measures how much of $Y$'s variation is explained by $X$.

---

**Exercise 6.**
**Variance estimation from a sample.** Show that for an i.i.d. sample $X_1, \ldots, X_n$, the **sample covariance** $\hat{\mathrm{Cov}}(X, Y) = \frac{1}{n-1}\sum_i (X_i - \bar X)(Y_i - \bar Y)$ is an unbiased estimator of $\mathrm{Cov}(X, Y)$.

??? success "Solution to Exercise 6"
    Expand: $\sum_i (X_i - \bar X)(Y_i - \bar Y) = \sum_i X_i Y_i - n \bar X \bar Y$.

    Take expectations:

    $\mathbb{E}\sum X_i Y_i = n(\mathrm{Cov}(X, Y) + \mu_X \mu_Y)$ (each term contributes $\mathbb{E}[X_i Y_i] = \mathrm{Cov}(X, Y) + \mu_X \mu_Y$).

    $\mathbb{E}[n \bar X \bar Y]$: using independence across observations,

    $$
    \mathbb{E}[\bar X \bar Y] = \frac{1}{n^2}\sum_{i, j} \mathbb{E}[X_i Y_j] = \frac{1}{n^2}\left[n(\mathrm{Cov}(X, Y) + \mu_X \mu_Y) + n(n - 1)\mu_X \mu_Y\right]
    $$

    $= (\mathrm{Cov}(X, Y) + \mu_X \mu_Y)/n + (n - 1)\mu_X \mu_Y / n = \mathrm{Cov}(X, Y)/n + \mu_X \mu_Y$.

    So $\mathbb{E}[n \bar X \bar Y] = \mathrm{Cov}(X, Y) + n\mu_X \mu_Y$.

    Subtracting: $\mathbb{E}[\sum_i (X_i - \bar X)(Y_i - \bar Y)] = n(\mathrm{Cov}(X, Y) + \mu_X \mu_Y) - \mathrm{Cov}(X, Y) - n\mu_X \mu_Y = (n - 1)\mathrm{Cov}(X, Y)$.

    Dividing by $n - 1$: $\mathbb{E}[\hat{\mathrm{Cov}}] = \mathrm{Cov}(X, Y)$. $\square$

    The $n - 1$ denominator is **Bessel's correction** for covariance, identical to the correction for sample variance: estimating $\mu_X$ and $\mu_Y$ from the data "uses up" one degree of freedom.
