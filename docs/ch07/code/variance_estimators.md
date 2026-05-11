# Variance Estimators

## Overview

Estimating the population variance $\sigma^2$ involves a fundamental choice of divisor: the naive MLE uses $1/n$, Bessel's correction uses $1/(n-1)$, and the MSE-optimal estimator (under normality) uses $1/(n+1)$. This page compares these three estimators, explores the degrees-of-freedom intuition, examines the benefit of knowing the true mean, and applies these ideas to financial volatility estimation.

## Three Variance Estimators

Given iid observations $X_1, \ldots, X_n$ from a population with variance $\sigma^2$, define the sum of squared deviations:

$$\text{SS} = \sum_{i=1}^n (X_i - \bar{X})^2$$

The three estimators are:

| Estimator | Formula | Bias | MSE (Normal) |
|-----------|---------|------|--------------|
| Naive (MLE) | $\tilde{S}^2 = \text{SS}/n$ | $-\sigma^2/n$ | $\frac{(2n-1)\sigma^4}{n^2}$ |
| Bessel's | $S^2 = \text{SS}/(n-1)$ | $0$ | $\frac{2\sigma^4}{n-1}$ |
| MSE-optimal | $\hat{S}^2 = \text{SS}/(n+1)$ | $-\frac{2\sigma^2}{n+1}$ | $\frac{2(n-1)\sigma^4 + 4\sigma^4}{(n+1)^2}$ |

## Bias Verification

The naive estimator has a predictable downward bias:

$$E[\tilde{S}^2] = \frac{n-1}{n}\sigma^2 \implies \text{Bias} = -\frac{\sigma^2}{n}$$

```python
import numpy as np

def bias_verification(sigma=3.0, n_sim=200_000, seed=42):
    rng = np.random.default_rng(seed)
    sigma2 = sigma**2
    sample_sizes = [3, 5, 10, 20, 50, 100, 500]

    for n in sample_sizes:
        samples = rng.normal(0, sigma, (n_sim, n))
        s_tilde2 = np.var(samples, axis=1, ddof=0)
        print(f"n={n:>4}  E[S̃²]={s_tilde2.mean():.4f}  "
              f"(n-1)/n·σ²={(n-1)/n*sigma2:.4f}  "
              f"Bias={s_tilde2.mean()-sigma2:.4f}  -σ²/n={-sigma2/n:.4f}")
```

!!! note "Bias shrinks with n"
    For $n = 3$ the bias is $-\sigma^2/3 = -3.0$, which is 33% of the true variance. By $n = 500$, the bias is negligible ($-0.018$). The bias matters most for small samples.

## MSE Comparison

The unbiased estimator ($1/(n-1)$) does **not** minimize MSE. The MSE-optimal estimator under normality uses $1/(n+1)$, trading a small bias for a larger reduction in variance.

```python
import matplotlib.pyplot as plt

def three_estimators_mse(sigma=3.0, n_sim=100_000, seed=42):
    rng = np.random.default_rng(seed)
    sigma2, sigma4 = sigma**2, sigma**4

    fig, ax = plt.subplots(figsize=(10, 6))
    ns = np.arange(3, 101)
    ax.plot(ns, (2*ns-1)/ns**2 * sigma4, 'b-', lw=2, label='1/n (naive / MLE)')
    ax.plot(ns, 2/(ns-1) * sigma4, 'r-', lw=2, label="1/(n-1) (Bessel's)")
    ax.plot(ns, (2*(ns-1)+4)/(ns+1)**2 * sigma4, 'g-', lw=2, label='1/(n+1) (MSE-optimal)')
    ax.set_xlabel('Sample Size n')
    ax.set_ylabel('MSE')
    ax.set_title('MSE of Variance Estimators (Normal Population)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

!!! info "Bias-variance tradeoff"
    The MSE-optimal estimator has the lowest MSE for all $n$, despite being biased. This is a clean illustration of the bias-variance tradeoff: sometimes accepting a small bias leads to lower overall estimation error.

## Degrees of Freedom Intuition

The $n$ deviations $d_i = X_i - \bar{X}$ satisfy the constraint:

$$\sum_{i=1}^n (X_i - \bar{X}) = 0$$

Only $n - 1$ of these deviations are free to vary independently. Dividing by the degrees of freedom corrects for the fact that $\bar{X}$ is closer to the data than $\mu$ is, systematically shrinking the sum of squares.

```python
def degrees_of_freedom_intuition(seed=42):
    rng = np.random.default_rng(seed)
    mu, sigma, n = 5.0, 2.0, 5
    sample = rng.normal(mu, sigma, n)
    x_bar = sample.mean()

    dev_xbar = sample - x_bar
    dev_mu   = sample - mu

    for i in range(n):
        print(f"  X_{i+1}={sample[i]:.3f}  "
              f"X_i-X̄={dev_xbar[i]:.3f}  X_i-μ={dev_mu[i]:.3f}")

    print(f"\n  Sum(X_i - X̄) = {sum(dev_xbar):.6f}  (always 0)")
    print(f"  Sum(X_i - μ)  = {sum(dev_mu):.3f}  (not 0)")
    print(f"  SS(X̄) = {np.sum(dev_xbar**2):.3f}")
    print(f"  SS(μ)  = {np.sum(dev_mu**2):.3f}")
    print(f"  Difference = n·(X̄−μ)² = {n*(x_bar-mu)**2:.3f}")
```

The key identity connecting the two sums of squares is:

$$\sum_{i=1}^n (X_i - \mu)^2 = \sum_{i=1}^n (X_i - \bar{X})^2 + n(\bar{X} - \mu)^2$$

Since $E[n(\bar{X} - \mu)^2] = \sigma^2$, the deviations from $\bar{X}$ underestimate the deviations from $\mu$ by exactly $\sigma^2$ on average.

## Known vs Unknown Mean

When the true mean $\mu$ is known, we can use:

$$\hat{\sigma}^2_{\text{known}} = \frac{1}{n}\sum_{i=1}^n (X_i - \mu)^2$$

This estimator is unbiased and has **lower variance** than $S^2$, because it does not lose a degree of freedom to estimate $\mu$.

```python
def known_vs_unknown_mean(sigma=3.0, n_sim=100_000, seed=42):
    rng = np.random.default_rng(seed)
    mu, sigma2 = 5.0, sigma**2
    sample_sizes = [5, 10, 25, 50, 100]

    for n in sample_sizes:
        samples = rng.normal(mu, sigma, (n_sim, n))
        est_known   = np.mean((samples - mu)**2, axis=1)
        est_unknown = np.var(samples, axis=1, ddof=0)
        mse_k = np.mean((est_known - sigma2)**2)
        mse_u = np.mean((est_unknown - sigma2)**2)
        print(f"n={n:>4}  MSE(known μ)={mse_k:.4f}  "
              f"MSE(unknown)={mse_u:.4f}  Ratio={mse_u/mse_k:.3f}")
```

## Financial Application: Volatility Estimation

In finance, volatility is typically estimated as the annualized standard deviation of returns. The choice of divisor ($n$ vs $n-1$) matters most for short estimation windows.

```python
def volatility_estimation_finance(seed=42):
    rng = np.random.default_rng(seed)
    annual_vol = 0.20
    daily_vol = annual_vol / np.sqrt(252)
    daily_mu = 0.08 / 252
    n_sim = 30_000

    windows = [5, 10, 21, 63, 126, 252]

    for w in windows:
        vol_n, vol_n1 = [], []
        for _ in range(n_sim):
            r = rng.normal(daily_mu, daily_vol, w)
            vol_n.append(np.sqrt(np.var(r, ddof=0) * 252))
            vol_n1.append(np.sqrt(np.var(r, ddof=1) * 252))
        print(f"Window={w:>4}  Vol(1/n)={np.mean(vol_n)*100:.2f}%  "
              f"Vol(1/(n-1))={np.mean(vol_n1)*100:.2f}%  "
              f"Diff={(np.mean(vol_n1)-np.mean(vol_n))/np.mean(vol_n)*100:.2f}%")
```

!!! warning "Short windows amplify the difference"
    With a 5-day window, the Bessel-corrected volatility is roughly 12% higher than the naive estimate. For quarterly (63-day) and longer windows, the difference is negligible. In practice, many financial applications use $n-1$ by default.

## Interpretation

- The **naive estimator** ($1/n$) is biased downward, underestimating $\sigma^2$ by exactly $\sigma^2/n$.
- **Bessel's correction** ($1/(n-1)$) removes the bias but does not minimize MSE.
- The **MSE-optimal** estimator ($1/(n+1)$ for normal data) accepts a small bias for a larger reduction in variance — a classic illustration of the bias-variance tradeoff.
- **Degrees of freedom** provide the intuition: estimating the mean "uses up" one degree of freedom.
- Knowing the **true mean** yields a better variance estimator. In practice, this rarely happens, but it motivates ideas like shrinkage estimation.
- In finance, the divisor choice matters mainly for **short estimation windows** (weekly or biweekly).

## Exercises

**Exercise 1.**
Prove that $E[\tilde{S}^2] = \frac{n-1}{n}\sigma^2$ where $\tilde{S}^2 = \frac{1}{n}\sum_{i=1}^n(X_i - \bar{X})^2$, using the identity $\sum(X_i - \bar{X})^2 = \sum(X_i - \mu)^2 - n(\bar{X} - \mu)^2$.

??? success "Solution to Exercise 1"
    Starting from the identity:

    $$\sum_{i=1}^n(X_i - \bar{X})^2 = \sum_{i=1}^n(X_i - \mu)^2 - n(\bar{X} - \mu)^2$$

    Taking expectations:

    $$E\left[\sum_{i=1}^n(X_i - \bar{X})^2\right] = \sum_{i=1}^n E[(X_i - \mu)^2] - nE[(\bar{X} - \mu)^2] = n\sigma^2 - n\cdot\frac{\sigma^2}{n} = (n-1)\sigma^2$$

    Therefore:

    $$E[\tilde{S}^2] = E\left[\frac{1}{n}\sum_{i=1}^n(X_i - \bar{X})^2\right] = \frac{(n-1)\sigma^2}{n} = \frac{n-1}{n}\sigma^2$$

    The bias is $E[\tilde{S}^2] - \sigma^2 = -\sigma^2/n$. $\square$

---

**Exercise 2.**
Compute the MSE of the Bessel-corrected estimator $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$ for normal data, using the fact that $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$.

??? success "Solution to Exercise 2"
    Let $Q = (n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$. Then $S^2 = Q\sigma^2/(n-1)$.

    Since $S^2$ is unbiased, $\text{MSE}(S^2) = \text{Var}(S^2)$.

    $$\text{Var}(S^2) = \frac{\sigma^4}{(n-1)^2}\text{Var}(Q) = \frac{\sigma^4}{(n-1)^2}\cdot 2(n-1) = \frac{2\sigma^4}{n-1}$$

    where we used $\text{Var}(\chi^2_k) = 2k$ with $k = n - 1$. $\square$

---

**Exercise 3.**
Show that the MSE-optimal divisor for estimating $\sigma^2$ under normality is $n + 1$ by minimizing $\text{MSE}(\text{SS}/d)$ over $d > 0$.

??? success "Solution to Exercise 3"
    Let the estimator be $\hat{\sigma}^2 = \text{SS}/d$ where $\text{SS} = \sum(X_i - \bar{X})^2$ and $\text{SS}/\sigma^2 \sim \chi^2_{n-1}$.

    Then $E[\text{SS}] = (n-1)\sigma^2$ and $\text{Var}(\text{SS}) = 2(n-1)\sigma^4$.

    $$\text{Bias} = \frac{(n-1)\sigma^2}{d} - \sigma^2 = \sigma^2\left(\frac{n-1}{d} - 1\right)$$

    $$\text{Var}\left(\frac{\text{SS}}{d}\right) = \frac{2(n-1)\sigma^4}{d^2}$$

    $$\text{MSE} = \text{Bias}^2 + \text{Var} = \sigma^4\left(\frac{n-1}{d} - 1\right)^2 + \frac{2(n-1)\sigma^4}{d^2}$$

    Setting $\frac{d(\text{MSE}/\sigma^4)}{dd} = 0$:

    $$2\left(\frac{n-1}{d} - 1\right)\left(-\frac{n-1}{d^2}\right) + 2(n-1)\left(-\frac{2}{d^3}\right)\sigma^4/\sigma^4 = 0$$

    Simplifying (multiply through by $d^3$):

    $$-2(n-1)\left[(n-1) - d\right] - 4(n-1) = 0$$

    $$(n-1) - d + 2 = 0 \implies d = n + 1$$

    $\square$

---

**Exercise 4.**
A portfolio manager estimates daily volatility from 21 trading days (one month) of returns. If the true daily volatility is 1.26%, compute the expected annualized volatility estimate using both the $1/n$ and $1/(n-1)$ divisors. Which one is closer to the true annualized volatility of 20%?

??? success "Solution to Exercise 4"
    True daily volatility: $\sigma_d = 0.0126$. True annualized: $\sigma_a = 0.0126 \times \sqrt{252} \approx 0.20$ (20%).

    With 21 days of data:

    **Using $1/n$:** $E[\tilde{S}^2] = \frac{n-1}{n}\sigma_d^2 = \frac{20}{21}\sigma_d^2$. Expected annualized vol:

    $$\sqrt{\frac{20}{21}} \times 20\% \approx \sqrt{0.9524} \times 20\% \approx 0.976 \times 20\% = 19.52\%$$

    **Using $1/(n-1)$:** $E[S^2] = \sigma_d^2$. But note that $E[S] < \sigma_d$ by Jensen's inequality (since square root is concave). The expected annualized vol is:

    $$E[\sqrt{S^2 \times 252}] = \sqrt{252}\cdot E[S] < \sqrt{252}\cdot \sigma_d = 20\%$$

    So neither estimator is unbiased for $\sigma$ (as opposed to $\sigma^2$). However, the $1/(n-1)$ estimator is closer to the true value. The bias of $S$ can be corrected using the $c_4$ factor from the chi-squared distribution. $\square$

---

**Exercise 5.**
Explain why knowing the true mean $\mu$ reduces the MSE of variance estimation. Quantify the improvement for $n = 5$.

??? success "Solution to Exercise 5"
    When $\mu$ is known, we use $\hat{\sigma}^2 = \frac{1}{n}\sum(X_i - \mu)^2$, which is unbiased with:

    $$\text{Var}(\hat{\sigma}^2) = \frac{1}{n^2}\text{Var}\left(\sum(X_i-\mu)^2\right) = \frac{1}{n^2}\cdot n\cdot\text{Var}((X-\mu)^2)$$

    For normal data, $(X-\mu)^2/\sigma^2 \sim \chi^2_1$, so $\text{Var}((X-\mu)^2) = 2\sigma^4$, giving:

    $$\text{MSE}(\hat{\sigma}^2_{\text{known}}) = \frac{2\sigma^4}{n}$$

    When $\mu$ is unknown, the best unbiased estimator has:

    $$\text{MSE}(S^2) = \frac{2\sigma^4}{n-1}$$

    For $n = 5$:

    $$\frac{\text{MSE}(S^2)}{\text{MSE}(\hat{\sigma}^2_{\text{known}})} = \frac{2\sigma^4/4}{2\sigma^4/5} = \frac{5}{4} = 1.25$$

    Knowing $\mu$ reduces MSE by 20%. The improvement comes from using all $n$ degrees of freedom for variance estimation instead of $n - 1$. As $n$ grows, the ratio approaches 1. $\square$
