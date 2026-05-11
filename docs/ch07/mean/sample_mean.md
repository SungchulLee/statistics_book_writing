# Sample Mean as Estimator

## Introduction

The **sample mean** $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ is the most fundamental estimator in all of statistics. It serves as the natural estimator for the population mean $\mu = E[X]$ and plays a central role in estimation theory, hypothesis testing, and nearly every branch of applied statistics and finance. Understanding its properties — why it works, when it works optimally, and when it fails — is essential for statistical practice.

## Definition

Given a random sample $X_1, X_2, \ldots, X_n$ from a population with mean $\mu$ and variance $\sigma^2$, the **sample mean** is:

$$\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$$

The sample mean is a **statistic** (a function of the data) and hence a random variable. Its value changes with each sample drawn.

## Properties

### Expectation

$$E[\bar{X}] = E\left[\frac{1}{n}\sum_{i=1}^n X_i\right] = \frac{1}{n}\sum_{i=1}^n E[X_i] = \frac{1}{n} \cdot n\mu = \mu$$

The sample mean is an **unbiased estimator** of $\mu$ regardless of the population distribution, sample size, or whether observations are identically distributed (as long as they share the same mean).

### Variance

For iid observations:

$$\text{Var}(\bar{X}) = \text{Var}\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) = \frac{\sigma^2}{n}$$

**Key implications:**

- Variance decreases linearly in $n$
- Standard error: $\text{SE}(\bar{X}) = \sigma/\sqrt{n}$
- To halve the standard error, quadruple the sample size

### Mean Squared Error

Since $\bar{X}$ is unbiased:

$$\text{MSE}(\bar{X}) = \text{Var}(\bar{X}) = \frac{\sigma^2}{n}$$

### Distribution of the Sample Mean

**For normal populations:** If $X_i \sim N(\mu, \sigma^2)$, then:

$$\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{exactly, for all } n$$

**Central Limit Theorem (any population):** For large $n$:

$$\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0, 1)$$

or equivalently $\bar{X} \approx N(\mu, \sigma^2/n)$ for large $n$.

## Optimality Properties

### MVUE for Normal Populations

For $X_i \sim N(\mu, \sigma^2)$ with known $\sigma^2$, the sample mean $\bar{X}$ is the **Minimum Variance Unbiased Estimator (MVUE)** of $\mu$. It achieves the Cramér-Rao Lower Bound:

$$\text{Var}(\bar{X}) = \frac{\sigma^2}{n} = \frac{1}{nI_1(\mu)}$$

where $I_1(\mu) = 1/\sigma^2$ is the Fisher information per observation.

### Sufficiency

For the normal distribution, $\bar{X}$ is a **sufficient statistic** for $\mu$ (when $\sigma^2$ is known). By the Rao-Blackwell theorem, any unbiased estimator can be improved (in terms of MSE) by conditioning on a sufficient statistic. Since $\bar{X}$ is already based on the sufficient statistic, it cannot be improved.

### Gauss-Markov Theorem

In the linear regression context $Y = X\beta + \epsilon$, the OLS estimator (which includes $\bar{X}$ as a special case when $X = \mathbf{1}$) is the **Best Linear Unbiased Estimator (BLUE)**: it has the smallest variance among all linear unbiased estimators, regardless of the error distribution.

### Efficiency Relative to Other Estimators

For normal populations, compare the sample mean with alternatives:

| Estimator | Variance (Normal) | Relative Efficiency |
|-----------|-------------------|-------------------|
| Sample Mean $\bar{X}$ | $\sigma^2/n$ | 1.000 (reference) |
| Sample Median | $\frac{\pi}{2} \cdot \frac{\sigma^2}{n}$ | $\frac{2}{\pi} \approx 0.637$ |
| Midrange | $\frac{\sigma^2}{2\log n}$ (approx.) | Inconsistent |
| 10% Trimmed Mean | $\approx 1.05 \cdot \frac{\sigma^2}{n}$ | $\approx 0.952$ |

For normal data, the sample mean is strictly the best. For heavy-tailed data, alternatives like the trimmed mean or median can be better.

## Sample Mean for Non-iid Data

### Correlated Observations

If observations are correlated (common in time series), the variance formula changes:

$$\text{Var}(\bar{X}) = \frac{1}{n^2}\sum_{i=1}^n\sum_{j=1}^n \text{Cov}(X_i, X_j)$$

For stationary data with autocorrelation $\rho_k = \text{Corr}(X_t, X_{t+k})$:

$$\text{Var}(\bar{X}) = \frac{\sigma^2}{n}\left(1 + 2\sum_{k=1}^{n-1}\left(1 - \frac{k}{n}\right)\rho_k\right)$$

With positive autocorrelation, $\text{Var}(\bar{X}) > \sigma^2/n$ — the effective sample size is smaller than $n$.

### Weighted Mean

When observations have unequal reliability, use the **weighted mean**:

$$\bar{X}_w = \frac{\sum_{i=1}^n w_i X_i}{\sum_{i=1}^n w_i}$$

If $\text{Var}(X_i) = \sigma_i^2$ and observations are independent, the optimal weights are $w_i = 1/\sigma_i^2$ (inverse variance weighting), yielding:

$$\text{Var}(\bar{X}_w) = \frac{1}{\sum_{i=1}^n 1/\sigma_i^2}$$

## When the Sample Mean Fails

The sample mean is not always the best estimator:

1. **Heavy-tailed distributions** (e.g., Cauchy, Student-t with few df): The sample mean may have infinite variance or not even exist. The median is more robust.

2. **Contaminated data** (outliers): A single extreme observation can dramatically shift $\bar{X}$. Robust alternatives (trimmed mean, Huber estimator) are preferable.

3. **Skewed distributions**: For highly skewed data, the mean may not be the most useful summary. The median or a transformed mean may be better.

4. **Small samples from non-normal populations**: The CLT approximation may be poor, leading to unreliable confidence intervals based on $\bar{X}$.

## Connections to Finance

The sample mean is ubiquitous in finance, but its limitations are especially important:

- **Expected return estimation**: The sample mean of historical returns is the standard estimator for expected returns, but it is notoriously imprecise. With typical annual return volatility of 20% and 50 years of data, $\text{SE}(\bar{X}) \approx 20\%/\sqrt{50} \approx 2.8\%$, which is large relative to typical risk premia.

- **Sharpe ratio**: $\hat{SR} = \bar{X}/S$ inherits the imprecision of $\bar{X}$. The estimation error in the numerator dominates.

- **Time series dependence**: Financial returns exhibit volatility clustering and other forms of dependence, making the standard $\sigma/\sqrt{n}$ formula an underestimate of uncertainty.

- **Portfolio optimization**: The sensitivity of mean-variance optimization to estimated means is well known (estimation error dominates, leading to extreme weights). Shrinkage estimators and Bayesian approaches address this.

## Summary

The sample mean is the simplest and most natural estimator of a population mean. For normal populations, it is optimal by every criterion: unbiased, minimum variance, sufficient, and efficient. For large samples from any distribution with finite variance, it is consistent and approximately normal (by the CLT). However, its sensitivity to outliers and heavy tails, and the slowness of its convergence ($1/\sqrt{n}$), make it important to understand alternatives and to use it wisely, especially in financial applications where estimation precision is at a premium.

## Key Formulas

| Quantity | Formula |
|----------|---------|
| Sample mean | $\bar{X} = \frac{1}{n}\sum X_i$ |
| $E[\bar{X}]$ | $\mu$ (unbiased) |
| $\text{Var}(\bar{X})$ | $\sigma^2/n$ |
| Standard error | $\sigma/\sqrt{n}$ |
| For normal: exact distribution | $\bar{X} \sim N(\mu, \sigma^2/n)$ |
| CLT | $\sqrt{n}(\bar{X} - \mu)/\sigma \to N(0,1)$ |
| Cramér-Rao bound | $\sigma^2/n$ (achieved) |

## Exercises

**Exercise 1.**
Strategy: 5% mean annual return, 18% volatility. (a) SE from 10 years. (b) Years until 95% CI excludes 0. (c) Does monthly data help?

??? success "Solution to Exercise 1"
    (a) $\mathrm{SE} = 0.18/\sqrt{10} \approx 0.057$. Larger than the point estimate 0.05.

    (b) Need $|\mu|/\mathrm{SE} > 1.96$: $\sqrt n > 1.96 \cdot 0.18/0.05 \approx 7.06 \Rightarrow n \ge 50$ years.

    (c) Under i.i.d. assumption, switching to monthly data scales both mean (by $1/12$) and variance (by $1/12$). The $t$-statistic stays the same — sampling more frequently within the same calendar window doesn't help. The "drift estimation problem" in finance is fundamentally about *time span*, not data density.

---

**Exercise 2.**
**Prove $\bar X$ is BLUE** (Best Linear Unbiased Estimator) of $\mu$ for any distribution with finite variance.

??? success "Solution to Exercise 2"
    Consider linear unbiased estimators $\hat\mu = \sum w_i X_i$ with $\sum w_i = 1$ (for unbiasedness).

    $\mathrm{Var}(\hat\mu) = \sum w_i^2 \sigma^2 = \sigma^2 \sum w_i^2$, subject to $\sum w_i = 1$.

    By Cauchy-Schwarz or Lagrange multipliers, $\sum w_i^2$ is minimized at $w_i = 1/n$ (uniform weights). So $\bar X = (1/n)\sum X_i$ has minimum variance among linear unbiased estimators.

    $\bar X$ is BLUE for i.i.d. samples with finite variance, regardless of the distribution shape. Under normality, $\bar X$ is also UMVUE (uniformly minimum variance among *all* unbiased estimators, not just linear).

---

**Exercise 3.**
**Robustness failure.** Show that adding one outlier to a sample arbitrarily shifts $\bar X$.

??? success "Solution to Exercise 3"
    Sample $\{X_1, \ldots, X_n, M\}$ where $M$ is a single outlier. New mean: $(\sum X_i + M)/(n + 1)$. As $M \to \infty$, new mean $\to \infty$.

    So the **breakdown point** of $\bar X$ is $1/(n+1) \to 0$: even a single corrupted observation can move $\bar X$ arbitrarily far. Zero asymptotic breakdown.

    Compare with the median: breakdown point $\approx 1/2$ — half the data must be corrupted. The price for the median is lower efficiency under normality (~64% relative to mean).

    For real data with potential outliers, prefer robust location estimators (median, trimmed mean, M-estimators) over $\bar X$.

---

**Exercise 4.**
**Bessel's correction.** Why does $S^2 = \sum(X_i - \bar X)^2/(n-1)$ have $n - 1$ in the denominator?

??? success "Solution to Exercise 4"
    Two views:

    **Algebraic:** $\sum(X_i - \bar X)^2 = \sum X_i^2 - n\bar X^2$. Taking expectations:

    $\mathbb{E}[\sum X_i^2] = n(\sigma^2 + \mu^2)$, $\mathbb{E}[n\bar X^2] = \sigma^2 + n\mu^2$. Subtracting: $(n-1)\sigma^2$.

    Dividing by $n - 1$: $\mathbb{E}[S^2] = \sigma^2$. Unbiased.

    **Geometric (degrees of freedom):** observations $X_i$ are unconstrained ($n$ dof), but residuals $X_i - \bar X$ satisfy $\sum(X_i - \bar X) = 0$ — one linear constraint. So they have $n - 1$ effective dof. Variance estimation divides by effective dof.

    With $\mu$ known (e.g., for centered data), use $\sum X_i^2/n$ — no correction since no dof is consumed.

---

**Exercise 5.**
**Sample mean of dependent data.** $X_1, \ldots, X_n$ are AR(1): $X_t = \rho X_{t-1} + \varepsilon_t$. Derive $\mathrm{Var}(\bar X)$ in terms of $\rho$, $n$, and $\sigma^2_\varepsilon$.

??? success "Solution to Exercise 5"
    For stationary AR(1): $X_t = \sum_{k=0}^\infty \rho^k \varepsilon_{t-k}$, with $\mathrm{Var}(X_t) = \sigma^2_\varepsilon/(1 - \rho^2)$ and $\mathrm{Cov}(X_s, X_t) = \rho^{|s-t|} \sigma^2_\varepsilon/(1-\rho^2)$.

    $\mathrm{Var}(\bar X) = (1/n^2)\sum_{s, t} \mathrm{Cov}(X_s, X_t)$. After algebra:

    $\mathrm{Var}(\bar X) \approx \frac{\sigma^2_X}{n} \cdot \frac{1 + \rho}{1 - \rho}$ (for large $n$, with $\sigma^2_X = \sigma^2_\varepsilon/(1-\rho^2)$).

    **Inflation factor** $(1+\rho)/(1-\rho)$ > 1 for $\rho > 0$. For $\rho = 0.5$: factor 3 — equivalent to using $n/3$ independent observations.

    **Implication:** dependent data is less informative than i.i.d. data with the same nominal sample size. The **effective sample size** is $n_{\text{eff}} = n(1-\rho)/(1+\rho)$ for AR(1).

    This is critical in time-series analysis: naively computing $\mathrm{SE} = \sigma/\sqrt n$ underestimates uncertainty for autocorrelated data.

---

**Exercise 6.**
**Weighted sample mean.** When observations have different variances ($\mathrm{Var}(X_i) = \sigma_i^2$), the inverse-variance-weighted mean $\hat\mu = \sum w_i X_i$ with $w_i \propto 1/\sigma_i^2$ minimizes variance. Derive the optimal $w_i$.

??? success "Solution to Exercise 6"
    Constraint: $\sum w_i = 1$ for unbiasedness. Variance: $\mathrm{Var}(\hat\mu) = \sum w_i^2 \sigma_i^2$.

    Minimize $\sum w_i^2 \sigma_i^2$ subject to $\sum w_i = 1$. Lagrangian: $L = \sum w_i^2 \sigma_i^2 - \lambda(\sum w_i - 1)$.

    $\partial L/\partial w_i = 2 w_i \sigma_i^2 - \lambda = 0 \Rightarrow w_i = \lambda/(2\sigma_i^2)$.

    Normalize: $\lambda = 2/\sum(1/\sigma_i^2)$, so $w_i^* = (1/\sigma_i^2)/\sum_j(1/\sigma_j^2)$.

    **Inverse-variance weighting.** Variance of the optimal estimator: $\mathrm{Var}(\hat\mu^*) = 1/\sum(1/\sigma_i^2)$.

    Used in meta-analysis (combining studies of different precision), Kalman filtering (combining sensor measurements), and weighted least squares regression.
