# Bessel's Correction

## Introduction

**Bessel's correction** refers to the use of $n-1$ instead of $n$ in the denominator of the sample variance formula, yielding an unbiased estimator of the population variance. Named after Friedrich Bessel, this correction accounts for the fact that estimating the mean from the same data "uses up" one degree of freedom, causing the naive estimator (dividing by $n$) to systematically underestimate the true variance.

## Definition

The **Bessel-corrected sample variance** is:

$$S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$$

This is the standard "sample variance" used in most statistical software (e.g., `numpy.var(ddof=1)`, R's `var()`).

## Unbiasedness Proof

### Main Result

$$E[S^2] = \sigma^2$$

### Proof

Starting from the key identity:

$$\sum_{i=1}^n (X_i - \bar{X})^2 = \sum_{i=1}^n (X_i - \mu)^2 - n(\bar{X} - \mu)^2$$

Taking expectations:

$$E\left[\sum_{i=1}^n (X_i - \bar{X})^2\right] = n\sigma^2 - n \cdot \frac{\sigma^2}{n} = (n-1)\sigma^2$$

Therefore:

$$E\left[\frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2\right] = \frac{(n-1)\sigma^2}{n-1} = \sigma^2$$

### Why n-1? The Degrees of Freedom Argument
The $n$ deviations $d_i = X_i - \bar{X}$ are subject to the constraint:

$$\sum_{i=1}^n d_i = \sum_{i=1}^n (X_i - \bar{X}) = 0$$

This means only $n-1$ of the deviations are free to vary independently. We say there are $n-1$ **degrees of freedom**. Dividing by the degrees of freedom ($n-1$) instead of the number of observations ($n$) corrects the bias.

**General principle:** When estimating a variance using $k$ estimated parameters, divide by $n - k$:

- Mean unknown, variance of $X$: divide by $n - 1$
- Regression with $p$ coefficients: residual variance uses $n - p$

## Distribution of S-squared
### For Normal Populations

If $X_i \sim N(\mu, \sigma^2)$, then:

$$\frac{(n-1)S^2}{\sigma^2} = \frac{\sum_{i=1}^n (X_i - \bar{X})^2}{\sigma^2} \sim \chi^2_{n-1}$$

This exact distributional result gives us:

$$E[S^2] = \sigma^2 \cdot \frac{n-1}{n-1} = \sigma^2 \quad \text{(confirming unbiasedness)}$$

$$\text{Var}(S^2) = \frac{2\sigma^4}{n-1}$$

### Independence of X-bar and S-squared
For normal populations, $\bar{X}$ and $S^2$ are **independent**. This is a remarkable property unique to the normal distribution (by Cochran's theorem) and is crucial for the derivation of the $t$-distribution used in hypothesis testing.

## Properties

### Variance and MSE

For normal populations:

$$\text{Var}(S^2) = \frac{2\sigma^4}{n-1}$$

$$\text{MSE}(S^2) = \text{Var}(S^2) = \frac{2\sigma^4}{n-1} \quad \text{(since bias = 0)}$$

### Consistency

$S^2$ is consistent for $\sigma^2$:

$$S^2 \xrightarrow{p} \sigma^2 \quad \text{as } n \to \infty$$

### MSE Comparison with Alternatives

For normal populations:

$$\text{MSE}(S^2) = \frac{2}{n-1}\sigma^4 > \frac{2n-1}{n^2}\sigma^4 = \text{MSE}(\tilde{S}^2)$$

The unbiased estimator has **higher MSE** than the biased naive estimator. This is because unbiasedness comes at the cost of increased variance, and the variance increase outweighs the bias reduction (in MSE terms).

## Practical Significance

### When Does It Matter?

The difference between dividing by $n$ and $n-1$:

| $n$ | $(n-1)/n$ | Relative error |
|-----|-----------|---------------|
| 3 | 0.667 | 33.3% |
| 5 | 0.800 | 20.0% |
| 10 | 0.900 | 10.0% |
| 30 | 0.967 | 3.3% |
| 100 | 0.990 | 1.0% |
| 1000 | 0.999 | 0.1% |

For $n > 30$, the practical difference is small. For $n < 10$, the correction is substantial.

### Standard Deviation Bias

While $S^2$ is unbiased for $\sigma^2$, the sample standard deviation $S = \sqrt{S^2}$ is **not** unbiased for $\sigma$. By Jensen's inequality (since $\sqrt{\cdot}$ is concave):

$$E[S] = E[\sqrt{S^2}] < \sqrt{E[S^2]} = \sigma$$

For normal populations:

$$E[S] = \sigma \cdot \sqrt{\frac{2}{n-1}} \cdot \frac{\Gamma(n/2)}{\Gamma((n-1)/2)}$$

The correction factor $c_4 = \sqrt{2/(n-1)} \cdot \Gamma(n/2)/\Gamma((n-1)/2)$ can be used to obtain an unbiased estimator of $\sigma$: $\hat{\sigma} = S/c_4$.

## Software Implementation

Different software has different defaults:

| Software | `var()` default | Divisor |
|----------|----------------|---------|
| Python `numpy.var()` | $n$ (population) | `ddof=0` |
| Python `numpy.var(ddof=1)` | $n-1$ (sample) | `ddof=1` |
| R `var()` | $n-1$ (sample) | — |
| Excel `VAR.S()` | $n-1$ (sample) | — |
| Excel `VAR.P()` | $n$ (population) | — |
| pandas `.var()` | $n-1$ (sample) | `ddof=1` |

**Common pitfall:** Using `numpy.var()` without `ddof=1` gives the biased (naive) estimator. Always specify `ddof=1` when you want the unbiased sample variance.

## Generalization: Degrees of Freedom in Regression

In linear regression $Y = X\beta + \epsilon$, the residual variance estimator is:

$$\hat{\sigma}^2 = \frac{1}{n-p}\sum_{i=1}^n (Y_i - \hat{Y}_i)^2 = \frac{\text{RSS}}{n-p}$$

where $p$ is the number of estimated coefficients. This generalizes Bessel's correction: we lose one degree of freedom for each estimated parameter.

## Connections to Finance

- **Realized volatility**: When computing daily realized volatility from intraday returns, the choice of $n$ vs $n-1$ is often irrelevant (many observations). But for monthly volatility from daily data (~21 observations), the correction matters.

- **Tracking error**: Computing tracking error of a portfolio vs. benchmark uses $S = \sqrt{\frac{1}{n-1}\sum(r_p - r_b)^2}$ with Bessel's correction.

- **Risk budgeting**: Variance decomposition in portfolio risk uses the unbiased covariance matrix, which divides by $n-1$.

## Summary

Bessel's correction ($n-1$ in the denominator) produces an unbiased estimator of the population variance. The correction compensates for the "lost" degree of freedom from estimating the mean. While the unbiased estimator has higher MSE than the naive one, unbiasedness is often preferred for its theoretical properties and is the standard in most statistical software. For large samples, the choice between $n$ and $n-1$ is inconsequential.

## Key Formulas

| Quantity | Formula |
|----------|---------|
| Bessel-corrected variance | $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$ |
| Unbiasedness | $E[S^2] = \sigma^2$ |
| Distribution (Normal) | $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$ |
| Variance (Normal) | $\text{Var}(S^2) = 2\sigma^4/(n-1)$ |
| $S$ is biased for $\sigma$ | $E[S] < \sigma$ (Jensen's inequality) |
| General regression | $\hat{\sigma}^2 = \text{RSS}/(n-p)$ |

## Exercises

**Exercise 1.**
**Chi-squared distribution of $S^2$** for normal data. (a) Show $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$. (b) 95% CI for $\sigma^2$ with $n = 20, S^2 = 16$. (c) Why asymmetric?

??? success "Solution to Exercise 1"
    (a) $\sum(X_i - \bar X)^2/\sigma^2 = \sum(X_i - \mu)^2/\sigma^2 - n(\bar X - \mu)^2/\sigma^2$. The first is $\chi^2_n$, the second is $\chi^2_1$ and independent (Cochran's theorem). So the difference is $\chi^2_{n-1}$.

    (b) $\chi^2_{19, 0.025} = 8.907$, $\chi^2_{19, 0.975} = 32.852$. CI: $(19 \cdot 16/32.852, 19 \cdot 16/8.907) = (9.25, 34.13)$.

    (c) Chi-squared is right-skewed; quantiles are asymmetric. Upper bound farther from $S^2$ than lower. The asymmetry shrinks as $n \to \infty$ (chi-squared approaches normal by CLT applied to squared standard normals).

---

**Exercise 2.**
**Bias of $S$.** (a) Why $\mathbb{E}[S] < \sigma$. (b) For $n = 5$, compute $c_4$. (c) Used in practice?

??? success "Solution to Exercise 2"
    (a) $\sqrt{\cdot}$ is concave. Jensen: $\mathbb{E}[\sqrt{S^2}] < \sqrt{\mathbb{E}[S^2]} = \sigma$. Strict because $S^2$ has nontrivial variance.

    (b) $c_4 = \sqrt{2/(n-1)} \cdot \Gamma(n/2)/\Gamma((n-1)/2)$. For $n = 5$: $c_4 = \sqrt{2/4} \cdot \Gamma(2.5)/\Gamma(2) = (1/\sqrt 2) \cdot (3\sqrt\pi/4)/1 \approx 0.940$.

    Bias-corrected SD: $S/c_4$.

    (c) Rarely used; bias is small ($\sim 6\%$ at $n = 5$, $< 1\%$ at $n \ge 30$). SPC charts (Shewhart) use $c_4$; most other applications report $S$ uncorrected.

---

**Exercise 3.**
**Realized volatility.** 78 five-minute returns per day. (a) Naive bias fraction. (b) Does Bessel matter? (c) 21-day rolling window?

??? success "Solution to Exercise 3"
    (a) Naive bias = $-\sigma^2/n = -\sigma^2/78 \approx -1.3\%$ of true variance. Negligible.

    (b) Bessel correction: factor $78/77 \approx 1.013$. Smaller than microstructure noise and intraday volatility patterns. Doesn't matter.

    (c) 21-day rolling = 21 daily variance estimates. SE of mean variance estimate: $\sqrt{2 \sigma^4/(n_{\text{day}} - 1)}/\sqrt{21}$. Trade-off: smoother estimates but lag volatility regime changes by 10 days on average. Half-life of regime shift: ~10 days.

---

**Exercise 4.**
**Bessel's correction generalization.** For regression with $p$ parameters, residual variance estimator divides by $n - p$. Why?

??? success "Solution to Exercise 4"
    Residuals $e_i = Y_i - \hat Y_i$ satisfy linear constraints (normal equations: $\mathbf X^T \mathbf e = 0$). $\mathbf X$ has $p$ columns, so there are $p$ linear constraints, leaving $n - p$ effective dof.

    $\mathbb{E}[\mathrm{SSE}/\sigma^2] = n - p$ (chi-squared distribution under normality). So $\hat\sigma^2 = \mathrm{SSE}/(n - p)$ is unbiased.

    Generalizes Bessel's correction: $n - 1$ is the case $p = 1$ (intercept only).

    Connection to t-test/F-test: residual variance with $n - p$ dof determines the appropriate $t$ critical value (df $= n - p$).

---

**Exercise 5.**
**Why divide by $n - 1$ exactly?** Could other denominators be justified?

??? success "Solution to Exercise 5"
    **Unbiasedness:** $n - 1$ is the unique divisor making $S^2$ unbiased.

    **MSE-optimal:** $n + 1$ minimizes MSE (Exercise from earlier section).

    **MLE:** $n$ is the MLE divisor (for normal data).

    **Justification for $n - 1$:** unbiasedness is a "clean" property — averages of unbiased estimators are unbiased; expectations are interpretable. Statistical convention favors $n - 1$ even though it's not MSE-optimal.

    Different fields make different choices:

    - Engineering / SPC: $n - 1$ for sample variance, $c_4$-corrected for SD.
    - Machine learning: often $n$ (MLE) without thinking about bias.
    - Bayesian: posterior credible intervals based on $\chi^2_{n-1}$.
    - Robust statistics: median absolute deviation, irrelevant divisor choice.

---

**Exercise 6.**
**Why bias correction matters for very small samples but not large samples.** Concrete demonstration with $n = 3, 10, 100$.

??? success "Solution to Exercise 6"
    Naive vs unbiased estimator difference: $\hat\sigma^2_{\text{naive}}/\hat\sigma^2_{\text{unbiased}} = (n-1)/n$.

    | $n$ | $(n-1)/n$ | Discrepancy |
    |---|---|---|
    | 3 | 0.667 | 33% |
    | 10 | 0.900 | 10% |
    | 100 | 0.990 | 1% |
    | 1000 | 0.999 | 0.1% |

    For $n = 3$: missing the Bessel correction means underestimating variance by 1/3 — a serious error.

    For $n = 100$: 1% — within rounding error of most measurements. Bessel correction is mostly conventional.

    **Practical implication:** for small samples (< 30), always use Bessel correction. For large samples (> 100), the choice is mostly academic. Real-data noise typically dominates the small bias correction.
