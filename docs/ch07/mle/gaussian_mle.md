# MLE of μ and σ²

## Introduction

The **Maximum Likelihood Estimators of the Gaussian (Normal) distribution parameters** are among the most important results in statistics. For $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$, the MLE provides closed-form estimators for both the mean $\mu$ and variance $\sigma^2$. This section derives these estimators, analyzes their properties, and connects the results to the broader theory of estimation.

## The Normal Log-Likelihood

For an iid sample $x_1, \ldots, x_n$ from $N(\mu, \sigma^2)$, the log-likelihood is:

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

## Deriving the MLEs

### MLE of mu
Differentiate with respect to $\mu$:

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = \frac{n}{\sigma^2}(\bar{x} - \mu)$$

Setting to zero:

$$\bar{x} - \mu = 0 \implies \boxed{\hat{\mu}_{\text{MLE}} = \bar{X} = \frac{1}{n}\sum_{i=1}^n X_i}$$

The MLE of the mean is the **sample mean**.

### MLE of sigma-squared
Differentiate with respect to $\sigma^2$ (treating $\sigma^2$ as a single variable):

$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n (x_i - \mu)^2$$

Setting to zero and substituting $\hat{\mu} = \bar{x}$:

$$-\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n (x_i - \bar{x})^2 = 0$$

$$\sigma^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$$

$$\boxed{\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2}$$

The MLE of the variance divides by $n$, **not** $n-1$.

### Verification: Second-Order Conditions

The Hessian matrix evaluated at $(\hat{\mu}, \hat{\sigma}^2)$ is:

$$H = \begin{pmatrix} -n/\hat{\sigma}^2 & 0 \\ 0 & -n/(2\hat{\sigma}^4) \end{pmatrix}$$

This is negative definite (both diagonal entries are negative), confirming a maximum.

## Properties of the Gaussian MLEs

### Properties of mu-hat = X-bar
| Property | Result |
|----------|--------|
| Bias | $E[\hat{\mu}] = \mu$ (unbiased) |
| Variance | $\text{Var}(\hat{\mu}) = \sigma^2/n$ |
| Distribution | $\hat{\mu} \sim N(\mu, \sigma^2/n)$ exactly |
| Efficiency | Achieves CRLB; MVUE |
| Sufficiency | Sufficient for $\mu$ (given $\sigma^2$) |
| Consistency | $\hat{\mu} \xrightarrow{p} \mu$ |

### Properties of sigma-squared (MLE)
| Property | Result |
|----------|--------|
| Bias | $E[\hat{\sigma}^2] = \frac{n-1}{n}\sigma^2$ (biased) |
| Bias magnitude | $\text{Bias} = -\sigma^2/n$ |
| Distribution | $n\hat{\sigma}^2/\sigma^2 \sim \chi^2_{n-1}$ |
| Variance | $\text{Var}(\hat{\sigma}^2) = \frac{2(n-1)}{n^2}\sigma^4$ |
| MSE | $\frac{2n-1}{n^2}\sigma^4$ |
| Consistency | $\hat{\sigma}^2 \xrightarrow{p} \sigma^2$ |
| Asymptotically unbiased | $E[\hat{\sigma}^2] \to \sigma^2$ as $n \to \infty$ |

### Independence

$\hat{\mu}$ and $\hat{\sigma}^2$ are **independent** (by Cochran's theorem). This is a special property of the normal distribution and is crucial for deriving the $t$-distribution.

## Fisher Information Matrix

The Fisher information matrix for $(\mu, \sigma^2)$ is:

$$I(\mu, \sigma^2) = \begin{pmatrix} n/\sigma^2 & 0 \\ 0 & n/(2\sigma^4) \end{pmatrix}$$

The zero off-diagonal entries confirm that $\mu$ and $\sigma^2$ carry independent information.

### Cramér-Rao Lower Bounds

$$\text{Var}(\hat{\mu}) \geq \frac{\sigma^2}{n}, \quad \text{Var}(\hat{\sigma}^2) \geq \frac{2\sigma^4}{n}$$

The MLE of $\mu$ achieves the CRLB exactly. The MLE of $\sigma^2$ does *not* achieve the CRLB in finite samples (it has variance $2(n-1)\sigma^4/n^2 < 2\sigma^4/n$), but it does asymptotically.

## Alternative Parametrization: (mu, sigma)
If we parametrize by $(\mu, \sigma)$ instead of $(\mu, \sigma^2)$, the MLE of $\sigma$ is:

$$\hat{\sigma}_{\text{MLE}} = \sqrt{\hat{\sigma}^2_{\text{MLE}}} = \sqrt{\frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2}$$

This follows from the **invariance property** of MLEs: if $\hat{\theta}$ is the MLE of $\theta$, then $g(\hat{\theta})$ is the MLE of $g(\theta)$.

Note that $\hat{\sigma}_{\text{MLE}}$ is biased for $\sigma$ (by Jensen's inequality, $E[\sqrt{X}] < \sqrt{E[X]}$).

## Bias-Corrected Estimator

The unbiased estimator of $\sigma^2$ is:

$$S^2 = \frac{n}{n-1}\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$$

Comparison:

| Estimator | Formula | $E[\cdot]$ | MSE |
|-----------|---------|------------|-----|
| MLE | $\frac{1}{n}\sum(X_i - \bar{X})^2$ | $\frac{n-1}{n}\sigma^2$ | $\frac{2n-1}{n^2}\sigma^4$ |
| Bessel's | $\frac{1}{n-1}\sum(X_i - \bar{X})^2$ | $\sigma^2$ | $\frac{2}{n-1}\sigma^4$ |
| MSE-optimal | $\frac{1}{n+1}\sum(X_i - \bar{X})^2$ | $\frac{n-1}{n+1}\sigma^2$ | minimum |

## Log-Likelihood Surface

The log-likelihood function $\ell(\mu, \sigma^2)$ forms a surface over the $(\mu, \sigma^2)$ plane:

- For fixed $\sigma^2$: $\ell$ is a downward-opening parabola in $\mu$, maximized at $\bar{X}$
- For fixed $\mu$: $\ell$ is a concave function of $\sigma^2$
- The global maximum is at $(\bar{X}, \hat{\sigma}^2)$
- Contours of constant log-likelihood are ellipses centered at the MLE (approximately, for large $n$)

## Confidence Regions from the Likelihood

### For mu (σ² known)

$$\bar{X} \pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}$$

### For mu (σ² unknown)

$$\bar{X} \pm t_{n-1, \alpha/2}\frac{S}{\sqrt{n}}$$

where $S = \sqrt{S^2}$ and $t_{n-1}$ is the Student's $t$-distribution with $n-1$ degrees of freedom.

### For sigma-squared

$$\left(\frac{(n-1)S^2}{\chi^2_{n-1, \alpha/2}}, \quad \frac{(n-1)S^2}{\chi^2_{n-1, 1-\alpha/2}}\right)$$

## MLE Under Constraints

### Known Mean

If $\mu = \mu_0$ is known, the constrained MLE of $\sigma^2$ is:

$$\hat{\sigma}^2_{\mu_0} = \frac{1}{n}\sum_{i=1}^n (X_i - \mu_0)^2$$

This is unbiased (unlike the case when $\mu$ is estimated).

### Equal Means (Pooled Variance)

For two groups $X_1, \ldots, X_{n_1} \sim N(\mu_1, \sigma^2)$ and $Y_1, \ldots, Y_{n_2} \sim N(\mu_2, \sigma^2)$ with common variance, the MLE of $\sigma^2$ is:

$$\hat{\sigma}^2_{\text{pooled}} = \frac{\sum(X_i - \bar{X})^2 + \sum(Y_j - \bar{Y})^2}{n_1 + n_2}$$

The unbiased version divides by $n_1 + n_2 - 2$.

## Connections to Finance

- **Return modeling**: Assuming log-returns $r_t \sim N(\mu, \sigma^2)$ is the foundation of many financial models. The MLEs $\hat{\mu} = \bar{r}$ and $\hat{\sigma}^2 = \frac{1}{n}\sum(r_t - \bar{r})^2$ are the standard estimates.

- **Black-Scholes**: The model assumes $\log(S_T/S_t) \sim N((\mu - \sigma^2/2)(T-t), \sigma^2(T-t))$. MLE of volatility from historical returns is a key input.

- **VaR estimation**: Under normality, $\text{VaR}_\alpha = -(\hat{\mu} + z_\alpha \hat{\sigma})$, which uses the Gaussian MLEs directly.

- **Portfolio theory**: Markowitz optimization uses $\hat{\mu}$ and $\hat{\Sigma}$ (the sample mean vector and covariance matrix), which are the multivariate Gaussian MLEs.

- **Normality testing**: Before using Gaussian MLE, one should test whether the normal distribution is appropriate. Financial returns often exhibit fat tails, making the Gaussian MLE suboptimal.

## Summary

The Gaussian MLEs — $\hat{\mu} = \bar{X}$ and $\hat{\sigma}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$ — are closed-form, computationally trivial, and have excellent properties. The mean estimator is unbiased and efficient; the variance estimator is biased but consistent and has lower MSE than the unbiased alternative. Their independence (unique to the normal distribution) enables exact inference via $t$ and $\chi^2$ distributions. These estimators form the foundation of classical statistical inference and are the starting point for financial parameter estimation.

## Key Formulas

| Quantity | Formula |
|----------|---------|
| $\hat{\mu}_{\text{MLE}}$ | $\bar{X}$ |
| $\hat{\sigma}^2_{\text{MLE}}$ | $\frac{1}{n}\sum(X_i - \bar{X})^2$ |
| Fisher info for $\mu$ | $I_n(\mu) = n/\sigma^2$ |
| Fisher info for $\sigma^2$ | $I_n(\sigma^2) = n/(2\sigma^4)$ |
| $\hat{\mu}$ distribution | $N(\mu, \sigma^2/n)$ |
| $n\hat{\sigma}^2/\sigma^2$ distribution | $\chi^2_{n-1}$ |
| $t$-statistic | $(\bar{X}-\mu)/(S/\sqrt{n}) \sim t_{n-1}$ |

## Exercises

**Exercise 1.**
Derive $\hat\mu_{\text{MLE}} = \bar X$ and $\hat\sigma^2_{\text{MLE}} = (1/n)\sum(X_i - \bar X)^2$ for $N(\mu, \sigma^2)$, and verify second-order conditions.

??? success "Solution to Exercise 1"
    Log-lik: $\ell(\mu, \sigma^2) = -(n/2)\ln(2\pi\sigma^2) - (1/(2\sigma^2))\sum(x_i - \mu)^2$.

    $\partial\ell/\partial\mu = (1/\sigma^2)\sum(x_i - \mu) = 0 \Rightarrow \hat\mu = \bar X$.

    $\partial\ell/\partial\sigma^2 = -n/(2\sigma^2) + (1/(2\sigma^4))\sum(x_i - \mu)^2 = 0$; plugging in $\hat\mu$ gives $\hat\sigma^2_{\text{MLE}} = (1/n)\sum(x_i - \bar X)^2$.

    **Hessian:** $\partial^2\ell/\partial\mu^2 = -n/\sigma^2 < 0$; $\partial^2\ell/\partial(\sigma^2)^2 = -n/(2\sigma^4) < 0$ at the MLE; mixed partial vanishes in expectation. Negative-definite — confirms a maximum.

---

**Exercise 2.**
**Fisher information matrix** for $N(\mu, \sigma^2)$. Show off-diagonals are zero and verify $\bar X$ achieves the CRLB exactly.

??? success "Solution to Exercise 2"
    $I_{\mu\mu} = n/\sigma^2$, $I_{\sigma^2 \sigma^2} = n/(2\sigma^4)$, $I_{\mu \sigma^2} = \mathbb{E}[-(X-\mu)/\sigma^4] = 0$.

    Diagonal Fisher info: $\mu$ and $\sigma^2$ are **orthogonal parameters**. Estimating one doesn't affect the asymptotic variance of estimating the other.

    $\mathrm{Var}(\bar X) = \sigma^2/n = 1/I_{\mu\mu}$ — $\bar X$ achieves the CRLB **exactly** for any $n$, not just asymptotically. The MLE for $\mu$ is fully efficient.

    For $\sigma^2$: $\mathrm{Var}(\hat\sigma^2_{\text{MLE}}) = 2(n-1)\sigma^4/n^2$, CRLB = $2\sigma^4/n$. Slightly above CRLB — asymptotically efficient.

---

**Exercise 3.**
**MLE invariance.** Find the MLE of (a) $\sigma$, (b) $\mathrm{CV} = \sigma/\mu$, (c) the 99th percentile $\mu + 2.326\sigma$.

??? success "Solution to Exercise 3"
    By MLE invariance, $\widehat{g(\theta)} = g(\hat\theta_{\text{MLE}})$:

    (a) $\hat\sigma = \sqrt{\hat\sigma^2_{\text{MLE}}}$.

    (b) $\widehat{\mathrm{CV}} = \hat\sigma/\bar X$.

    (c) $\widehat{q_{0.99}} = \bar X + 2.326 \hat\sigma$.

    **Caveat:** invariance preserves MLE point estimates but not unbiasedness. $\hat\sigma$ is biased (Jensen's inequality on concave $\sqrt{\cdot}$); bias correction available via the $c_4$ constant.

---

**Exercise 4.**
**Constrained MLE with $\mu = 0$.** Derive $\hat\sigma^2$ when $\mu$ is known to be zero. Compare its variance with the unconstrained MLE.

??? success "Solution to Exercise 4"
    With $\mu = 0$: $\hat\sigma^2_0 = (1/n) \sum X_i^2$. $\mathbb{E}[\hat\sigma^2_0] = (1/n) \cdot n\sigma^2 = \sigma^2$ — **unbiased** (no Bessel correction needed because $\mu$ is known, not estimated).

    $n\hat\sigma^2_0/\sigma^2 \sim \chi^2_n$, so $\mathrm{Var}(\hat\sigma^2_0) = 2\sigma^4/n$.

    Unconstrained: $\mathrm{Var}(\hat\sigma^2_{\text{MLE}}) = 2\sigma^4(n-1)/n^2$.

    Constrained has slightly larger variance per observation (one extra dof), but is unbiased. The "cost" of estimating $\mu$ is a $-1$ in dof.

---

**Exercise 5.**
**Parametric VaR.** Given $\hat\mu = 0.0003$, $\hat\sigma = 0.012$ from 252 daily returns: (a) 1-day 99% VaR. (b) 10-day VaR via square-root rule. (c) If true kurtosis is 3 (excess), does normal VaR over- or under-estimate?

??? success "Solution to Exercise 5"
    (a) $\mathrm{VaR}_{0.99}^{\text{1-day}} = -(\hat\mu + z_{0.01} \hat\sigma) = -(0.0003 - 2.326 \cdot 0.012) = 0.0276$ (2.76% loss).

    (b) $\mathrm{VaR}_{0.99}^{\text{10-day}} = \sqrt{10} \cdot 0.0276 \approx 0.0873$ (8.73%). Valid under i.i.d. zero-drift assumption.

    (c) Heavy tails (excess kurtosis 3 > 0) mean the true 99th percentile loss is *larger* than normal predicts. Normal VaR **underestimates** real risk. Conservative practice: use $t$-distribution-based VaR or empirical quantiles for risk management.

---

**Exercise 6.**
**Monte Carlo verification.** Simulate 10000 samples of $n = 20$ from $N(5, 9)$. Verify $\mathbb{E}[\hat\mu], \mathbb{E}[\hat\sigma^2_{\text{MLE}}], \mathbb{E}[S^2]$.

??? success "Solution to Exercise 6"
    ```python
    import numpy as np
    rng = np.random.default_rng(0)
    R, n, mu, var = 10_000, 20, 5.0, 9.0
    samples = rng.normal(mu, np.sqrt(var), (R, n))
    mu_hat = samples.mean(axis=1)
    sig2_mle = samples.var(axis=1, ddof=0)
    s2 = samples.var(axis=1, ddof=1)
    print(f"E[mu_hat]   = {mu_hat.mean():.4f}   (true {mu})")
    print(f"E[sig2_MLE] = {sig2_mle.mean():.4f} (true {(n-1)/n*var:.4f})")
    print(f"E[S^2]      = {s2.mean():.4f}       (true {var})")
    ```

    Expected:

    - $\mathbb{E}[\bar X] \approx 5.00$ (unbiased).
    - $\mathbb{E}[\hat\sigma^2_{\text{MLE}}] \approx 8.55 = (19/20) \cdot 9$ (biased downward by $\sigma^2/n$).
    - $\mathbb{E}[S^2] \approx 9.00$ (Bessel-corrected, unbiased).

    Confirms theoretical results and demonstrates the bias of the MLE for variance.
