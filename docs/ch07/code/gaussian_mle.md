# Gaussian Maximum Likelihood

## Overview

Maximum likelihood estimation (MLE) for the normal distribution yields closed-form estimators: $\hat{\mu} = \bar{X}$ for the mean and $\hat{\sigma}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$ for the variance. This page verifies the analytical MLEs against numerical optimization, visualizes the log-likelihood surface, quantifies finite-sample bias, derives the Cramer-Rao lower bound, validates confidence interval coverage, and applies Gaussian MLE to Value at Risk estimation.

## Analytical MLE

For iid observations $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$, the log-likelihood is:

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n(X_i - \mu)^2$$

Setting the partial derivatives to zero gives the MLEs:

$$\hat{\mu}_{\text{MLE}} = \bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$$

$$\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2$$

Note: the variance MLE divides by $n$, not $n-1$.

```python
import numpy as np
from scipy import optimize

def mle_analytical_vs_numerical(seed=42):
    rng = np.random.default_rng(seed)
    mu_true, sigma_true = 5.0, 2.0
    n = 50
    data = rng.normal(mu_true, sigma_true, n)

    # Analytical MLE
    mu_mle = data.mean()
    sigma2_mle = np.mean((data - mu_mle)**2)

    # Numerical MLE (parameterize log(sigma^2) for unconstrained optimization)
    def neg_ll(params):
        mu, ls2 = params
        s2 = np.exp(ls2)
        return n/2*np.log(2*np.pi*s2) + np.sum((data-mu)**2)/(2*s2)

    res = optimize.minimize(neg_ll, [0, 0], method='Nelder-Mead')
    mu_num, s2_num = res.x[0], np.exp(res.x[1])

    print(f"Analytical: mu={mu_mle:.6f}, sigma²={sigma2_mle:.6f}")
    print(f"Numerical:  mu={mu_num:.6f}, sigma²={s2_num:.6f}")
```

!!! tip "Agreement"
    The analytical and numerical solutions agree to many decimal places, confirming the closed-form derivation.

## Log-Likelihood Surface

The log-likelihood forms a smooth, concave surface with a unique maximum at $(\hat{\mu}, \hat{\sigma}^2)$. Profile likelihoods allow visualization of each parameter separately.

```python
import matplotlib.pyplot as plt
from scipy import stats

def loglikelihood_surface(seed=42):
    rng = np.random.default_rng(seed)
    n = 30
    mu_true, sigma_true = 5.0, 2.0
    data = rng.normal(mu_true, sigma_true, n)

    mu_mle = data.mean()
    s2_mle = np.mean((data - mu_mle)**2)

    mu_r = np.linspace(mu_mle - 2, mu_mle + 2, 200)
    s2_r = np.linspace(s2_mle * 0.3, s2_mle * 3, 200)
    MU, S2 = np.meshgrid(mu_r, s2_r)

    LL = np.zeros_like(MU)
    for i in range(LL.shape[0]):
        for j in range(LL.shape[1]):
            LL[i, j] = (-n/2*np.log(2*np.pi*S2[i,j])
                        - np.sum((data-MU[i,j])**2)/(2*S2[i,j]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Contour plot
    axes[0].contour(MU, S2, LL, levels=30, cmap='viridis')
    axes[0].plot(mu_mle, s2_mle, 'r*', ms=15, label='MLE')
    axes[0].set_xlabel('mu'); axes[0].set_ylabel('sigma²')
    axes[0].set_title('Log-Likelihood Contours')
    axes[0].legend()

    # Profile for mu
    prof_mu = [-np.sum((data-m)**2)/(2*s2_mle) for m in mu_r]
    prof_mu = np.array(prof_mu) - max(prof_mu)
    axes[1].plot(mu_r, prof_mu, 'b-', lw=2)
    axes[1].axvline(mu_mle, color='red', ls='--')
    axes[1].set_xlabel('mu'); axes[1].set_title('Profile for mu')

    # Profile for sigma²
    prof_s = [-n/2*np.log(s)-np.sum((data-mu_mle)**2)/(2*s) for s in s2_r]
    prof_s = np.array(prof_s) - max(prof_s)
    axes[2].plot(s2_r, prof_s, 'b-', lw=2)
    axes[2].axvline(s2_mle, color='red', ls='--')
    axes[2].set_xlabel('sigma²'); axes[2].set_title('Profile for sigma²')

    plt.tight_layout()
    plt.show()
```

## Finite-Sample Bias

The mean MLE $\hat{\mu}$ is unbiased, but the variance MLE $\hat{\sigma}^2_{\text{MLE}}$ is biased downward:

$$E[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2$$

The bias is $-\sigma^2/n$, which vanishes as $n \to \infty$ (so the MLE is asymptotically unbiased).

```python
def finite_sample_bias(n_sim=200_000, seed=42):
    rng = np.random.default_rng(seed)
    mu_true, sigma_true = 5.0, 3.0
    sigma2 = sigma_true**2
    sample_sizes = [3, 5, 10, 20, 50, 100, 500]

    for n in sample_sizes:
        samp = rng.normal(mu_true, sigma_true, (n_sim, n))
        s2_mle = np.var(samp, axis=1, ddof=0)
        s2_ub  = np.var(samp, axis=1, ddof=1)
        print(f"n={n:>4}  E[sigma²_MLE]={s2_mle.mean():.4f}  "
              f"E[S²]={s2_ub.mean():.4f}  Bias(MLE)={s2_mle.mean()-sigma2:.4f}")
```

## Fisher Information and Cramer-Rao Lower Bound

The **Fisher information matrix** for $N(\mu, \sigma^2)$ is:

$$I_n(\mu, \sigma^2) = \begin{pmatrix} n/\sigma^2 & 0 \\ 0 & n/(2\sigma^4) \end{pmatrix}$$

The **Cramer-Rao Lower Bound (CRLB)** gives the minimum variance of any unbiased estimator:

$$\text{Var}(\hat{\mu}) \geq \frac{\sigma^2}{n}, \qquad \text{Var}(\hat{\sigma}^2) \geq \frac{2\sigma^4}{n}$$

The mean MLE achieves the CRLB exactly. The variance MLE approaches it asymptotically.

```python
def fisher_information_crlb(sigma=3.0, n_sim=100_000, seed=42):
    rng = np.random.default_rng(seed)
    sample_sizes = [10, 25, 50, 100, 500]

    print("For mu: CRLB = sigma²/n")
    for n in sample_sizes:
        mu_h = np.array([rng.normal(5, sigma, n).mean() for _ in range(n_sim)])
        print(f"  n={n:>4}  Var(mu_hat)={mu_h.var():.6f}  "
              f"CRLB={sigma**2/n:.6f}  Ratio={mu_h.var()/(sigma**2/n):.4f}")

    print("\nFor sigma²: CRLB = 2*sigma⁴/n")
    for n in sample_sizes:
        s2_h = np.array([np.var(rng.normal(5, sigma, n)) for _ in range(n_sim)])
        print(f"  n={n:>4}  Var(sigma²_hat)={s2_h.var():.6f}  "
              f"CRLB={2*sigma**4/n:.6f}  Ratio={s2_h.var()/(2*sigma**4/n):.4f}")
```

!!! info "Efficiency"
    The ratio $\text{Var}/\text{CRLB}$ is exactly 1 for $\hat{\mu}$ (it is efficient at all sample sizes) and converges to 1 for $\hat{\sigma}^2$ as $n \to \infty$ (asymptotically efficient).

## Confidence Interval Coverage

Three types of confidence intervals arise from the Gaussian model:

| Parameter | Known | Interval Type | Pivotal Quantity |
|-----------|-------|---------------|-----------------|
| $\mu$ | $\sigma$ known | $z$-interval | $\frac{\bar{X}-\mu}{\sigma/\sqrt{n}} \sim N(0,1)$ |
| $\mu$ | $\sigma$ unknown | $t$-interval | $\frac{\bar{X}-\mu}{S/\sqrt{n}} \sim t_{n-1}$ |
| $\sigma^2$ | $\mu$ unknown | $\chi^2$-interval | $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}$ |

```python
def confidence_interval_coverage(seed=42):
    rng = np.random.default_rng(seed)
    mu_true, sigma_true = 10.0, 3.0
    n, alpha, n_sim = 25, 0.05, 50_000

    z_ok = t_ok = chi_ok = 0
    for _ in range(n_sim):
        d = rng.normal(mu_true, sigma_true, n)
        xb, s, s2 = d.mean(), d.std(ddof=1), d.var(ddof=1)

        # z-interval (sigma known)
        z_c = stats.norm.ppf(1 - alpha/2)
        if xb - z_c*sigma_true/np.sqrt(n) <= mu_true <= xb + z_c*sigma_true/np.sqrt(n):
            z_ok += 1

        # t-interval (sigma unknown)
        t_c = stats.t.ppf(1 - alpha/2, n-1)
        if xb - t_c*s/np.sqrt(n) <= mu_true <= xb + t_c*s/np.sqrt(n):
            t_ok += 1

        # chi-squared interval for sigma²
        lo = (n-1)*s2 / stats.chi2.ppf(1-alpha/2, n-1)
        hi = (n-1)*s2 / stats.chi2.ppf(alpha/2, n-1)
        if lo <= sigma_true**2 <= hi:
            chi_ok += 1

    print(f"z-interval (mu, sigma known):  {z_ok/n_sim:.1%} (target: {1-alpha:.1%})")
    print(f"t-interval (mu, sigma unknown): {t_ok/n_sim:.1%} (target: {1-alpha:.1%})")
    print(f"chi²-interval (sigma²):        {chi_ok/n_sim:.1%} (target: {1-alpha:.1%})")
```

!!! success "Coverage matches"
    All three intervals achieve their nominal 95% coverage, confirming the theoretical derivations.

## Financial Application: Value at Risk

**Value at Risk (VaR)** at level $\alpha$ is the loss exceeded with probability $\alpha$. Under a Gaussian model for daily returns $R \sim N(\hat{\mu}, \hat{\sigma}^2)$:

$$\text{VaR}_\alpha = -(\hat{\mu} + z_\alpha \hat{\sigma})$$

where $z_\alpha = \mathcal{N}^{-1}(\alpha)$ is the normal quantile.

```python
def var_estimation_finance(seed=42):
    rng = np.random.default_rng(seed)
    mu_d = 0.08/252
    sig_d = 0.20/np.sqrt(252)
    n = 504
    df = 5

    # Simulate t-distributed returns (heavier tails than normal)
    returns = mu_d + sig_d * rng.standard_t(df, n) / np.sqrt(df/(df-2))
    mu_hat = returns.mean()
    sig_hat = np.sqrt(np.mean((returns - mu_hat)**2))

    for alpha in [0.01, 0.025, 0.05, 0.10]:
        v_p = -(mu_hat + stats.norm.ppf(alpha) * sig_hat)
        v_h = -np.percentile(returns, alpha * 100)
        print(f"alpha={alpha:.3f}  Parametric VaR={v_p*100:.3f}%  "
              f"Historical VaR={v_h*100:.3f}%  Ratio={v_h/v_p:.3f}")
```

!!! warning "Model risk"
    When the true return distribution has heavier tails than the normal (as is typical in finance), the Gaussian VaR **underestimates** tail risk. The historical VaR at the 1% level is typically larger than the parametric VaR, reflecting the true distribution's fatter tails.

## Interpretation

- The Gaussian MLE has elegant **closed-form solutions** and the mean estimator is globally efficient (achieves the CRLB).
- The **variance MLE is biased** by a factor of $(n-1)/n$, but this bias vanishes asymptotically and can be corrected by Bessel's factor.
- The **log-likelihood surface** is concave with a unique maximum, making optimization straightforward.
- The Fisher information provides a fundamental limit on estimation precision through the **Cramer-Rao lower bound**.
- All three standard confidence intervals ($z$, $t$, $\chi^2$) achieve their nominal coverage under normality.
- In finance, the Gaussian assumption leads to simple VaR formulas but systematically **underestimates tail risk**.

## Exercises

**Exercise 1.**
Derive the MLE for $\mu$ and $\sigma^2$ by differentiating the log-likelihood and solving the first-order conditions.

??? success "Solution to Exercise 1"
    The log-likelihood for iid $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$ is:

    $$\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n(X_i - \mu)^2$$

    **For $\mu$:** $\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n(X_i - \mu) = 0$ gives $\sum X_i = n\mu$, so $\hat{\mu} = \bar{X}$.

    **For $\sigma^2$:** $\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n(X_i - \mu)^2 = 0$.

    Solving: $n\sigma^2 = \sum(X_i - \mu)^2$, and substituting $\hat{\mu} = \bar{X}$:

    $$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n(X_i - \bar{X})^2$$

    The second-order conditions confirm this is a maximum (the Hessian is negative definite at the MLE). $\square$

---

**Exercise 2.**
Show that the Fisher information for $\mu$ in the $N(\mu, \sigma^2)$ model is $I(\mu) = 1/\sigma^2$ per observation, and that the CRLB for estimating $\mu$ from $n$ observations is $\sigma^2/n$.

??? success "Solution to Exercise 2"
    The log-likelihood for a single observation is:

    $$\ell(\mu; x) = -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$$

    The score function is:

    $$\frac{\partial \ell}{\partial \mu} = \frac{x - \mu}{\sigma^2}$$

    The Fisher information per observation is:

    $$I_1(\mu) = E\left[\left(\frac{\partial \ell}{\partial \mu}\right)^2\right] = E\left[\frac{(X-\mu)^2}{\sigma^4}\right] = \frac{\sigma^2}{\sigma^4} = \frac{1}{\sigma^2}$$

    For $n$ iid observations, $I_n(\mu) = nI_1(\mu) = n/\sigma^2$. The CRLB is:

    $$\text{Var}(\hat{\mu}) \geq \frac{1}{I_n(\mu)} = \frac{\sigma^2}{n}$$

    Since $\text{Var}(\bar{X}) = \sigma^2/n$, the sample mean achieves the CRLB exactly and is therefore an **efficient** estimator. $\square$

---

**Exercise 3.**
Construct a 95% confidence interval for the mean of a normal population when $n = 25$, $\bar{x} = 12.4$, and $s = 3.1$. Compare the $z$-interval (pretending $\sigma$ is known) with the correct $t$-interval.

??? success "Solution to Exercise 3"
    **$z$-interval** (treating $s$ as $\sigma$): $z_{0.025} = 1.960$.

    $$\bar{x} \pm z_{0.025}\frac{s}{\sqrt{n}} = 12.4 \pm 1.960 \times \frac{3.1}{\sqrt{25}} = 12.4 \pm 1.216$$

    $$\text{CI}_z = [11.184, 13.616]$$

    **$t$-interval** (correct): $t_{24, 0.025} = 2.064$.

    $$\bar{x} \pm t_{24, 0.025}\frac{s}{\sqrt{n}} = 12.4 \pm 2.064 \times \frac{3.1}{\sqrt{25}} = 12.4 \pm 1.280$$

    $$\text{CI}_t = [11.120, 13.680]$$

    The $t$-interval is wider (by about 5%) because it accounts for the additional uncertainty from estimating $\sigma$. For $n = 25$, the difference is modest; for smaller $n$, it would be more substantial. $\square$

---

**Exercise 4.**
A portfolio has 504 daily returns with sample mean $\hat{\mu} = 0.035\%$ and sample standard deviation $\hat{\sigma} = 1.30\%$. Compute the 1% and 5% parametric (Gaussian) VaR. If the true returns have a $t$-distribution with 5 degrees of freedom, would you expect the Gaussian VaR to over- or underestimate the true VaR?

??? success "Solution to Exercise 4"
    **Gaussian VaR:**

    $$\text{VaR}_{1\%} = -(\hat{\mu} + z_{0.01}\hat{\sigma}) = -(0.035\% + (-2.326)(1.30\%)) = -(0.035\% - 3.024\%) = 2.989\%$$

    $$\text{VaR}_{5\%} = -(\hat{\mu} + z_{0.05}\hat{\sigma}) = -(0.035\% + (-1.645)(1.30\%)) = -(0.035\% - 2.139\%) = 2.103\%$$

    **Effect of heavy tails:** The $t$-distribution with 5 degrees of freedom has heavier tails than the normal. Its 1st percentile is $t_{5, 0.01} = -3.365$ (compared to $z_{0.01} = -2.326$). The Gaussian VaR **underestimates** the true tail risk because the normal model does not capture the excess probability in the tails.

    This is a systematic problem: Gaussian VaR is anti-conservative for fat-tailed distributions, which is exactly the situation encountered in financial returns. $\square$

---

**Exercise 5.**
Prove that the MLE for $\sigma^2$ is asymptotically efficient, i.e., $n \cdot \text{Var}(\hat{\sigma}^2_{\text{MLE}}) \to 2\sigma^4$ as $n \to \infty$.

??? success "Solution to Exercise 5"
    We have $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum(X_i - \bar{X})^2 = \frac{n-1}{n}S^2$.

    For normal data, $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$, so $\text{Var}(S^2) = 2\sigma^4/(n-1)$.

    $$\text{Var}(\hat{\sigma}^2_{\text{MLE}}) = \left(\frac{n-1}{n}\right)^2 \text{Var}(S^2) = \left(\frac{n-1}{n}\right)^2 \cdot \frac{2\sigma^4}{n-1} = \frac{2(n-1)\sigma^4}{n^2}$$

    Therefore:

    $$n \cdot \text{Var}(\hat{\sigma}^2_{\text{MLE}}) = \frac{2(n-1)\sigma^4}{n} \to 2\sigma^4 \text{ as } n \to \infty$$

    The CRLB for $\sigma^2$ is $1/I_n(\sigma^2) = 2\sigma^4/n$, so $n \cdot \text{CRLB} = 2\sigma^4$.

    Since the asymptotic variance equals the CRLB, $\hat{\sigma}^2_{\text{MLE}}$ is asymptotically efficient. $\square$
