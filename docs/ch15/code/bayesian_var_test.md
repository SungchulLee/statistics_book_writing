# Bayesian Variance Test

## Overview

This page presents a Bayesian approach to comparing variances across two groups. Rather than computing a single $p$-value, the Bayesian framework produces a full posterior distribution for each group's variance and for the variance ratio. We use a conjugate Normal--Inverse-Gamma model, draw posterior samples via Monte Carlo, and summarize the results with credible intervals and posterior probabilities.

---

## The Normal--Inverse-Gamma Model

For each group $i$, we model the data as $x_{ij} \mid \mu_i, \sigma_i^2 \sim \mathcal{N}(\mu_i, \sigma_i^2)$ and place conjugate priors:

$$
\mu_i \mid \sigma_i^2 \sim \mathcal{N}\!\left(m_0,\; \frac{\sigma_i^2}{\kappa_0}\right), \qquad \sigma_i^2 \sim \text{Inv-Gamma}(\alpha_0, \beta_0)
$$

where $m_0$, $\kappa_0$, $\alpha_0$, $\beta_0$ are hyperparameters. With vague priors ($\kappa_0 \approx 0$, $\alpha_0 \approx 0$, $\beta_0 \approx 0$), the posterior is dominated by the data.

## Posterior Updates

Given $n$ observations with sample mean $\bar{x}$ and sum of squared deviations $S = \sum_{j=1}^{n}(x_j - \bar{x})^2$, the posterior parameters are:

$$
\kappa_n = \kappa_0 + n, \qquad m_n = \frac{\kappa_0 m_0 + n \bar{x}}{\kappa_n}
$$

$$
\alpha_n = \alpha_0 + \frac{n}{2}, \qquad \beta_n = \beta_0 + \frac{1}{2}\left(S + \frac{\kappa_0 n}{\kappa_n}(\bar{x} - m_0)^2\right)
$$

The marginal posterior for the variance is:

$$
\sigma_i^2 \mid \mathbf{x}_i \sim \text{Inv-Gamma}(\alpha_n,\, \beta_n)
$$

---

## Implementation

The following code computes the posterior parameters and draws samples from the posterior of $\sigma^2$:

```python
import numpy as np
from scipy.stats import invgamma

def posterior_params(x, m0=0.0, k0=1e-6, a0=1e-2, b0=1e-2):
    x = np.asarray(x, dtype=float)
    n = x.size
    xbar = x.mean()
    S = np.sum((x - xbar)**2)
    k_n = k0 + n
    m_n = (k0 * m0 + n * xbar) / k_n
    a_n = a0 + n / 2.0
    b_n = b0 + 0.5 * (S + (k0 * n / k_n) * (xbar - m0)**2)
    return m_n, k_n, a_n, b_n

def draw_posterior_sigma2(x, n_draws=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    m_n, k_n, a_n, b_n = posterior_params(x)
    sig2 = invgamma(a=a_n, scale=b_n).rvs(size=n_draws, random_state=rng)
    return sig2
```

---

## Comparing Two Groups

To compare $\sigma_1^2$ and $\sigma_2^2$, we draw independently from each posterior and form the ratio:

$$
\rho = \frac{\sigma_1^2}{\sigma_2^2}
$$

A 95% credible interval for $\rho$ that excludes 1 provides evidence that the variances differ.

```python
x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
x2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)

rng = np.random.default_rng(0)
s1 = draw_posterior_sigma2(x1, n_draws=20000, rng=rng)
s2 = draw_posterior_sigma2(x2, n_draws=20000, rng=rng)
ratio = s1 / s2

print(f"Posterior mean of sigma1^2: {s1.mean():.3f}")
print(f"Posterior mean of sigma2^2: {s2.mean():.3f}")
print(f"Posterior mean of ratio:    {ratio.mean():.3f}")
print(f"95% credible interval:      ({np.percentile(ratio, 2.5):.3f}, "
      f"{np.percentile(ratio, 97.5):.3f})")
print(f"P(sigma1^2 > sigma2^2):     {np.mean(ratio > 1.0):.4f}")
```

---

## Interpretation

- The posterior distribution of each $\sigma_i^2$ summarizes all information about the group variance given the data and prior.
- The posterior probability $P(\sigma_1^2 > \sigma_2^2 \mid \text{data})$ directly answers the question of interest without requiring a fixed significance level.
- With vague priors, the Bayesian credible interval closely approximates the frequentist confidence interval, but the interpretation differs: the credible interval says "there is a 95% probability that the true parameter lies in this interval" (given the model and prior).
- Unlike the classical $F$-test, this approach does not assume normality in the testing procedure itself, though the likelihood model does assume normality.

---

## Exercises

**Exercise 1.** Starting from the Inverse-Gamma prior $\sigma^2 \sim \text{Inv-Gamma}(\alpha_0, \beta_0)$ and the normal likelihood, derive the posterior parameters $\alpha_n$ and $\beta_n$ given above. Show each step of the conjugate update.

??? success "Solution to Exercise 1"

    The likelihood for $n$ observations from $\mathcal{N}(\mu, \sigma^2)$ with known $\mu$ is:

    $$
    L(\sigma^2) \propto (\sigma^2)^{-n/2} \exp\!\left(-\frac{S}{2\sigma^2}\right)
    $$

    where $S = \sum(x_j - \mu)^2$. The Inverse-Gamma prior density is:

    $$
    p(\sigma^2) \propto (\sigma^2)^{-\alpha_0 - 1} \exp\!\left(-\frac{\beta_0}{\sigma^2}\right)
    $$

    Multiplying:

    $$
    p(\sigma^2 \mid \mathbf{x}) \propto (\sigma^2)^{-(\alpha_0 + n/2) - 1} \exp\!\left(-\frac{\beta_0 + S/2}{\sigma^2}\right)
    $$

    This is the kernel of an $\text{Inv-Gamma}(\alpha_0 + n/2,\; \beta_0 + S/2)$ density. With unknown $\mu$ and the normal--Inverse-Gamma joint prior, the additional term $\frac{\kappa_0 n}{2\kappa_n}(\bar{x} - m_0)^2$ appears in $\beta_n$ from integrating out $\mu$. $\square$

---

**Exercise 2.** Modify the code to use an informative prior: set $\alpha_0 = 3$, $\beta_0 = 10$, which centers the prior around $\sigma^2 = 5$. Compare the posterior credible intervals to those obtained with the vague prior. How does the informative prior affect the results when the sample size is small?

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy.stats import invgamma

    def posterior_params(x, m0=0.0, k0=1e-6, a0=1e-2, b0=1e-2):
        x = np.asarray(x, dtype=float)
        n = x.size
        xbar = x.mean()
        S = np.sum((x - xbar)**2)
        k_n = k0 + n
        a_n = a0 + n / 2.0
        b_n = b0 + 0.5 * (S + (k0 * n / k_n) * (xbar - m0)**2)
        return a_n, b_n

    x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
    rng = np.random.default_rng(0)

    # Vague prior
    a_v, b_v = posterior_params(x1, a0=1e-2, b0=1e-2)
    draws_v = invgamma(a=a_v, scale=b_v).rvs(20000, random_state=rng)

    # Informative prior
    a_i, b_i = posterior_params(x1, a0=3, b0=10)
    draws_i = invgamma(a=a_i, scale=b_i).rvs(20000, random_state=rng)

    print(f"Vague:       mean={draws_v.mean():.2f}, "
          f"CI=({np.percentile(draws_v,2.5):.2f}, {np.percentile(draws_v,97.5):.2f})")
    print(f"Informative: mean={draws_i.mean():.2f}, "
          f"CI=({np.percentile(draws_i,2.5):.2f}, {np.percentile(draws_i,97.5):.2f})")
    ```

    With only 8 observations, the informative prior pulls the posterior toward $\sigma^2 = 5$ (the prior mean $\beta_0/(\alpha_0 - 1) = 5$). The credible interval with the informative prior is narrower and shifted compared to the vague prior. As $n$ grows, the influence of the prior diminishes and both posteriors converge. $\square$

---

**Exercise 3.** Generate $n_1 = 30$ observations from $\mathcal{N}(0, 4)$ and $n_2 = 30$ from $\mathcal{N}(0, 9)$. Compute the posterior for the variance ratio $\rho = \sigma_1^2 / \sigma_2^2$ and determine $P(\rho < 1 \mid \text{data})$. Compare this to the $p$-value from a classical $F$-test.

??? success "Solution to Exercise 3"

    ```python
    import numpy as np
    from scipy.stats import invgamma, f as f_dist

    def posterior_params(x, m0=0.0, k0=1e-6, a0=1e-2, b0=1e-2):
        n = x.size
        xbar = x.mean()
        S = np.sum((x - xbar)**2)
        k_n = k0 + n
        a_n = a0 + n / 2.0
        b_n = b0 + 0.5 * (S + (k0 * n / k_n) * (xbar - m0)**2)
        return a_n, b_n

    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 2, 30)
    x2 = rng.normal(0, 3, 30)

    a1, b1 = posterior_params(x1)
    a2, b2 = posterior_params(x2)
    s1 = invgamma(a=a1, scale=b1).rvs(50000, random_state=rng)
    s2 = invgamma(a=a2, scale=b2).rvs(50000, random_state=rng)
    ratio = s1 / s2

    print(f"P(rho < 1 | data) = {np.mean(ratio < 1):.4f}")

    # Classical F-test
    F_stat = np.var(x1, ddof=1) / np.var(x2, ddof=1)
    p_val = 2 * min(f_dist.cdf(F_stat, 29, 29), 1 - f_dist.cdf(F_stat, 29, 29))
    print(f"F-test p-value = {p_val:.4f}")
    ```

    Both the Bayesian posterior probability and the frequentist $p$-value should indicate that $\sigma_1^2 < \sigma_2^2$. The posterior probability $P(\rho < 1)$ will typically be high (e.g., above 0.90), consistent with a small $p$-value from the $F$-test. $\square$

---

**Exercise 4.** Explain why the Bayesian approach uses the log of the variance ratio for summarization (e.g., $\log(\sigma_1^2/\sigma_2^2)$) rather than the ratio itself. Draw 50,000 posterior samples and plot histograms of both $\rho$ and $\log(\rho)$. Which is closer to symmetric?

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from scipy.stats import invgamma
    import matplotlib.pyplot as plt

    def posterior_params(x, m0=0.0, k0=1e-6, a0=1e-2, b0=1e-2):
        n = x.size; xbar = x.mean(); S = np.sum((x - xbar)**2)
        k_n = k0 + n
        a_n = a0 + n / 2.0
        b_n = b0 + 0.5 * (S + (k0 * n / k_n) * (xbar - m0)**2)
        return a_n, b_n

    x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
    x2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)
    rng = np.random.default_rng(0)

    a1, b1 = posterior_params(x1)
    a2, b2 = posterior_params(x2)
    s1 = invgamma(a=a1, scale=b1).rvs(50000, random_state=rng)
    s2 = invgamma(a=a2, scale=b2).rvs(50000, random_state=rng)
    ratio = s1 / s2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(ratio, bins=60, edgecolor='k', alpha=0.7)
    axes[0].set_title("Posterior of ratio")
    axes[1].hist(np.log(ratio), bins=60, edgecolor='k', alpha=0.7)
    axes[1].set_title("Posterior of log(ratio)")
    plt.tight_layout()
    plt.show()
    ```

    The ratio $\rho = \sigma_1^2/\sigma_2^2$ is bounded below by zero and has a heavy right tail, making it right-skewed. The log transform maps $(0, \infty)$ to $(-\infty, \infty)$ and produces a distribution that is much closer to symmetric. Symmetric distributions are easier to summarize with a mean and credible interval, and percentile-based intervals on the log scale transform back to intervals that are more balanced in relative terms. $\square$

---

**Exercise 5.** Describe a scenario where the Bayesian variance test and the classical $F$-test would give substantially different conclusions. Consider the role of sample size, prior informativeness, and departures from normality.

??? success "Solution to Exercise 5"

    The two approaches diverge most in the following scenarios:

    1. **Small sample size with informative prior.** If $n$ is very small (say 5 per group) and the researcher has strong prior information favoring equal variances, the Bayesian posterior for $\rho$ will be concentrated near 1 even if the sample variances differ moderately. The $F$-test ignores prior information and may either reject or fail to reject based solely on the noisy sample estimates.

    2. **Non-normal data.** The $F$-test is highly sensitive to departures from normality: heavy-tailed data inflate the sample variances and produce inflated Type I error. The Bayesian model as presented also assumes normality in the likelihood, so it suffers from the same misspecification. However, the Bayesian framework can be extended to use robust likelihoods (e.g., Student-$t$), which gracefully handles outliers.

    3. **One-sided questions.** The Bayesian approach naturally answers $P(\sigma_1^2 > \sigma_2^2)$, while the $F$-test is typically set up as a two-sided test and requires explicit modification for one-sided alternatives. When the question is directional, the posterior probability provides a more direct and interpretable answer. $\square$
