# Estimation Methods Comparison

## Overview

This page compares the major approaches to point estimation: the **Method of Moments (MoM)** and **Maximum Likelihood Estimation (MLE)**. Using both analytical derivations and Monte Carlo simulations, we examine the bias, variance, and MSE of each method for several distributions. Understanding when and why MLE outperforms MoM -- and the computational cost of that advantage -- is central to applied statistical practice.

## Method of Moments

The Method of Moments equates population moments to their sample counterparts and solves for the unknown parameters. For a distribution with parameters $\theta_1, \ldots, \theta_k$:

$$
\mu_r'(\theta_1, \ldots, \theta_k) = \frac{1}{n}\sum_{i=1}^n X_i^r, \quad r = 1, \ldots, k
$$

!!! info "Advantages and Disadvantages"
    **Advantages:** Simple closed-form expressions; no optimization required; always consistent under mild conditions.

    **Disadvantages:** May produce estimates outside the parameter space; generally less efficient than MLE; does not use the full likelihood information.

## Maximum Likelihood Estimation

MLE finds the parameter values that maximize the likelihood of the observed data:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \prod_{i=1}^n f(x_i; \theta)
$$

In practice, we maximize the log-likelihood:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^n \log f(x_i; \theta)
$$

## Comparing Variance Estimators

A foundational comparison involves three estimators of the population variance $\sigma^2$ from a normal sample of size $n$:

| Estimator | Divisor | Bias | MSE |
|-----------|---------|------|-----|
| MLE $\hat{\sigma}^2_n$ | $n$ | $-\sigma^2/n$ | $\frac{2n-1}{n^2}\sigma^4$ |
| Bessel $S^2_{n-1}$ | $n-1$ | $0$ | $\frac{2}{n-1}\sigma^4$ |
| MSE-optimal $\hat{\sigma}^2_{n+1}$ | $n+1$ | $-\frac{2\sigma^2}{n+1}$ | $\frac{2}{n+1}\sigma^4$ |

The following simulation verifies these results empirically.

```python
import numpy as np

def compare_variance_estimators(mu=5, sigma2=4, n=10, n_sim=50_000):
    sigma = np.sqrt(sigma2)
    rng = np.random.default_rng(42)

    results = {}
    samples = rng.normal(mu, sigma, (n_sim, n))
    ss = np.sum((samples - samples.mean(axis=1, keepdims=True)) ** 2, axis=1)

    estimators = {
        "MLE (n)":      ss / n,
        "Bessel (n-1)": ss / (n - 1),
        "MSE-opt (n+1)": ss / (n + 1),
    }

    for name, vals in estimators.items():
        bias = vals.mean() - sigma2
        var = vals.var()
        mse = np.mean((vals - sigma2) ** 2)
        results[name] = {"bias": bias, "var": var, "mse": mse}
        print(f"{name:18s}  bias={bias:+.4f}  var={var:.4f}  MSE={mse:.4f}")

    return results

compare_variance_estimators()
```

!!! note "Key Observation"
    The unbiased estimator $S^2_{n-1}$ has the **largest** MSE among the three. Dividing by $n+1$ introduces bias but achieves the minimum MSE, illustrating the bias--variance tradeoff.

## Shrinkage Estimator Demonstration

A shrinkage estimator $\hat{\mu}_\lambda = \lambda \bar{X}$ trades bias for reduced variance. The MSE decomposes as:

$$
\text{MSE}(\hat{\mu}_\lambda) = \lambda^2 \frac{\sigma^2}{n} + (1 - \lambda)^2 \mu^2
$$

The MSE-optimal shrinkage factor is:

$$
\lambda^* = \frac{\mu^2}{\mu^2 + \sigma^2/n}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

def shrinkage_mse(mu_true=3, sigma2=4, n=20):
    lambdas = np.linspace(0.01, 1.5, 200)
    bias_sq = (lambdas - 1) ** 2 * mu_true ** 2
    variance = lambdas ** 2 * sigma2 / n
    mse = bias_sq + variance

    lambda_opt = mu_true ** 2 / (mu_true ** 2 + sigma2 / n)
    print(f"Optimal lambda = {lambda_opt:.4f}")
    print(f"MSE at lambda=1 (unbiased): {sigma2 / n:.4f}")
    print(f"MSE at lambda*:             {lambda_opt**2 * sigma2/n + (1 - lambda_opt)**2 * mu_true**2:.4f}")

shrinkage_mse()
```

## MLE for the Normal Distribution

For $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$, the MLEs have closed-form solutions:

$$
\hat{\mu}_{\text{MLE}} = \bar{X}, \qquad \hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
$$

We can also verify these by numerical optimization of the negative log-likelihood:

$$
-\ell(\mu, \sigma^2) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2
$$

```python
import numpy as np
from scipy import optimize

def mle_normal_demo(n=100):
    rng = np.random.default_rng(42)
    mu_true, sigma_true = 5.0, 2.0
    data = rng.normal(mu_true, sigma_true, n)

    # Closed-form MLE
    mu_hat = data.mean()
    sigma2_hat = np.mean((data - mu_hat) ** 2)

    # Numerical MLE via optimization
    def neg_log_lik(params, x):
        mu, log_sigma2 = params
        sigma2 = np.exp(log_sigma2)
        n = len(x)
        return 0.5 * n * np.log(2 * np.pi * sigma2) + np.sum((x - mu) ** 2) / (2 * sigma2)

    result = optimize.minimize(neg_log_lik, x0=[0, 0], args=(data,), method="Nelder-Mead")
    mu_num, sigma2_num = result.x[0], np.exp(result.x[1])

    print(f"True:        mu = {mu_true:.4f}, sigma^2 = {sigma_true**2:.4f}")
    print(f"Closed-form: mu = {mu_hat:.4f}, sigma^2 = {sigma2_hat:.4f}")
    print(f"Numerical:   mu = {mu_num:.4f}, sigma^2 = {sigma2_num:.4f}")

mle_normal_demo()
```

## MLE vs Method of Moments for the Gamma Distribution

For $X \sim \text{Gamma}(\alpha, \beta)$ with $E[X] = \alpha\beta$ and $\text{Var}(X) = \alpha\beta^2$, the MoM estimators are:

$$
\hat{\alpha}_{\text{MoM}} = \frac{\bar{X}^2}{S^2}, \qquad \hat{\beta}_{\text{MoM}} = \frac{S^2}{\bar{X}}
$$

The MLE has no closed form and requires numerical optimization.

```python
import numpy as np
from scipy import stats

def mle_vs_mom_gamma(alpha_true=3, beta_true=2, n=200, n_sim=5000):
    rng = np.random.default_rng(42)
    mle_alpha, mom_alpha = [], []

    for _ in range(n_sim):
        data = rng.gamma(alpha_true, beta_true, n)

        # Method of Moments
        m1 = data.mean()
        v = data.var(ddof=0)
        mom_alpha.append(m1 ** 2 / v)

        # MLE (scipy)
        a_mle, _, _ = stats.gamma.fit(data, floc=0)
        mle_alpha.append(a_mle)

    mle_alpha = np.array(mle_alpha)
    mom_alpha = np.array(mom_alpha)

    for name, vals in [("MLE", mle_alpha), ("MoM", mom_alpha)]:
        bias = vals.mean() - alpha_true
        mse = np.mean((vals - alpha_true) ** 2)
        print(f"{name}: bias={bias:+.4f}, MSE={mse:.6f}")

mle_vs_mom_gamma()
```

!!! success "MLE Wins on MSE"
    For the Gamma distribution, MLE has smaller MSE than MoM for both $\alpha$ and $\beta$. This aligns with the asymptotic theory: MLE is efficient (achieves the Cramer--Rao bound), while MoM generally does not.

## Cramer-Rao Lower Bound Verification

The Cramer-Rao inequality states that for any unbiased estimator $\hat{\theta}$:

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}
$$

where $I(\theta)$ is the Fisher information. For estimating the mean of $N(\mu, \sigma^2)$, the CRLB is $\sigma^2/n$, and the sample mean achieves this bound exactly.

```python
import numpy as np

def cramer_rao_demo(n=50, n_sim=20_000):
    rng = np.random.default_rng(42)
    mu_true, sigma = 5.0, 2.0
    crlb = sigma ** 2 / n

    means = np.array([rng.normal(mu_true, sigma, n).mean() for _ in range(n_sim)])
    medians = np.array([np.median(rng.normal(mu_true, sigma, n)) for _ in range(n_sim)])

    print(f"CRLB = sigma^2/n = {crlb:.6f}")
    print(f"Var(X_bar)       = {means.var():.6f}  (ratio to CRLB: {means.var()/crlb:.4f})")
    print(f"Var(median)      = {medians.var():.6f}  (ratio to CRLB: {medians.var()/crlb:.4f})")

cramer_rao_demo()
```

## Interpretation

The simulations confirm several theoretical results:

1. **Bias--variance tradeoff is real**: The MSE-optimal variance estimator divides by $n+1$, not $n-1$, despite being biased.
2. **MLE is asymptotically efficient**: For the Gamma distribution, MLE achieves lower MSE than MoM as theory predicts.
3. **Shrinkage can help**: Pulling the sample mean toward zero reduces MSE when the signal-to-noise ratio $\mu/(\sigma/\sqrt{n})$ is moderate.
4. **The sample mean is CRLB-efficient**: Its variance matches the Cramer-Rao lower bound exactly for the normal mean.

## Exercises

**Exercise 1.** For $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Exp}(\lambda)$, derive the Method of Moments and MLE estimators for $\lambda$. Are they the same?

??? success "Solution to Exercise 1"
    The exponential distribution has $E[X] = 1/\lambda$, so the MoM estimator sets $\bar{X} = 1/\hat{\lambda}$, giving $\hat{\lambda}_{\text{MoM}} = 1/\bar{X}$.

    The log-likelihood is $\ell(\lambda) = n\log\lambda - \lambda \sum x_i$. Setting $\ell'(\lambda) = n/\lambda - \sum x_i = 0$ gives $\hat{\lambda}_{\text{MLE}} = n/\sum x_i = 1/\bar{X}$.

    The two estimators are identical for the exponential distribution. This happens because the exponential has a single parameter determined by a single moment. $\square$

---

**Exercise 2.** Show that the MSE-optimal estimator of $\sigma^2$ in the family $\hat{\sigma}^2_c = \frac{1}{c}\sum_{i=1}^n(X_i - \bar{X})^2$ has $c^* = n+1$ for a normal population.

??? success "Solution to Exercise 2"
    Let $Q = \sum(X_i - \bar{X})^2$. For $X_i \sim N(\mu, \sigma^2)$, we have $Q/\sigma^2 \sim \chi^2_{n-1}$, so $E[Q] = (n-1)\sigma^2$ and $\text{Var}(Q) = 2(n-1)\sigma^4$.

    The MSE of $Q/c$ is:

    $$
    \text{MSE}(Q/c) = \text{Var}(Q/c) + [\text{Bias}(Q/c)]^2 = \frac{2(n-1)\sigma^4}{c^2} + \left(\frac{n-1}{c} - 1\right)^2\sigma^4
    $$

    Taking the derivative with respect to $c$ and setting it to zero:

    $$
    \frac{d}{dc}\text{MSE} = -\frac{4(n-1)\sigma^4}{c^3} - \frac{2(n-1)\sigma^4}{c^2}\left(\frac{n-1}{c} - 1\right) = 0
    $$

    Simplifying: $-4(n-1)/c^3 + 2(n-1)(c - n + 1)/c^3 = 0$, which gives $2(c - n + 1) = 4$, so $c = n + 1$. $\square$

---

**Exercise 3.** Run a Monte Carlo simulation comparing MLE and MoM for the Beta distribution $\text{Beta}(\alpha, \beta)$ with $\alpha = 2, \beta = 5$ and sample size $n = 50$. Which method has lower MSE for estimating $\alpha$?

??? success "Solution to Exercise 3"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(42)
    a_true, b_true, n, n_sim = 2, 5, 50, 10_000
    mle_a, mom_a = [], []

    for _ in range(n_sim):
        data = rng.beta(a_true, b_true, n)
        m1 = data.mean()
        m2 = np.mean(data ** 2)
        v = m2 - m1 ** 2
        common = m1 * (1 - m1) / v - 1
        mom_a.append(m1 * common)

        a_mle, b_mle, _, _ = stats.beta.fit(data, floc=0, fscale=1)
        mle_a.append(a_mle)

    mle_a, mom_a = np.array(mle_a), np.array(mom_a)
    print(f"MLE: MSE = {np.mean((mle_a - a_true)**2):.6f}")
    print(f"MoM: MSE = {np.mean((mom_a - a_true)**2):.6f}")
    ```

    MLE will typically have lower MSE, consistent with its asymptotic efficiency. $\square$

---

**Exercise 4.** Prove that the MLE is invariant under reparameterization: if $\hat{\theta}$ is the MLE of $\theta$, then $g(\hat{\theta})$ is the MLE of $g(\theta)$ for any function $g$.

??? success "Solution to Exercise 4"
    Let $\eta = g(\theta)$ where $g$ is a one-to-one function (the general case extends via the induced likelihood). The likelihood as a function of $\eta$ is:

    $$
    L^*(\eta) = L(g^{-1}(\eta))
    $$

    Since $L(\theta)$ is maximized at $\hat{\theta}$, $L^*(g(\hat{\theta})) = L(\hat{\theta}) \geq L(\theta)$ for all $\theta$. Hence $L^*(\eta) \leq L^*(g(\hat{\theta}))$ for all $\eta$ in the range of $g$, so $g(\hat{\theta})$ maximizes $L^*$.

    For general (not necessarily one-to-one) $g$, define $\hat{\eta} = \sup_{\{\theta: g(\theta) = \eta\}} L(\theta)$ and take the maximizer, which equals $g(\hat{\theta})$ by construction. $\square$

---

**Exercise 5.** The Fisher information for the Bernoulli parameter $p$ is $I(p) = 1/[p(1-p)]$. Verify numerically that the variance of the sample proportion $\hat{p} = \bar{X}$ achieves the Cramer-Rao bound $1/[nI(p)]$ for $p = 0.3$ and $n = 100$.

??? success "Solution to Exercise 5"
    ```python
    import numpy as np

    rng = np.random.default_rng(42)
    p, n, n_sim = 0.3, 100, 100_000
    p_hats = np.array([rng.binomial(n, p) / n for _ in range(n_sim)])

    crlb = p * (1 - p) / n
    empirical_var = p_hats.var()
    print(f"CRLB = p(1-p)/n = {crlb:.6f}")
    print(f"Var(p_hat)      = {empirical_var:.6f}")
    print(f"Ratio           = {empirical_var / crlb:.4f}")
    ```

    The ratio should be very close to 1.0, confirming that $\hat{p}$ is an efficient estimator achieving the CRLB. $\square$
