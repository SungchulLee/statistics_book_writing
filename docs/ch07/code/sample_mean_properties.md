# Sample Mean Properties

## Overview

The sample mean $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ is the most fundamental estimator in statistics. This page verifies its core properties through simulation: unbiasedness across distributions, the variance formula $\text{Var}(\bar{X}) = \sigma^2/n$, mean squared error, standard error convergence at the $1/\sqrt{n}$ rate, efficiency relative to alternative location estimators, and inverse-variance weighting.

## Unbiasedness

The sample mean is unbiased for the population mean regardless of the underlying distribution:

$$E[\bar{X}] = \mu$$

This follows from linearity of expectation. The proof requires only that each $X_i$ has the same mean $\mu$; no distributional assumption is needed.

The following simulation verifies this across six different distributions by computing $\bar{X}$ from 100,000 replications and checking that the average is close to the true mean.

```python
import numpy as np

def verify_unbiasedness(mu=10.0, sigma=3.0, n=20, n_sim=100_000, seed=42):
    rng = np.random.default_rng(seed)

    distributions = {
        f'Normal({mu}, {sigma}²)': (lambda: rng.normal(mu, sigma, n), mu),
        'Exp(λ=0.5)':             (lambda: rng.exponential(2, n), 2.0),
        'Poisson(3.7)':           (lambda: rng.poisson(3.7, n), 3.7),
        'Uniform(2, 8)':          (lambda: rng.uniform(2, 8, n), 5.0),
        'Bernoulli(0.4)':         (lambda: rng.binomial(1, 0.4, n), 0.4),
        'Chi²(df=5)':             (lambda: rng.chisquare(5, n), 5.0),
    }

    for name, (sampler, true_mu) in distributions.items():
        estimates = np.array([sampler().mean() for _ in range(n_sim)])
        bias = estimates.mean() - true_mu
        print(f"{name:<22} True μ={true_mu:.4f}  E[X̄]={estimates.mean():.4f}  Bias={bias:.6f}")
```

!!! tip "Key takeaway"
    All biases are negligibly small (within Monte Carlo noise), confirming $E[\bar{X}] = \mu$ for every distribution tested.

## Variance and MSE

For iid observations with variance $\sigma^2$:

$$\text{Var}(\bar{X}) = \frac{\sigma^2}{n}, \qquad \text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}}$$

Since $\bar{X}$ is unbiased, its MSE equals its variance:

$$\text{MSE}(\bar{X}) = \text{Bias}^2 + \text{Var}(\bar{X}) = 0 + \frac{\sigma^2}{n} = \frac{\sigma^2}{n}$$

```python
def verify_variance_and_mse(mu=10.0, sigma=3.0, n_sim=100_000, seed=42):
    rng = np.random.default_rng(seed)
    sample_sizes = [5, 10, 25, 50, 100, 500]

    for n in sample_sizes:
        samples = rng.normal(mu, sigma, (n_sim, n))
        x_bars = samples.mean(axis=1)

        var_xbar   = x_bars.var(ddof=0)
        theory_var = sigma**2 / n
        mse        = np.mean((x_bars - mu)**2)
        se         = x_bars.std(ddof=0)
        theory_se  = sigma / np.sqrt(n)

        print(f"n={n:>4}  Var(X̄)={var_xbar:.6f}  σ²/n={theory_var:.6f}  "
              f"MSE={mse:.6f}  SE={se:.6f}  σ/√n={theory_se:.6f}")
```

## Efficiency Comparison

The sample mean is the most efficient location estimator for normal data, but not for heavy-tailed distributions. **Relative efficiency** of estimator $T$ relative to $\bar{X}$ is:

$$\text{RE}(T, \bar{X}) = \frac{\text{MSE}(\bar{X})}{\text{MSE}(T)}$$

When $\text{RE} > 1$, the alternative $T$ is *more* efficient.

```python
from scipy import stats

def efficiency_comparison(n=30, n_sim=50_000, seed=42):
    rng = np.random.default_rng(seed)

    distributions = {
        'Normal(0,1)':           lambda: rng.standard_normal(n),
        't(df=3)':               lambda: rng.standard_t(3, n),
        'Contaminated Normal':   lambda: np.where(
            rng.uniform(0, 1, n) < 0.1,
            rng.normal(0, 10, n),
            rng.standard_normal(n)),
    }

    for dist_name, sampler in distributions.items():
        est = {'Mean': [], 'Median': [], 'Trim10%': [], 'Trim20%': []}
        for _ in range(n_sim):
            s = sampler()
            est['Mean'].append(np.mean(s))
            est['Median'].append(np.median(s))
            est['Trim10%'].append(stats.trim_mean(s, 0.1))
            est['Trim20%'].append(stats.trim_mean(s, 0.2))

        mse_mean = np.mean(np.array(est['Mean'])**2)
        print(f"\n{dist_name}:")
        for name, vals in est.items():
            mse = np.mean(np.array(vals)**2)
            re  = mse_mean / mse
            print(f"  {name:<12} MSE={mse:.6f}  Rel.Eff.={re:.4f}")
```

!!! note "When the mean loses"
    For heavy-tailed distributions like $t(3)$ and contaminated normals, the trimmed mean and median have lower MSE than the sample mean. The mean's sensitivity to outliers makes it inefficient in these settings.

## Inverse-Variance Weighted Mean

When observations have unequal variances $\sigma_i^2$, the **optimal weights** are inversely proportional to the variances:

$$w_i = \frac{1/\sigma_i^2}{\sum_{j=1}^k 1/\sigma_j^2}, \qquad \bar{X}_w = \sum_{i=1}^k w_i X_i$$

This minimizes $\text{Var}(\bar{X}_w)$ among all weighted averages that sum to the true mean.

```python
def weighted_mean_demo(mu=5.0, n_sim=50_000, seed=42):
    rng = np.random.default_rng(seed)
    sigmas = np.array([1.0, 2.0, 5.0, 10.0, 0.5])

    optimal_weights = 1 / sigmas**2
    optimal_weights /= optimal_weights.sum()

    unweighted, weighted = [], []
    for _ in range(n_sim):
        obs = rng.normal(mu, sigmas)
        unweighted.append(obs.mean())
        weighted.append(np.sum(optimal_weights * obs))

    unweighted = np.array(unweighted)
    weighted   = np.array(weighted)

    print(f"Unweighted:  Var={unweighted.var():.6f}  MSE={np.mean((unweighted-mu)**2):.6f}")
    print(f"IV-Weighted: Var={weighted.var():.6f}  MSE={np.mean((weighted-mu)**2):.6f}")
    print(f"Variance reduction: {(1 - weighted.var()/unweighted.var())*100:.1f}%")
```

## Standard Error Convergence Rate

On a log-log plot, the standard error $\text{SE}(\bar{X}) = \sigma/\sqrt{n}$ appears as a straight line with slope $-1/2$, confirming the $O(1/\sqrt{n})$ convergence rate.

```python
import matplotlib.pyplot as plt

def convergence_rate_plot(mu=5.0, sigma=3.0, n_sim=50_000, seed=42):
    rng = np.random.default_rng(seed)
    sample_sizes = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

    empirical_se = []
    for n in sample_sizes:
        estimates = np.array([rng.normal(mu, sigma, n).mean() for _ in range(n_sim)])
        empirical_se.append(estimates.std())

    theoretical_se = [sigma / np.sqrt(n) for n in sample_sizes]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(sample_sizes, empirical_se, 'bo-', label='Empirical SE')
    ax.loglog(sample_sizes, theoretical_se, 'r--', label='σ/√n', linewidth=2)
    ax.set_xlabel('Sample Size n')
    ax.set_ylabel('Standard Error')
    ax.set_title('Convergence Rate of Sample Mean')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()
```

## Interpretation

- **Unbiasedness** is distribution-free: it holds for any population with a finite mean.
- **Variance** decreases as $1/n$, so the **standard error** decreases as $1/\sqrt{n}$. To cut the standard error in half, you need four times as many observations.
- **MSE equals variance** because the bias is zero. This is the simplest case of the bias-variance tradeoff.
- **Efficiency** depends on the population shape. For normal data the mean is optimal; for heavy-tailed data, trimmed means and medians can have lower MSE.
- **Inverse-variance weighting** is the correct way to combine observations of differing precision; it can dramatically reduce variance compared to a simple average.

## Exercises

**Exercise 1.**
Prove analytically that $E[\bar{X}] = \mu$ for any distribution with finite mean, using only the linearity of expectation.

??? success "Solution to Exercise 1"
    Let $X_1, \ldots, X_n$ be iid with $E[X_i] = \mu$. Then:

    $$E[\bar{X}] = E\left[\frac{1}{n}\sum_{i=1}^n X_i\right] = \frac{1}{n}\sum_{i=1}^n E[X_i] = \frac{1}{n} \cdot n\mu = \mu$$

    The first equality is the definition of $\bar{X}$, the second uses linearity of expectation, and the third uses $E[X_i] = \mu$ for all $i$. $\square$

---

**Exercise 2.**
Show that $\text{Var}(\bar{X}) = \sigma^2/n$ for iid observations. Then explain why you need to quadruple the sample size to halve the standard error.

??? success "Solution to Exercise 2"
    For iid $X_1, \ldots, X_n$ with $\text{Var}(X_i) = \sigma^2$:

    $$\text{Var}(\bar{X}) = \text{Var}\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) = \frac{1}{n^2}\cdot n\sigma^2 = \frac{\sigma^2}{n}$$

    The second step uses independence (so the variance of a sum is the sum of variances). The standard error is $\text{SE} = \sigma/\sqrt{n}$. To halve it:

    $$\frac{\sigma}{\sqrt{n'}} = \frac{1}{2}\cdot\frac{\sigma}{\sqrt{n}} \implies \sqrt{n'} = 2\sqrt{n} \implies n' = 4n$$

    $\square$

---

**Exercise 3.**
Consider five observations from sources with standard deviations $\sigma_1 = 1, \sigma_2 = 2, \sigma_3 = 5, \sigma_4 = 10, \sigma_5 = 0.5$. Compute the optimal inverse-variance weights and the variance of the weighted mean. Compare to the variance of the unweighted mean.

??? success "Solution to Exercise 3"
    The unnormalized weights are $w_i^* = 1/\sigma_i^2$:

    $$w_1^* = 1, \quad w_2^* = 0.25, \quad w_3^* = 0.04, \quad w_4^* = 0.01, \quad w_5^* = 4$$

    The sum is $W = 1 + 0.25 + 0.04 + 0.01 + 4 = 5.3$. Normalized weights: $w_i = w_i^*/W$.

    The variance of the weighted mean is:

    $$\text{Var}(\bar{X}_w) = \sum_{i=1}^5 w_i^2 \sigma_i^2 = \frac{1}{W^2}\sum_{i=1}^5 \frac{\sigma_i^2}{\sigma_i^4} = \frac{1}{W^2}\sum_{i=1}^5 \frac{1}{\sigma_i^2} = \frac{W}{W^2} = \frac{1}{W} = \frac{1}{5.3} \approx 0.1887$$

    The unweighted mean has variance:

    $$\text{Var}(\bar{X}) = \frac{1}{25}\sum_{i=1}^5 \sigma_i^2 = \frac{1 + 4 + 25 + 100 + 0.25}{25} = \frac{130.25}{25} = 5.21$$

    The inverse-variance weighted mean has roughly $5.21/0.189 \approx 27.6$ times smaller variance. $\square$

---

**Exercise 4.**
For a normal population, the sample mean achieves the Cramer-Rao lower bound $\sigma^2/n$. Show that the asymptotic relative efficiency of the sample median to the sample mean is $2/\pi \approx 0.637$.

??? success "Solution to Exercise 4"
    For $X_i \sim N(\mu, \sigma^2)$, the sample mean has variance $\sigma^2/n$. The sample median $\tilde{X}$ has asymptotic variance:

    $$\text{Var}(\tilde{X}) \approx \frac{1}{4n[f(\mu)]^2}$$

    where $f$ is the population density. For the normal distribution, $f(\mu) = \frac{1}{\sigma\sqrt{2\pi}}$, so:

    $$\text{Var}(\tilde{X}) \approx \frac{1}{4n \cdot \frac{1}{2\pi\sigma^2}} = \frac{2\pi\sigma^2}{4n} = \frac{\pi\sigma^2}{2n}$$

    The asymptotic relative efficiency is:

    $$\text{ARE}(\tilde{X}, \bar{X}) = \frac{\text{Var}(\bar{X})}{\text{Var}(\tilde{X})} = \frac{\sigma^2/n}{\pi\sigma^2/(2n)} = \frac{2}{\pi} \approx 0.637$$

    This means the median "wastes" about 36% of the data compared to the mean when the population is truly normal. $\square$

---

**Exercise 5.**
Suppose you observe $X_1 \sim N(\mu, 1)$ and $X_2 \sim N(\mu, 9)$ independently. Find the weighted estimator $\hat{\mu} = aX_1 + bX_2$ (with $a + b = 1$) that minimizes variance. What is its variance?

??? success "Solution to Exercise 5"
    With $b = 1 - a$, the variance is:

    $$\text{Var}(\hat{\mu}) = a^2 \cdot 1 + (1-a)^2 \cdot 9 = a^2 + 9(1-a)^2$$

    Differentiating and setting to zero:

    $$\frac{d}{da}\left[a^2 + 9(1-a)^2\right] = 2a - 18(1-a) = 2a - 18 + 18a = 20a - 18 = 0$$

    So $a = 9/10$ and $b = 1/10$. This matches the inverse-variance weights: $w_1 \propto 1/1 = 1$ and $w_2 \propto 1/9$, normalized to $(9/10, 1/10)$.

    The minimum variance is:

    $$\text{Var}(\hat{\mu}) = \left(\frac{9}{10}\right)^2 + 9\left(\frac{1}{10}\right)^2 = \frac{81}{100} + \frac{9}{100} = \frac{90}{100} = \frac{9}{10}$$

    Compare with the unweighted mean: $\text{Var}\!\left(\frac{X_1+X_2}{2}\right) = \frac{1+9}{4} = 2.5$, which is much larger. $\square$
