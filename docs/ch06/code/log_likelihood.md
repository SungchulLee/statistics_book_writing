# Log-Likelihood Visualization

## Overview

The **log-likelihood function** is the logarithm of the likelihood, and it serves as the primary tool for maximum likelihood estimation. Working with the log-likelihood rather than the likelihood itself avoids numerical underflow when multiplying many small probabilities and converts products into sums, simplifying both computation and differentiation. This page demonstrates log-likelihood construction, visualization, and MLE extraction using a Bernoulli coin-flipping example.

## From Likelihood to Log-Likelihood

Given iid observations $x_1, \ldots, x_n$ from a distribution $f(x; \theta)$, the **likelihood** is:

$$
L(\theta) = \prod_{i=1}^n f(x_i; \theta)
$$

The **log-likelihood** is:

$$
\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log f(x_i; \theta)
$$

Since $\log$ is a strictly increasing function, the MLE is the same for both:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta L(\theta) = \arg\max_\theta \ell(\theta)
$$

!!! warning "Why Not Use the Likelihood Directly?"
    For $n = 100$ Bernoulli observations with $p = 0.7$, the likelihood involves a product of 100 numbers between 0 and 1. This product is of order $10^{-30}$ -- far below the threshold for floating-point underflow. The log-likelihood avoids this by working with sums of log-probabilities.

## Bernoulli Log-Likelihood

For $X_i \sim \text{Bernoulli}(p)$, the PMF is:

$$
f(x; p) = p^x(1-p)^{1-x}, \quad x \in \{0, 1\}
$$

The log-probability of a single observation is:

$$
\log f(x; p) = x\log p + (1-x)\log(1-p)
$$

The log-likelihood for $n$ observations is:

$$
\ell(p) = \sum_{i=1}^n [x_i \log p + (1 - x_i)\log(1-p)] = k\log p + (n-k)\log(1-p)
$$

where $k = \sum_{i=1}^n x_i$ is the number of successes.

## Deriving the MLE

Setting the score (derivative of the log-likelihood) to zero:

$$
\ell'(p) = \frac{k}{p} - \frac{n-k}{1-p} = 0
$$

$$
k(1-p) = (n-k)p \quad \Rightarrow \quad k = np \quad \Rightarrow \quad \hat{p}_{\text{MLE}} = \frac{k}{n}
$$

The second derivative confirms this is a maximum:

$$
\ell''(p) = -\frac{k}{p^2} - \frac{n-k}{(1-p)^2} < 0
$$

## Implementation and Visualization

```python
import numpy as np

def compute_log_prob(coin, p):
    """Log-probability of a single Bernoulli outcome."""
    return coin * np.log(p) + (1 - coin) * np.log(1 - p)


def compute_log_likelihood(coins, p):
    """Log-likelihood for a sequence of Bernoulli trials."""
    return sum(compute_log_prob(coin, p) for coin in coins)


# Simulate coin flips
rng = np.random.default_rng(1)
p_true = 0.7
n_samples = 100
coins = rng.binomial(n=1, p=p_true, size=n_samples)

k = coins.sum()
print(f"Observed: {k} heads out of {n_samples} flips")
print(f"MLE: p_hat = {k / n_samples:.4f}")

# Evaluate log-likelihood over a grid
ps = np.linspace(0.01, 0.99, 200)
log_liks = np.array([compute_log_likelihood(coins, p) for p in ps])

# Find MLE numerically
idx = np.argmax(log_liks)
mle_p = ps[idx]
print(f"Grid-search MLE: p_hat = {mle_p:.4f}")
print(f"Max log-likelihood: {log_liks[idx]:.4f}")
```

!!! note "Log-Likelihood Shape"
    The Bernoulli log-likelihood is a concave function of $p$ on $(0, 1)$, guaranteeing a unique global maximum. This concavity follows from $\ell''(p) < 0$ for all $p \in (0, 1)$.

## Vectorized Computation

The log-likelihood can also be computed efficiently without a loop:

```python
import numpy as np

def log_likelihood_vectorized(coins, p):
    """Vectorized log-likelihood computation."""
    k = coins.sum()
    n = len(coins)
    return k * np.log(p) + (n - k) * np.log(1 - p)

# Compare
rng = np.random.default_rng(1)
coins = rng.binomial(1, 0.7, 100)
ps = np.linspace(0.01, 0.99, 200)

ll_vec = np.array([log_likelihood_vectorized(coins, p) for p in ps])
idx = np.argmax(ll_vec)
print(f"Vectorized MLE: p = {ps[idx]:.4f}")
```

## Likelihood vs Log-Likelihood Comparison

To illustrate why the log-transform is essential, consider the raw likelihood values:

```python
import numpy as np

rng = np.random.default_rng(1)
coins = rng.binomial(1, 0.7, 100)
k = coins.sum()
n = len(coins)

p = 0.7
raw_likelihood = p**k * (1-p)**(n-k)
log_likelihood = k * np.log(p) + (n-k) * np.log(1-p)

print(f"Raw likelihood at p=0.7: {raw_likelihood:.2e}")
print(f"Log-likelihood at p=0.7: {log_likelihood:.4f}")
```

The raw likelihood is an astronomically small number, while the log-likelihood is a manageable negative number.

## Interpretation

- The **log-likelihood function** transforms a product of probabilities into a sum, providing numerical stability and analytical convenience.
- The **MLE** is the parameter value at the peak of the log-likelihood curve.
- For Bernoulli data, the MLE $\hat{p} = k/n$ (sample proportion) can be found analytically, but the log-likelihood visualization reveals the full shape of the inference landscape.
- The **curvature** of the log-likelihood at the MLE is related to the Fisher information and determines the precision of the estimate.

## Exercises

**Exercise 1.** For $n = 20$ Bernoulli trials with $k = 14$ successes, compute the log-likelihood at $p = 0.5, 0.6, 0.7, 0.8$. Which value has the highest log-likelihood? How does this compare to the MLE?

??? success "Solution to Exercise 1"
    Using $\ell(p) = 14\log p + 6\log(1-p)$:

    | $p$ | $\ell(p)$ |
    |-----|-----------|
    | 0.5 | $14\log 0.5 + 6\log 0.5 = 20\log 0.5 = -13.863$ |
    | 0.6 | $14\log 0.6 + 6\log 0.6 = 14(-0.511) + 6(-0.511) = -7.148 + (-3.065) = -10.213$ |
    | 0.7 | $14\log 0.7 + 6\log 0.7 = 14(-0.357) + 6(-1.204) = -4.993 + (-7.225) = ... $ |

    Computing precisely:

    - $\ell(0.5) = 20 \ln 0.5 = -13.863$
    - $\ell(0.6) = 14 \ln 0.6 + 6 \ln 0.4 = -7.148 - 5.498 = -12.646$
    - $\ell(0.7) = 14 \ln 0.7 + 6 \ln 0.3 = -4.993 - 7.225 = -12.218$
    - $\ell(0.8) = 14 \ln 0.8 + 6 \ln 0.2 = -3.124 - 9.657 = -12.781$

    The highest log-likelihood is at $p = 0.7$, which is the MLE: $\hat{p} = 14/20 = 0.7$. $\square$

---

**Exercise 2.** Show that the log-likelihood of the Bernoulli model is concave in $p$. Why does concavity guarantee that any critical point is a global maximum?

??? success "Solution to Exercise 2"
    The second derivative of the log-likelihood is:

    $$
    \ell''(p) = -\frac{k}{p^2} - \frac{n-k}{(1-p)^2}
    $$

    Since $k \geq 0$, $n - k \geq 0$, $p^2 > 0$, and $(1-p)^2 > 0$, both terms are non-positive. For $0 < k < n$ (at least one success and one failure), both terms are strictly negative, so $\ell''(p) < 0$ for all $p \in (0, 1)$.

    A function with strictly negative second derivative is strictly concave. For a strictly concave function on an interval, any critical point (where $\ell'(p) = 0$) must be a global maximum, because concavity means the function curves downward everywhere. There cannot be any other local maxima or saddle points. $\square$

---

**Exercise 3.** Explain why using $\log L(\theta)$ instead of $L(\theta)$ is essential for numerical computation. Give a specific example where $L(\theta)$ would underflow to zero on a computer.

??? success "Solution to Exercise 3"
    IEEE 754 double-precision floating point has a minimum positive value of approximately $5 \times 10^{-324}$. Consider $n = 1000$ iid Bernoulli$(0.5)$ observations. The likelihood at $p = 0.5$ is:

    $$
    L(0.5) = 0.5^{1000} = 2^{-1000} \approx 9.3 \times 10^{-302}
    $$

    This is representable, but for $n = 1100$ we get $2^{-1100} \approx 10^{-331}$, which is below the minimum and would underflow to exactly 0.0 in floating point.

    The log-likelihood avoids this: $\ell(0.5) = -1100 \ln 2 \approx -762.5$, which is a perfectly representable number. Even for $n = 10^6$, the log-likelihood remains numerically stable. $\square$

---

**Exercise 4.** For the Poisson distribution with $n$ observations, write the log-likelihood $\ell(\lambda)$ and derive the MLE. Verify that $\ell''(\hat{\lambda}) < 0$.

??? success "Solution to Exercise 4"
    The Poisson PMF is $f(x; \lambda) = e^{-\lambda}\lambda^x/x!$, so:

    $$
    \ell(\lambda) = \sum_{i=1}^n [-\lambda + x_i \log\lambda - \log(x_i!)] = -n\lambda + \left(\sum x_i\right)\log\lambda - \sum\log(x_i!)
    $$

    Score: $\ell'(\lambda) = -n + \frac{\sum x_i}{\lambda} = 0$, giving $\hat{\lambda} = \bar{X}$.

    Second derivative: $\ell''(\lambda) = -\frac{\sum x_i}{\lambda^2}$.

    At $\hat{\lambda} = \bar{X}$: $\ell''(\bar{X}) = -\frac{n\bar{X}}{\bar{X}^2} = -\frac{n}{\bar{X}} < 0$ (assuming $\bar{X} > 0$).

    This confirms the log-likelihood is concave and the MLE is a maximum. $\square$

---

**Exercise 5.** The observed Fisher information is $\hat{I}(\theta) = -\ell''(\hat{\theta})$. For the Bernoulli model, show that the observed information at the MLE equals $n/[\hat{p}(1-\hat{p})]$. Use this to construct an approximate 95% confidence interval for $p$ when $n = 100$ and $k = 72$.

??? success "Solution to Exercise 5"
    From Exercise 2, $\ell''(p) = -k/p^2 - (n-k)/(1-p)^2$.

    At $\hat{p} = k/n$:

    $$
    -\ell''(\hat{p}) = \frac{k}{\hat{p}^2} + \frac{n-k}{(1-\hat{p})^2} = \frac{n\hat{p}}{\hat{p}^2} + \frac{n(1-\hat{p})}{(1-\hat{p})^2} = \frac{n}{\hat{p}} + \frac{n}{1-\hat{p}} = \frac{n}{\hat{p}(1-\hat{p})}
    $$

    For $n = 100, k = 72$: $\hat{p} = 0.72$ and $\hat{I} = 100/(0.72 \times 0.28) = 495.87$.

    The approximate variance of the MLE is $1/\hat{I} = 0.72 \times 0.28/100 = 0.002016$.

    The standard error is $\sqrt{0.002016} = 0.04490$.

    The 95% confidence interval is:

    $$
    0.72 \pm 1.96 \times 0.04490 = [0.632, 0.808]
    $$

    $\square$
