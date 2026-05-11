# Geometric and Poisson Maximum Likelihood

## Overview

Maximum likelihood estimation (MLE) provides a principled way to fit parametric models to observed data. This page derives and demonstrates the MLE for two fundamental discrete distributions -- the Geometric and the Poisson -- and compares the parametric MLE fit against a nonparametric empirical PMF on held-out test data. The analysis highlights why parametric models generalize better when the model is correctly specified.

## Geometric Distribution MLE

### The Model

The Geometric distribution models the number of consecutive successes before the first failure. With parameter $p$ (probability of success on each trial):

$$
P(X = k) = (1 - p)\, p^k, \quad k = 0, 1, 2, \ldots
$$

The mean is $E[X] = p / (1 - p)$.

### Deriving the MLE

Given observations $x_1, \ldots, x_n$, the log-likelihood is:

$$
\ell(p) = \sum_{i=1}^n \left[ x_i \log p + \log(1 - p) \right] = n_s \log p + n \log(1 - p)
$$

where $n_s = \sum_{i=1}^n x_i$ is the total number of successes. Taking the derivative and setting it to zero:

$$
\frac{d\ell}{dp} = \frac{n_s}{p} - \frac{n}{1 - p} = 0
$$

Solving gives:

$$
\hat{p}_{\text{MLE}} = \frac{n_s}{n_s + n} = \frac{\bar{x}}{1 + \bar{x}}
$$

### Demonstration

```python
import numpy as np

np.random.seed(42)

def geometric_mle_demo(n_train=1000, n_test=1000, p_true=0.12):
    """Geometric MLE: parametric vs nonparametric comparison."""
    # Generate data: number of successes before first failure
    train = np.random.geometric(1 - p_true, n_train) - 1  # 0-indexed
    test = np.random.geometric(1 - p_true, n_test) - 1

    # MLE
    p_hat = train.mean() / (1 + train.mean())
    k_max = max(train.max(), test.max()) + 1
    k_vals = np.arange(k_max)

    # Parametric PMF from MLE
    pmf_param = (1 - p_hat) * p_hat ** k_vals

    # Empirical PMF (nonparametric)
    pmf_train = np.bincount(train, minlength=k_max) / n_train
    pmf_test = np.bincount(test, minlength=k_max) / n_test

    # Test-set RMSE
    err_param = np.sqrt(np.mean((pmf_param - pmf_test)**2))
    err_nonparam = np.sqrt(np.mean((pmf_train - pmf_test)**2))

    print(f"True p = {p_true:.3f}")
    print(f"MLE p_hat = {p_hat:.4f}")
    print(f"Test RMSE -- parametric: {err_param:.5f}")
    print(f"Test RMSE -- nonparametric: {err_nonparam:.5f}")

geometric_mle_demo()
```

### Log-Likelihood Surface

The log-likelihood is a concave function of $p$, confirming a unique global maximum:

```python
import matplotlib.pyplot as plt

train = np.random.geometric(1 - 0.12, 1000) - 1
n_success = train.sum()
n_fail = len(train)

theta_grid = np.linspace(0.01, 0.99, 200)
ll = n_success * np.log(theta_grid) + n_fail * np.log(1 - theta_grid)

p_hat = train.mean() / (1 + train.mean())

plt.plot(theta_grid, ll, "k-", lw=2)
plt.axvline(p_hat, color="red", linestyle="--", label=f"MLE = {p_hat:.3f}")
plt.axvline(0.12, color="blue", linestyle=":", label="True = 0.120")
plt.xlabel("theta")
plt.ylabel("Log-likelihood")
plt.title("Geometric: Log-Likelihood Surface")
plt.legend()
plt.show()
```

## Poisson Distribution MLE

### The Model

The Poisson distribution models the number of events in a fixed interval:

$$
P(X = k) = \frac{e^{-\lambda}\, \lambda^k}{k!}, \quad k = 0, 1, 2, \ldots
$$

The mean and variance are both equal to $\lambda$.

### Deriving the MLE

Given observations $x_1, \ldots, x_n$, the log-likelihood (up to constants not depending on $\lambda$) is:

$$
\ell(\lambda) = \left(\sum_{i=1}^n x_i\right) \log \lambda - n\lambda
$$

Taking the derivative:

$$
\frac{d\ell}{d\lambda} = \frac{\sum x_i}{\lambda} - n = 0
$$

Solving gives the well-known result:

$$
\hat{\lambda}_{\text{MLE}} = \bar{x}
$$

The MLE for the Poisson rate parameter is simply the sample mean.

### Demonstration

```python
from scipy import stats

def poisson_mle_demo(n_train=200, n_test=200, lam_true=4.5):
    """Poisson MLE: parametric vs nonparametric comparison."""
    np.random.seed(42)
    train = np.random.poisson(lam_true, n_train)
    test = np.random.poisson(lam_true, n_test)

    lam_hat = train.mean()
    k_max = max(train.max(), test.max()) + 1
    k_vals = np.arange(k_max)

    # Parametric PMF from MLE
    pmf_param = stats.poisson.pmf(k_vals, lam_hat)

    # Empirical PMF (nonparametric)
    pmf_train = np.bincount(train, minlength=k_max) / n_train
    pmf_test = np.bincount(test, minlength=k_max) / n_test

    # Test-set RMSE
    err_param = np.sqrt(np.mean((pmf_param - pmf_test)**2))
    err_nonparam = np.sqrt(np.mean((pmf_train - pmf_test)**2))

    print(f"True lambda = {lam_true:.2f}")
    print(f"MLE lambda_hat = {lam_hat:.4f}")
    print(f"Test RMSE -- parametric: {err_param:.5f}")
    print(f"Test RMSE -- nonparametric: {err_nonparam:.5f}")

poisson_mle_demo()
```

### Comparing Parametric and Nonparametric Fits

```python
poi = poisson_mle_demo.__code__  # (see full script for plotting code)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Train fit
axes[0].bar(k_vals, pmf_param, color="white", edgecolor="black",
            lw=1.5, label="Parametric (MLE)")
axes[0].plot(k_vals, pmf_train, "ko", ms=5, label="Empirical (train)")
axes[0].set_title("Poisson: Train Fit")
axes[0].legend()

# Test fit
axes[1].bar(k_vals, pmf_param, color="white", edgecolor="black",
            lw=1.5, label="Parametric (MLE)")
axes[1].plot(k_vals, pmf_test, "ro", ms=5, label="Empirical (test)")
axes[1].set_title("Poisson: Test Fit")
axes[1].legend()

plt.tight_layout()
plt.show()
```

## Interpretation

- **Parametric models generalize better.** When the assumed model family is correct (data truly come from a Geometric or Poisson distribution), the MLE-based parametric PMF typically achieves lower test-set RMSE than the nonparametric empirical PMF. The parametric model "borrows strength" across all values of $k$ through the functional form.
- **Nonparametric models are more flexible but noisier.** The empirical PMF assigns zero probability to values not observed in training. With smaller training samples ($n = 200$ for Poisson), this discreteness effect is pronounced in the tails.
- **The log-likelihood surface is concave** for both distributions, guaranteeing that gradient-based optimization converges to the unique global MLE.
- **Model misspecification risk.** If the true data-generating process is not Geometric or Poisson, the parametric model can systematically misfit the data, and the nonparametric approach may then outperform it.

!!! warning "Parametric Assumptions Matter"
    The superior test-set performance of the parametric fit depends on the model being correctly specified. Always check goodness-of-fit (e.g., chi-squared test, QQ-plot) before trusting a parametric model over the empirical distribution.

## Exercises

**Exercise 1.** Derive the MLE for the Geometric distribution directly from the method-of-moments perspective. Show that the MLE and method-of-moments estimator coincide for this distribution.

??? success "Solution to Exercise 1"
    The mean of $\text{Geometric}(p)$ (with $P(X = k) = (1-p)p^k$) is:

    $$
    E[X] = \frac{p}{1 - p}
    $$

    Setting $E[X] = \bar{x}$ and solving for $p$:

    $$
    \bar{x} = \frac{p}{1 - p} \implies \bar{x}(1 - p) = p \implies \bar{x} = p(1 + \bar{x})
    $$

    $$
    \hat{p}_{\text{MoM}} = \frac{\bar{x}}{1 + \bar{x}}
    $$

    This is identical to $\hat{p}_{\text{MLE}} = \bar{x}/(1 + \bar{x})$ derived by maximizing the log-likelihood. For single-parameter exponential family distributions, the MLE is always a function of the sufficient statistic, and when the moment equation involves only that statistic, MLE and MoM coincide. $\square$

---

**Exercise 2.** For the Poisson MLE, show that $\hat{\lambda} = \bar{x}$ is not only a critical point but a global maximum by verifying the second derivative condition.

??? success "Solution to Exercise 2"
    The log-likelihood is:

    $$
    \ell(\lambda) = \left(\sum x_i\right) \log \lambda - n\lambda + C
    $$

    where $C$ does not depend on $\lambda$. The first derivative is:

    $$
    \frac{d\ell}{d\lambda} = \frac{\sum x_i}{\lambda} - n
    $$

    Setting this to zero gives $\hat{\lambda} = \bar{x}$.

    The second derivative is:

    $$
    \frac{d^2\ell}{d\lambda^2} = -\frac{\sum x_i}{\lambda^2}
    $$

    Since $\sum x_i \geq 0$ and $\lambda > 0$, we have $d^2\ell/d\lambda^2 \leq 0$ for all $\lambda > 0$. When $\sum x_i > 0$, the inequality is strict, confirming that $\hat{\lambda} = \bar{x}$ is a global maximum. (If all observations are zero, then $\hat{\lambda} = 0$, which is a boundary maximum.) $\square$

---

**Exercise 3.** You observe $n = 500$ i.i.d. draws from a Poisson distribution with sample mean $\bar{x} = 3.2$. Construct an approximate 95% confidence interval for $\lambda$ using the Fisher information.

??? success "Solution to Exercise 3"
    The Fisher information for a single Poisson observation is:

    $$
    I(\lambda) = \frac{1}{\lambda}
    $$

    For $n$ observations, the total Fisher information is $nI(\lambda) = n/\lambda$. By the asymptotic normality of the MLE:

    $$
    \hat{\lambda} \dot{\sim} N\!\left(\lambda, \frac{1}{nI(\lambda)}\right) = N\!\left(\lambda, \frac{\lambda}{n}\right)
    $$

    Plugging in $\hat{\lambda} = 3.2$ and $n = 500$:

    $$
    \text{SE} = \sqrt{\frac{\hat{\lambda}}{n}} = \sqrt{\frac{3.2}{500}} = \sqrt{0.0064} = 0.08
    $$

    The 95% confidence interval is:

    $$
    \hat{\lambda} \pm 1.96 \cdot \text{SE} = 3.2 \pm 1.96(0.08) = 3.2 \pm 0.157 = [3.043, 3.357]
    $$

    $\square$

---

**Exercise 4.** Explain why the nonparametric (empirical PMF) estimator tends to have higher test-set RMSE than the parametric MLE when the model is correctly specified, despite the nonparametric estimator being unbiased. What role does the bias-variance trade-off play?

??? success "Solution to Exercise 4"
    The empirical PMF estimates each probability $P(X = k)$ independently using the proportion $\hat{p}_k = n_k / n$. Each $\hat{p}_k$ is unbiased with variance:

    $$
    \text{Var}(\hat{p}_k) = \frac{P(X = k)(1 - P(X = k))}{n}
    $$

    The parametric MLE estimates a single parameter ($p$ or $\lambda$), from which the entire PMF is derived. The parametric estimator of $P(X = k)$ is biased in finite samples (due to the nonlinear plug-in), but its variance is much lower because it estimates only one degree of freedom rather than one for each value of $k$.

    The total MSE decomposes as:

    $$
    \text{MSE} = \text{Bias}^2 + \text{Variance}
    $$

    For the nonparametric estimator: bias is zero, but variance is high (especially in the tails where $n_k$ is small). For the parametric MLE: bias is negligible under correct specification, and variance is low because the functional form constrains the PMF shape. The parametric model achieves a favorable bias-variance trade-off, resulting in lower RMSE on test data.

    When the model is misspecified, the parametric estimator carries non-vanishing bias, and the nonparametric estimator can win. $\square$

---

**Exercise 5.** Suppose you observe the following data that you believe came from a Geometric distribution: $x = (0, 2, 1, 0, 3, 1, 0, 0, 1, 2)$. Compute the MLE $\hat{p}$. Then compute the log-likelihood at $\hat{p}$ and at $p = 0.3$ and $p = 0.7$, and verify that the MLE yields the highest log-likelihood.

??? success "Solution to Exercise 5"
    The data are $x = (0, 2, 1, 0, 3, 1, 0, 0, 1, 2)$ with $n = 10$ and $\sum x_i = 10$.

    The MLE is:

    $$
    \hat{p} = \frac{\bar{x}}{1 + \bar{x}} = \frac{1}{1 + 1} = 0.5
    $$

    The log-likelihood is $\ell(p) = n_s \log p + n \log(1 - p)$ where $n_s = 10$ and $n = 10$:

    **At** $\hat{p} = 0.5$:

    $$
    \ell(0.5) = 10 \log(0.5) + 10 \log(0.5) = 20 \log(0.5) = -20 \times 0.6931 = -13.863
    $$

    **At** $p = 0.3$:

    $$
    \ell(0.3) = 10\log(0.3) + 10\log(0.7) = 10(-1.2040) + 10(-0.3567) = -15.607
    $$

    **At** $p = 0.7$:

    $$
    \ell(0.7) = 10\log(0.7) + 10\log(0.3) = 10(-0.3567) + 10(-1.2040) = -15.607
    $$

    Indeed $\ell(0.5) > \ell(0.3) = \ell(0.7)$, confirming the MLE maximizes the log-likelihood. The symmetry $\ell(0.3) = \ell(0.7)$ follows from the log-likelihood being symmetric around $\hat{p} = 0.5$ when $n_s = n$. $\square$
