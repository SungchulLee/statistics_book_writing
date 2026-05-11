# Parameters vs. Statistics

## Overview

The single most important distinction in statistical inference is between **parameters** — fixed but unknown quantities that describe a population — and **statistics** — computable quantities derived from a sample that serve as estimates of those parameters. Every confidence interval, hypothesis test, and regression coefficient in this book rests on this distinction. Forgetting it leads to two of the most common errors in applied work: treating a sample quantity as if it were known exactly (ignoring estimation uncertainty), and treating a parameter as if it had a probability distribution (a frequentist error fix only by Bayesian methods).

## Definitions

| Term | Scope | Notation (typical) | Known? |
|---|---|---|---|
| **Parameter** | Population | $\mu,\; \sigma^2,\; p,\; \beta,\; \rho$ | Usually unknown |
| **Statistic** | Sample | $\bar{x},\; s^2,\; \hat{p},\; \hat{\beta},\; r$ | Computable from data |

A **parameter** is a fixed numerical characteristic of the population — for example, the true average annual return of all NYSE-listed stocks. A **statistic** is the analogous quantity computed from a sample — for example, the average return of 50 randomly chosen NYSE stocks. The parameter is what you want to know; the statistic is what you have.

The standard convention: Greek letters for parameters, Roman letters (often with a hat $\hat{}$) for their estimators. Bayesian methods relax this by treating parameters as random variables with prior distributions, but for the frequentist majority of this book, parameters are fixed constants we are trying to learn.

## Why the Distinction Matters

We almost never observe the full population, so we rely on sample statistics to **estimate** population parameters. The quality of that estimation is judged by:

- **Bias**: $\mathrm{bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta$. Zero is best.
- **Variance**: $\mathrm{Var}(\hat{\theta})$. Smaller is better, controlled by $n$ and population variance.
- **Mean squared error**: $\mathrm{MSE}(\hat{\theta}) = \mathrm{Var}(\hat{\theta}) + \mathrm{bias}(\hat{\theta})^2$ — the standard scalar summary.
- **Consistency**: $\hat{\theta}_n \to \theta$ in probability as $n \to \infty$. A minimum requirement for any reasonable estimator.

These properties are studied at length in Chapter 6.

## Common Parameter–Statistic Pairs

### Mean

$$
\text{Parameter: } \mu = \frac{1}{N}\sum_{i=1}^{N} x_i \qquad\longleftrightarrow\qquad \text{Statistic: } \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

When the sample is i.i.d. from the population, $\bar{x}$ is unbiased ($\mathbb{E}[\bar{x}] = \mu$) and has variance $\mathrm{Var}(\bar{x}) = \sigma^2 / n$. The standard error $\sigma / \sqrt{n}$ is the most-used quantity in elementary inference.

### Variance

$$
\text{Parameter: } \sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2 \qquad\longleftrightarrow\qquad \text{Statistic: } s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2
$$

The sample variance divides by $n - 1$ (Bessel's correction) precisely so that $\mathbb{E}[s^2] = \sigma^2$. Dividing by $n$ would systematically underestimate $\sigma^2$ by a factor of $(n - 1)/n$, because plugging in $\bar{x}$ (the value that minimizes the sum of squared deviations) instead of the true $\mu$ "uses up" one degree of freedom.

### Proportion

$$
\text{Parameter: } p \qquad\longleftrightarrow\qquad \text{Statistic: } \hat{p} = \frac{\#\text{ successes in sample}}{n}
$$

With binary data, $\hat{p}$ is unbiased and $\mathrm{Var}(\hat{p}) = p(1-p)/n$, attaining its maximum at $p = 1/2$ (the hardest case for estimating a proportion).

### Regression coefficient

$$
\text{Parameter: } \beta_1 \qquad\longleftrightarrow\qquad \text{Statistic: } \hat{\beta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}
$$

Under the standard assumptions, $\hat{\beta}_1$ is unbiased with variance $\sigma^2 / \sum(x_i - \bar{x})^2$. The denominator shows why more spread in $x$ produces tighter slope estimates — a principle behind experimental design.

## Sampling Variability

Because a statistic is computed from a random sample, it is itself a random variable. Resampling produces a different value. The distribution of a statistic over all possible samples of size $n$ is its **sampling distribution**. Two of its features that we cite constantly:

- **Standard error** — the standard deviation of the sampling distribution. For the mean, $\mathrm{SE}(\bar{x}) = \sigma/\sqrt{n}$.
- **Sampling bias** — systematic discrepancy from the parameter, often introduced by non-random sampling.

The Central Limit Theorem says the sampling distribution of $\bar{x}$ is approximately $N(\mu, \sigma^2/n)$ for large $n$, regardless of the population's shape. This single fact powers most large-sample confidence intervals.

## Python Example

```python
"""Illustrate the sampling distribution of the mean."""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# === True population parameters ===
mu_true = 100
sigma_true = 15
population = rng.normal(mu_true, sigma_true, size=500_000)

# === Repeated sampling: compute sample means ===
n = 50
num_samples = 5_000
sample_means = np.array([
    rng.choice(population, n, replace=False).mean()
    for _ in range(num_samples)
])

# === The sampling distribution of x-bar ===
print(f"True μ:                {mu_true}")
print(f"Mean of sample means:  {sample_means.mean():.3f}")
print(f"Theoretical SE:        {sigma_true / np.sqrt(n):.3f}")
print(f"Observed SE:           {sample_means.std(ddof=1):.3f}")

fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(sample_means, bins=40, density=True, alpha=0.7, edgecolor="black")
ax.axvline(mu_true, color="red", lw=2, label=fr"$\mu = {mu_true}$")
ax.set_xlabel("Sample mean")
ax.set_ylabel("Density")
ax.set_title(rf"Sampling distribution of $\bar{{x}}$ (n={n})")
ax.legend()
fig.tight_layout()
plt.show()
```

## Key Takeaways

- Parameters describe populations; statistics describe samples.
- Statistics are random variables; their sampling distributions are the bridge from data to inference.
- An estimator's usefulness is judged by bias, variance, and consistency.
- The standard error decreases as $1/\sqrt{n}$ — quadrupling the sample size only halves the SE.

## Exercises

**Exercise 1.**
Classify each quantity as a **parameter** or a **statistic**.

**(a)** The average height of all adults in a country.
**(b)** The median income computed from a survey of 2,000 households.
**(c)** The proportion of defective items in an entire factory's production run.
**(d)** The standard deviation of test scores for 50 sampled students.
**(e)** The probability that a fair coin lands heads.
**(f)** The slope coefficient in a regression fit to data from 1,000 trades.

??? success "Solution to Exercise 1"
    (a) Parameter — describes the entire population of adults.
    (b) Statistic — computed from the sample of 2,000.
    (c) Parameter — describes the entire production run.
    (d) Statistic — computed from 50 students.
    (e) Parameter — a property of the coin (a "population" of conceptual repetitions).
    (f) Statistic — $\hat{\beta}_1$ is an estimator of the population parameter $\beta_1$.

---

**Exercise 2.**
Show that the sample variance with Bessel's correction is unbiased: $\mathbb{E}[s^2] = \sigma^2$, where $s^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar X)^2$ and $X_1, \dots, X_n$ are i.i.d. with mean $\mu$ and variance $\sigma^2$.

??? success "Solution to Exercise 2"
    Use the identity $\sum_i (X_i - \bar{X})^2 = \sum_i X_i^2 - n\bar{X}^2$. Taking expectations,

    $$
    \mathbb{E}\!\left[\sum_i X_i^2\right] = n(\sigma^2 + \mu^2), \qquad \mathbb{E}[n\bar{X}^2] = n\!\left(\frac{\sigma^2}{n} + \mu^2\right) = \sigma^2 + n\mu^2
    $$

    Subtracting,

    $$
    \mathbb{E}\!\left[\sum_i (X_i - \bar{X})^2\right] = n\sigma^2 + n\mu^2 - \sigma^2 - n\mu^2 = (n-1)\sigma^2
    $$

    Therefore $\mathbb{E}[s^2] = \frac{(n-1)\sigma^2}{n-1} = \sigma^2$. $\square$

    Dividing by $n$ instead would give $\mathbb{E}[\tilde s^2] = \frac{n-1}{n}\sigma^2$ — a systematically downward-biased estimator, especially harmful for small $n$.

---

**Exercise 3.**
Compute the standard error of $\bar X$ for an i.i.d. sample of size $n$ from a population with variance $\sigma^2 = 100$ at $n = 4, 25, 100, 400$. How does the SE change when $n$ quadruples?

??? success "Solution to Exercise 3"
    $\mathrm{SE}(\bar X) = \sigma / \sqrt{n} = 10/\sqrt{n}$:

    | $n$ | $\mathrm{SE}(\bar X)$ |
    |---|---|
    | 4 | 5.000 |
    | 25 | 2.000 |
    | 100 | 1.000 |
    | 400 | 0.500 |

    Each quadrupling of $n$ halves the SE — the standard $\sqrt{n}$ rate. Substantial precision gains require *order-of-magnitude* increases in sample size, which is the basic economics of statistical sample-size planning.

---

**Exercise 4.**
A survey reports $\hat p = 0.40$ from $n = 100$ respondents. Treat $\hat p$ as approximately $N(p, p(1-p)/n)$ and explain whether the statement "the parameter $p$ has a 95% probability of lying in $(0.30, 0.50)$" is a correct interpretation in the frequentist sense.

??? success "Solution to Exercise 4"
    Not correct in the frequentist sense. In frequentist inference, $p$ is a fixed (but unknown) constant, not a random variable, so probability statements like "$p$ is in $(a, b)$ with probability 0.95" do not apply.

    The correct interpretation of a 95% confidence interval is a statement about the *procedure*: if we repeated the sampling and CI construction many times, 95% of the resulting intervals would contain the true $p$. The particular interval $(0.30, 0.50)$ either contains $p$ or it does not — there is no probability statement about $p$ itself.

    Bayesian inference *does* let you make probability statements about parameters, but only after specifying a prior and reporting a posterior — a different mode of reasoning. The two interpretations are often confused in popular reporting.

---

**Exercise 5.**
For an i.i.d. sample of size $n$ from $\mathrm{Bernoulli}(p)$, the sample proportion is $\hat p = (1/n)\sum X_i$. Show $\mathbb{E}[\hat p] = p$ and $\mathrm{Var}(\hat p) = p(1-p)/n$. Where (in $p$) is the variance maximized, and why does this matter for sample-size planning?

??? success "Solution to Exercise 5"
    Linearity of expectation: $\mathbb{E}[\hat p] = (1/n)\sum \mathbb{E}[X_i] = p$.

    Independence: $\mathrm{Var}(\hat p) = (1/n^2)\sum \mathrm{Var}(X_i) = (1/n^2) \cdot n\, p(1-p) = p(1-p)/n$.

    The function $p \mapsto p(1-p)$ is a downward parabola maximized at $p = 1/2$ (where $p(1-p) = 0.25$). For sample-size planning we typically take the worst case $p(1-p) = 0.25$ when $p$ is unknown — leading to the familiar formula $n \approx 1/(4 \mathrm{ME}^2)$ for a target margin of error ME at 95% confidence.

---

**Exercise 6.**
The **mean squared error** decomposes as $\mathrm{MSE}(\hat \theta) = \mathrm{Var}(\hat \theta) + [\mathrm{bias}(\hat \theta)]^2$. Give an example of a biased estimator that has lower MSE than an unbiased one, and explain why this can happen.

??? success "Solution to Exercise 6"
    Consider estimating $\sigma^2$ from an i.i.d. normal sample. The MLE divides by $n$ rather than $n - 1$:

    $$
    \tilde s^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar X)^2
    $$

    This is biased: $\mathbb{E}[\tilde s^2] = \frac{n-1}{n}\sigma^2$. Its variance is also smaller than the unbiased $s^2$. Computing,

    $$
    \mathrm{MSE}(\tilde s^2) = \mathrm{Var}(\tilde s^2) + \mathrm{bias}(\tilde s^2)^2 = \frac{2(n-1)\sigma^4}{n^2} + \left(\frac{\sigma^2}{n}\right)^{\!2} = \frac{(2n-1)\sigma^4}{n^2}
    $$

    while

    $$
    \mathrm{MSE}(s^2) = \mathrm{Var}(s^2) = \frac{2\sigma^4}{n-1} = \frac{2\sigma^4 n}{n(n-1)}
    $$

    For finite $n$, $\mathrm{MSE}(\tilde s^2) < \mathrm{MSE}(s^2)$ — the MLE has lower MSE despite being biased.

    The general lesson: **bias trades against variance**. Shrinkage estimators (James–Stein, ridge regression) exploit this by accepting some bias to achieve large variance reductions. Bias is not always bad; it is one knob in a two-dimensional accuracy budget.
