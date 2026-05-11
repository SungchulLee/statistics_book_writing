# Statistics as Random Variables

## Overview

A **statistic** is any function of the observed data. Because the data arise from random sampling, the statistic itself is a **random variable** — its value changes from sample to sample. Recognizing this is the conceptual foundation of all sampling-distribution theory.

$$
\text{Population}
\;\xrightarrow{\text{draw sample}}\;
\mathbf{x} = (x_1, x_2, \dots, x_n)
\;\xrightarrow{\text{compute}}\;
T(\mathbf{x})
$$

Before the sample is drawn, $T(\mathbf{X})$ is a random variable; after the sample is observed, $T(\mathbf{x})$ is a realized number.

## From Population to Statistic

### Population, Sample, and Statistic

| Concept | Symbol | Description |
|---------|--------|-------------|
| Population | — | The entire collection of units of interest |
| Parameter | $\theta$ | A fixed but unknown numerical summary of the population (e.g., $\mu$, $\sigma^2$, $p$) |
| Sample | $\mathbf{X} = (X_1, \dots, X_n)$ | A random subset drawn from the population |
| Statistic | $T(\mathbf{X})$ | Any function of the sample (no unknown parameters) |
| Estimate | $T(\mathbf{x})$ | The numerical value of the statistic for one particular sample |

### Key Distinction

- **Parameter** $\theta$: fixed, unknown, describes the population.
- **Statistic** $T(\mathbf{X})$: random, observable, computed from sample data.
- **Estimator** $\hat{\theta}(\mathbf{X})$: a statistic used specifically to estimate $\theta$.

$$
\begin{array}{ccccc}
\text{Population}
&\rightarrow&
\text{Sample } \mathbf{X}
&\rightarrow&
\text{Statistic } T(\mathbf{X}) \\[6pt]
\text{Population}
&\rightarrow&
\text{Sample } \mathbf{X}
&\rightarrow&
\text{Estimator } \hat{\theta}(\mathbf{X})
\end{array}
$$

## Common Statistics and Their Targets

| Statistic | Formula | Target Parameter |
|-----------|---------|-----------------|
| Sample mean | $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ | Population mean $\mu$ |
| Sample variance | $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$ | Population variance $\sigma^2$ |
| Sample proportion | $\hat{p} = \frac{1}{n}\sum_{i=1}^n X_i$ (binary data) | Population proportion $p$ |
| Sample median | $\text{Med}(\mathbf{X})$ | Population median |

Each of these is a random variable whose distribution depends on the population distribution and the sample size $n$.

## Estimators and Their Properties

### Unbiased Estimator

An estimator $\hat{\theta}$ is **unbiased** if its expected value equals the true parameter:

$$
E[\hat{\theta}(\mathbf{X})] = \theta
$$

Unbiasedness means that across infinitely many repeated samples, the estimator is correct *on average* — it neither systematically overestimates nor underestimates $\theta$.

### Example: Ping Pong Balls — Assessing Unbiasedness

**Setup.** Ping pong balls numbered 0 to 32 are placed in an urn. The population median is 16. In each trial, 5 balls are drawn without replacement and the sample median is recorded. This is repeated 50 times.

**Question.** Is the sample median an unbiased estimator of the population median?

**Simulation.**

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
num_samples = 50

def main():
    balls = np.arange(33)
    print(f"Population median: {np.median(balls)}")

    data = []
    for _ in range(num_samples):
        sample = np.random.choice(balls, size=5, replace=False)
        data.append(np.median(sample))

    print(f"Mean of sample medians: {np.mean(data):.2f}")

    # Count frequencies
    data_dict = {}
    for num in data:
        data_dict[num] = data_dict.get(num, 0) + 1

    fig, ax = plt.subplots(figsize=(12, 3))
    for num, freq in data_dict.items():
        ax.plot([num] * freq, range(1, freq + 1), 'ok')
    ax.plot([16, 16], [0, 5], "--r", alpha=0.3, label="True median")
    ax.legend()
    ax.set_title('Simulation-Based Distribution of Sample Median')
    ax.set_xlabel('Sample Median')
    ax.set_ylabel('Number of Samples')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position("zero")
    plt.show()

if __name__ == "__main__":
    main()
```

**Conclusion.** The sampling distribution of the sample median is approximately symmetric and centered around the true median of 16, suggesting the sample median is an unbiased estimator of the population median.

## Maximum Likelihood Estimation (MLE)

### Introduction

Maximum Likelihood Estimation (MLE) is a method for estimating parameters by finding the values that make the observed data most probable. It is often preferred for its desirable asymptotic properties, including consistency and efficiency.

### Mathematical Formulation

Given i.i.d. observations $\mathbf{x} = (x_1, \dots, x_n)$ from a distribution $f(x \mid \theta)$, the MLE is:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \; L(\theta \mid \mathbf{x})
= \arg\max_{\theta} \prod_{i=1}^n f(x_i \mid \theta)
$$

For computational convenience, we maximize the **log-likelihood**:

$$
\ell(\theta \mid \mathbf{x}) = \sum_{i=1}^n \log f(x_i \mid \theta)
$$

### MLE for Normal Distribution Parameters

Let $x^{(1)}, \dots, x^{(m)}$ be i.i.d. draws from $N(\mu, \sigma^2)$.

**Likelihood:**

$$
L(\mu, \sigma^2) = \prod_{i=1}^m \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x^{(i)} - \mu)^2}{2\sigma^2}\right)
$$

**Log-likelihood:**

$$
\ell(\mu, \sigma^2) = -\frac{1}{2\sigma^2}\sum_{i=1}^m (x^{(i)} - \mu)^2 - \frac{m}{2}\log\sigma^2 + \text{const.}
$$

**MLE solutions:**

$$
\hat{\mu} = \frac{1}{m}\sum_{i=1}^m x^{(i)}, \qquad
\hat{\sigma}^2 = \frac{1}{m}\sum_{i=1}^m (x^{(i)} - \hat{\mu})^2
$$

!!! note
    The MLE for $\sigma^2$ divides by $m$ (not $m-1$), so it is biased. The unbiased estimator $S^2$ divides by $m-1$ (Bessel's correction).

### MLE for Bernoulli Parameter

Let $x^{(1)}, \dots, x^{(m)}$ be i.i.d. draws from $\text{Bernoulli}(p)$.

**Likelihood:**

$$
L(p) = \prod_{i=1}^m p^{x^{(i)}}(1-p)^{1-x^{(i)}}
$$

**Log-likelihood:**

$$
\ell(p) = \sum_{i=1}^m \left[ x^{(i)} \log p + (1-x^{(i)})\log(1-p) \right]
$$

**MLE solution:**

$$
\hat{p} = \frac{1}{m}\sum_{i=1}^m x^{(i)}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
p_true = 0.7
n_samples = 100

# Simulate coin flips
coins = np.random.binomial(n=1, p=p_true, size=n_samples)

# Compute log-likelihood over a grid of p values
ps = np.linspace(0.01, 0.99, 100)
log_likelihoods = np.array([
    np.sum(coins * np.log(p) + (1 - coins) * np.log(1 - p))
    for p in ps
])

# Find MLE
idx = np.argmax(log_likelihoods)
mle_p = ps[idx]

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(ps, log_likelihoods, label="Log-likelihood")
ax.axvline(mle_p, color='r', linestyle='--', label=f"MLE: p = {mle_p:.2f}")
ax.legend(loc="lower right")
ax.set_xlabel("Probability (p)")
ax.set_ylabel("Log-likelihood")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
```

### MLE for Capture–Recapture

The **capture–recapture method** estimates population size $N$ using two sampling stages:

1. Capture $M$ individuals, mark them, and release.
2. Recapture $n$ individuals; $m$ of them are marked.

The number of marked individuals in the recapture follows a **hypergeometric distribution**:

$$
P(m \mid N) = \frac{\binom{M}{m}\binom{N-M}{n-m}}{\binom{N}{n}}
$$

The MLE of $N$ is:

$$
\hat{N} = \frac{M \cdot n}{m}
$$

This follows from the proportionality argument $m/n \approx M/N$.

**Example.** If $M = 50$ fish are marked, and a second sample of $n = 40$ yields $m = 10$ marked fish:

$$
\hat{N} = \frac{50 \times 40}{10} = 200
$$

```python
import matplotlib.pyplot as plt
from scipy import special

def prob(n, c, r, t):
    """Hypergeometric probability for capture-recapture."""
    return special.comb(n - c, r - t) * special.comb(c, t) / special.comb(n, r)

def capture_recapture(c=50, r=40, t=10):
    min_n = c + r - t
    ns = range(min_n, 10 * min_n)
    probs = [prob(n, c, r, t) for n in ns]

    mle_idx = probs.index(max(probs))
    mle_n = mle_idx + min_n
    print(f"MLE of N: {mle_n}")
    return list(ns), probs, mle_n

ns, probs, mle_n = capture_recapture()

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(ns, probs, label='Likelihood')
ax.axvline(mle_n, color='r', linestyle='--', label=f'MLE: N = {mle_n}')
ax.set_xlabel('Population Size (N)')
ax.set_ylabel('Probability')
ax.set_title('Capture–Recapture: Likelihood vs Population Size')
ax.legend()
plt.show()
```

## Summary

| Concept | Meaning |
|---------|---------|
| Statistic | Any function of the sample; a random variable before data are observed |
| Estimator | A statistic used to estimate a population parameter |
| Unbiased | $E[\hat{\theta}] = \theta$ — correct on average |
| MLE | The parameter value maximizing the likelihood of the observed data |

Understanding that statistics are random variables is the gateway to all of inferential statistics: confidence intervals, hypothesis tests, and prediction intervals all rely on knowing — or approximating — the distribution of the relevant statistic.

## Exercises

**Exercise 1.**
**Capture-recapture.** Mark $M = 50$ fish, release; later catch $n = 40$ and find $m = 10$ marked. (a) Derive the hypergeometric likelihood $P(m \mid N)$. (b) Find the MLE $\hat N$.

??? success "Solution to Exercise 1"
    (a) $P(m \mid N) = \binom{M}{m} \binom{N - M}{n - m} / \binom{N}{n}$ — hypergeometric. With given values: $P(N) = \binom{50}{10}\binom{N-50}{30}/\binom{N}{40}$.

    (b) Differentiating log-likelihood and solving: $\hat N = Mn/m = 50 \cdot 40 / 10 = 200$. The MLE has the intuitive form "(marked in pop) × (caught total) / (marked caught)" — a proportional reasoning argument.

    Capture-recapture is the foundational method in wildlife population estimation; modifications (closed/open populations, multiple recaptures, mark loss) give a rich family of estimators.

---

**Exercise 2.**
**MLE for Bernoulli.** Coin tossed 100 times, 40 heads. (a) Write the likelihood $L(p)$. (b) Find $\hat p_{\text{MLE}}$.

??? success "Solution to Exercise 2"
    (a) $L(p) = \binom{100}{40} p^{40}(1-p)^{60} \propto p^{40}(1-p)^{60}$ (constant prefactor doesn't affect maximization).

    (b) Log-likelihood: $\ell(p) = 40\ln p + 60\ln(1-p)$. Derivative: $40/p - 60/(1-p) = 0 \Rightarrow p = 0.4$.

    $\hat p_{\text{MLE}} = 0.4 = $ sample proportion. In general for $X \sim \mathrm{Binomial}(n, p)$, the MLE is $\hat p = X/n$.

---

**Exercise 3.**
**MLE of normal parameters.** Given i.i.d. $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$, find both MLEs.

??? success "Solution to Exercise 3"
    Log-likelihood: $\ell(\mu, \sigma^2) = -(n/2)\ln(2\pi\sigma^2) - (1/(2\sigma^2))\sum(X_i - \mu)^2$.

    $\partial \ell/\partial \mu = (1/\sigma^2)\sum(X_i - \mu) = 0 \Rightarrow \hat\mu = \bar X$.

    $\partial \ell/\partial \sigma^2 = -n/(2\sigma^2) + (1/(2\sigma^4))\sum(X_i - \hat\mu)^2 = 0 \Rightarrow \hat\sigma^2 = (1/n)\sum(X_i - \bar X)^2$.

    **Note:** MLE divides by $n$, not $n - 1$. So $\hat\sigma^2_{\text{MLE}}$ is biased: $\mathbb{E}[\hat\sigma^2] = ((n-1)/n)\sigma^2$. For unbiased estimation use $s^2 = \sum(X_i - \bar X)^2/(n - 1)$ (Bessel's correction).

    The MLE/unbiased distinction is a recurring theme: MLE is asymptotically optimal but may be finite-sample biased.

---

**Exercise 4.**
**Sufficient statistics.** A statistic $T(X)$ is **sufficient** for $\theta$ if the conditional distribution $X \mid T$ doesn't depend on $\theta$. State the **Fisher-Neyman factorization theorem** and use it to verify $\sum X_i$ is sufficient for $\lambda$ when $X_i \sim \mathrm{Poisson}(\lambda)$.

??? success "Solution to Exercise 4"
    **Fisher-Neyman factorization theorem:** $T(X)$ is sufficient for $\theta$ iff the joint density factors as

    $$
    f(x \mid \theta) = g(T(x), \theta) \cdot h(x)
    $$

    where $g$ depends on $\theta$ only through $T(x)$, and $h$ doesn't depend on $\theta$.

    **For Poisson:** $f(x_1, \ldots, x_n \mid \lambda) = \prod_i \frac{e^{-\lambda} \lambda^{x_i}}{x_i!} = e^{-n\lambda} \lambda^{\sum x_i} / \prod_i x_i!$.

    The likelihood factors as $g(\sum x_i, \lambda) \cdot h(x) = (e^{-n\lambda} \lambda^{\sum x_i}) \cdot (1/\prod x_i!)$. Hence $T(X) = \sum X_i$ is sufficient.

    **Significance:** all the information in $X_1, \ldots, X_n$ about $\lambda$ is concentrated in $\sum X_i$. The MLE depends on the data only through $T$; the full data is unnecessary for inference. This is the foundation of efficient estimation via the Rao-Blackwell theorem.

---

**Exercise 5.**
**Fisher information.** For $X \sim N(\mu, \sigma^2)$ with $\sigma^2$ known, compute the Fisher information about $\mu$. Why does this quantity matter?

??? success "Solution to Exercise 5"
    Score function: $\partial \log f/\partial \mu = (x - \mu)/\sigma^2$.

    Fisher information: $I(\mu) = \mathbb{E}\!\left[(\partial \log f / \partial \mu)^2\right] = \mathbb{E}[(X - \mu)^2/\sigma^4] = \sigma^2/\sigma^4 = 1/\sigma^2$.

    For an i.i.d. sample of size $n$: $I_n(\mu) = n/\sigma^2$.

    **Why it matters: Cramér-Rao bound.** The variance of any unbiased estimator of $\mu$ is at least $1/I_n(\mu) = \sigma^2/n$. Since $\mathrm{Var}(\bar X) = \sigma^2/n$ exactly, $\bar X$ achieves the Cramér-Rao lower bound — **efficient**. No unbiased estimator can be better.

    Fisher information quantifies the "information content" of the sample about the parameter and provides a lower bound on estimator variance. Used in MLE asymptotics, experimental design, and information geometry.

---

**Exercise 6.**
**Asymptotic normality of the MLE.** State the general result: $\sqrt n (\hat\theta_{\text{MLE}} - \theta) \xrightarrow{d} N(0, 1/I(\theta))$ where $I(\theta)$ is the Fisher information per observation. Verify for Poisson.

??? success "Solution to Exercise 6"
    For Poisson($\lambda$): $\hat\lambda_{\text{MLE}} = \bar X$. Fisher info per observation: $I(\lambda) = 1/\lambda$ (since $\partial \log f/\partial \lambda = X/\lambda - 1$, and $\mathbb{E}[(X/\lambda - 1)^2] = \mathrm{Var}(X)/\lambda^2 = 1/\lambda$).

    Asymptotic distribution: $\sqrt n(\bar X - \lambda) \xrightarrow{d} N(0, \lambda)$, which matches $1/I(\lambda) = \lambda$.

    Direct check via CLT: $\bar X = (1/n)\sum X_i$ with $\mathrm{Var}(X_i) = \lambda$, so $\sqrt n(\bar X - \lambda) \to N(0, \lambda)$ by CLT. ✓

    **General significance:** MLEs are asymptotically normal with variance equal to the inverse Fisher information per observation. This gives:

    - **Asymptotic CIs:** $\hat\theta \pm 1.96/\sqrt{n I(\hat\theta)}$ (using $I(\hat\theta)$ as a plug-in estimate of $I(\theta)$).
    - **Asymptotic efficiency:** MLEs achieve the Cramér-Rao bound asymptotically.

    These results justify the central role of likelihood-based inference in modern statistics.
