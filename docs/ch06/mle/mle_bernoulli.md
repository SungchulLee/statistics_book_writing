# MLE for Bernoulli Distribution

## Overview

Let $x^{(i)}$ be $m$ i.i.d. samples from $B(p)$. Then, $p$ can be estimated by $\hat{p}$ where:

$$
\hat{p} = \frac{\sum_{i=1}^m x^{(i)}}{m}
$$

## Derivation

### Data

$$
\{x^{(i)} : i = 1, \ldots, m\}
$$

### Model

$$
x^{(i)} \sim B(p)
$$

### Likelihood Function

$$
L(p) = \prod_{i=1}^m p^{x^{(i)}} (1 - p)^{1 - x^{(i)}}
$$

### Log-Likelihood Function

$$
\ell(p) = \sum_{i=1}^m x^{(i)} \log(p) + (1 - x^{(i)}) \log(1 - p)
$$

### Cost Function

$$
J(p) = -\sum_{i=1}^m x^{(i)} \log(p) + (1 - x^{(i)}) \log(1 - p)
$$

!!! note "Connection to Cross-Entropy"
    The cost function $J(p)$ is exactly the **binary cross-entropy loss** used in logistic regression and neural networks. MLE for the Bernoulli distribution is the theoretical foundation of cross-entropy-based training.

### Maximum Likelihood Principle

$$
\text{argmax}_{p}\; L
\quad\Leftrightarrow\quad
\text{argmax}_{p}\; \ell
\quad\Leftrightarrow\quad
\text{argmin}_{p}\; J
$$

### MLE Solution

$$
\begin{array}{llcll}
\displaystyle\frac{\partial J}{\partial p} = 0
&\Rightarrow&
\displaystyle\sum_{i=1}^m \frac{x^{(i)}}{p} - \frac{1 - x^{(i)}}{1 - p} = 0
&\Rightarrow&
\displaystyle\hat{p} = \frac{\sum_{i=1}^m x^{(i)}}{m}
\end{array}
$$

## Python Implementation: Log-Likelihood and MLE

```python
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
seed = 1
np.random.seed(seed)

# Define probability of heads and sample size
p = 0.7
n_samples = 100

def load_data():
    """
    Simulate coin flips based on a binomial distribution.

    Returns:
    - numpy array: Array of coin flips (1 for heads, 0 for tails).
    """
    return np.random.binomial(n=1, p=p, size=(n_samples,))  # Shape (100,)

def compute_prob(coin, p):
    """
    Compute the probability of a single coin flip outcome.

    Parameters:
    - coin: Outcome of the coin flip (1 for heads, 0 for tails).
    - p: Probability of heads.

    Returns:
    - float: Probability of observing the outcome.
    """
    return p**coin * (1 - p)**(1 - coin)

def compute_log_prob(coin, p):
    """
    Compute the log-probability of a single coin flip outcome.

    Parameters:
    - coin: Outcome of the coin flip (1 for heads, 0 for tails).
    - p: Probability of heads.

    Returns:
    - float: Log-probability of observing the outcome.
    """
    return coin * np.log(p) + (1 - coin) * np.log(1 - p)

def compute_likelihood(coins, p):
    """
    Compute the joint probability of all coin flips for a given probability.

    Parameters:
    - coins: Array of coin flip outcomes.
    - p: Probability of heads.

    Returns:
    - float: Joint probability of observing all outcomes.
    """
    joint_prob = 1.0
    for coin in coins:
        joint_prob *= compute_prob(coin, p)
    return joint_prob

def compute_log_likelihood(coins, p):
    """
    Compute the log-likelihood of all coin flips for a given probability.

    Parameters:
    - coins: Array of coin flip outcomes.
    - p: Probability of heads.

    Returns:
    - float: Log-likelihood of observing all outcomes.
    """
    log_joint_prob = 0.0
    for coin in coins:
        log_joint_prob += compute_log_prob(coin, p)
    return log_joint_prob

# Simulate coin flips
coins = load_data()

# Define range of probabilities to test for MLE
ps = np.linspace(0.01, 0.99, 100)

# Calculate log-likelihood for each probability
log_likelihood_list = [compute_log_likelihood(coins, p) for p in ps]
log_likelihood = np.array(log_likelihood_list)

# Find the probability with the highest log-likelihood (MLE)
idx = np.argmax(log_likelihood)
mle_p = ps[idx]
log_likelihood_max = log_likelihood[idx]
print(f"MLE index: {idx}")
print(f"MLE probability (p): {mle_p:.4f}")
print(f"Max log-likelihood: {log_likelihood_max:.4f}\n")

# Plot log-likelihood function and mark the MLE
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(ps, log_likelihood, label="Log-likelihood")
ax.plot([mle_p, mle_p], [0, log_likelihood_max], '--or', label="MLE")
ax.legend(loc="lower right")

# Customize plot appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_position("zero")
ax.spines['left'].set_position("zero")
ax.set_xlabel("Probability (p)")
ax.set_ylabel("Log-likelihood")
plt.show()
```

## Exercises

**Exercise 1.**
A coin is flipped 20 times, producing 13 heads and 7 tails. Write the log-likelihood function and find the MLE $\hat{p}$ analytically.

??? success "Solution to Exercise 1"
    Let $k = 13$ heads out of $n = 20$ flips. The log-likelihood is:

    $$
    \ell(p) = k \log p + (n-k) \log(1-p) = 13\log p + 7\log(1-p)
    $$

    Setting the derivative to zero:

    $$
    \frac{d\ell}{dp} = \frac{13}{p} - \frac{7}{1-p} = 0
    $$

    $$
    13(1-p) = 7p \implies 13 - 13p = 7p \implies 13 = 20p \implies \hat{p} = \frac{13}{20} = 0.65
    $$

    The second derivative is $-13/p^2 - 7/(1-p)^2 < 0$, confirming this is a maximum.

---

**Exercise 2.**
For the Bernoulli MLE, show that $\hat{p} = \bar{x}$ (the sample proportion) is always the MLE, regardless of the sample size or the observed data.

??? success "Solution to Exercise 2"
    For $n$ independent Bernoulli trials with $k = \sum x_i$ successes, the log-likelihood is:

    $$
    \ell(p) = k \log p + (n-k)\log(1-p)
    $$

    Taking the derivative and setting to zero:

    $$
    \frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0 \implies k(1-p) = (n-k)p \implies k = np
    $$

    $$
    \hat{p} = \frac{k}{n} = \frac{\sum x_i}{n} = \bar{x}
    $$

    This holds for any values of $k$ and $n$ with $0 \leq k \leq n$. $\square$

---

**Exercise 3.**
Compute the Fisher information for a single Bernoulli observation and derive the asymptotic variance of $\hat{p}$.

??? success "Solution to Exercise 3"
    For a single Bernoulli$(p)$ observation, the log-likelihood is $\ell(p) = x\log p + (1-x)\log(1-p)$. The second derivative is:

    $$
    \frac{d^2\ell}{dp^2} = -\frac{x}{p^2} - \frac{1-x}{(1-p)^2}
    $$

    Taking the negative expectation (using $E[X] = p$):

    $$
    I(p) = -E\!\left[\frac{d^2\ell}{dp^2}\right] = \frac{p}{p^2} + \frac{1-p}{(1-p)^2} = \frac{1}{p} + \frac{1}{1-p} = \frac{1}{p(1-p)}
    $$

    The asymptotic variance of $\hat{p}$ based on $n$ observations is:

    $$
    \text{Var}(\hat{p}) \approx \frac{1}{nI(p)} = \frac{p(1-p)}{n}
    $$

    This is the familiar formula for the variance of a sample proportion.

---

**Exercise 4.**
If you observe 0 heads in 10 flips, the MLE gives $\hat{p} = 0$. Explain why this is problematic and describe one alternative approach.

??? success "Solution to Exercise 4"
    The MLE $\hat{p} = 0$ implies the coin can never land heads, which is an extreme conclusion from only 10 observations. The problem is that MLE can produce boundary estimates that are unreasonable, especially in small samples.

    One alternative is **Laplace smoothing** (or the Bayesian approach with a uniform prior): add one "pseudo-success" and one "pseudo-failure" to the data, giving $\hat{p}_{\text{Laplace}} = (0+1)/(10+2) = 1/12 \approx 0.083$. This avoids the zero estimate while remaining data-driven. More formally, this corresponds to the posterior mean under a Beta$(1,1)$ (uniform) prior, yielding $\hat{p}_{\text{Bayes}} = (k+1)/(n+2)$.
