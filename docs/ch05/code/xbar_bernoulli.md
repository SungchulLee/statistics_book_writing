# Sampling Distribution of X-bar (Bernoulli)

## Overview

When the population consists of binary outcomes (success/failure), the sample mean $\bar{X}$ equals the sample proportion $\hat{p}$, the fraction of successes in the sample. This page explores the sampling distribution of $\hat{p}$ drawn from Bernoulli populations with different success probabilities. By the Central Limit Theorem, $\hat{p}$ is approximately normal for large $n$, and we verify this with simulation.

## Population Model

Each observation is a Bernoulli trial with success probability $p$:

$$
X_i \sim \text{Bernoulli}(p), \qquad P(X_i = 1) = p, \quad P(X_i = 0) = 1 - p
$$

The population mean and variance are:

$$
\mu = E[X_i] = p, \qquad \sigma^2 = \text{Var}(X_i) = p(1 - p)
$$

## The Sample Proportion

For a sample of size $n$, the sample proportion is:

$$
\hat{p} = \bar{X} = \frac{1}{n}\sum_{i=1}^n X_i
$$

Its sampling distribution has:

$$
E[\hat{p}] = p, \qquad \text{Var}(\hat{p}) = \frac{p(1 - p)}{n}
$$

The standard error of $\hat{p}$ is:

$$
\text{SE}(\hat{p}) = \sqrt{\frac{p(1 - p)}{n}}
$$

## Normal Approximation

By the Central Limit Theorem, for sufficiently large $n$:

$$
\hat{p} \;\dot{\sim}\; N\!\left(p,\; \frac{p(1 - p)}{n}\right)
$$

!!! tip "Rule of Thumb"
    The normal approximation to $\hat{p}$ is generally considered reliable when both $np \ge 10$ and $n(1-p) \ge 10$. This ensures the distribution is not too skewed.

## Simulation

The following code simulates the sampling distribution of $\hat{p}$ for several values of $p$, drawing samples of size $n = 100$ from Bernoulli populations and overlaying the theoretical normal approximation.

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

np.random.seed(1)

n_population = 10_000
n_sample = 100
n_sim = 1_000
p_values = [0.4, 0.5, 0.6, 0.7]

fig, axes = plt.subplots(1, len(p_values), figsize=(14, 3.5))

for ax, p in zip(axes, p_values):
    # Create a Bernoulli population
    population = stats.binom(n=1, p=p).rvs(n_population, random_state=1)

    # Simulate sampling distribution of p-hat
    p_hat_sims = np.array([
        np.random.choice(population, size=n_sample, replace=False).mean()
        for _ in range(n_sim)
    ])

    # Histogram of simulated values
    _, bins, _ = ax.hist(p_hat_sims, density=True, bins=15,
                         alpha=0.5, edgecolor="white",
                         label=r"simulated $\hat{p}$")

    # Normal approximation overlay
    se = np.sqrt(p * (1 - p) / n_sample)
    x_grid = np.linspace(bins[0], bins[-1], 200)
    pdf = stats.norm(loc=p, scale=se).pdf(x_grid)
    ax.plot(x_grid, pdf, "--r", lw=2, alpha=0.7, label="Normal approx.")
    ax.set_title(f"p = {p}")
    ax.set_xlabel(r"$\hat{p}$")

axes[0].set_ylabel("Density")
axes[-1].legend(fontsize=8)
plt.tight_layout()
plt.show()
```

## Interpretation

!!! note "Key Observations"

    1. For all four values of $p$ ($0.4, 0.5, 0.6, 0.7$), the simulated sampling distribution of $\hat{p}$ closely matches the normal approximation when $n = 100$.
    2. The distribution is most symmetric when $p = 0.5$ (maximum variance) and becomes slightly more skewed as $p$ moves toward 0 or 1.
    3. The spread varies with $p$: the standard error $\sqrt{p(1-p)/n}$ is maximized at $p = 0.5$ and decreases as $p$ moves away from $0.5$.
    4. With $n = 100$, the rule of thumb $np \ge 10$ and $n(1-p) \ge 10$ is satisfied for all four values, so the normal approximation is expected to work well.

### Standard Errors by Population Proportion

| $p$ | $\text{SE}(\hat{p})$ |
|---|---|
| 0.4 | $\sqrt{0.24 / 100} = 0.0490$ |
| 0.5 | $\sqrt{0.25 / 100} = 0.0500$ |
| 0.6 | $\sqrt{0.24 / 100} = 0.0490$ |
| 0.7 | $\sqrt{0.21 / 100} = 0.0458$ |

## Exercises

**Exercise 1.** Derive the variance of $\hat{p}$ from first principles, starting from $\text{Var}(X_i) = p(1-p)$.

??? success "Solution to Exercise 1"
    Since $\hat{p} = \frac{1}{n}\sum_{i=1}^n X_i$ and the $X_i$ are independent:

    $$
    \text{Var}(\hat{p}) = \text{Var}\!\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2} \sum_{i=1}^n \text{Var}(X_i) = \frac{1}{n^2} \cdot n \cdot p(1-p) = \frac{p(1-p)}{n}
    $$

    $\square$

---

**Exercise 2.** A poll surveys $n = 400$ voters. The sample proportion favouring a candidate is $\hat{p} = 0.53$. Construct a 95% confidence interval for the true proportion $p$.

??? success "Solution to Exercise 2"
    Using the normal approximation, the 95% confidence interval is:

    $$
    \hat{p} \pm z_{0.025} \cdot \text{SE}(\hat{p})
    $$

    The estimated standard error is:

    $$
    \widehat{\text{SE}} = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} = \sqrt{\frac{0.53 \times 0.47}{400}} = \sqrt{\frac{0.2491}{400}} \approx 0.02495
    $$

    With $z_{0.025} = 1.96$:

    $$
    0.53 \pm 1.96 \times 0.02495 = 0.53 \pm 0.0489
    $$

    The 95% confidence interval is approximately $(0.481, 0.579)$. Since this interval contains 0.5, we cannot conclude that the candidate has majority support at the 95% level. $\square$

---

**Exercise 3.** Show that $p(1-p)$ is maximized at $p = 0.5$ and equals $1/4$. Explain why this means the "worst-case" standard error for $\hat{p}$ is $1/(2\sqrt{n})$.

??? success "Solution to Exercise 3"
    Let $g(p) = p(1-p) = p - p^2$ for $p \in [0, 1]$.

    $$
    g'(p) = 1 - 2p = 0 \implies p = \frac{1}{2}
    $$

    Since $g''(p) = -2 < 0$, this is a maximum. The maximum value is:

    $$
    g\!\left(\frac{1}{2}\right) = \frac{1}{2} \cdot \frac{1}{2} = \frac{1}{4}
    $$

    Therefore, the standard error satisfies:

    $$
    \text{SE}(\hat{p}) = \sqrt{\frac{p(1-p)}{n}} \le \sqrt{\frac{1/4}{n}} = \frac{1}{2\sqrt{n}}
    $$

    This upper bound is useful when planning sample sizes: regardless of the unknown $p$, the standard error never exceeds $1/(2\sqrt{n})$. For example, to guarantee $\text{SE} \le 0.03$, we need $n \ge 1/(4 \times 0.03^2) \approx 278$. $\square$

---

**Exercise 4.** How large must $n$ be so that the 95% margin of error for $\hat{p}$ is at most 0.02, regardless of the true $p$?

??? success "Solution to Exercise 4"
    The margin of error is $E = z_{0.025} \cdot \text{SE}(\hat{p}) = 1.96 \sqrt{p(1-p)/n}$.

    Using the worst case $p(1-p) \le 1/4$:

    $$
    E \le 1.96 \cdot \frac{1}{2\sqrt{n}}
    $$

    Setting $E \le 0.02$:

    $$
    1.96 \cdot \frac{1}{2\sqrt{n}} \le 0.02 \implies \sqrt{n} \ge \frac{1.96}{0.04} = 49 \implies n \ge 2401
    $$

    A sample size of at least $n = 2401$ guarantees a margin of error of 0.02 or less. $\square$

---

**Exercise 5.** When $p = 0.01$ and $n = 100$, compute $np$ and $n(1-p)$. Does the normal approximation satisfy the rule of thumb? Suggest an alternative approach for this setting.

??? success "Solution to Exercise 5"
    We compute:

    $$
    np = 100 \times 0.01 = 1, \qquad n(1-p) = 100 \times 0.99 = 99
    $$

    Since $np = 1 < 10$, the rule of thumb is **not** satisfied, and the normal approximation is unreliable. The distribution of $n\hat{p} = \sum X_i$ is $\text{Binomial}(100, 0.01)$, which is heavily right-skewed and concentrated near 0.

    Alternative approaches include:

    - **Exact binomial methods**: Use the exact binomial distribution for confidence intervals and tests (e.g., the Clopper--Pearson interval).
    - **Poisson approximation**: Since $n$ is large and $p$ is small, $\sum X_i \approx \text{Poisson}(\lambda = np = 1)$, which is often simpler to work with.
    - **Wilson interval**: A modified confidence interval that performs better than the Wald (normal-based) interval when $p$ is near 0 or 1.

    In general, when the event of interest is rare, one should either increase $n$ substantially or use methods that do not rely on the normal approximation. $\square$
