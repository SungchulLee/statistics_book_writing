# Central Limit Theorem Multi-Distribution Visualization

## Overview

The Central Limit Theorem (CLT) states that the sampling distribution of the sample mean converges to a normal distribution as $n$ grows, regardless of the parent distribution, provided the population has finite mean and variance. This page demonstrates the CLT visually by drawing repeated samples from three non-normal distributions and plotting the resulting sampling distributions of $\bar{X}$.

---

## Setup

We use three parent distributions with distinctly non-normal shapes:

| Distribution | Shape | Mean | Variance |
|---|---|---|---|
| Uniform(2, 8) | Flat, symmetric | 5 | 3 |
| Beta(6, 2) | Left-skewed, bounded on $[0, 1]$ | 0.75 | 0.0208 |
| Gamma(6, 1) | Right-skewed, unbounded | 6 | 6 |

For each distribution and each sample size $n \in \{2, 10, 100\}$, we draw 2000 independent samples, compute the sample mean of each, and plot the resulting histogram.

---

## The Sampling Procedure

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
N_REPS = 2000

def sample_means(dist_rvs, sample_sizes, n_reps=N_REPS):
    """For each sample size, draw n_reps samples and return their means."""
    results = {}
    for n in sample_sizes:
        means = np.array([dist_rvs(n).mean() for _ in range(n_reps)])
        results[n] = means
    return results
```

The key idea: each entry in `results[n]` is a single realization of $\bar{X}_n$. Plotting 2000 such realizations approximates the **sampling distribution** of $\bar{X}_n$.

---

## Distributions and Visualization

```python
sample_sizes = [2, 10, 100]

distributions = {
    "Uniform(2, 8)": {
        "rvs": lambda n: np.random.uniform(2, 8, n),
        "color": "tomato",
        "pop_x": np.linspace(2, 8, 200),
        "pop_pdf": lambda x: np.ones_like(x) / 6,
    },
    "Beta(6, 2)": {
        "rvs": lambda n: stats.beta.rvs(6, 2, size=n),
        "color": "seagreen",
        "pop_x": np.linspace(0, 1, 200),
        "pop_pdf": lambda x: stats.beta.pdf(x, 6, 2),
    },
    "Gamma(6, 1)": {
        "rvs": lambda n: stats.gamma.rvs(6, size=n),
        "color": "steelblue",
        "pop_x": np.linspace(0, 25, 200),
        "pop_pdf": lambda x: stats.gamma.pdf(x, 6),
    },
}

n_dists = len(distributions)
n_rows = 1 + len(sample_sizes)
fig, axes = plt.subplots(n_rows, n_dists, figsize=(6 * n_dists, 4 * n_rows))

for col, (name, d) in enumerate(distributions.items()):
    c = d["color"]

    # Row 0: population PDF
    ax = axes[0, col]
    ax.plot(d["pop_x"], d["pop_pdf"](d["pop_x"]), lw=3, color=c)
    ax.fill_between(d["pop_x"], d["pop_pdf"](d["pop_x"]), alpha=0.3, color=c)
    ax.set_title(name, fontsize=14, fontweight="bold")
    if col == 0:
        ax.set_ylabel("Population PDF", fontsize=11)

    # Rows 1–3: sampling distributions of the mean
    means_dict = sample_means(d["rvs"], sample_sizes)
    for row, n in enumerate(sample_sizes, start=1):
        ax = axes[row, col]
        ax.hist(means_dict[n], bins=30, color=c, alpha=0.5,
                edgecolor="white", density=True)
        ax.set_title(f"n = {n}", fontsize=11)
        if col == 0:
            ax.set_ylabel(f"Sampling Dist (n={n})", fontsize=10)

plt.suptitle("Central Limit Theorem: Sampling Distribution of x̄",
             fontsize=15, y=1.01)
plt.tight_layout()
plt.show()
```

The figure has a $4 \times 3$ grid: the top row shows the three parent distributions, and the remaining rows show the sampling distribution of $\bar{X}$ for $n = 2, 10, 100$.

---

## Interpretation

### What to observe

- **Top row:** The three parent distributions are visibly non-normal — flat, left-skewed, and right-skewed respectively.
- **$n = 2$:** The sampling distributions still reflect the parent shape. For two observations, averaging does little to smooth the original distribution.
- **$n = 10$:** The histograms are noticeably more bell-shaped, though some skewness may remain (especially for the Gamma).
- **$n = 100$:** All three sampling distributions are approximately normal, regardless of the parent distribution. This is the CLT in action.

### Quantitative check

The CLT predicts that as $n$ grows:

$$
\text{std}(\bar{X}) \approx \frac{\sigma}{\sqrt{n}}
$$

For the Uniform(2, 8) with $\sigma^2 = 3$:

| $n$ | Predicted $\text{std}(\bar{X})$ | Simulated $\text{std}(\bar{X})$ |
|---|---|---|
| 2 | $\sqrt{3/2} \approx 1.225$ | $\approx 1.22$ |
| 10 | $\sqrt{3/10} \approx 0.548$ | $\approx 0.55$ |
| 100 | $\sqrt{3/100} \approx 0.173$ | $\approx 0.17$ |

The simulated standard deviations match the $1/\sqrt{n}$ prediction closely, confirming that the sampling distribution tightens at the expected rate.

---

## Statistical Insight

The CLT requires two conditions:

1. The observations are **independent and identically distributed**.
2. The population has **finite mean and finite variance**.

When either condition fails, the CLT does not apply:

- **Dependent data** (e.g., time series with strong autocorrelation) may converge more slowly or to a different limit.
- **Infinite variance** (e.g., Cauchy distribution) means the sample mean does not stabilize at all.

!!! tip "Rate of Convergence"
    How quickly the sampling distribution becomes normal depends on the parent distribution's skewness. Symmetric distributions converge faster; heavily skewed distributions may need larger $n$. The Berry–Esseen theorem quantifies this: the approximation error is bounded by $O(1/\sqrt{n})$.

---

## Exercises

**Exercise 1.**
If $X_1, \ldots, X_n$ are i.i.d. Uniform(0, 1), write down the exact mean and variance of $\bar{X}$. For $n = 12$, what is the standard deviation of $\bar{X}$?

??? success "Solution to Exercise 1"
    For Uniform(0, 1): $\mu = 1/2$, $\sigma^2 = 1/12$.

    $$
    E[\bar{X}] = \mu = \frac{1}{2}, \qquad \text{Var}(\bar{X}) = \frac{\sigma^2}{n} = \frac{1}{12n}
    $$

    For $n = 12$:

    $$
    \text{Var}(\bar{X}) = \frac{1}{144}, \qquad \text{std}(\bar{X}) = \frac{1}{12} \approx 0.0833
    $$

---

**Exercise 2.**
Suppose you draw samples of size $n$ from a Gamma(2, 3) distribution ($\mu = 6$, $\sigma^2 = 18$). How large must $n$ be so that $P(|\bar{X} - 6| < 0.5) \ge 0.95$ using the CLT approximation?

??? success "Solution to Exercise 2"
    By the CLT, $\bar{X} \approx N(\mu, \sigma^2/n)$. We need:

    $$
    P\left(\left|\frac{\bar{X} - 6}{\sqrt{18/n}}\right| < \frac{0.5}{\sqrt{18/n}}\right) \ge 0.95
    $$

    This requires $\frac{0.5}{\sqrt{18/n}} \ge 1.96$, so:

    $$
    \sqrt{18/n} \le \frac{0.5}{1.96} \approx 0.2551
    $$

    $$
    \frac{18}{n} \le 0.06506 \implies n \ge \frac{18}{0.06506} \approx 276.7
    $$

    So $n \ge 277$.

---

**Exercise 3.**
Explain why the CLT does not apply to the Cauchy distribution. What happens to the sample mean of i.i.d. Cauchy random variables as $n$ increases?

??? success "Solution to Exercise 3"
    The Cauchy distribution has density $f(x) = \frac{1}{\pi(1 + x^2)}$. Its mean does not exist (the integral $\int x f(x)\, dx$ diverges), and consequently neither does its variance.

    Since the CLT requires finite mean and variance, it does not apply. In fact, for i.i.d. Cauchy $X_1, \ldots, X_n$, the sample mean $\bar{X}$ has the same Cauchy distribution as a single observation — averaging does not reduce the spread at all. This is because the Cauchy distribution is a stable distribution with index $\alpha = 1$.

---

**Exercise 4.**
Using the CLT, derive an approximate 95% confidence interval for the population mean $\mu$ based on $\bar{X}$ and $s$ (the sample standard deviation).

??? success "Solution to Exercise 4"
    By the CLT, for large $n$:

    $$
    \frac{\bar{X} - \mu}{s / \sqrt{n}} \approx N(0, 1)
    $$

    A 95% interval requires $|Z| \le 1.96$:

    $$
    P\left(-1.96 \le \frac{\bar{X} - \mu}{s/\sqrt{n}} \le 1.96\right) \approx 0.95
    $$

    Rearranging:

    $$
    \bar{X} - 1.96 \frac{s}{\sqrt{n}} \le \mu \le \bar{X} + 1.96 \frac{s}{\sqrt{n}}
    $$

    The approximate 95% confidence interval is $\bar{X} \pm 1.96 \, s / \sqrt{n}$.

---

**Exercise 5.**
Prove that the variance of $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$ equals $\sigma^2 / n$ when the $X_i$ are i.i.d. with variance $\sigma^2$.

??? success "Solution to Exercise 5"
    By linearity of variance for independent random variables:

    $$
    \text{Var}(\bar{X}_n) = \text{Var}\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2} \sum_{i=1}^n \text{Var}(X_i) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}
    $$

    The second equality uses independence (so the variance of the sum equals the sum of variances) and the fact that each $X_i$ has the same variance $\sigma^2$. $\square$

---

**Exercise 6.**
**Continuity correction** for normal approximation to a discrete distribution: when approximating $P(X \le k)$ for integer $X$ by a normal, use $\Phi((k + 0.5 - \mu)/\sigma)$. Why?

??? success "Solution to Exercise 6"
    A discrete RV places mass at integers; a continuous approximation distributes mass smoothly. Without correction, $P(X \le k) \approx \Phi((k - \mu)/\sigma)$ effectively excludes the mass at $X = k$.

    The continuity correction treats each integer as an interval of width 1 centered on the integer. For $P(X \le k)$, use the upper endpoint $k + 0.5$:

    $$
    P(X \le k) \approx \Phi\!\left(\frac{k + 0.5 - \mu}{\sigma}\right)
    $$

    **Improvement:** error rate improves from $O(1/\sqrt n)$ to $O(1/n)$.

    **Example:** Binomial(20, 0.5), $\mu = 10$, $\sigma \approx 2.236$. Exact $P(X \le 12) = 0.8684$. Without correction: $\Phi(0.894) = 0.814$ (error 0.054). With correction: $\Phi(1.118) = 0.868$ (error 0.000).

    Always apply continuity correction for discrete-to-continuous approximations, especially with modest $n$.

---

**Exercise 7.**
**CLT does not apply to maxima.** Show that the maximum $M_n = \max_i X_i$ of i.i.d. samples does not have a Gaussian limit. What is its limiting distribution?

??? success "Solution to Exercise 7"
    $F_{M_n}(x) = F(x)^n$. As $n \to \infty$, this degenerates to a step function — not a Gaussian.

    Proper rescaling gives a non-degenerate limit: for sequences $a_n > 0$ and $b_n$,

    $$
    P\!\left(\frac{M_n - b_n}{a_n} \le x\right) \to G(x)
    $$

    By the **Fisher-Tippett-Gnedenko theorem**, $G$ must be one of three extreme-value distributions:

    - **Gumbel** for light-tailed $F$ (normal, exponential).
    - **Fréchet** for heavy-tailed $F$ (Pareto, $t$).
    - **Weibull** for bounded-support $F$ (uniform).

    **Practical use:** extreme-value theory governs maximum river levels (hydrology), maximum financial losses (risk management), and minimum-life reliability problems. The Gaussian CLT covers sums; extreme-value theory covers maxima. They are distinct asymptotic frameworks for different statistics of the same data.
