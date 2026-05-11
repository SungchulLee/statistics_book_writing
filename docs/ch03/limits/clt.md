# Central Limit Theorem

## Overview

The **Central Limit Theorem (CLT)** is one of the most important results in all of probability and statistics. It states that the sampling distribution of the sample mean of a sufficiently large number of i.i.d. random variables is approximately normal, **regardless** of the original distribution, provided the population has finite mean and variance.

---

## Statement of the CLT

If $X_1, X_2, \ldots, X_n$ are i.i.d. random variables with mean $\mu$ and variance $\sigma^2$, then the standardized sample mean converges in distribution to a standard normal:

$$
\frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} N(0, 1) \quad \text{as } n \to \infty
$$

Equivalently, the sample mean is approximately normally distributed for large $n$:

$$
\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right)
$$

Or in terms of the sum $S_n = \sum_{i=1}^n X_i$:

$$
S_n \approx N(n\mu, \, n\sigma^2)
$$

---

## From LLN to CLT

The Law of Large Numbers tells us **where** the sample mean converges: $\bar{X} \to \mu$. The CLT tells us **how fast** and **in what shape** the fluctuations around $\mu$ behave:

$$
\frac{\sqrt{n}}{\sigma}(\bar{X} - \mu) = \frac{S_n - n\mu}{\sqrt{n\sigma^2}} \xrightarrow{d} N(0, 1)
$$

The LLN says the deviation $\bar{X} - \mu \to 0$. By rescaling by $\sqrt{n}$, the CLT reveals that these deviations have a non-trivial normal structure.

---

## Practical Guidelines

### Minimum Sample Size (n >= 30)
A commonly cited rule of thumb is that $n \geq 30$ is "large enough" for the CLT approximation to hold:

- If the population is approximately symmetric, even $n \approx 15$–20 may suffice.
- If the population is skewed or heavy-tailed, $n \geq 40$–50 may be needed.
- The number 30 is a **convention**, not a theorem.

### Normal Approximation for Proportions

When dealing with sample proportions (binomial data), the CLT requires:

$$
np \geq 5 \quad \text{and} \quad n(1-p) \geq 5
$$

Some textbooks use the stricter rule $np \geq 10$ and $n(1-p) \geq 10$.

### The 10% Condition for Independence

When sampling without replacement from a finite population of size $N$, draws are not independent. The finite population correction is:

$$
\text{Var}(\bar{X}_n) = \frac{\sigma^2}{n} \cdot \frac{N - n}{N - 1}
$$

The correction is negligible if the sampling fraction is small:

$$
\frac{n}{N} \leq 10\%
$$

If this holds, we can safely treat the sample as i.i.d.

---

## Normal Approximation

### Approximation to the Binomial

For $X \sim \text{Binomial}(n, p)$ with large $n$:

$$
X \approx N(np, \, np(1-p))
$$

With **continuity correction**:

$$
P(X \leq k) \approx P\left(Z \leq \frac{k + 0.5 - np}{\sqrt{np(1-p)}}\right)
$$

### Approximation to the Poisson

For $X \sim \text{Poisson}(\lambda)$ with large $\lambda$:

$$
X \approx N(\lambda, \lambda)
$$

---

## CLT in Action: Uniform and Exponential Distributions

The CLT works regardless of the original distribution. However, the **rate of convergence** depends on the distribution's shape:

- **Symmetric distributions** (e.g., uniform) converge to normality very quickly.
- **Skewed distributions** (e.g., exponential) converge more slowly—extreme values have a more pronounced effect, requiring larger sample sizes.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def demonstrate_clt(distribution_type, sample_size, n_simulations=10_000):
    """Demonstrate CLT convergence for a given distribution."""
    np.random.seed(0)

    if distribution_type == 'uniform':
        data = np.mean(stats.uniform().rvs((sample_size, n_simulations)), axis=0)
        label = 'Uniform(0,1)'
    elif distribution_type == 'exponential':
        data = np.mean(stats.expon().rvs((sample_size, n_simulations)), axis=0)
        label = 'Exponential(1)'

    mu, sigma = data.mean(), data.std()

    fig, ax = plt.subplots(figsize=(12, 3))
    _, bins, _ = ax.hist(data, bins=100, density=True, alpha=0.3, color='blue',
                         label=f'Sample Means (n={sample_size})')
    ax.plot(bins, stats.norm(mu, sigma).pdf(bins), '--r', lw=2, label='Normal PDF')
    ax.set_title(f'CLT: Sample Means from {label}')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.show()

# Demonstrate with both distributions
demonstrate_clt('uniform', sample_size=5)
demonstrate_clt('exponential', sample_size=5)
```

---

## Applications

The CLT underpins many core statistical procedures:

- **Hypothesis testing:** $z$-tests and $t$-tests assume the sampling distribution of the test statistic is approximately normal.
- **Confidence intervals:** Constructed using normal quantiles, justified by the CLT.
- **Quality control:** Assessing whether sample means of product measurements meet standards.

---

## Putting It All Together

Before applying a normal approximation, check these conditions:

| Condition | Rule of Thumb |
|:---|:---|
| Sample size | $n \geq 30$ (unless population is nearly normal) |
| Proportions | $np \geq 5$ and $n(1-p) \geq 5$ |
| Finite population sampling | $n/N \leq 10\%$ |

These are not strict theorems but widely adopted **practical guidelines** that bridge the ideal mathematical world and real data analysis.

---

## Key Takeaways

- The CLT guarantees that sample means are approximately normal for large $n$, regardless of the population distribution.
- The rate of convergence depends on the skewness of the original distribution.
- Practical conditions ($n \geq 30$, success/failure counts, 10% rule) ensure the approximation is reliable.
- The CLT is the theoretical backbone of confidence intervals, hypothesis tests, and much of applied statistics.

## Exercises

**Exercise 1.**
A machine fills bottles with $\mu = 500$ ml, $\sigma = 10$ ml (non-normal). (a) Distribution of $\bar X_{36}$ by CLT? (b) $P(\bar X_{36} > 503)$? (c) $n$ for $P(|\bar X_n - 500| < 2) \ge 0.95$?

??? success "Solution to Exercise 1"
    (a) $\bar X_{36} \approx N(500, 100/36) = N(500, 2.778)$. SE $= 10/6 \approx 1.67$.

    (b) $Z = (503 - 500)/(10/6) = 1.8$. $P(Z > 1.8) \approx 1 - 0.9641 = 0.0359$. About 3.6%.

    (c) Need $2/(\sigma/\sqrt n) \ge z_{0.025} = 1.96 \Rightarrow 2\sqrt n / 10 \ge 1.96 \Rightarrow n \ge 96.04$. So $n \ge 97$.

---

**Exercise 2.**
**Prove the CLT via the MGF approach** for i.i.d. $X_i$ with mean 0, variance 1, and finite MGF $M(t)$ in a neighborhood of zero. Show that the MGF of $\sqrt n \bar X_n$ converges to $e^{t^2/2}$.

??? success "Solution to Exercise 2"
    Standardized sum: $Z_n = \sqrt n \bar X_n = (X_1 + \cdots + X_n)/\sqrt n$.

    Its MGF: $M_{Z_n}(t) = \mathbb{E}[e^{t Z_n}] = \prod_i \mathbb{E}[e^{t X_i / \sqrt n}] = [M(t/\sqrt n)]^n$.

    Expand $M(t/\sqrt n)$ around 0: $M(u) = 1 + \mu u + (\sigma^2 + \mu^2) u^2/2 + O(u^3) = 1 + u^2/2 + O(u^3)$ (using $\mu = 0$, $\sigma^2 = 1$).

    So $M(t/\sqrt n) = 1 + t^2/(2n) + O(n^{-3/2})$.

    Take $n$-th power: $M_{Z_n}(t) = (1 + t^2/(2n) + O(n^{-3/2}))^n \to e^{t^2/2}$ as $n \to \infty$ (using $(1 + a/n + o(1/n))^n \to e^a$).

    The limiting MGF $e^{t^2/2}$ is that of $N(0, 1)$. By the MGF convergence theorem, $Z_n \xrightarrow{d} N(0, 1)$. $\square$

    Note: this requires finite MGF in a neighborhood, which excludes some heavy-tailed distributions. The characteristic-function proof (using $\phi(t) = \mathbb{E}[e^{itX}]$) generalizes to all finite-variance distributions.

---

**Exercise 3.**
**Demonstrate slow CLT convergence** for a heavily skewed distribution: $X_i$ is exponential with mean 1. What is the skewness of $\bar X_n$ for $n = 30$? Compare with the symmetry of the normal limit.

??? success "Solution to Exercise 3"
    Exponential(1) has skewness 2 (right-skewed). For an i.i.d. sum, skewness scales as $\gamma_n = \gamma_1 / \sqrt n$ for the *standardized sum*:

    $$
    \mathrm{Skew}(\bar X_n) = \frac{\mathrm{Skew}(X)}{\sqrt n}
    $$

    For $n = 30$: $\mathrm{Skew}(\bar X_{30}) = 2/\sqrt{30} \approx 0.365$.

    This is still substantially non-zero — the normal limit has skewness 0. Sample means from an exponential distribution at $n = 30$ are visibly right-skewed. Practical implication: the conventional $n \ge 30$ rule of thumb is *not* sufficient for heavily skewed distributions; in practice one needs $n \ge 100$ or more, or alternative techniques (bootstrap, exact methods).

    The convergence rate is governed by the **Berry–Esseen theorem**: $\sup_x |F_{\bar X_n}(x) - \Phi(x)| \le C \cdot \mathbb{E}|X|^3 / (\sigma^3 \sqrt n)$. The third moment in the numerator captures skewness/asymmetry: heavily skewed distributions have larger third absolute moment and slower CLT convergence.

---

**Exercise 4.**
**Multivariate CLT.** Let $\mathbf X_i \in \mathbb{R}^d$ be i.i.d. with mean $\boldsymbol\mu$ and covariance $\boldsymbol\Sigma$. State the multivariate CLT and explain why it justifies multivariate normal-based confidence ellipsoids.

??? success "Solution to Exercise 4"
    **Multivariate CLT:**

    $$
    \sqrt n (\bar{\mathbf X}_n - \boldsymbol\mu) \xrightarrow{d} N_d(\mathbf 0, \boldsymbol\Sigma)
    $$

    where convergence is in distribution in $\mathbb{R}^d$ (joint distributions of all components).

    Proof sketch: use the **Cramér–Wold device** — multivariate convergence holds iff every linear projection converges (univariately). For any $\mathbf a \in \mathbb{R}^d$, $\sqrt n \, \mathbf a^T(\bar{\mathbf X}_n - \boldsymbol\mu) \xrightarrow{d} N(0, \mathbf a^T \boldsymbol\Sigma \mathbf a)$ by univariate CLT applied to the scalar projection. Equivalent multivariate convergence to $N_d(\mathbf 0, \boldsymbol\Sigma)$ follows.

    **Justification of confidence ellipsoids:** if $\bar{\mathbf X}_n \approx N_d(\boldsymbol\mu, \boldsymbol\Sigma/n)$, then $n(\bar{\mathbf X}_n - \boldsymbol\mu)^T \boldsymbol\Sigma^{-1}(\bar{\mathbf X}_n - \boldsymbol\mu) \approx \chi^2_d$. The set $\{\boldsymbol\mu : n(\bar{\mathbf X}_n - \boldsymbol\mu)^T \boldsymbol\Sigma^{-1}(\bar{\mathbf X}_n - \boldsymbol\mu) \le \chi^2_{d, 0.95}\}$ is an ellipsoid covering the true mean with 95% asymptotic probability. Hotelling's $T^2$ test and multivariate confidence regions both rest on this.

---

**Exercise 5.**
**The Lindeberg CLT** allows non-identically-distributed but independent random variables. State the **Lindeberg condition** and explain when it holds.

??? success "Solution to Exercise 5"
    Let $X_1, X_2, \ldots$ be independent (not necessarily identically distributed) with $\mathbb{E}[X_i] = 0$, $\mathrm{Var}(X_i) = \sigma_i^2$, $s_n^2 = \sum_{i=1}^n \sigma_i^2$. The **Lindeberg condition** is

    $$
    \forall\, \varepsilon > 0: \quad \frac{1}{s_n^2}\sum_{i=1}^n \mathbb{E}\!\left[X_i^2 \mathbf 1(|X_i| > \varepsilon s_n)\right] \to 0
    $$

    Intuition: the total variance contributed by the "tail" of any individual $X_i$ (i.e., the part where $|X_i|$ exceeds $\varepsilon s_n$) becomes negligible. No single $X_i$ dominates the variance.

    **When it holds:** (1) the i.i.d. case with finite variance trivially; (2) uniformly bounded $X_i$ with $s_n \to \infty$; (3) any sequence where the maximal variance $\max_i \sigma_i^2 / s_n^2 \to 0$.

    **When it fails:** if one term contributes a non-vanishing share of the total variance (e.g., $X_n = n \cdot Y$ for a Gaussian $Y$, where $X_n$ dominates) — the limit is not Gaussian but stable.

    Lindeberg CLT subsumes the i.i.d. CLT and applies to sums of independent but heterogeneous contributions — central to regression with non-identically-distributed errors.

---

**Exercise 6.**
The CLT requires **finite variance**. Discuss why and what happens when variance is infinite (heavy-tailed distributions). Mention the **stable distributions** and **alpha-stable CLT**.

??? success "Solution to Exercise 6"
    **Why finite variance is essential:** the standardization $(S_n - n\mu)/\sqrt{n\sigma^2}$ implicitly assumes $\sigma^2 < \infty$. With infinite variance, this standardization is undefined. The Lindeberg-Feller framework breaks down because the "negligibility" of individual contributions cannot be established.

    **Heavy-tailed behavior:** distributions in the domain of attraction of a **stable distribution** $S_\alpha(\sigma, \beta)$ with $\alpha < 2$ have power-law tails $P(|X| > x) \sim x^{-\alpha}$. The variance is infinite when $\alpha \le 2$ and mean is infinite when $\alpha \le 1$.

    **Alpha-stable CLT:** for $X_i$ i.i.d. with tail index $0 < \alpha \le 2$, the normalized sums

    $$
    \frac{S_n - n a_n}{n^{1/\alpha}} \xrightarrow{d} S_\alpha(\sigma, \beta)
    $$

    converge to an $\alpha$-stable distribution. The normalization is $n^{1/\alpha}$, not $\sqrt n$: for $\alpha = 2$ this recovers the Gaussian CLT; for $\alpha = 1$ we get a Cauchy limit; for $\alpha < 1$ even the mean fails to converge.

    **Practical implications:** financial returns often have $\alpha \approx 1.5$–$1.8$ (heavy tails but finite variance, so Gaussian CLT applies — slowly). Network packet sizes, file sizes, and queueing delays often have $\alpha < 2$ (genuinely heavy-tailed), making Gaussian-based confidence intervals invalid. The right tools are heavy-tail aware: bootstrap with care, quantile estimators, alpha-stable models.
