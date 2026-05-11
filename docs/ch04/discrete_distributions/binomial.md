# Bernoulli and Binomial Distributions

## Overview

The **Bernoulli distribution** models a single trial with two outcomes (success/failure), while the **binomial distribution** extends this to count the number of successes in $n$ independent trials. Together, they form the foundation of discrete probability modeling.

---

## Bernoulli Distribution

### Definition

A random variable $X$ follows a Bernoulli distribution if it takes value 1 (success) with probability $p$ and value 0 (failure) with probability $1 - p$:

$$
X \sim \text{Bernoulli}(p), \qquad P(X = x) = p^x (1 - p)^{1-x}, \quad x \in \{0, 1\}
$$

### Properties

$$
\begin{aligned}
E[X] &= p \\
\text{Var}(X) &= p(1 - p) \\
\text{SD}(X) &= \sqrt{p(1 - p)}
\end{aligned}
$$

### Derivation of Variance

$$
E[X^2] = 0^2 \cdot (1-p) + 1^2 \cdot p = p
$$

$$
\text{Var}(X) = E[X^2] - (E[X])^2 = p - p^2 = p(1 - p)
$$

---

## Binomial Distribution

### Definition

If $X_1, X_2, \ldots, X_n$ are independent $\text{Bernoulli}(p)$ random variables, then $Y = \sum_{i=1}^n X_i$ follows a **binomial distribution**:

$$
Y \sim \text{Binomial}(n, p), \qquad P(Y = k) = \binom{n}{k} p^k (1 - p)^{n-k}, \quad k = 0, 1, \ldots, n
$$

The binomial coefficient $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ counts the number of ways to choose $k$ successes from $n$ trials.

### Properties

$$
\begin{aligned}
E[Y] &= np \\
\text{Var}(Y) &= np(1 - p) \\
\text{SD}(Y) &= \sqrt{np(1 - p)}
\end{aligned}
$$

### Derivation of Mean and Variance

Since $Y = \sum_{i=1}^n X_i$ where $X_i \overset{\text{iid}}{\sim} \text{Bernoulli}(p)$:

$$
E[Y] = \sum_{i=1}^n E[X_i] = np
$$

By independence:

$$
\text{Var}(Y) = \sum_{i=1}^n \text{Var}(X_i) = np(1 - p)
$$

### Verifying the PMF Sums to 1

By the binomial theorem:

$$
\sum_{k=0}^n \binom{n}{k} p^k (1-p)^{n-k} = (p + (1-p))^n = 1^n = 1
$$

---

## Binomial Coefficient Identities

Several identities are useful for working with binomial distributions:

$$
\begin{aligned}
(1) &\quad \binom{n}{k} = \binom{n}{n-k} \quad \text{(symmetry)} \\[4pt]
(2) &\quad \binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k} \quad \text{(Pascal's rule)} \\[4pt]
(3) &\quad k\binom{n}{k} = n\binom{n-1}{k-1} \quad \text{(absorption identity)}
\end{aligned}
$$

The absorption identity is particularly useful for computing $E[Y]$ directly from the PMF:

$$
E[Y] = \sum_{k=0}^n k \binom{n}{k} p^k (1-p)^{n-k} = np \sum_{k=1}^n \binom{n-1}{k-1} p^{k-1} (1-p)^{n-k} = np
$$

---

## Worked Example

**Problem:** A stock has a 60% chance of rising on any given day (independent across days). Over 10 trading days, what is the probability it rises on exactly 7 days?

**Solution:**

$$
P(Y = 7) = \binom{10}{7} (0.6)^7 (0.4)^3 = 120 \cdot 0.0280 \cdot 0.064 = 0.2150
$$

Expected number of up days: $E[Y] = 10 \times 0.6 = 6$.

---

## Python: PMF, CDF, and Sampling

### PMF and CDF

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

n, p = 10, 0.6
x = np.arange(0, n + 1)

fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(x - 0.15, stats.binom(n, p).pmf(x), width=0.3, label='PMF', alpha=0.7)
ax.bar(x + 0.15, stats.binom(n, p).cdf(x), width=0.3, label='CDF', alpha=0.7)
ax.set_xlabel('k')
ax.set_xticks(x)
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

### Comparing Different Parameters

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

fig, ax = plt.subplots(figsize=(12, 3))
for n, p in [(10, 0.5), (20, 0.5), (20, 0.7)]:
    x = np.arange(0, n + 1)
    ax.plot(x, stats.binom(n, p).pmf(x), 'o-', label=f'n={n}, p={p}', markersize=4)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('k')
ax.legend()
plt.show()
```

### Sampling and Verification

```python
import numpy as np
from scipy import stats

np.random.seed(42)
n, p = 10, 0.6
samples = stats.binom(n, p).rvs(100_000)

print(f"Theoretical mean: {n*p:.4f},  Sample mean: {samples.mean():.4f}")
print(f"Theoretical var:  {n*p*(1-p):.4f},  Sample var:  {samples.var():.4f}")
```

---

## Normal Approximation to the Binomial

For large $n$, the binomial distribution is well approximated by a normal distribution:

$$
Y \sim \text{Binomial}(n, p) \approx N(np, \, np(1-p)) \quad \text{when } np \geq 5 \text{ and } n(1-p) \geq 5
$$

With continuity correction, $P(Y \leq k) \approx \mathcal{N}\left(\frac{k + 0.5 - np}{\sqrt{np(1-p)}}\right)$.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

n, p = 50, 0.4
x_disc = np.arange(0, n + 1)
x_cont = np.linspace(0, n, 200)

fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(x_disc, stats.binom(n, p).pmf(x_disc), alpha=0.5, label='Binomial PMF')
ax.plot(x_cont, stats.norm(n*p, np.sqrt(n*p*(1-p))).pdf(x_cont),
        'r-', lw=2, label='Normal approx.')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

---

## Key Takeaways

- The Bernoulli distribution models a single binary trial; the binomial counts successes over $n$ independent trials.
- The binomial PMF uses the binomial coefficient to account for all possible orderings of successes.
- Mean $np$ and variance $np(1-p)$ follow directly from the sum-of-independent-Bernoullis representation.
- For large $n$, the binomial is well approximated by the normal distribution, connecting discrete and continuous probability.

## Exercises

**Exercise 1.**
10 items, each independently defective with $p = 0.15$. (a) Distribution of $X$ = #defective? (b) $P(X = 2)$. (c) $P(X \ge 3)$. (d) Mean and variance.

??? success "Solution to Exercise 1"
    (a) $X \sim \mathrm{Binomial}(10, 0.15)$.

    (b) $P(X = 2) = \binom{10}{2}(0.15)^2(0.85)^8 = 45 \cdot 0.0225 \cdot 0.2725 \approx 0.276$.

    (c) $P(X \ge 3) = 1 - P(X \le 2)$. Compute $P(X = 0) = (0.85)^{10} \approx 0.197$, $P(X = 1) = 10 \cdot 0.15 \cdot (0.85)^9 \approx 0.347$, $P(X = 2) \approx 0.276$. So $P(X \ge 3) = 1 - 0.820 = 0.180$.

    (d) $\mathbb{E}[X] = np = 1.5$. $\mathrm{Var}(X) = np(1-p) = 10 \cdot 0.15 \cdot 0.85 = 1.275$.

---

**Exercise 2.**
**Prove $\mathbb{E}[Y] = np$ and $\mathrm{Var}(Y) = np(1-p)$** for $Y \sim \mathrm{Binomial}(n, p)$ using the indicator representation $Y = \sum_{i=1}^n X_i$ with $X_i \sim \mathrm{Bernoulli}(p)$.

??? success "Solution to Exercise 2"
    **Mean.** Linearity of expectation:

    $$
    \mathbb{E}[Y] = \mathbb{E}\!\sum_{i=1}^n X_i = \sum_{i=1}^n \mathbb{E}[X_i] = \sum_{i=1}^n p = np
    $$

    **Variance.** By independence:

    $$
    \mathrm{Var}(Y) = \sum_{i=1}^n \mathrm{Var}(X_i) = \sum_{i=1}^n p(1-p) = np(1-p)
    $$

    The indicator/sum representation is the cleanest derivation. Direct computation from the PMF works but requires the absorption identity $k\binom{n}{k} = n\binom{n-1}{k-1}$. $\square$

---

**Exercise 3.**
**Sum of two independent binomials.** Let $X \sim \mathrm{Binomial}(n_1, p)$ and $Y \sim \mathrm{Binomial}(n_2, p)$ be independent. Show $X + Y \sim \mathrm{Binomial}(n_1 + n_2, p)$.

??? success "Solution to Exercise 3"
    Each binomial is itself a sum of i.i.d. Bernoulli($p$) trials. $X$ is the sum of $n_1$ Bernoulli($p$), $Y$ of $n_2$ Bernoulli($p$). Independence between $X$ and $Y$ means their underlying Bernoullis are independent across the two groups.

    So $X + Y$ is the sum of $n_1 + n_2$ i.i.d. Bernoulli($p$) trials, which is Binomial$(n_1 + n_2, p)$. $\square$

    **MGF verification:** $M_{X+Y}(t) = M_X(t) M_Y(t) = (1 - p + pe^t)^{n_1}(1 - p + pe^t)^{n_2} = (1 - p + pe^t)^{n_1 + n_2}$, which is the MGF of Binomial$(n_1 + n_2, p)$.

    **Caveat:** the *common $p$* is essential. If $p$ differs, the sum is not binomial (it has a Poisson-binomial distribution).

---

**Exercise 4.**
**Normal approximation with continuity correction.** For Binomial(100, 0.4), approximate $P(35 \le Y \le 45)$ using the normal approximation with and without continuity correction. Compare to the exact binomial value (0.7287).

??? success "Solution to Exercise 4"
    $\mu = 40$, $\sigma = \sqrt{100 \cdot 0.4 \cdot 0.6} = \sqrt{24} \approx 4.899$.

    **Without continuity correction:**

    $$
    P(35 \le Y \le 45) \approx \Phi\!\left(\frac{45 - 40}{4.899}\right) - \Phi\!\left(\frac{35 - 40}{4.899}\right) = \Phi(1.021) - \Phi(-1.021) = 0.8463 - 0.1537 = 0.6926
    $$

    Error: $|0.6926 - 0.7287| = 0.036$.

    **With continuity correction:**

    $$
    P(35 \le Y \le 45) \approx \Phi\!\left(\frac{45.5 - 40}{4.899}\right) - \Phi\!\left(\frac{34.5 - 40}{4.899}\right) = \Phi(1.122) - \Phi(-1.122) = 0.8691 - 0.1309 = 0.7382
    $$

    Error: $|0.7382 - 0.7287| = 0.010$ — three times smaller.

    Always use continuity correction when approximating discrete distributions by continuous ones.

---

**Exercise 5.**
**Bernoulli variance is maximized at $p = 1/2$.** Prove this analytically and explain the practical implication for confidence-interval calculations.

??? success "Solution to Exercise 5"
    $\mathrm{Var}(X) = p(1 - p)$. Differentiate with respect to $p$:

    $$
    \frac{d}{dp}\, p(1-p) = 1 - 2p
    $$

    Set to zero: $p = 1/2$. Second derivative $= -2 < 0$, so it's a maximum. Maximum variance is $1/4$.

    **Practical implication:** for a binomial proportion CI, the worst-case variance is $p(1-p) \le 1/4$. The conservative SE is $\sqrt{1/(4n)} = 1/(2\sqrt n)$, so a 95% margin of error of at most $1.96/(2\sqrt n) \approx 1/\sqrt n$.

    Setting margin of error $\le 0.03$: $n \ge 1/(0.03)^2 \approx 1111$ — the origin of the "n ≈ 1000" rule for national opinion polls. The actual $p$ is usually away from 0.5, so the conservative bound is somewhat slack, but it provides a sample-size estimate that works regardless of $p$.

---

**Exercise 6.**
**Inverse problem: find $p$ from a sample.** From $n = 100$ trials we observe $Y = 35$ successes. Construct an approximate 95% CI for $p$ using two methods: (a) **Wald** ($\hat p \pm 1.96 \sqrt{\hat p(1 - \hat p)/n}$); (b) **Wilson score interval**. Compare.

??? success "Solution to Exercise 6"
    $\hat p = 35/100 = 0.35$.

    **(a) Wald interval:** $\hat p \pm 1.96 \sqrt{\hat p(1 - \hat p)/n} = 0.35 \pm 1.96 \sqrt{0.35 \cdot 0.65 / 100} = 0.35 \pm 1.96 \cdot 0.0477 = 0.35 \pm 0.094 = (0.256, 0.444)$.

    **(b) Wilson interval** (solves for $p$ from the inequality $|\hat p - p|/\sqrt{p(1-p)/n} \le 1.96$):

    $$
    p_{\text{Wilson}} = \frac{\hat p + z^2/(2n) \pm z\sqrt{\hat p(1-\hat p)/n + z^2/(4n^2)}}{1 + z^2/n}
    $$

    For $z = 1.96$, $\hat p = 0.35$, $n = 100$:

    Numerator center: $0.35 + 0.0192 = 0.3692$. Numerator half-width: $1.96 \sqrt{0.002275 + 9.6e-5} = 1.96 \sqrt{0.002371} \approx 0.0954$.

    Denominator: $1 + 0.0384 = 1.0384$.

    CI: $((0.3692 - 0.0954)/1.0384, (0.3692 + 0.0954)/1.0384) = (0.264, 0.448)$.

    **Comparison:** the Wilson interval is asymmetric around $\hat p$ (slightly shifted toward 0.5) and has guaranteed coverage even when $\hat p$ is near 0 or 1. The Wald interval can degenerate (extend below 0 or above 1) for extreme $\hat p$; Wilson never does. Modern practice prefers Wilson over Wald for binomial CIs, especially for small samples.
