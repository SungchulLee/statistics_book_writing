# Law of Large Numbers

## Overview

The **Law of Large Numbers (LLN)** is a fundamental theorem in probability theory describing how the average of a large number of independent and identically distributed (i.i.d.) random variables converges to the expected value as the sample size increases. It provides the theoretical basis for why sample averages are reliable estimators of population means.

---

## The Weak Law of Large Numbers (WLLN)

For a sequence of i.i.d. random variables $X_1, X_2, \ldots, X_n$ with finite mean $\mu$, the sample mean converges to the population mean **in probability**:

$$
\bar{X} = \frac{S_n}{n} \xrightarrow{P} \mu \quad \text{as } n \to \infty
$$

where $S_n = X_1 + X_2 + \cdots + X_n$.

More precisely, for any fixed $\varepsilon > 0$:

$$
P\left(\left|\frac{S_n}{n} - \mu\right| > \varepsilon\right) \to 0 \quad \text{as } n \to \infty
$$

**Interpretation:** As we collect more data, the probability that the sample mean deviates from $\mu$ by more than any fixed amount $\varepsilon$ goes to zero. For example, flipping a fair coin many times will result in the proportion of heads converging to 0.5.

---

## The Strong Law of Large Numbers (SLLN)

The SLLN provides a stronger form of convergence known as **almost sure convergence**:

$$
\bar{X} = \frac{S_n}{n} \xrightarrow{\text{a.s.}} \mu \quad \text{as } n \to \infty
$$

Equivalently, for any fixed $\varepsilon > 0$:

$$
P\left(\omega \in \Omega : \frac{S_n(\omega)}{n} \to \mu\right) = 1
$$

**Difference from WLLN:** The WLLN says the probability of a large deviation goes to zero for each $n$, but individual sample paths might still occasionally deviate. The SLLN guarantees that **every sample path** (except a set of probability zero) converges to $\mu$.

---

## Python Demonstration

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate die rolls and track the running average
n_rolls = 10_000
rolls = np.random.randint(1, 7, size=n_rolls)
running_avg = np.cumsum(rolls) / np.arange(1, n_rolls + 1)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(running_avg, alpha=0.8, label='Running Average')
ax.axhline(y=3.5, color='r', linestyle='--', label='E[X] = 3.5')
ax.set_xlabel('Number of Rolls')
ax.set_ylabel('Sample Mean')
ax.set_title('Law of Large Numbers: Fair Die')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
plt.show()
```

---

## Key Takeaways

- The **WLLN** says the sample mean converges to $\mu$ in probability.
- The **SLLN** says the sample mean converges to $\mu$ almost surely (a stronger guarantee).
- Both require i.i.d. samples with finite mean.
- The LLN is the theoretical justification for using sample averages to estimate population parameters.

## Exercises

**Exercise 1.**
$X_i \sim \mathrm{Uniform}(0,1)$ i.i.d. (a) $\mathbb{E}[X]$, $\mathrm{Var}(X)$? (b) State WLLN for $\bar X_n$. (c) Chebyshev bound on $P(|\bar X_{100} - 1/2| \ge 0.05)$. (d) Bound for $n = 10\,000$.

??? success "Solution to Exercise 1"
    (a) $\mathbb{E}[X] = 1/2$, $\mathrm{Var}(X) = 1/12$.

    (b) For all $\varepsilon > 0$, $P(|\bar X_n - 1/2| \ge \varepsilon) \to 0$ as $n \to \infty$.

    (c) $\mathrm{Var}(\bar X_n) = 1/(12n)$. Chebyshev: $P(|\bar X_{100} - 1/2| \ge 0.05) \le (1/1200)/(0.05)^2 = (1/1200)/0.0025 = 1/3 \approx 0.333$.

    (d) For $n = 10\,000$: bound $= (1/120000)/0.0025 = 1/300 \approx 0.00333$. The bound shrinks 100× as $n$ scales 100× — confirming $O(1/n)$ convergence. The actual probability is much smaller (Chebyshev uses only mean and variance; the uniform is far from worst case). A CLT-based normal approximation gives a tighter estimate.

---

**Exercise 2.**
**Prove the WLLN** from Chebyshev's inequality for i.i.d. $X_i$ with $\mathbb{E}[X] = \mu$ and finite $\mathrm{Var}(X) = \sigma^2$.

??? success "Solution to Exercise 2"
    Chebyshev: $P(|\bar X_n - \mu| \ge \varepsilon) \le \mathrm{Var}(\bar X_n)/\varepsilon^2 = \sigma^2/(n\varepsilon^2)$.

    For any fixed $\varepsilon > 0$, the right side $\to 0$ as $n \to \infty$, so $\bar X_n \to \mu$ in probability. $\square$

    Note this proof requires only finite variance, not the full Kolmogorov SLLN conditions. The WLLN can be sharpened to hold under just finite first moment (a more delicate proof using characteristic functions); the Khinchin WLLN does so.

---

**Exercise 3.**
**Distinguish convergence in probability and almost surely.** Construct a sequence $X_n \to 0$ in probability but **not** almost surely.

??? success "Solution to Exercise 3"
    The classic "moving block" example. On $\Omega = [0, 1]$ with uniform measure, define:

    - $X_1 = \mathbf 1_{[0, 1]}$
    - $X_2 = \mathbf 1_{[0, 1/2]}$, $X_3 = \mathbf 1_{[1/2, 1]}$
    - $X_4 = \mathbf 1_{[0, 1/4]}$, $X_5 = \mathbf 1_{[1/4, 1/2]}$, $X_6 = \mathbf 1_{[1/2, 3/4]}$, $X_7 = \mathbf 1_{[3/4, 1]}$
    - ... (continue with intervals of length $1/2^k$ tiling $[0,1]$)

    Then $P(X_n \ne 0) = $ length of the indicator interval $\to 0$, so $X_n \to 0$ in probability.

    But for **every** $\omega \in [0, 1]$, $X_n(\omega) = 1$ infinitely often (every $\omega$ is hit by every length scale). So $X_n(\omega) \not\to 0$ for any $\omega$ — convergence almost surely fails at every point.

    The WLLN gives in-probability convergence; the SLLN gives the stronger almost-sure convergence. Sequence behavior at individual sample paths can differ between the two regimes.

---

**Exercise 4.**
**Convergence rate.** For i.i.d. $X_i$ with mean $\mu$ and variance $\sigma^2$, show that $\sqrt n (\bar X_n - \mu) = O_P(1)$. Why is this *not* fast enough for almost-sure convergence?

??? success "Solution to Exercise 4"
    By CLT, $\sqrt n (\bar X_n - \mu) \xrightarrow{d} N(0, \sigma^2)$, so the sequence is tight (bounded in probability) — that is, $O_P(1)$.

    Implication for $\bar X_n - \mu$: it is $O_P(n^{-1/2})$ — shrinks at rate $1/\sqrt n$.

    **Why $1/\sqrt n$ is not fast enough for a.s. convergence:** the random fluctuations of $\bar X_n - \mu$ are roughly Gaussian with SD $\sigma/\sqrt n$. The maximum over $n_0 \le n \le 2 n_0$ scales like $\sqrt{\log n_0}/\sqrt{n_0}$ — so $\sqrt n (\bar X_n - \mu)$ visits arbitrarily large values infinitely often (the **law of the iterated logarithm**: $\limsup \sqrt n (\bar X_n - \mu) / \sqrt{2\sigma^2 \log\log n} = 1$ a.s.).

    A.s. convergence requires the deviations to eventually stay small for all subsequent $n$, not just become small in probability. Stronger conditions (e.g., bounded fourth moment, or just bounded first moment by Etemadi's proof) are required.

---

**Exercise 5.**
**LLN fails for the Cauchy distribution.** The standard Cauchy has density $f(x) = 1/(\pi(1+x^2))$. Why is $\mathbb{E}[X]$ undefined, and what happens to $\bar X_n$ as $n \to \infty$?

??? success "Solution to Exercise 5"
    $\mathbb{E}[X] = \int x \, f(x) \, dx$ requires the integral to converge. For the Cauchy, $\int_0^\infty x/(1+x^2) \, dx = (1/2) \ln(1 + x^2) |_0^\infty = \infty$. The positive and negative parts both diverge, so $\mathbb{E}[X]$ is undefined (not just infinite, but not even formally well-defined).

    **Behavior of $\bar X_n$:** for the Cauchy, the sample mean $\bar X_n$ has the *same distribution* as a single $X_i$ — a consequence of the Cauchy's stability under averaging (sum of $n$ Cauchys is $n$ times a Cauchy, divided by $n$ gives a Cauchy). So $\bar X_n$ does *not* concentrate as $n$ grows; it has heavy tails for every $n$.

    Lesson: the LLN requires finite mean. Heavy-tailed distributions can have unstable sample averages — the average of many observations is no better than a single observation. In practice this affects estimators in fields where extreme values are common (finance, network traffic) and motivates robust alternatives (median, trimmed mean) that have well-defined population analogs even when the mean does not.

---

**Exercise 6.**
The **WLLN justifies frequentist probability**: $P(A) = \lim_{n \to \infty} (1/n) \sum_{i=1}^n \mathbf 1(\omega_i \in A)$. State the limit precisely and explain why this *defines* probability rather than just computing it.

??? success "Solution to Exercise 6"
    Let $X_i = \mathbf 1(\omega_i \in A)$ for i.i.d. samples from $\Omega$. Then $\mathbb{E}[X_i] = P(A)$ and $\mathrm{Var}(X_i) = P(A)(1 - P(A)) < \infty$. By WLLN,

    $$
    \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{P} P(A)
    $$

    So the long-run frequency converges to the probability.

    **Why this *defines* probability rather than just computing it:** in the **frequentist interpretation**, probability is defined as the long-run frequency. The WLLN is then a tautology — the limit exists because we set $P$ to be that limit. In axiomatic measure theory (Kolmogorov), probability is a set function satisfying axioms, and the WLLN becomes a theorem connecting frequentist intuition to axioms.

    Practical use: this is why we can estimate $P(A)$ from data by computing sample proportions. The WLLN ensures the estimate is consistent — it converges to the truth as $n$ grows. Without the LLN, frequentist statistics would have no formal justification.
