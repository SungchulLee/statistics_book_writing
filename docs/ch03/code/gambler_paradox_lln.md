# Gambler's Paradox: When the Law of Large Numbers Fails

## Overview

The Law of Large Numbers (LLN) guarantees that the sample mean converges to the population mean — but only when the population mean is **finite**. The **St. Petersburg paradox** provides a classic counterexample: a game with infinite expected value where the sample mean diverges rather than stabilizes. This page simulates the paradox and contrasts it with a bounded variant where the LLN holds.

---

## 1. The St. Petersburg Game

### Rules

A fair coin is flipped repeatedly until the first heads appears. If heads first appears on flip $k$, the payout is $2^k$ dollars.

The expected payout is:

$$
E[X] = \sum_{k=1}^{\infty} 2^k \cdot \left(\frac{1}{2}\right)^k = \sum_{k=1}^{\infty} 1 = \infty
$$

Since $E[X] = \infty$, the LLN does not apply: there is no finite value for the sample mean to converge to.

### The Paradox

Despite the infinite expected value, most individual rounds pay very little (50% pay \$2, 75% pay \$4 or less). But the rare event of a long run of tails produces an enormous payout that dominates the average. No matter how many rounds are played, a single extreme outcome can shift the sample mean dramatically.

---

## 2. Simulation: Infinite Mean

We simulate 100 independent sequences, each playing up to 10,000 rounds, and track the running sample mean:

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def st_petersburg_sample_means(n_max=10_000, tries=100, n_grid=200):
    """Each round: first heads on flip k → win 2^k. E[X] = infinity."""
    n_vals = np.unique(np.logspace(1, np.log10(n_max), n_grid).astype(int))
    results = []
    for n in n_vals:
        flips = np.random.geometric(0.5, size=(n, tries))
        winnings = 2.0 ** flips
        means = winnings.mean(axis=0)
        results.append((n, means))
    return results

infinite_results = st_petersburg_sample_means()
```

---

## 3. The Bounded Variant: Finite Mean

Now cap the payout at $2^{10} = 1024$ dollars. This truncation makes $E[X]$ finite, so the LLN applies:

$$
E[X_{\text{bounded}}] = \sum_{k=1}^{10} 2^k \cdot \left(\frac{1}{2}\right)^k + 1024 \cdot \sum_{k=11}^{\infty} \left(\frac{1}{2}\right)^k = 10 + 1024 \cdot \frac{1}{1024} = 11
$$

```python
def bounded_game_sample_means(n_max=10_000, tries=100, n_grid=200):
    """Same game but capped at 2^10 = 1024. E[X] is now finite."""
    n_vals = np.unique(np.logspace(1, np.log10(n_max), n_grid).astype(int))
    results = []
    for n in n_vals:
        flips = np.random.geometric(0.5, size=(n, tries))
        winnings = np.minimum(2.0 ** flips, 1024.0)
        means = winnings.mean(axis=0)
        results.append((n, means))
    return results

bounded_results = bounded_game_sample_means()
```

---

## 4. Visualization

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: infinite mean — divergence
ax = axes[0]
for n, means in infinite_results:
    ax.loglog(n * np.ones(len(means)), means, ".", color="black", ms=2, alpha=0.5)
ax.set_xlabel("n (number of rounds)")
ax.set_ylabel("Sample mean of winnings")
ax.set_title("St. Petersburg Game (E[X] = ∞)\nSample mean does NOT converge")

# Panel 2: finite mean — convergence
ax = axes[1]
for n, means in bounded_results:
    ax.semilogx(n * np.ones(len(means)), means, ".", color="steelblue", ms=2, alpha=0.5)
true_mean = 11.0
ax.axhline(true_mean, color="red", linestyle="--", lw=2,
           label=f"E[X] = {true_mean:.1f}")
ax.set_xlabel("n (number of rounds)")
ax.set_ylabel("Sample mean of winnings")
ax.set_title("Bounded Game (E[X] < ∞)\nSample mean converges (LLN)")
ax.legend()

plt.tight_layout()
plt.show()
```

---

## 5. Interpretation

### Left panel (infinite mean)

The cloud of sample means does **not** tighten as $n$ increases — it continues to spread on the log-log scale. Different simulation runs give wildly different averages even at $n = 10{,}000$. This is the hallmark of LLN failure: the sample mean is not a consistent estimator when the population mean is infinite.

### Right panel (finite mean)

The cloud of sample means **contracts** around the red line ($E[X] = 11$) as $n$ grows. By $n = 10{,}000$, essentially all 100 simulation runs agree on a value near 11. This is the LLN working as expected.

### The critical condition

The difference between the two panels is a single mathematical condition: **finite expected value**. The St. Petersburg game and its bounded variant differ only in tail behavior, yet this produces qualitatively opposite statistical behavior.

!!! warning "Finite Mean Is Not Optional"
    The LLN is often stated informally as "averages converge." This is misleading. Averages converge only when the population mean exists and is finite. Heavy-tailed distributions in finance, insurance, and network traffic can violate this condition.

---

## 6. Connection to Theory

The LLN comes in two forms:

- **Weak LLN (WLLN):** Requires finite variance (or weaker: finite mean via truncation arguments). Gives convergence in probability.
- **Strong LLN (SLLN):** Requires only finite mean ($E[|X|] < \infty$). Gives almost sure convergence.

The St. Petersburg game has $E[X] = \infty$, so neither form applies. The bounded variant has finite mean and finite variance, so both forms hold.

!!! note "Heavy Tails vs Infinite Mean"
    A distribution can have heavy tails (e.g., Pareto with $\alpha > 1$) yet still have a finite mean, in which case the LLN applies. The distinction is between heavy tails (slow decay) and non-integrable tails (infinite mean). Only the latter breaks the LLN.

---

## Exercises

**Exercise 1.**
Compute the expected value of the St. Petersburg game payout from the definition. At what step does the standard convergence test for series fail?

??? success "Solution to Exercise 1"
    The payout is $X = 2^K$ where $K \sim \text{Geometric}(1/2)$. The expected value is:

    $$
    E[X] = \sum_{k=1}^{\infty} 2^k \cdot P(K = k) = \sum_{k=1}^{\infty} 2^k \cdot \frac{1}{2^k} = \sum_{k=1}^{\infty} 1
    $$

    This is the harmonic-type series $1 + 1 + 1 + \cdots$, which diverges. The terms do not approach zero, so even the basic divergence test (if $a_k \not\to 0$ then $\sum a_k$ diverges) confirms infinite expected value.

---

**Exercise 2.**
Suppose the payout is capped at $2^M$ for some integer $M \ge 1$. Derive a formula for $E[X_{\text{bounded}}]$ as a function of $M$.

??? success "Solution to Exercise 2"
    For $k \le M$, the payout is $2^k$ with probability $(1/2)^k$. For $k > M$, the payout is $2^M$ with probability $(1/2)^k$. So:

    $$
    E[X_{\text{bounded}}] = \sum_{k=1}^{M} 2^k \cdot \frac{1}{2^k} + 2^M \sum_{k=M+1}^{\infty} \frac{1}{2^k}
    $$

    The first sum is $M$. The second sum is a geometric series:

    $$
    2^M \cdot \frac{1/2^{M+1}}{1 - 1/2} = 2^M \cdot \frac{1}{2^M} = 1
    $$

    Therefore $E[X_{\text{bounded}}] = M + 1$.

    For $M = 10$: $E[X] = 11$, matching the simulation.

---

**Exercise 3.**
The Pareto distribution with parameter $\alpha$ has density $f(x) = \alpha / x^{\alpha+1}$ for $x \ge 1$. For what values of $\alpha$ does $E[X]$ exist? For what values does $\text{Var}(X)$ exist? Which forms of the LLN apply in each case?

??? success "Solution to Exercise 3"
    The $r$-th moment is:

    $$
    E[X^r] = \int_1^{\infty} \frac{\alpha \, x^r}{x^{\alpha+1}} \, dx = \alpha \int_1^{\infty} x^{r - \alpha - 1} \, dx
    $$

    This converges if and only if $r - \alpha - 1 < -1$, i.e., $r < \alpha$.

    - $E[X]$ exists $\iff \alpha > 1$. The SLLN applies.
    - $\text{Var}(X)$ exists $\iff E[X^2] < \infty \iff \alpha > 2$. The WLLN (with finite variance) applies, and the CLT also applies.
    - For $1 < \alpha \le 2$: the mean is finite but the variance is infinite. The SLLN still holds, but the CLT does not apply in its standard form (generalized CLT with stable distributions is needed).
    - For $\alpha \le 1$: the mean is infinite and neither LLN applies.

---

**Exercise 4.**
Prove the Weak Law of Large Numbers using Chebyshev's inequality, assuming finite variance $\sigma^2$.

??? success "Solution to Exercise 4"
    Let $X_1, \ldots, X_n$ be i.i.d. with mean $\mu$ and variance $\sigma^2$. Then $E[\bar{X}] = \mu$ and $\text{Var}(\bar{X}) = \sigma^2 / n$.

    By Chebyshev's inequality:

    $$
    P(|\bar{X} - \mu| \ge \varepsilon) \le \frac{\text{Var}(\bar{X})}{\varepsilon^2} = \frac{\sigma^2}{n\varepsilon^2}
    $$

    As $n \to \infty$, the right side goes to 0 for any fixed $\varepsilon > 0$:

    $$
    P(|\bar{X} - \mu| \ge \varepsilon) \to 0
    $$

    This is exactly convergence in probability: $\bar{X} \xrightarrow{P} \mu$. $\square$

---

**Exercise 5.**
In the simulation, the bounded game uses sampling with replacement (`np.random.geometric`). If instead you played a fixed sequence of $n$ rounds and computed the running average, would the plot look different? Explain the distinction between the simulation design and a single gambler's experience.

??? success "Solution to Exercise 5"
    The simulation draws 100 **independent** sequences of length $n$ for each grid point and plots the sample mean of each. This shows the **sampling distribution** of $\bar{X}_n$ — i.e., the variability across many hypothetical gamblers.

    A single gambler playing $n$ rounds would produce one running average path. This path would show the **almost sure** convergence guaranteed by the SLLN: a single trajectory that eventually stabilizes near $E[X]$.

    The plot would look different: instead of a cloud of dots at each $n$, you would see 100 individual trajectories (lines), each converging toward the true mean. The cloud representation emphasizes the **distribution** of the estimator; the trajectory representation emphasizes the **path-wise** behavior. Both illustrate the LLN, but from complementary perspectives.

---

**Exercise 6.**
The **gambler's fallacy** is the mistaken belief that, after a run of bad outcomes, "good ones are due." Show formally why the LLN does *not* justify this. State the correct interpretation of "averages converge to the expected value."

??? success "Solution to Exercise 6"
    Let $X_1, X_2, \ldots$ be i.i.d. fair coin flips. Each $X_i$ is independent, so $P(X_{n+1} = \text{H} \mid X_1, \ldots, X_n) = P(X_{n+1} = \text{H}) = 1/2$ regardless of history. After 10 tails in a row, flip 11 is still 50/50. The coin has no memory.

    **What LLN actually says:** $\bar X_n \to \mu$ almost surely. The *average* approaches $\mu$. But the *sum* $\sum X_i - n\mu$ does not return to zero — by the **law of the iterated logarithm**, $\limsup |\sum X_i - n\mu|/\sqrt{2n\log\log n} = \sigma$ a.s. The sum's fluctuations grow like $\sqrt n$, unbounded.

    So after 10 tails, the average $\bar X_{10} = -1$ will drift toward $\mu$ over many more flips, but *not because future flips compensate*. The earlier 10 tails get diluted, not corrected. Every flip stands alone.

    **The gambler's fallacy** misunderstands this as "tails owe me heads now." Casinos exploit this in roulette, slot machines, and lottery strategies. The correct statement: the *long-run frequency* equals the *probability* — but no particular short run is "due" for anything.

---

**Exercise 7.**
For **bounded** random variables ($|X_i| \le M$), show the SLLN follows from the **Borel-Cantelli lemma** applied to $\{|\bar X_n - \mu| > \varepsilon\}$.

??? success "Solution to Exercise 7"
    By Hoeffding's inequality (for bounded $X_i$ with range $\le 2M$):

    $$
    P(|\bar X_n - \mu| > \varepsilon) \le 2 e^{-n\varepsilon^2/(2M^2)}
    $$

    Sum over $n$: $\sum_n P(|\bar X_n - \mu| > \varepsilon) < \infty$ for any $\varepsilon > 0$ (geometric-tail summable).

    By **Borel-Cantelli lemma I**, $P(|\bar X_n - \mu| > \varepsilon \text{ infinitely often}) = 0$. So with probability 1, only finitely many of these events occur — equivalently, $\bar X_n \to \mu$ almost surely. $\square$

    Kolmogorov's general SLLN requires only finite first moment, but the proof is more delicate (truncation arguments). The Hoeffding-Borel-Cantelli proof is the cleanest path under the boundedness assumption.
