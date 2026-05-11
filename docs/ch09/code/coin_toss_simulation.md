# Coin Toss Simulation

## Overview

Simulation-based hypothesis testing replaces analytical formulas with repeated random experiments. To test whether a coin is fair, we simulate many sequences of coin tosses under the null hypothesis $H_0\colon p = 0.5$ and estimate the p-value as the fraction of simulations yielding a result at least as extreme as the observed data. This approach illustrates the core logic of hypothesis testing without requiring knowledge of the binomial distribution.

## Setup

A coin is tossed $n = 30$ times and we observe $k = 24$ heads. Under $H_0\colon p = 0.5$ (fair coin), we ask: how unusual is this result?

The one-sided p-value is

$$
p\text{-value} = P(X \geq 24 \mid X \sim \text{Bin}(30, 0.5)).
$$

Instead of computing this analytically, we estimate it by simulation.

## Code

### Single Experiment

```python
import numpy as np

np.random.seed(42)

TOTAL_TOSSES = 30
OBSERVED_HEADS = 24
PROB_HEAD_FAIR = 0.5
NUM_SIMULATIONS = 100_000

def single_experiment(n_tosses=TOTAL_TOSSES, p=PROB_HEAD_FAIR):
    """Simulate one round of n_tosses fair-coin flips; return head count."""
    return np.random.binomial(n_tosses, p)
```

### Repeated Simulation

```python
def simulate_coin_tosses(n_simulations=NUM_SIMULATIONS,
                         n_tosses=TOTAL_TOSSES,
                         p=PROB_HEAD_FAIR):
    """Repeat the experiment n_simulations times. Returns array of head counts."""
    return np.random.binomial(n_tosses, p, size=n_simulations)

head_counts = simulate_coin_tosses()
extreme = np.sum(head_counts >= OBSERVED_HEADS)
pct = extreme / NUM_SIMULATIONS * 100

print(f"Times with >= {OBSERVED_HEADS} heads: {extreme:,}")
print(f"Percentage: {pct:.4f}%")
```

### Exact Comparison

```python
from scipy.stats import binom

p_exact = 1 - binom.cdf(OBSERVED_HEADS - 1, TOTAL_TOSSES, PROB_HEAD_FAIR)
print(f"Exact binomial P(X >= {OBSERVED_HEADS}): {p_exact:.6f}")
```

### Visualization

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.arange(0, TOTAL_TOSSES + 2) - 0.5
ax.hist(head_counts, bins=bins, edgecolor="white", alpha=0.7,
        label="Simulated head counts")
ax.axvline(OBSERVED_HEADS, color="red", linestyle="--", linewidth=2,
           label=f"Observed = {OBSERVED_HEADS}")
ax.set_xlabel("Number of heads")
ax.set_ylabel("Frequency")
ax.set_title(f"Coin Toss Simulation ({NUM_SIMULATIONS:,} runs)")
ax.legend()
plt.tight_layout()
plt.show()
```

### Interpretation

With 100,000 simulations, the fraction of runs producing 24 or more heads is well below 5%. The exact binomial p-value $P(X \geq 24 \mid n=30, p=0.5) \approx 0.0003$. Since this is far below any reasonable significance level, we reject $H_0$ and conclude the coin is likely biased toward heads.

## Exercises

**Exercise 1.** Modify the simulation to test whether the coin is biased in either direction (two-sided). That is, estimate $P(X \leq 6 \text{ or } X \geq 24 \mid n=30, p=0.5)$ by simulation.

??? success "Solution to Exercise 1"

    ```python
    head_counts = np.random.binomial(30, 0.5, size=100_000)
    extreme_two_sided = np.sum((head_counts >= 24) | (head_counts <= 6))
    p_two_sided = extreme_two_sided / 100_000
    print(f"Two-sided simulated p-value: {p_two_sided:.4f}")
    ```

    By symmetry, $P(X \leq 6) = P(X \geq 24)$, so the two-sided p-value is approximately $2 \times 0.0003 = 0.0006$. The simulation should yield a value close to this. $\square$

---

**Exercise 2.** Compute the exact binomial p-value $P(X \geq 24 \mid n=30, p=0.5)$ by hand using the complement and the binomial PMF for the last few terms.

??? success "Solution to Exercise 2"

    $$
    P(X \geq 24) = \sum_{k=24}^{30} \binom{30}{k} (0.5)^{30}.
    $$

    Since $(0.5)^{30} = 1/1{,}073{,}741{,}824$:

    - $\binom{30}{24} = \binom{30}{6} = 593{,}775$
    - $\binom{30}{25} = \binom{30}{5} = 142{,}506$
    - $\binom{30}{26} = \binom{30}{4} = 27{,}405$
    - $\binom{30}{27} = \binom{30}{3} = 4{,}060$
    - $\binom{30}{28} = \binom{30}{2} = 435$
    - $\binom{30}{29} = \binom{30}{1} = 30$
    - $\binom{30}{30} = 1$

    Sum: $593{,}775 + 142{,}506 + 27{,}405 + 4{,}060 + 435 + 30 + 1 = 768{,}212$.

    $$
    P(X \geq 24) = \frac{768{,}212}{1{,}073{,}741{,}824} \approx 0.000716.
    $$

    This is consistent with the simulation estimate. $\square$

---

**Exercise 3.** Explain why the simulation-based p-value converges to the exact p-value as the number of simulations increases. What is the standard error of the simulated p-value?

??? success "Solution to Exercise 3"

    Each simulation produces an indicator $I_i = \mathbf{1}(X_i \geq k)$ where $P(I_i = 1) = p^*$ (the true p-value). The simulated p-value is $\hat{p} = \bar{I} = \sum I_i / N$. By the law of large numbers, $\hat{p} \to p^*$ as $N \to \infty$.

    The standard error of $\hat{p}$ is

    $$
    SE = \sqrt{\frac{p^*(1-p^*)}{N}}.
    $$

    For $p^* \approx 0.0007$ and $N = 100{,}000$:

    $$
    SE = \sqrt{\frac{0.0007 \times 0.9993}{100{,}000}} \approx 0.000084.
    $$

    A 95% CI for the simulated p-value is approximately $0.0007 \pm 0.00016$. More simulations reduce this uncertainty. $\square$

---

**Exercise 4.** How many simulations are needed so that the 95% confidence interval for a simulated p-value of $p^* = 0.05$ has half-width no larger than 0.005?

??? success "Solution to Exercise 4"

    We need $1.96 \times SE \leq 0.005$, so $SE \leq 0.00255$. Setting

    $$
    \sqrt{\frac{0.05 \times 0.95}{N}} \leq 0.00255,
    $$

    $$
    \frac{0.0475}{N} \leq 0.00255^2 = 6.5025 \times 10^{-6},
    $$

    $$
    N \geq \frac{0.0475}{6.5025 \times 10^{-6}} \approx 7{,}305.
    $$

    At least 7,305 simulations are needed. In practice, $N = 10{,}000$ is a common minimum. $\square$

---

**Exercise 5.** Suppose 24 heads in 30 tosses is observed. Using a Bayesian approach with a $\text{Beta}(1,1)$ (uniform) prior on $p$, compute the posterior distribution and the posterior probability $P(p > 0.5 \mid \text{data})$.

??? success "Solution to Exercise 5"

    With a $\text{Beta}(1,1)$ prior and observing $k=24$ heads in $n=30$ trials, the posterior is

    $$
    p \mid \text{data} \sim \text{Beta}(1 + 24,\; 1 + 6) = \text{Beta}(25, 7).
    $$

    The posterior probability that the coin is biased toward heads is

    $$
    P(p > 0.5 \mid \text{data}) = 1 - I_{0.5}(25, 7),
    $$

    where $I_x(a,b)$ is the regularized incomplete beta function. Using Python: `1 - stats.beta.cdf(0.5, 25, 7)` $\approx 0.9997$. There is a 99.97% posterior probability that $p > 0.5$. $\square$
