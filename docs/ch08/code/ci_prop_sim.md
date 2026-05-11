# Proportion Confidence Interval Coverage Simulation

## Overview

This page investigates the coverage performance of four methods for constructing confidence intervals for a population proportion $p$: the Wald, Wilson score, Agresti--Coull, and Clopper--Pearson (exact) intervals. A Monte Carlo simulation repeatedly generates Bernoulli samples, builds a CI with each method, and records whether the interval captures the true $p$. The results highlight why the Wald interval is unreliable for small $n$ or extreme $p$.

## Four Interval Methods

### Wald Interval

$$
\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$

Simple but can severely under-cover when $n$ is small or $p$ is near 0 or 1.

### Wilson Score Interval

$$
\frac{\hat{p} + \frac{z^2}{2n}}{1 + \frac{z^2}{n}}
\;\pm\;
\frac{z}{1 + \frac{z^2}{n}}
\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}
$$

Adjusts the center away from $\hat{p}$ and provides good coverage even for moderate $n$. This is the recommended default.

### Agresti--Coull Interval

Add $z^2/2$ pseudo-successes and $z^2/2$ pseudo-failures to form adjusted counts:

$$
\tilde{n} = n + z^2, \quad \tilde{p} = \frac{k + z^2/2}{\tilde{n}}
$$

Then apply the Wald formula to $(\tilde{p}, \tilde{n})$:

$$
\tilde{p} \pm z_{\alpha/2} \sqrt{\frac{\tilde{p}(1-\tilde{p})}{\tilde{n}}}
$$

Coverage is very close to Wilson; computation is even simpler.

### Clopper--Pearson (Exact) Interval

Inverts the binomial test using Beta quantiles:

$$
\left(\text{Beta}\!\left(\frac{\alpha}{2};\; k,\; n-k+1\right),\;\;
      \text{Beta}\!\left(1-\frac{\alpha}{2};\; k+1,\; n-k\right)\right)
$$

This is conservative: actual coverage is at least $(1-\alpha)100\%$, but intervals tend to be wider.

## Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, beta

n_simulations = 100
n = 20
p_true = 0.20
alpha = 0.05
method = "wilson"  # 'wald' | 'wilson' | 'ac' | 'cp'

k = np.random.binomial(n=n, p=p_true, size=n_simulations)
phat = k / n
z = norm.ppf(1 - alpha / 2)

lower = np.empty(n_simulations)
upper = np.empty(n_simulations)

for i, ki in enumerate(k):
    p = ki / n
    if method == "wald":
        se = np.sqrt(p * (1 - p) / n)
        lo, hi = p - z * se, p + z * se
    elif method == "wilson":
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        lo, hi = center - half, center + half
    elif method == "ac":
        n_tilde = n + z**2
        p_tilde = (ki + 0.5 * z**2) / n_tilde
        se_tilde = np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
        lo, hi = p_tilde - z * se_tilde, p_tilde + z * se_tilde
    else:  # cp
        lo = 0.0 if ki == 0 else beta.ppf(alpha / 2, ki, n - ki + 1)
        hi = 1.0 if ki == n else beta.ppf(1 - alpha / 2, ki + 1, n - ki)

    lower[i] = max(0.0, lo)
    upper[i] = min(1.0, hi)

covered = (lower <= p_true) & (p_true <= upper)
coverage_pct = 100.0 * covered.mean()
print(f"{method} coverage: {coverage_pct:.1f}%")
```

### Plotting the Intervals

```python
fig, ax = plt.subplots(figsize=(12, 12))
for i in range(n_simulations):
    color = "k" if covered[i] else "r"
    ax.plot([lower[i], upper[i]], [i, i], lw=2, color=color)
    ax.plot(phat[i], i, marker="o", ms=3, color=color)

ax.axvline(p_true, linestyle="--", linewidth=1.5)
ax.set_title(f"{n_simulations} {method.upper()} CIs | n={n}, p={p_true}, CL=95%")
ax.set_yticks([])
ax.set_xlabel("Proportion value")
plt.tight_layout()
plt.show()
```

## Interpretation

- **Wald** intervals can have dramatically low coverage when $n$ is small or $p$ is near the boundaries 0 or 1. A common rule of thumb requires $n\hat{p} \ge 10$ and $n(1-\hat{p}) \ge 10$, but even this is not always sufficient.
- **Wilson** and **Agresti--Coull** intervals shift the center toward $1/2$ and widen the interval slightly, yielding much more reliable coverage across a wide range of $n$ and $p$.
- **Clopper--Pearson** guarantees at least the nominal coverage level by construction, but is conservative (wider than necessary), especially for small $n$.
- When sampling without replacement from a finite population of size $N$, the binomial model is an approximation valid when $n \le 0.10 N$. For larger sampling fractions, a hypergeometric-based interval is more appropriate.

## Exercises

**Exercise 1.** Run the simulation with $n = 20$, $p_{\text{true}} = 0.05$, and $n_{\text{sim}} = 10{,}000$ for the Wald interval. Report the empirical coverage and explain why it deviates from 95 %.

??? success "Solution to Exercise 1"

    With $p_{\text{true}} = 0.05$ and $n = 20$, we have $np = 1.0$, which violates the rule of thumb $np \ge 10$. Many samples produce $k = 0$ or $k = 1$ successes, giving $\hat{p}$ values near 0 where the Wald standard error collapses to near zero. As a result, the intervals are extremely narrow (or degenerate at 0), and coverage drops to roughly 70--80 %. The Wald interval's reliance on the normal approximation to the binomial fails when the distribution of $\hat{p}$ is highly skewed. $\square$

---

**Exercise 2.** Derive the Wilson score interval starting from the inequality $|Z| \le z_{\alpha/2}$ where $Z = (\hat{p} - p)/\sqrt{p(1-p)/n}$.

??? success "Solution to Exercise 2"

    The test-inversion approach solves for all values of $p$ satisfying

    $$
    \left|\frac{\hat{p} - p}{\sqrt{p(1-p)/n}}\right| \le z_{\alpha/2}
    $$

    Squaring both sides:

    $$
    \frac{(\hat{p} - p)^2}{p(1-p)/n} \le z^2
    $$

    $$
    n(\hat{p} - p)^2 \le z^2 p(1-p)
    $$

    Expanding and collecting terms in $p$:

    $$
    n\hat{p}^2 - 2n\hat{p}\,p + np^2 \le z^2 p - z^2 p^2
    $$

    $$
    (n + z^2)p^2 - (2n\hat{p} + z^2)p + n\hat{p}^2 \le 0
    $$

    This is a quadratic in $p$. Applying the quadratic formula gives

    $$
    p = \frac{2n\hat{p} + z^2 \pm \sqrt{z^4 + 4n z^2 \hat{p}(1-\hat{p})}}{2(n + z^2)}
    $$

    which simplifies to the Wilson interval endpoints. $\square$

---

**Exercise 3.** Show that the Agresti--Coull interval at the 95 % level adds approximately 2 pseudo-successes and 2 pseudo-failures to the data.

??? success "Solution to Exercise 3"

    At the 95 % confidence level, $\alpha = 0.05$ and $z_{\alpha/2} = z_{0.025} \approx 1.96$. The Agresti--Coull method adds $z^2/2$ pseudo-successes and $z^2/2$ pseudo-failures, giving a total of $z^2$ additional observations. Computing:

    $$
    \frac{z^2}{2} = \frac{(1.96)^2}{2} = \frac{3.8416}{2} \approx 1.92
    $$

    So approximately 2 pseudo-successes and 2 pseudo-failures are added, and the adjusted sample size is $\tilde{n} = n + z^2 \approx n + 4$. This "add 2 successes and 2 failures" rule is why the Agresti--Coull method is sometimes called the "plus-four" interval. $\square$

---

**Exercise 4.** A medical study observes 3 adverse events in 200 patients. Compute the Wald, Wilson, and Clopper--Pearson 95 % intervals. Comment on the differences.

??? success "Solution to Exercise 4"

    Here $k = 3$, $n = 200$, $\hat{p} = 0.015$, and $z = 1.96$.

    **Wald:** $\text{SE} = \sqrt{0.015 \times 0.985 / 200} = 0.00860$. CI: $0.015 \pm 1.96 \times 0.00860 = (-0.0019, 0.0319)$. Clipped to $(0, 0.0319)$.

    **Wilson:** Numerator center: $(0.015 + 3.8416/400)/(1 + 3.8416/200) = 0.02461/1.01921 \approx 0.02415$. Half-width $\approx 0.01939$. CI $\approx (0.0048, 0.0435)$.

    **Clopper--Pearson:** Lower $= \text{Beta}(0.025;\, 3,\, 198) \approx 0.00311$. Upper $= \text{Beta}(0.975;\, 4,\, 197) \approx 0.0433$. CI $\approx (0.0031, 0.0433)$.

    The Wald interval includes negative values (which are impossible for a proportion) and is narrower. Wilson and Clopper--Pearson give similar and more sensible intervals. The Clopper--Pearson lower bound is slightly smaller, reflecting its conservative nature. $\square$

---

**Exercise 5.** Prove that the Clopper--Pearson interval has coverage at least $(1-\alpha)$ for every value of $p \in (0,1)$.

??? success "Solution to Exercise 5"

    The Clopper--Pearson interval $[L(k), U(k)]$ is defined by inverting two one-sided binomial tests. Specifically, $L(k)$ is the value of $p$ such that

    $$
    P(X \ge k \mid p = L(k)) = \alpha/2, \quad X \sim \text{Binomial}(n, p)
    $$

    and $U(k)$ is the value of $p$ such that

    $$
    P(X \le k \mid p = U(k)) = \alpha/2
    $$

    For any true $p$, the event $p \notin [L(K), U(K)]$ means either $p < L(K)$ or $p > U(K)$. By construction, $p < L(K)$ implies $K$ is "too large" given $p$, and this tail probability is at most $\alpha/2$. Similarly, $p > U(K)$ implies $K$ is "too small," with tail probability at most $\alpha/2$. Since the binomial CDF is a step function, the tail probabilities can be strictly less than $\alpha/2$ for some $p$, but never more. Therefore

    $$
    P(p \notin [L(K), U(K)]) \le \frac{\alpha}{2} + \frac{\alpha}{2} = \alpha
    $$

    and coverage $P(p \in [L(K), U(K)]) \ge 1 - \alpha$ for all $p \in (0,1)$. $\square$
