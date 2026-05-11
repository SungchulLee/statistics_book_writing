# P-Hacking Demonstration

## Overview

P-hacking occurs when researchers exploit flexibility in data collection and analysis to obtain statistically significant results from data that contain no real effect. Common forms include testing many outcome variables and reporting only the significant ones, stopping data collection as soon as $p < 0.05$, and selectively choosing subgroups or analysis methods. This page simulates each of these practices to show how dramatically they inflate the false positive rate above the nominal $\alpha = 0.05$.

## Honest Testing Under the Null

When $H_0$ is true and we test at level $\alpha$, exactly a fraction $\alpha$ of tests will reject. The p-values follow a $\text{Uniform}(0,1)$ distribution:

$$
P(p \leq t \mid H_0) = t \quad \text{for } t \in [0,1].
$$

### Code

```python
import numpy as np
from scipy import stats

np.random.seed(42)

n_experiments = 10_000
n_per_group = 30
pvals = np.zeros(n_experiments)

for i in range(n_experiments):
    a = np.random.normal(0, 1, n_per_group)
    b = np.random.normal(0, 1, n_per_group)
    _, pvals[i] = stats.ttest_ind(a, b)

false_pos_rate = np.mean(pvals < 0.05)
print(f"False positive rate: {false_pos_rate:.4f}  (expected: 0.05)")
```

The histogram of p-values is essentially flat, confirming uniformity under $H_0$.

## Cherry-Picking Multiple Outcomes

If a researcher tests $k$ independent outcomes and reports only the smallest p-value, the probability of finding at least one "significant" result under $H_0$ is

$$
P(\min(p_1, \ldots, p_k) < \alpha) = 1 - (1 - \alpha)^k.
$$

For $k = 20$ and $\alpha = 0.05$:

$$
1 - (1 - 0.05)^{20} = 1 - 0.95^{20} \approx 0.64.
$$

The false positive rate jumps from 5% to 64%.

### Code

```python
n_outcomes = 20
min_pvals = np.zeros(1000)

for i in range(1000):
    ps = []
    for _ in range(n_outcomes):
        a = np.random.normal(0, 1, 30)
        b = np.random.normal(0, 1, 30)
        _, p = stats.ttest_ind(a, b)
        ps.append(p)
    min_pvals[i] = min(ps)

phack_rate = np.mean(min_pvals < 0.05)
print(f"Cherry-pick rate: {phack_rate:.4f}  (theoretical: 0.6415)")
```

## Optional Stopping

Another form of p-hacking is to peek at the data repeatedly during collection and stop as soon as $p < 0.05$. Even though each individual peek uses a valid test, the sequential peeking inflates the overall false positive rate.

### Code

```python
n_experiments = 1000
n_max = 200
check_interval = 10

stopped_pvals = []
for _ in range(n_experiments):
    a, b = [], []
    for n in range(check_interval, n_max + 1, check_interval):
        a.extend(np.random.normal(0, 1, check_interval).tolist())
        b.extend(np.random.normal(0, 1, check_interval).tolist())
        _, p = stats.ttest_ind(a, b)
        if p < 0.05:
            stopped_pvals.append(p)
            break
    else:
        stopped_pvals.append(p)

stop_rate = np.mean(np.array(stopped_pvals) < 0.05)
print(f"Optional stopping rate: {stop_rate:.4f}")
```

With up to 20 peeks (checking every 10 observations up to 200), the false positive rate can exceed 20%.

## Interpretation

| Method | Expected False Positive Rate |
|---|---|
| Honest single test | $\alpha = 0.05$ |
| Cherry-pick from 20 outcomes | $\approx 0.64$ |
| Optional stopping (20 peeks) | $\approx 0.20$ |

The core lesson is that $\alpha = 0.05$ only controls the Type I error rate when the analysis plan is fixed before looking at the data. Any post-hoc flexibility -- choosing outcomes, stopping rules, or subgroups -- inflates the true error rate, sometimes dramatically.

**Remedies** include pre-registration of analysis plans, correction for multiple comparisons (Bonferroni, BH), and sequential testing methods (alpha spending functions) that formally account for interim analyses.

## Exercises

**Exercise 1.** Derive the formula $P(\min(p_1, \ldots, p_k) < \alpha) = 1 - (1 - \alpha)^k$ for independent p-values under $H_0$. What assumption is critical?

??? success "Solution to Exercise 1"

    Under $H_0$, each $p_i \sim \text{Uniform}(0,1)$. The minimum exceeds $\alpha$ only if all p-values exceed $\alpha$:

    $$
    P(\min(p_1,\ldots,p_k) \geq \alpha) = \prod_{i=1}^k P(p_i \geq \alpha) = (1-\alpha)^k.
    $$

    By the complement,

    $$
    P(\min < \alpha) = 1 - (1-\alpha)^k.
    $$

    The critical assumption is **independence** of the p-values. If the outcomes are correlated (e.g., overlapping measurements), the actual probability may be lower than this formula predicts. $\square$

---

**Exercise 2.** How many independent outcomes must a researcher cherry-pick from to have a greater than 90% chance of finding at least one "significant" result under $H_0$ at $\alpha = 0.05$?

??? success "Solution to Exercise 2"

    We need $1 - 0.95^k > 0.90$, i.e., $0.95^k < 0.10$. Taking logarithms:

    $$
    k > \frac{\ln 0.10}{\ln 0.95} = \frac{-2.3026}{-0.05129} \approx 44.9.
    $$

    So $k \geq 45$ outcomes suffice. Testing 45 independent variables under the null gives a 90%+ chance of at least one false positive at $\alpha = 0.05$. $\square$

---

**Exercise 3.** Explain why the p-value distribution under $H_0$ is $\text{Uniform}(0,1)$ for a continuous test statistic. What happens if the test statistic is discrete?

??? success "Solution to Exercise 3"

    For a continuous test statistic $T$ with CDF $F_0$ under $H_0$, the p-value is $p = 1 - F_0(T)$ (or $2\min(F_0(T), 1-F_0(T))$ for two-sided tests). By the probability integral transform, $F_0(T) \sim \text{Uniform}(0,1)$ when $F_0$ is the true CDF, so $p \sim \text{Uniform}(0,1)$.

    For discrete test statistics, the CDF is a step function, so $F_0(T)$ takes only finitely many values. The p-value distribution is then **stochastically larger** than $\text{Uniform}(0,1)$:

    $$
    P(p \leq \alpha) \leq \alpha \quad \text{for all } \alpha,
    $$

    with strict inequality at most values. This makes discrete tests conservative. $\square$

---

**Exercise 4.** Modify the optional stopping simulation so that the researcher checks after every single new observation (rather than every 10). How does this affect the false positive rate?

??? success "Solution to Exercise 4"

    ```python
    stopped = []
    for _ in range(1000):
        a, b = [], []
        for n in range(2, 201):  # need at least 2 per group
            a.append(np.random.normal(0, 1))
            b.append(np.random.normal(0, 1))
            if len(a) >= 2:
                _, p = stats.ttest_ind(a, b)
                if p < 0.05:
                    stopped.append(p)
                    break
        else:
            stopped.append(p)
    rate = np.mean(np.array(stopped) < 0.05)
    print(f"Rate with every-observation peeking: {rate:.4f}")
    ```

    Checking after every observation gives the maximum number of peeks and therefore the highest false positive rate. The rate can exceed 30% or more, far above the nominal 5%. This is the worst-case scenario for optional stopping. $\square$

---

**Exercise 5.** Propose a correction for optional stopping. If you plan to peek $K$ times during data collection, how should you adjust $\alpha$ at each interim analysis to maintain an overall Type I error rate of 0.05? (Hint: Bonferroni is one option; alpha spending is another.)

??? success "Solution to Exercise 5"

    **Bonferroni approach:** Test at level $\alpha/K$ at each peek. If $K = 20$ peeks, each uses $\alpha = 0.05/20 = 0.0025$. This is simple but conservative.

    **Pocock boundary:** Use the same adjusted threshold $\alpha^*$ at every peek, chosen so the overall Type I error is 0.05. For $K = 20$, $\alpha^* \approx 0.003$ (computed via simulation or sequential analysis tables).

    **O'Brien-Fleming boundary:** Start with a very strict threshold at early peeks and relax it as data accumulate. At peek $k$ out of $K$, the critical $z$-value is approximately $z_{\alpha/2}/\sqrt{k/K}$. This spends almost no alpha early on, preserving power at the final analysis.

    **Alpha spending function (Lan-DeMets):** A flexible framework where a function $\alpha^*(t)$ specifies how much of the total $\alpha = 0.05$ to "spend" by information fraction $t \in [0,1]$. This generalizes Pocock and O'Brien-Fleming and does not require pre-specifying the exact peek times. $\square$
