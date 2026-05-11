# Kaplan Meier

## Overview

This page presents a self-contained Python implementation of the Kaplan-Meier survival
curve estimator and the two-sample log-rank test, using only NumPy and SciPy.  The
implementation mirrors the mathematical definitions step by step, making it suitable
for learning the mechanics of survival analysis before turning to production libraries.

## Kaplan-Meier Estimator

### Mathematical Definition

Given $n$ subjects with observed times and censoring indicators, the Kaplan-Meier
estimator of the survival function is

$$
\hat{S}(t) = \prod_{j:\, t_{(j)} \leq t} \left(1 - \frac{d_j}{n_j}\right)
$$

where $t_{(1)} < t_{(2)} < \cdots < t_{(K)}$ are the distinct event times, $d_j$ is
the number of events at $t_{(j)}$, and $n_j$ is the number of subjects at risk just
before $t_{(j)}$.

### Implementation

```python
import numpy as np

def kaplan_meier(times, censored):
    """
    Compute the Kaplan-Meier survival function estimate.

    Parameters
    ----------
    times    : 1-d array   Observed times (event or censoring).
    censored : 1-d array   1 = censored (no event), 0 = event observed.

    Returns
    -------
    t_plot : array   Time points for step-plot (includes 0 and max time).
    s_plot : array   Survival probabilities matching t_plot.
    """
    order = np.argsort(times)
    times = times[order]
    censored = censored[order]

    event_times = times[censored == 0]
    unique_events = np.unique(event_times)

    n_total = len(times)
    s = 1.0
    t_list = [0.0]
    s_list = [1.0]

    for t_j in unique_events:
        n_at_risk = np.sum(times >= t_j)
        d_j = np.sum((times == t_j) & (censored == 0))
        s *= (n_at_risk - d_j) / n_at_risk
        t_list.append(t_j)
        s_list.append(s)

    t_list.append(times.max())
    s_list.append(s_list[-1])

    return np.array(t_list), np.array(s_list)
```

**Algorithm walkthrough:**

1. **Sort** observations by time.
2. **Extract** distinct event times (times where `censored == 0`).
3. **For each event time** $t_j$:
     - Count subjects still at risk: $n_j = \sum \mathbf{1}(t_i \geq t_j)$.
     - Count events: $d_j = \sum \mathbf{1}(t_i = t_j \text{ and event observed})$.
     - Update: $\hat{S} \leftarrow \hat{S} \times (1 - d_j / n_j)$.
4. **Extend** the curve to the maximum observed time for plotting.

!!! note "Censored Observations"

    Censored subjects do not trigger a drop in $\hat{S}(t)$, but they reduce the
    risk set $n_j$ at subsequent event times.  This is the mechanism by which
    the Kaplan-Meier estimator incorporates partial information from incomplete
    observations.

## Log-Rank Test

### Hypotheses

The two-sample log-rank test evaluates

$$
H_0 : S_1(t) = S_2(t) \quad \text{for all } t \geq 0
$$

$$
H_1 : S_1(t) \neq S_2(t) \quad \text{for some } t \geq 0
$$

### Test Statistic

At each pooled event time $t_{(j)}$, define $r_{1j}$, $r_{2j}$ as the risk set sizes,
$d_{1j}$, $d_{2j}$ as the event counts, $r_j = r_{1j} + r_{2j}$, and
$d_j = d_{1j} + d_{2j}$.  The expected events in group 1 and the variance contribution
are

$$
e_{1j} = d_j \cdot \frac{r_{1j}}{r_j}, \qquad v_j = \frac{r_{1j} \, r_{2j} \, d_j \, (r_j - d_j)}{r_j^2 \, (r_j - 1)}
$$

The test statistic is

$$
\chi^2_{\text{LR}} = \frac{(O_1 - E_1)^2}{V_1} \;\xrightarrow{d}\; \chi^2_1
$$

where $O_1 = \sum d_{1j}$, $E_1 = \sum e_{1j}$, $V_1 = \sum v_j$.

### Implementation

```python
from scipy import stats

def logrank_test(times_1, censored_1, times_2, censored_2):
    """
    Two-sample log-rank test.

    Returns
    -------
    chi2    : float   Test statistic (chi-square with 1 df).
    p_value : float   p-value from chi-square(1).
    """
    event_1 = times_1[censored_1 == 0]
    event_2 = times_2[censored_2 == 0]
    all_event_times = np.unique(np.concatenate([event_1, event_2]))

    O1 = 0.0
    E1 = 0.0
    V  = 0.0

    for t_j in all_event_times:
        r1 = np.sum(times_1 >= t_j)
        r2 = np.sum(times_2 >= t_j)
        r  = r1 + r2

        d1 = np.sum(event_1 == t_j)
        d2 = np.sum(event_2 == t_j)
        d  = d1 + d2

        e1 = r1 * d / r if r > 0 else 0
        v  = r1 * r2 * d * (r - d) / (r**2 * (r - 1)) if r > 1 else 0

        O1 += d1
        E1 += e1
        V  += v

    chi2 = (O1 - E1)**2 / V if V > 0 else 0
    p_value = stats.chi2(1).sf(chi2)
    return chi2, p_value
```

The implementation pools event times from both groups, then loops through each event
time to accumulate observed events, expected events, and variance for group 1.

## Simulation and Visualization

The following code generates simulated data and plots the Kaplan-Meier curves.

```python
import matplotlib.pyplot as plt

def main():
    np.random.seed(0)

    # Group 1: slower event rate (mean = 20)
    n1 = 40
    times_1 = np.random.exponential(scale=20, size=n1)
    censored_1 = (np.random.rand(n1) < 0.2).astype(int)

    # Group 2: faster event rate (mean = 12)
    n2 = 40
    times_2 = np.random.exponential(scale=12, size=n2)
    censored_2 = (np.random.rand(n2) < 0.2).astype(int)

    # Kaplan-Meier curves
    t1, s1 = kaplan_meier(times_1, censored_1)
    t2, s2 = kaplan_meier(times_2, censored_2)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(t1, s1, where="post", linewidth=2, label="Group 1 (slow)")
    ax.step(t2, s2, where="post", linewidth=2, label="Group 2 (fast)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Kaplan-Meier Survival Curves")
    ax.set_ylim(-0.02, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Log-rank test
    chi2, p = logrank_test(times_1, censored_1, times_2, censored_2)
    print(f"Log-Rank Test:  chi2 = {chi2:.4f},  p = {p:.4f}")
```

Group 1 is drawn from $\text{Exp}(\lambda = 1/20)$ and group 2 from
$\text{Exp}(\lambda = 1/12)$, with approximately 20% random censoring in each group.
The visual separation of the curves and the log-rank p-value together indicate whether
the survival difference is statistically significant.

## Interpretation

- **Step-function output**: The Kaplan-Meier curve is a step function that drops only
  at observed event times.  Flat segments correspond to intervals with no events.
- **Censoring**: Censored subjects exit the risk set without causing a survival drop.
  Heavy censoring at late times leads to wider confidence intervals.
- **Log-rank test**: A significant p-value (e.g., $p < 0.05$) indicates that the
  survival distributions differ.  The test is most powerful under proportional hazards.
- **Limitations**: The Kaplan-Meier estimator is univariate --- it cannot adjust for
  covariates.  For covariate-adjusted survival analysis, the Cox proportional hazards
  model is needed.

!!! warning "Crossing Survival Curves"

    If the two Kaplan-Meier curves cross, the log-rank test may fail to detect a
    significant difference even when the curves differ substantially.  In this
    case, consider a weighted log-rank test (e.g., Wilcoxon) that gives more
    weight to early event times.

## Exercises

**Exercise 1.**
Manual Kaplan-Meier Computation

Eight subjects have the following data:

| Subject | Time | Censored (1 = yes) |
|:-------:|:----:|:------------------:|
| 1 | 1 | 0 |
| 2 | 3 | 1 |
| 3 | 4 | 0 |
| 4 | 5 | 0 |
| 5 | 5 | 1 |
| 6 | 7 | 0 |
| 7 | 10 | 1 |
| 8 | 12 | 0 |

Compute $\hat{S}(t)$ at each event time.

??? success "Solution to Exercise 1"

    Distinct event times: 1, 4, 5, 7, 12.

    | $t_{(j)}$ | $n_j$ | $d_j$ | $1 - d_j/n_j$ | $\hat{S}(t_{(j)})$ |
    |:----------:|:-----:|:-----:|:--------------:|:-------------------:|
    | 1 | 8 | 1 | 7/8 = 0.875 | 0.875 |
    | 4 | 6 | 1 | 5/6 = 0.833 | 0.875 $\times$ 0.833 = 0.729 |
    | 5 | 5 | 1 | 4/5 = 0.800 | 0.729 $\times$ 0.800 = 0.583 |
    | 7 | 3 | 1 | 2/3 = 0.667 | 0.583 $\times$ 0.667 = 0.389 |
    | 12 | 1 | 1 | 0/1 = 0.000 | 0.000 |

    At $t_{(2)} = 4$: subject 1 had an event at $t = 1$ and subject 2 was censored at
    $t = 3$, leaving 6 subjects at risk.

    At $t_{(3)} = 5$: subject 5 is censored at $t = 5$ but is included in the risk set
    (convention: censorings at $t_j$ are processed after events).  So $n_3 = 5$ and
    $d_3 = 1$ (only subject 4 has an event).

---

**Exercise 2.**
Log-Rank Test Computation

Two groups:

**Group A:** 2, 5, 8+ (+ = censored)

**Group B:** 1, 4, 6

Compute the log-rank test statistic $\chi^2_{\text{LR}}$ and the p-value.

??? success "Solution to Exercise 2"

    Pooled distinct event times: 1, 2, 4, 5, 6.

    | $t_{(j)}$ | $r_{Aj}$ | $r_{Bj}$ | $r_j$ | $d_{Aj}$ | $d_{Bj}$ | $d_j$ | $e_{Aj}$ | $v_j$ |
    |:----------:|:--------:|:--------:|:-----:|:--------:|:--------:|:-----:|:--------:|:-----:|
    | 1 | 3 | 3 | 6 | 0 | 1 | 1 | 0.500 | 0.250 |
    | 2 | 3 | 2 | 5 | 1 | 0 | 1 | 0.600 | 0.240 |
    | 4 | 2 | 2 | 4 | 0 | 1 | 1 | 0.500 | 0.250 |
    | 5 | 2 | 1 | 3 | 1 | 0 | 1 | 0.667 | 0.222 |
    | 6 | 1 | 1 | 2 | 0 | 1 | 1 | 0.500 | 0.250 |

    $O_A = 2$, $E_A = 0.500 + 0.600 + 0.500 + 0.667 + 0.500 = 2.767$.

    $V_A = 0.250 + 0.240 + 0.250 + 0.222 + 0.250 = 1.212$.

    $$
    \chi^2_{\text{LR}} = \frac{(2 - 2.767)^2}{1.212} = \frac{0.589}{1.212} = 0.486
    $$

    $p = P(\chi^2_1 \geq 0.486) = 0.486$.  The p-value is large, so we do not reject
    $H_0$.  No significant difference is detected (the sample is very small).

---

**Exercise 3.**
Censoring Coding Convention

In the implementation above, `censored = 1` means censored and `censored = 0` means
the event was observed.  Many survival analysis packages use the opposite convention
(`event = 1`).

**(a)** Rewrite the core Kaplan-Meier loop using the `event` coding convention.

**(b)** Explain why the coding convention does not affect the mathematical result.

??? success "Solution to Exercise 3"

    **(a)** With `event` indicator (1 = event, 0 = censored):

    ```python
    event_times = times[event == 1]
    unique_events = np.unique(event_times)

    s = 1.0
    for t_j in unique_events:
        n_at_risk = np.sum(times >= t_j)
        d_j = np.sum((times == t_j) & (event == 1))
        s *= (n_at_risk - d_j) / n_at_risk
    ```

    The only change is replacing `censored == 0` with `event == 1`.

    **(b)** The mathematical quantities $n_j$ and $d_j$ are defined in terms of which
    observations are events and which are censored.  Whether we label events as 0 or 1
    is purely a software convention.  As long as the code correctly identifies events
    and censored observations, the computed $\hat{S}(t)$ is identical.

---

**Exercise 4.**
Properties of the Kaplan-Meier Estimator

**(a)** Why is $\hat{S}(t)$ called a "product-limit" estimator?

**(b)** Show that if there is no censoring, the Kaplan-Meier estimator reduces to the
empirical survival function $\hat{S}(t) = (\text{number of } t_i > t) / n$.

??? success "Solution to Exercise 4"

    **(a)** It is called a product-limit estimator because $\hat{S}(t)$ is defined as
    a **product** of conditional survival probabilities $(1 - d_j/n_j)$ over all event
    times up to $t$.  The "limit" refers to the connection to continuous-time survival:
    as the time partition becomes finer, the discrete product approaches the
    continuous survival function.

    **(b)** Without censoring, at each event time $t_{(j)}$ there are $n_j = n - j + 1$
    subjects at risk and $d_j = 1$ event (assuming no ties for simplicity).  Then

    $$
    \hat{S}(t_{(j)}) = \prod_{k=1}^{j} \frac{n - k + 1 - 1}{n - k + 1} = \prod_{k=1}^{j} \frac{n - k}{n - k + 1}
    $$

    This is a telescoping product:

    $$
    \hat{S}(t_{(j)}) = \frac{n-1}{n} \cdot \frac{n-2}{n-1} \cdots \frac{n-j}{n-j+1} = \frac{n - j}{n}
    $$

    Since $n - j$ is the number of subjects with $t_i > t_{(j)}$, we have
    $\hat{S}(t_{(j)}) = (\text{number of } t_i > t_{(j)}) / n$, which is the
    empirical survival function. $\square$

---

**Exercise 5.**
Variance of the Kaplan-Meier Estimator

Greenwood's formula gives the variance of $\hat{S}(t)$:

$$
\widehat{\text{Var}}(\hat{S}(t)) = \hat{S}(t)^2 \sum_{j:\, t_{(j)} \leq t} \frac{d_j}{n_j(n_j - d_j)}
$$

**(a)** For the data in Exercise 1, compute $\widehat{\text{Var}}(\hat{S}(5))$.

**(b)** Construct a 95% confidence interval for $S(5)$ using the log-transformation.

??? success "Solution to Exercise 5"

    **(a)** From Exercise 1, $\hat{S}(5) = 0.583$.  The sum of $d_j / [n_j(n_j - d_j)]$
    over event times up to 5:

    - $t = 1$: $1 / (8 \times 7) = 1/56 = 0.01786$
    - $t = 4$: $1 / (6 \times 5) = 1/30 = 0.03333$
    - $t = 5$: $1 / (5 \times 4) = 1/20 = 0.05000$

    Sum $= 0.01786 + 0.03333 + 0.05000 = 0.10119$.

    $$
    \widehat{\text{Var}}(\hat{S}(5)) = 0.583^2 \times 0.10119 = 0.3399 \times 0.10119 = 0.0344
    $$

    **(b)** The log-transformation confidence interval uses
    $\theta = \ln(-\ln \hat{S}(t))$ with approximate standard error

    $$
    \text{se}(\theta) = \frac{1}{|\ln \hat{S}(t)|} \cdot \frac{\sqrt{\widehat{\text{Var}}(\hat{S}(t))}}{\hat{S}(t)}
    $$

    $\ln \hat{S}(5) = \ln 0.583 = -0.539$.  $|\ln \hat{S}(5)| = 0.539$.

    $$
    \text{se}(\theta) = \frac{1}{0.539} \cdot \frac{\sqrt{0.0344}}{0.583} = \frac{1}{0.539} \cdot \frac{0.1855}{0.583} = 1.855 \times 0.318 = 0.590
    $$

    $\theta = \ln(0.539) = -0.618$.

    CI for $\theta$: $-0.618 \pm 1.96 \times 0.590 = (-1.774, 0.538)$.

    Back-transforming: $S = \exp(-\exp(\theta))$.

    Lower: $\exp(-\exp(-1.774)) = \exp(-0.170) = 0.844$.

    Upper: $\exp(-\exp(0.538)) = \exp(-1.713) = 0.180$.

    The 95% CI for $S(5)$ is approximately $(0.180, 0.844)$.  The interval is wide
    due to the small sample size.
