# Kaplan-Meier Survival Curves and Log-Rank Test

## Overview

The Kaplan-Meier estimator is the cornerstone non-parametric method for estimating
survival functions from censored data.  Combined with the log-rank test, it provides
a complete toolkit for visualizing survival experience and comparing survival
distributions across groups.  This page walks through a Python implementation of both
techniques, using only NumPy and SciPy, and explains each step of the computation.

## The Kaplan-Meier Estimator

### Mathematical Foundation

Given $n$ subjects with observed times and censoring indicators, the Kaplan-Meier
estimator computes the survival function as a product of conditional survival
probabilities at each distinct event time $t_{(j)}$:

$$
\hat{S}(t) = \prod_{j:\, t_{(j)} \leq t} \left(1 - \frac{d_j}{n_j}\right)
$$

where $d_j$ is the number of events at time $t_{(j)}$ and $n_j$ is the number of
subjects at risk just before $t_{(j)}$.

### Implementation

The following function computes the Kaplan-Meier survival curve from raw data.

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
    t_plot : array   Time points for step-plot.
    s_plot : array   Survival probabilities matching t_plot.
    """
    order = np.argsort(times)
    times = times[order]
    censored = censored[order]

    event_times = times[censored == 0]
    unique_events = np.unique(event_times)

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

**Key steps in the algorithm:**

1. **Sort** the observations by time.
2. **Identify** distinct event times (times where `censored == 0`).
3. **At each event time** $t_j$, compute the risk set size $n_j$ (subjects with
   $t_i \geq t_j$) and the event count $d_j$.
4. **Update** the survival probability: $\hat{S}(t_j) = \hat{S}(t_{j-1}) \times (1 - d_j / n_j)$.
5. **Extend** the curve to the maximum observed time.

!!! note "Censored Observations"

    Censored subjects (those with `censored == 1`) do not trigger a drop in the
    survival curve, but they reduce the risk set at later event times.  This is
    how partial information from incomplete observations is incorporated.

## The Log-Rank Test

### Hypotheses

The log-rank test compares two survival curves:

$$
H_0 : S_1(t) = S_2(t) \quad \text{for all } t \geq 0
$$

$$
H_1 : S_1(t) \neq S_2(t) \quad \text{for some } t \geq 0
$$

### Test Statistic

At each distinct event time $t_{(j)}$ across both groups, let $r_{1j}$ and $r_{2j}$
be the risk set sizes, $d_{1j}$ and $d_{2j}$ the event counts, and
$r_j = r_{1j} + r_{2j}$, $d_j = d_{1j} + d_{2j}$.  The expected events in group 1
under the null are

$$
e_{1j} = d_j \cdot \frac{r_{1j}}{r_j}
$$

The variance contribution at $t_{(j)}$ is

$$
v_j = \frac{r_{1j} \, r_{2j} \, d_j \, (r_j - d_j)}{r_j^2 \, (r_j - 1)}
$$

The test statistic is

$$
\chi^2_{\text{LR}} = \frac{(O_1 - E_1)^2}{V_1} \;\xrightarrow{d}\; \chi^2_1
$$

where $O_1 = \sum d_{1j}$, $E_1 = \sum e_{1j}$, and $V_1 = \sum v_j$.

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

    O1, E1, V = 0.0, 0.0, 0.0

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

The implementation loops over each pooled event time, accumulates the observed events,
expected events, and variance for group 1, then computes the chi-squared statistic.

## Simulation and Visualization

The following code generates simulated survival data for two groups with different
event rates and plots the Kaplan-Meier curves.

```python
import matplotlib.pyplot as plt

np.random.seed(0)

# Group 1: slower event rate (scale = 20)
n1 = 40
times_1 = np.random.exponential(scale=20, size=n1)
censored_1 = (np.random.rand(n1) < 0.2).astype(int)

# Group 2: faster event rate (scale = 12)
n2 = 40
times_2 = np.random.exponential(scale=12, size=n2)
censored_2 = (np.random.rand(n2) < 0.2).astype(int)

# Compute Kaplan-Meier curves
t1, s1 = kaplan_meier(times_1, censored_1)
t2, s2 = kaplan_meier(times_2, censored_2)

# Plot
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
```

Group 1 is drawn from an exponential distribution with mean 20 (slower events),
and group 2 from mean 12 (faster events).  Approximately 20% of observations in each
group are randomly censored.

## Interpretation

- The Kaplan-Meier curve is a **step function** that drops at each observed event time.
  Flat regions correspond to intervals with no events (possibly containing censored
  observations that reduce the risk set).
- A curve that stays higher for longer indicates **better survival** in that group.
- The **log-rank test** provides a formal assessment of whether the visual separation
  between two curves is statistically significant.  A small p-value (e.g., $p < 0.05$)
  leads to rejection of the null hypothesis of equal survival distributions.
- The log-rank test is most powerful when the **proportional hazards** assumption
  holds (i.e., the hazard ratio between groups is roughly constant over time).  When
  survival curves cross, the test may fail to detect differences.

!!! tip "When to Use the Log-Rank Test"

    The log-rank test is appropriate for comparing two or more groups when no
    covariate adjustment is needed.  For multivariable analysis, use the Cox
    proportional hazards model instead.

## Exercises

**Exercise 1.**
Kaplan-Meier Computation

Six patients have the following survival data:

| Subject | Time | Status (0 = event, 1 = censored) |
|:-------:|:----:|:--------------------------------:|
| A | 2 | 0 |
| B | 3 | 1 |
| C | 5 | 0 |
| D | 5 | 0 |
| E | 8 | 1 |
| F | 10 | 0 |

Compute $\hat{S}(t)$ at each event time using the Kaplan-Meier estimator.

??? success "Solution to Exercise 1"

    Distinct event times: $t_{(1)} = 2$, $t_{(2)} = 5$, $t_{(3)} = 10$.

    | $t_{(j)}$ | $n_j$ | $d_j$ | $1 - d_j/n_j$ | $\hat{S}(t_{(j)})$ |
    |:----------:|:-----:|:-----:|:--------------:|:-------------------:|
    | 2 | 6 | 1 | 5/6 = 0.833 | 0.833 |
    | 5 | 4 | 2 | 2/4 = 0.500 | 0.833 $\times$ 0.500 = 0.417 |
    | 10 | 1 | 1 | 0/1 = 0.000 | 0.417 $\times$ 0.000 = 0.000 |

    At $t_{(2)} = 5$: subject A had an event at $t = 2$ and subject B was censored
    at $t = 3$, leaving $n_2 = 4$ at risk.

---

**Exercise 2.**
Log-Rank Test Setup

Two groups are observed:

**Group 1:** times 3, 6+, 9, 15 (+ = censored).

**Group 2:** times 1, 4, 8+, 12.

**(a)** List all distinct event times from both groups.

**(b)** At $t = 1$, compute $e_{11}$ (expected events for group 1) and the variance
contribution $v_1$.

??? success "Solution to Exercise 2"

    **(a)** Event times (excluding censorings): from group 1: 3, 9, 15; from group 2:
    1, 4, 12.  Pooled distinct event times: 1, 3, 4, 9, 12, 15.

    **(b)** At $t = 1$: $r_1 = 4$, $r_2 = 4$, $r = 8$, $d_1 = 0$, $d_2 = 1$,
    $d = 1$.

    $$
    e_{11} = d \cdot \frac{r_1}{r} = 1 \cdot \frac{4}{8} = 0.5
    $$

    $$
    v_1 = \frac{r_1 \cdot r_2 \cdot d \cdot (r - d)}{r^2 (r - 1)} = \frac{4 \cdot 4 \cdot 1 \cdot 7}{64 \cdot 7} = \frac{112}{448} = 0.25
    $$

---

**Exercise 3.**
Censoring Effects

Suppose 10 subjects all have the same underlying exponential distribution with rate
$\lambda = 0.1$.  In scenario A, none are censored.  In scenario B, 5 subjects are
censored at time $t = 5$.

**(a)** Explain qualitatively how the Kaplan-Meier curves will differ between the two
scenarios.

**(b)** Which scenario will produce wider confidence bands at later time points?  Why?

??? success "Solution to Exercise 3"

    **(a)** In scenario A, all 10 subjects contribute events, so the Kaplan-Meier
    curve is based on complete information and will step down steadily to zero.
    In scenario B, the 5 subjects censored at $t = 5$ leave the risk set at that
    point.  The curve will be identical to scenario A up to $t = 5$, but after
    $t = 5$ the risk set is smaller (only the 5 uncensored subjects remain), so
    each event causes a larger drop in the survival estimate.

    **(b)** Scenario B will have wider confidence bands after $t = 5$.  The variance
    of the Kaplan-Meier estimator (via Greenwood's formula) increases as the risk
    set shrinks.  With only 5 subjects at risk instead of the original 10, each
    event contributes more uncertainty to the estimate.

---

**Exercise 4.**
Log-Rank Test Interpretation

A log-rank test comparing a treatment group to a control group yields
$\chi^2_{\text{LR}} = 5.23$ with $p = 0.022$.

**(a)** State the conclusion at the $\alpha = 0.05$ significance level.

**(b)** If the observed events in the treatment group were $O_1 = 15$ and the expected
events were $E_1 = 21.3$, interpret the direction of the effect.

**(c)** What assumption of the log-rank test, if violated, could make this result
misleading?

??? success "Solution to Exercise 4"

    **(a)** Since $p = 0.022 < 0.05$, we reject $H_0$ at the 5% level and conclude
    that the survival distributions of the treatment and control groups are
    significantly different.

    **(b)** $O_1 = 15 < E_1 = 21.3$: the treatment group experienced fewer events
    than expected under the null hypothesis.  This indicates that the treatment group
    has better survival (lower hazard) compared to the control group.

    **(c)** The log-rank test assumes **proportional hazards** --- the hazard ratio
    between groups is constant over time.  If the survival curves cross (e.g., the
    treatment is beneficial early but harmful late), the log-rank test may fail to
    detect the difference or may give a misleading summary.

---

**Exercise 5.**
Deriving the Variance of the Log-Rank Statistic

Show that the variance of $d_{1j}$ (events in group 1 at time $t_{(j)}$) under the
null hypothesis follows a hypergeometric distribution, and derive the expression

$$
v_j = \frac{r_{1j} \, r_{2j} \, d_j \, (r_j - d_j)}{r_j^2 \, (r_j - 1)}
$$

??? success "Solution to Exercise 5"

    Under $H_0$, at time $t_{(j)}$ there are $r_j$ subjects at risk, of which
    $r_{1j}$ belong to group 1.  Among these $r_j$ subjects, $d_j$ events occur.
    The number of events in group 1, $d_{1j}$, follows a hypergeometric distribution:
    drawing $d_j$ "events" from a pool of $r_j$ subjects containing $r_{1j}$ from
    group 1.

    The variance of a hypergeometric random variable $X \sim \text{Hyper}(N, K, n)$ is

    $$
    \text{Var}(X) = n \cdot \frac{K}{N} \cdot \frac{N - K}{N} \cdot \frac{N - n}{N - 1}
    $$

    Substituting $N = r_j$, $K = r_{1j}$, $n = d_j$:

    $$
    v_j = d_j \cdot \frac{r_{1j}}{r_j} \cdot \frac{r_j - r_{1j}}{r_j} \cdot \frac{r_j - d_j}{r_j - 1}
    $$

    Since $r_j - r_{1j} = r_{2j}$:

    $$
    v_j = \frac{r_{1j} \, r_{2j} \, d_j \, (r_j - d_j)}{r_j^2 \, (r_j - 1)}
    $$

    $\square$
