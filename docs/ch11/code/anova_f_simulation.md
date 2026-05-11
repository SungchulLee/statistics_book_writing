# Analysis of Variance F-Statistic Simulation

## Overview

The one-way ANOVA F-test detects whether at least one group mean differs from the others, but its power depends heavily on the interplay between group-mean separation, within-group variance, and sample size. This page uses Monte-Carlo simulation to build intuition for these relationships by running 1000 replications of the F-test under nine carefully chosen parameter settings and examining how the F-statistic distribution and rejection rate change.

## The One-Way ANOVA F-Statistic

Given $k$ groups with sample sizes $n_1, \dots, n_k$ and total sample size $N = \sum_{i=1}^{k} n_i$, the F-statistic is

$$
F = \frac{\text{MST}}{\text{MSE}} = \frac{\text{SST} / (k - 1)}{\text{SSE} / (N - k)}
$$

where $\text{SST}$ is the between-group sum of squares and $\text{SSE}$ is the within-group sum of squares. Under the null hypothesis $H_0: \mu_1 = \mu_2 = \cdots = \mu_k$, the statistic follows an $F(k-1,\, N-k)$ distribution. We reject $H_0$ at significance level $\alpha$ when

$$
F > F_{\text{crit}} = F_{1-\alpha}(k-1,\, N-k)
$$

## Simulation Design

Each scenario generates three normal groups with specified means $\boldsymbol{\mu}$, standard deviations $\boldsymbol{\sigma}$, and sizes $\boldsymbol{n}$, then runs `scipy.stats.f_oneway` 1000 times. The nine scenarios span a grid of conditions:

| Scenario | Means | Std. Devs. | Sizes |
|---|---|---|---|
| Large separation, equal var | $(3, 6, 9)$ | $(6, 6, 6)$ | $(10, 20, 30)$ |
| Tiny separation, equal var | $(3, 3.1, 2.9)$ | $(6, 6, 6)$ | $(10, 20, 30)$ |
| Tiny separation, unequal var | $(3, 3.1, 2.9)$ | $(6, 12, 18)$ | $(10, 20, 30)$ |
| Large separation, large var | $(3, 6, 9)$ | $(10, 10, 10)$ | $(10, 20, 30)$ |
| Moderate sep., large var, small $n$ | $(3, 5, 6)$ | $(10, 10, 10)$ | $(10, 10, 10)$ |
| Moderate sep., large var, large $n$ | $(3, 5, 6)$ | $(10, 10, 10)$ | $(5000, 5000, 5000)$ |
| No separation, huge var | $(3, 3, 3)$ | $(100, 100, 100)$ | $(10, 10, 10)$ |
| No separation, unequal var | $(3, 3, 3)$ | $(1, 1, 2)$ | $(10, 20, 30)$ |
| No separation, small var | $(3, 3, 3)$ | $(1, 1, 2)$ | $(10, 20, 30)$ |

## Core Simulation Code

The `simulate_f` function draws groups from their respective normal distributions and computes the ANOVA F-statistic for each replication:

```python
import numpy as np
from scipy import stats

def simulate_f(mu, sigma, sizes, n_sim=1000):
    F_vals, p_vals = [], []
    for _ in range(n_sim):
        groups = [stats.norm.rvs(m, s, n)
                  for m, s, n in zip(mu, sigma, sizes)]
        F, p = stats.f_oneway(*groups)
        F_vals.append(F)
        p_vals.append(p)
    return np.array(F_vals), np.array(p_vals)
```

For each scenario the critical value and empirical rejection rate are computed:

```python
df1 = 2                          # k - 1
df2 = sum(sizes) - 3             # N - k
F_crit = stats.f.ppf(0.95, df1, df2)
reject_pct = np.mean(F_vals > F_crit) * 100
```

## Interpretation

The simulation reveals several key principles:

- **Signal-to-noise ratio drives power.** Scenario 1 (large mean separation, moderate variance) yields high rejection rates, whereas Scenario 4 (same separation but larger variance) shows a marked drop in power.
- **Sample size compensates for noise.** Comparing Scenarios 5 and 6 (identical means and variances, but $n = 10$ vs. $n = 5000$) demonstrates that large samples can detect even moderate differences reliably.
- **Null scenarios control Type I error.** Scenarios 7--9 have equal means, so any rejection is a false positive. With $\alpha = 0.05$, the empirical rejection rate should hover near 5%, confirming the test's calibration.
- **Unequal variances distort the test.** Comparing Scenarios 2 and 3 (tiny separation with equal vs. unequal variances) shows that heteroscedasticity can reduce power or inflate the Type I error rate, depending on the relationship between variance and group size.

The p-value histograms provide a complementary view: under the null hypothesis, p-values should be approximately uniform on $[0, 1]$, whereas under the alternative they pile up near zero.

## Exercises

**Exercise 1.**
In a simulation with $k = 3$ groups of sizes $(10, 20, 30)$ and equal means $\mu_i = 5$ with $\sigma_i = 1$, you observe an empirical rejection rate of 4.8% at $\alpha = 0.05$. Is this consistent with a correctly calibrated test? Justify your answer.

??? success "Solution to Exercise 1"
    Under the null hypothesis the expected rejection rate equals $\alpha = 0.05$. With $n_{\text{sim}} = 1000$ replications, the observed proportion follows approximately $\hat{p} \sim N(0.05,\, 0.05 \cdot 0.95 / 1000)$, giving a standard error of $\sqrt{0.0000475} \approx 0.0069$. The observed rate 0.048 is within one standard error of 0.05, so it is entirely consistent with a correctly calibrated test.

---

**Exercise 2.**
Explain intuitively why doubling the within-group standard deviation $\sigma$ has the same effect on the F-statistic as halving the between-group mean separation, assuming sample sizes stay fixed.

??? success "Solution to Exercise 2"
    The F-statistic can be written as

    $$
    F = \frac{\text{MST}}{\text{MSE}}
    $$

    The between-group mean square $\text{MST}$ is proportional to $n_i (\bar{y}_i - \bar{y})^2$, which scales with the square of the mean separation $\delta^2$. The within-group mean square $\text{MSE}$ is proportional to $\sigma^2$. Therefore $F \propto \delta^2 / \sigma^2$. Doubling $\sigma$ divides $F$ by 4, and halving $\delta$ also divides $F$ by 4. Both operations reduce the signal-to-noise ratio by the same factor.

---

**Exercise 3.**
Design a simulation that estimates the minimum sample size per group needed to achieve 80% power for detecting a mean difference of $\delta = 2$ among $k = 3$ groups when $\sigma = 5$. Describe the algorithm in pseudocode.

??? success "Solution to Exercise 3"
    ```
    for n in [10, 20, 30, 50, 100, 200, ...]:
        set mu = [0, delta, 0] = [0, 2, 0]
        set sigma = [5, 5, 5]
        set sizes = [n, n, n]
        run simulate_f(mu, sigma, sizes, n_sim=5000)
        compute rejection_rate = mean(p_vals < 0.05)
        if rejection_rate >= 0.80:
            return n
    ```

    The algorithm iterates over candidate sample sizes, running many replications at each size and computing the empirical power. The first $n$ for which the rejection rate reaches 80% is the estimated minimum sample size. Using $n_{\text{sim}} = 5000$ or more keeps the Monte-Carlo error small.

---

**Exercise 4.**
Suppose the three groups have equal means but standard deviations $(1, 1, 10)$ with sample sizes $(50, 50, 5)$. Would you expect the empirical Type I error rate to be close to the nominal $\alpha = 0.05$? Explain.

??? success "Solution to Exercise 4"
    No. The classical F-test assumes homoscedasticity. When the group with the largest variance ($\sigma = 10$) has the smallest sample size ($n = 5$), the pooled MSE underestimates the variance of that small group's mean. This causes the F-statistic to be inflated on average, producing a liberal test with a Type I error rate well above 0.05. This is an example of the variance-sample-size confounding that makes the standard ANOVA unreliable under heteroscedasticity; Welch's ANOVA would be the appropriate alternative.

---

**Exercise 5.**
Under the null hypothesis, the p-values from the F-test should follow a $\text{Uniform}(0, 1)$ distribution. Describe a formal test you could apply to the 1000 simulated p-values to verify this, and state its hypotheses.

??? success "Solution to Exercise 5"
    The Kolmogorov-Smirnov (KS) goodness-of-fit test is appropriate. The hypotheses are

    $$
    H_0: F_p = \text{Uniform}(0, 1), \qquad H_1: F_p \neq \text{Uniform}(0, 1)
    $$

    where $F_p$ is the distribution of the simulated p-values. In Python this is `scipy.stats.kstest(p_vals, 'uniform')`. The KS statistic measures the maximum discrepancy between the empirical CDF of the p-values and the $\text{Uniform}(0, 1)$ CDF. Failing to reject $H_0$ confirms that the F-test is calibrated at the nominal level under the null scenario. $\square$
