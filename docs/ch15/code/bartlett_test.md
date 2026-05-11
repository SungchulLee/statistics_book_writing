# Bartlett's Test

## Overview

Bartlett's test checks the null hypothesis that multiple groups share the same variance (homoscedasticity). Among classical tests for equal variances, Bartlett's test is uniformly most powerful when the data are truly normally distributed. However, it is highly sensitive to departures from normality, which means it can produce inflated false-positive rates when applied to skewed or heavy-tailed data.

## Test Setup

Given $k$ independent groups of sizes $n_1, \ldots, n_k$ drawn from normal populations, the hypotheses are

$$
H_0 : \sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2 \quad \text{versus} \quad H_1 : \text{not all } \sigma_i^2 \text{ are equal}.
$$

## Test Statistic

Let $S_i^2$ denote the sample variance of group $i$, let $N = \sum_{i=1}^k n_i$, and define the pooled variance

$$
S_p^2 = \frac{1}{N - k} \sum_{i=1}^{k} (n_i - 1) S_i^2.
$$

Bartlett's test statistic is

$$
T = \frac{(N - k)\ln S_p^2 - \sum_{i=1}^{k}(n_i - 1)\ln S_i^2}{1 + \frac{1}{3(k-1)}\left(\sum_{i=1}^{k}\frac{1}{n_i - 1} - \frac{1}{N - k}\right)}.
$$

Under $H_0$ and normality, $T \sim \chi^2(k-1)$ approximately.

## Code

SciPy provides `scipy.stats.bartlett` directly:

```python
import numpy as np
import scipy.stats as stats

size, seed = 100, 1
x = stats.norm(loc=0, scale=1).rvs(size, random_state=seed)

for scale in [1.00, 1.05, 1.10, 1.15, 1.20]:
    y = stats.norm(loc=1, scale=scale).rvs(size, random_state=seed)
    stat, pval = stats.bartlett(x, y)
    print(f"sigma_y={scale:.2f}  chi2={stat:.2f}  p={pval:.3f}")
```

## Interpretation

- When both groups have the same variance, the test statistic is close to zero and the $p$-value is large.
- As the variance ratio departs from unity, $T$ increases and the $p$-value decreases.
- Bartlett's test should only be applied when the normality assumption is well justified. Under non-normality, Levene's or the Brown--Forsythe test are safer choices.

## Exercises

**Exercise 1.** Three groups of sizes $n_1 = 10$, $n_2 = 12$, $n_3 = 15$ have sample variances $S_1^2 = 4.1$, $S_2^2 = 3.8$, $S_3^2 = 5.6$. Compute the pooled variance $S_p^2$ and the Bartlett test statistic $T$ by hand (you may use a calculator for logarithms).

??? success "Solution to Exercise 1"

    The total sample size is $N = 10 + 12 + 15 = 37$ and $k = 3$. The pooled variance is

    $$
    S_p^2 = \frac{9(4.1) + 11(3.8) + 14(5.6)}{37 - 3} = \frac{36.9 + 41.8 + 78.4}{34} = \frac{157.1}{34} \approx 4.621.
    $$

    The numerator of $T$ is

    $$
    34 \ln(4.621) - [9\ln(4.1) + 11\ln(3.8) + 14\ln(5.6)] \approx 34(1.531) - [9(1.411) + 11(1.335) + 14(1.723)]
    $$

    $$
    \approx 52.054 - [12.699 + 14.685 + 24.122] = 52.054 - 51.506 = 0.548.
    $$

    The correction factor is

    $$
    C = 1 + \frac{1}{6}\left(\frac{1}{9} + \frac{1}{11} + \frac{1}{14} - \frac{1}{34}\right) \approx 1 + \frac{1}{6}(0.111 + 0.091 + 0.071 - 0.029) \approx 1.041.
    $$

    Therefore $T \approx 0.548 / 1.041 \approx 0.527$. Compared against $\chi^2(2)$ this yields a large $p$-value, so we fail to reject equal variances.

---

**Exercise 2.** Using Python, generate three groups of size 50 from $N(0,1)$, $N(0,1.5)$, and $N(0,2)$ respectively. Apply Bartlett's test and report the test statistic and $p$-value. Is the result consistent with what you expect?

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    import scipy.stats as stats

    rng = np.random.default_rng(42)
    g1 = rng.normal(0, 1.0, 50)
    g2 = rng.normal(0, 1.5, 50)
    g3 = rng.normal(0, 2.0, 50)

    stat, pval = stats.bartlett(g1, g2, g3)
    print(f"Bartlett: chi2={stat:.3f}, p={pval:.6f}")
    ```

    Because the true variances are 1, 2.25, and 4 (substantially different), the $p$-value should be very small, correctly rejecting $H_0$.

---

**Exercise 3.** Explain why Bartlett's test is sensitive to non-normality. Specifically, discuss how kurtosis affects the distribution of $S^2$ and therefore the test statistic.

??? success "Solution to Exercise 3"

    The derivation of Bartlett's statistic assumes that $(n_i - 1)S_i^2/\sigma_i^2 \sim \chi^2(n_i - 1)$, which holds exactly only when the data are normal. For distributions with excess kurtosis $\kappa > 0$ (heavy tails), the variance of $S^2$ is inflated:

    $$
    \operatorname{Var}(S^2) = \frac{2\sigma^4}{n-1} + \frac{\kappa \sigma^4}{n}.
    $$

    The extra term $\kappa\sigma^4/n$ makes $S_i^2$ more variable than the $\chi^2$ reference distribution predicts. Consequently, the between-group variability in $\ln S_i^2$ is inflated, causing $T$ to be stochastically larger than $\chi^2(k-1)$ under $H_0$. This leads to rejection rates well above the nominal $\alpha$. $\square$

---

**Exercise 4.** Run a simulation with 5000 replications: generate three groups of size 20 from a standard lognormal distribution (equal variances under the null). Apply Bartlett's test at $\alpha = 0.05$ and estimate the false-positive rate. Compare with Levene's test.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    import scipy.stats as stats

    rng = np.random.default_rng(0)
    n_sims, n, alpha = 5000, 20, 0.05
    rej_bart, rej_lev = 0, 0

    for _ in range(n_sims):
        g1 = rng.lognormal(0, 1, n)
        g2 = rng.lognormal(0, 1, n)
        g3 = rng.lognormal(0, 1, n)
        _, p_b = stats.bartlett(g1, g2, g3)
        _, p_l = stats.levene(g1, g2, g3)
        if p_b < alpha:
            rej_bart += 1
        if p_l < alpha:
            rej_lev += 1

    print(f"Bartlett false-positive rate: {rej_bart/n_sims:.3f}")
    print(f"Levene   false-positive rate: {rej_lev/n_sims:.3f}")
    ```

    Bartlett's false-positive rate will be far above 0.05 (often 0.20+), while Levene's rate will be much closer to the nominal level, illustrating Bartlett's sensitivity to non-normality.

---

**Exercise 5.** Prove that when $k = 2$ and $n_1 = n_2 = n$, Bartlett's test statistic is a monotone function of the F-statistic $F = S_1^2/S_2^2$. That is, show that the two tests always agree on whether to reject $H_0$.

??? success "Solution to Exercise 5"

    With $k = 2$ and equal sample sizes, $S_p^2 = (S_1^2 + S_2^2)/2$. Writing $r = S_1^2/S_2^2$, we have $S_p^2 = S_2^2(1 + r)/2$. The numerator of $T$ becomes

    $$
    2(n-1)\ln\frac{S_p^2}{\sqrt{S_1^2 S_2^2}} \cdot (\text{factor}) = (n-1)\bigl[2\ln S_p^2 - \ln S_1^2 - \ln S_2^2\bigr].
    $$

    Substituting, this equals $(n-1)[2\ln((1+r)/2) - \ln r]$, which is a function of $r = F$ alone. Since this function is strictly convex with a unique minimum at $r = 1$ (where $T = 0$), $T$ increases as $F$ moves away from 1 in either direction. Therefore the rejection region $\{T > \chi^2_{1-\alpha}(1)\}$ maps to $\{F < c_L\} \cup \{F > c_U\}$ for some constants $c_L < 1 < c_U$, matching the two-sided F-test rejection region. The two tests always agree. $\square$
