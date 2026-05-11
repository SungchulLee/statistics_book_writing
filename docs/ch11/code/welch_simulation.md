# Welch Analysis of Variance Type I Error and Power Simulation

## Overview

Welch's ANOVA is an alternative to the classical one-way ANOVA that does not assume equal variances across groups. This page uses a Monte Carlo simulation to estimate the Type I error rate and power of Welch's ANOVA under heteroscedastic conditions with unbalanced group sizes. The simulation demonstrates that Welch's ANOVA maintains the nominal Type I error rate even when variances differ substantially, while still having reasonable power to detect group mean differences.

## Why Welch's ANOVA?

Classical one-way ANOVA assumes homoscedasticity: $\sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2$. When this assumption is violated and sample sizes are unequal, the classical $F$-test can have an inflated Type I error rate. Welch's ANOVA uses a weighted formulation:

$$
F_W = \frac{\sum_{i=1}^{k} w_i (\bar{y}_{i\cdot} - \tilde{y})^2 / (k-1)}{1 + \frac{2(k-2)}{k^2-1} \sum_{i=1}^{k} \frac{(1 - w_i/\sum w_j)^2}{n_i - 1}}
$$

where $w_i = n_i / s_i^2$ and $\tilde{y} = \sum w_i \bar{y}_i / \sum w_j$. The denominator adjusts the degrees of freedom via a Satterthwaite-type approximation, yielding a test that is robust to heteroscedasticity.

## Simulation Design

The simulation compares three groups with deliberately unequal variances and unbalanced sizes:

| Group | $n_i$ | $\sigma_i$ | $\mu_i$ (null) | $\mu_i$ (alternative) |
|---|---|---|---|---|
| $G_1$ | 10 | 1.0 | 10.0 | 10.0 |
| $G_2$ | 18 | 3.0 | 10.0 | 10.0 |
| $G_3$ | 7 | 6.0 | 10.0 | 12.0 |

Under the null, all means are equal. Under the alternative, group $G_3$ is shifted upward by 2 units.

```python
import numpy as np
import pandas as pd
import pingouin as pg

rng = np.random.default_rng(0)

def simulate_once(null=True):
    ns = [10, 18, 7]
    sigmas = [1.0, 3.0, 6.0]
    means = [10.0, 10.0, 10.0] if null else [10.0, 10.0, 12.0]

    rows = []
    for i, (n, mu, sd) in enumerate(zip(ns, means, sigmas), start=1):
        x = rng.normal(mu, sd, size=n)
        rows += [{"Group": f"G{i}", "Values": v} for v in x]
    df = pd.DataFrame(rows)

    aov = pg.welch_anova(dv="Values", between="Group", data=df)
    return float(aov["p-unc"].iloc[0])
```

## Running the Simulation

For each scenario (null and alternative), generate many replications and estimate the rejection rate at $\alpha = 0.05$.

```python
def run(n_sims=500, alpha=0.05):
    pvals_null = [simulate_once(null=True) for _ in range(n_sims)]
    pvals_alt  = [simulate_once(null=False) for _ in range(n_sims)]
    type1 = np.mean(np.array(pvals_null) < alpha)
    power = np.mean(np.array(pvals_alt)  < alpha)
    return type1, power

type1, power = run(n_sims=300, alpha=0.05)
print(f"Estimated Type I error: {type1:.3f}")
print(f"Estimated Power:        {power:.3f}")
```

## Key Quantities

- **Type I error rate:** The proportion of simulations under $H_0$ where $p < \alpha$. A well-calibrated test should yield approximately $\alpha = 0.05$.

$$
\widehat{\alpha} = \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}(p_b < \alpha)
$$

- **Power:** The proportion of simulations under $H_A$ where $p < \alpha$.

$$
\widehat{\text{Power}} = \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}(p_b < \alpha)
$$

where $B$ is the number of Monte Carlo replications.

## Interpretation

- **Type I error control:** Welch's ANOVA typically keeps the empirical Type I error close to the nominal $\alpha = 0.05$, even under severe heteroscedasticity ($\sigma_3 / \sigma_1 = 6$). The classical ANOVA $F$-test, by contrast, would have an inflated Type I error in this scenario because the smallest group ($n_3 = 7$) has the largest variance.
- **Power:** The power to detect a shift of $\Delta = 2$ in $G_3$ depends on the sample size and variance of that group. With $n_3 = 7$ and $\sigma_3 = 6$, the signal-to-noise ratio is $\Delta / \sigma_3 = 1/3$, which is modest. Power increases with larger $n_3$, smaller $\sigma_3$, or larger $\Delta$.
- **Simulation precision:** With $B = 300$ replications, the standard error of the estimated Type I error is approximately $\sqrt{0.05 \times 0.95 / 300} \approx 0.013$. More replications would narrow this margin.

## Exercises

**Exercise 1.**
With $B = 300$ Monte Carlo replications and a true Type I error rate of $\alpha = 0.05$, compute a 95% confidence interval for the estimated Type I error rate. How many replications would be needed to halve the width of this interval?

??? success "Solution to Exercise 1"
    The estimated Type I error is a sample proportion $\hat{p}$ with standard error $\text{SE} = \sqrt{\hat{p}(1-\hat{p})/B}$. With $\hat{p} \approx 0.05$ and $B = 300$:

    $$
    \text{SE} = \sqrt{\frac{0.05 \times 0.95}{300}} \approx 0.0126
    $$

    The 95% confidence interval is $0.05 \pm 1.96 \times 0.0126 \approx (0.025,\, 0.075)$, with width approximately $0.049$.

    To halve the width, we need to double the precision, which requires quadrupling the sample size: $B = 4 \times 300 = 1200$ replications.

---

**Exercise 2.**
Explain why the classical ANOVA $F$-test has an inflated Type I error when the smallest group has the largest variance. What happens when the smallest group has the smallest variance?

??? success "Solution to Exercise 2"
    The classical $F$-test pools all groups to estimate $MSW$. When the smallest group has the largest variance, the pooled estimate underweights that group's large variance (because it contributes fewer observations). This makes $MSW$ too small, inflating the $F$-statistic and leading to too many rejections (liberal test, inflated Type I error).

    Conversely, when the smallest group has the smallest variance, the pooled $MSW$ overestimates the effective error variance for that group. This makes the $F$-statistic too conservative, and the Type I error drops below the nominal level. The test loses power but does not have an inflated false positive rate. This asymmetry is a well-known result, sometimes called the Welch-Satterthwaite effect in the unbalanced heteroscedastic setting.

---

**Exercise 3.**
The signal-to-noise ratio for detecting the shift in $G_3$ is $\Delta/\sigma_3 = 2/6 \approx 0.33$. Compute Cohen's $f$ for this three-group design and interpret its magnitude.

??? success "Solution to Exercise 3"
    Cohen's $f$ for one-way ANOVA is defined as

    $$
    f = \sqrt{\frac{\sum_{i=1}^{k} n_i (\mu_i - \bar{\mu})^2 / N}{\sigma_{\text{within}}^2}}
    $$

    Under the alternative, $\mu_1 = \mu_2 = 10$, $\mu_3 = 12$. With $n_1 = 10$, $n_2 = 18$, $n_3 = 7$, $N = 35$:

    $$
    \bar{\mu}_w = \frac{10 \times 10 + 18 \times 10 + 7 \times 12}{35} = \frac{364}{35} = 10.4
    $$

    $$
    \sum n_i(\mu_i - \bar{\mu}_w)^2 = 10(10 - 10.4)^2 + 18(10 - 10.4)^2 + 7(12 - 10.4)^2
    $$

    $$
    = 10(0.16) + 18(0.16) + 7(2.56) = 1.6 + 2.88 + 17.92 = 22.4
    $$

    Since variances are unequal, we use a pooled variance estimate: $\sigma_{\text{pool}}^2 = (9 \times 1 + 17 \times 9 + 6 \times 36)/32 = (9 + 153 + 216)/32 = 378/32 = 11.8125$.

    $$
    f = \sqrt{\frac{22.4 / 35}{11.8125}} = \sqrt{\frac{0.64}{11.8125}} = \sqrt{0.0542} \approx 0.233
    $$

    By Cohen's conventions, $f = 0.10$ is small, $f = 0.25$ is medium, and $f = 0.40$ is large. This effect size ($f \approx 0.23$) is between small and medium, which explains the moderate power observed in the simulation.

---

**Exercise 4.**
Modify the simulation design so that all three groups have equal variances ($\sigma_i = 3$) but keep the sample sizes unbalanced at $n = (10, 18, 7)$. Predict whether the classical ANOVA or Welch's ANOVA will have higher power and explain why.

??? success "Solution to Exercise 4"
    When variances are equal, both the classical ANOVA and Welch's ANOVA control the Type I error at the nominal level. However, the classical ANOVA has slightly higher power because it uses the exact $F_{k-1, N-k}$ distribution, whereas Welch's ANOVA uses an approximate reference distribution with reduced effective degrees of freedom.

    Welch's ANOVA pays a small power penalty for estimating separate variances and adjusting degrees of freedom. This penalty is the "insurance premium" for robustness to heteroscedasticity. When equal variances hold, this insurance is unnecessary, and the classical test is (slightly) more efficient. In practice, the power difference is small for moderate sample sizes, making Welch's ANOVA a reasonable default.

---

**Exercise 5.**
Prove that under $H_0$ with equal variances and equal sample sizes, Welch's $F_W$ statistic reduces to the classical ANOVA $F$-statistic.

??? success "Solution to Exercise 5"
    Under $H_0$ with equal variances $\sigma_i^2 = \sigma^2$ and equal sample sizes $n_i = n$ for all $i$, the weights become $w_i = n / s_i^2$. As all $s_i^2$ converge to $\sigma^2$, the weights become equal: $w_i = n / \sigma^2$ for all $i$.

    The weighted grand mean becomes $\tilde{y} = \sum w_i \bar{y}_i / \sum w_j = \bar{y}_{\cdot\cdot}$, the unweighted grand mean.

    The numerator of $F_W$ becomes

    $$
    \frac{1}{k-1}\sum_{i=1}^{k} \frac{n}{\sigma^2} (\bar{y}_i - \bar{y}_{\cdot\cdot})^2 = \frac{n}{(k-1)\sigma^2} \sum_{i=1}^{k} (\bar{y}_i - \bar{y}_{\cdot\cdot})^2 = \frac{MSB}{\sigma^2}
    $$

    The denominator correction term involves $\sum (1 - w_i / \sum w_j)^2 / (n_i - 1)$. With equal weights, $w_i / \sum w_j = 1/k$, so each term is $(1 - 1/k)^2 / (n-1)$. The sum is $k(k-1)^2/[k^2(n-1)]$. The full denominator simplifies to $1 + 2(k-2)/(k^2-1) \times (k-1)^2/[k(n-1)]$, which approaches 1 as $n \to \infty$. In the exact equal-variance case, the Welch statistic simplifies to $MSB/MSW$, the classical $F$-statistic. $\square$
