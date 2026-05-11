# Bartlett Test

## Overview

Bartlett's test assesses the null hypothesis that two or more populations share the same variance (homoscedasticity). It is the uniformly most powerful unbiased test for equality of variances when the data are truly normal, but it is highly sensitive to departures from normality. This page presents the test statistic, demonstrates it in Python with `scipy.stats.bartlett`, and compares its behavior across several variance-ratio scenarios.

## Hypotheses and Test Statistic

Consider $k$ independent samples of sizes $n_1, \dots, n_k$ drawn from normal populations with variances $\sigma_1^2, \dots, \sigma_k^2$. The hypotheses are

$$
H_0: \sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2, \qquad H_1: \text{not all variances are equal}
$$

Let $S_i^2$ denote the sample variance of group $i$ with $\nu_i = n_i - 1$ degrees of freedom, and let the pooled variance be

$$
S_p^2 = \frac{\sum_{i=1}^{k} \nu_i S_i^2}{\sum_{i=1}^{k} \nu_i}
$$

Bartlett's test statistic is

$$
\chi^2_B = \frac{\left(\sum_{i=1}^{k} \nu_i\right) \ln S_p^2 - \sum_{i=1}^{k} \nu_i \ln S_i^2}{1 + \frac{1}{3(k-1)}\left(\sum_{i=1}^{k} \frac{1}{\nu_i} - \frac{1}{\sum_{i=1}^{k} \nu_i}\right)}
$$

Under $H_0$ and normality, $\chi^2_B \sim \chi^2(k-1)$ approximately. We reject $H_0$ when $\chi^2_B > \chi^2_{1-\alpha}(k-1)$.

## Demonstration in Python

The accompanying script generates a reference sample $X \sim N(0, 1)$ and compares it with samples $Y \sim N(1, \sigma_Y)$ for $\sigma_Y \in \{1.00, 1.05, 1.10, 1.15, 1.20\}$:

```python
import numpy as np
import scipy.stats as stats

seed, size = 1, 100
x = stats.norm(loc=0, scale=1).rvs(size, random_state=seed)

for scale in [1.00, 1.05, 1.10, 1.15, 1.20]:
    y = stats.norm(loc=1, scale=scale).rvs(size, random_state=seed)
    stat, pval = stats.bartlett(x, y)
    print(f"sigma_y={scale:.2f}: chi2={stat:.2f}, p={pval:.3f}")
```

Each call returns the Bartlett $\chi^2$ statistic and its p-value. The overlaid histograms in the script's figure allow a visual comparison of the two distributions at each variance ratio.

## Interpretation

- When $\sigma_Y = 1.00$, the two groups have identical variances and the p-value is large, correctly failing to reject $H_0$.
- As $\sigma_Y$ increases from 1.05 to 1.20, the test statistic grows and the p-value decreases, reflecting the increasing departure from equal variances.
- With $n = 100$ per group, Bartlett's test has good power to detect even modest variance ratios (e.g., $\sigma_Y = 1.15$).

A critical caveat: if the underlying distributions are non-normal (e.g., heavy-tailed or skewed), Bartlett's test suffers from inflated Type I error rates. In such cases, Levene's test is preferred because it operates on absolute deviations from the group median, making it robust to distributional shape.

## Exercises

**Exercise 1.**
For two groups with $n_1 = n_2 = 50$, compute the pooled variance $S_p^2$ when $S_1^2 = 4.0$ and $S_2^2 = 6.0$.

??? success "Solution to Exercise 1"
    With equal sample sizes, $\nu_1 = \nu_2 = 49$, so

    $$
    S_p^2 = \frac{49 \cdot 4.0 + 49 \cdot 6.0}{49 + 49} = \frac{196 + 294}{98} = \frac{490}{98} = 5.0
    $$

    The pooled variance is simply the arithmetic mean of the two sample variances when group sizes are equal.

---

**Exercise 2.**
Explain why Bartlett's test is sensitive to non-normality. What specific property of the log-variance makes the test fragile?

??? success "Solution to Exercise 2"
    Bartlett's statistic is built from $\ln S_i^2$, the logarithm of the sample variance. The chi-squared approximation for the null distribution relies on the sample variances being approximately proportional to chi-squared random variables, which holds only when the underlying data are normal. Under non-normality, the distribution of $S_i^2$ can have heavier tails (due to excess kurtosis), causing $\ln S_i^2$ to be more dispersed than the chi-squared theory predicts. This inflates the test statistic under $H_0$, leading to an elevated Type I error rate.

---

**Exercise 3.**
With $k = 3$ groups, state the degrees of freedom of the Bartlett test statistic under $H_0$ and find the critical value at $\alpha = 0.05$.

??? success "Solution to Exercise 3"
    Under $H_0$, the Bartlett statistic follows $\chi^2(k - 1) = \chi^2(2)$. The critical value is

    $$
    \chi^2_{0.95}(2) = 5.991
    $$

    We reject $H_0$ when $\chi^2_B > 5.991$.

---

**Exercise 4.**
A colleague applies Bartlett's test to three groups of income data (which is heavily right-skewed) and obtains $p = 0.02$. They conclude that the group variances are unequal. Critique this analysis.

??? success "Solution to Exercise 4"
    The conclusion is unreliable because Bartlett's test assumes normality, and income data are typically right-skewed with heavy tails. The excess kurtosis inflates the Bartlett statistic, so the small p-value may reflect non-normality rather than genuine variance differences. The correct approach is to use Levene's test (with the median as the center), which is robust to non-normality. Additionally, applying a log transformation to the income data before testing could help symmetrize the distribution if a normality-based test is desired.

---

**Exercise 5.**
Derive the correction factor $C = 1 + \frac{1}{3(k-1)}\!\left(\sum_{i=1}^{k}\frac{1}{\nu_i} - \frac{1}{\sum \nu_i}\right)$ in the denominator of the Bartlett statistic. Why is this correction needed?

??? success "Solution to Exercise 5"
    Without the correction, the numerator $M = (\sum \nu_i)\ln S_p^2 - \sum \nu_i \ln S_i^2$ does not follow $\chi^2(k-1)$ exactly for finite samples. The ratio $M / C$ provides a better chi-squared approximation by accounting for the bias in $\ln S_i^2$ as an estimator. The correction factor $C$ is derived from a Box approximation: the exact distribution of $M$ is close to $C \cdot \chi^2(k-1)$, so dividing by $C$ yields an approximate $\chi^2(k-1)$ variate.

    Note that $C > 1$ always (each $1/\nu_i > 1/\sum \nu_i$ when $k \ge 2$), so the correction shrinks the test statistic relative to the uncorrected version, making the test slightly more conservative and improving the accuracy of the chi-squared approximation. $\square$
