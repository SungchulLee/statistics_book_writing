# Shapiro-Wilk Test

## Overview

The Shapiro-Wilk test is widely considered the most powerful normality test for small to moderate sample sizes. It measures how well the ordered sample values match the expected normal order statistics, producing a statistic $W$ between 0 and 1. Values of $W$ close to 1 indicate consistency with normality, while significantly smaller values lead to rejection. SciPy limits the test to $n \leq 5000$.

## The Test Statistic

Given order statistics $X_{(1)} \leq X_{(2)} \leq \cdots \leq X_{(n)}$, the Shapiro-Wilk statistic is

$$
W = \frac{\left(\sum_{i=1}^{n} a_i\, X_{(i)}\right)^2}{\sum_{i=1}^{n} (X_i - \bar{X})^2},
$$

where the coefficients $a_1, \ldots, a_n$ are derived from the expected values and covariance matrix of the normal order statistics. Specifically, let $m = (m_1, \ldots, m_n)^T$ be the vector of expected standard normal order statistics and $V$ their covariance matrix. Then

$$
a = \frac{V^{-1} m}{(m^T V^{-1} V^{-1} m)^{1/2}}.
$$

## Intuition

The numerator $(\sum a_i X_{(i)})^2$ is a weighted linear combination of the order statistics that, under normality, should closely track the denominator (total sum of squares). If the data are normal, the ordered values align well with the expected normal order statistics, making $W \approx 1$. Departures from normality -- skewness, heavy tails, multimodality -- disrupt this alignment, reducing $W$.

## Hypotheses

$$
H_0: X_1, \ldots, X_n \sim \mathcal{N}(\mu, \sigma^2), \qquad H_1: \text{the data are not normally distributed}.
$$

Small values of $W$ (equivalently, small $p$-values) lead to rejection of $H_0$.

### Code

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(0, 1, size=240),
                    rng.lognormal(0, 0.6, size=60)])

W, p = stats.shapiro(x)
g1 = stats.skew(x, bias=False)
g2 = stats.kurtosis(x, fisher=True, bias=False)

print(f"Sample size n = {x.size}")
print(f"Shapiro-Wilk: W = {W:.4f}, p-value = {p:.4g}")
print(f"Skewness g1 = {g1:.4f}, Excess kurtosis g2 = {g2:.4f}")
if p < 0.05:
    print("=> Reject normality at alpha = 0.05.")
else:
    print("=> Fail to reject normality at alpha = 0.05.")
```

## Strengths and Limitations

**Strengths:**

- Highest power among common normality tests for $n \leq 5000$.
- Sensitive to a wide range of alternatives (skewness, kurtosis, multimodality).
- Well-established critical values and $p$-value approximations.

**Limitations:**

- SciPy warns that $p$-values may be inaccurate for $n > 5000$.
- For very large $n$, even trivial departures from normality lead to rejection; graphical methods and effect sizes become more informative.
- Does not indicate the *type* of departure (pair with Q-Q plots for diagnosis).

## Interpretation

In the mixture example, the lognormal component introduces skewness and heavy tails. The $W$ statistic will be noticeably below 1 and the $p$-value essentially zero. Reporting both $W$ and the auxiliary summaries ($g_1$, $g_2$) helps the reader understand why normality was rejected.

## Exercises

**Exercise 1.** Generate $n = 100$ standard normal observations. Compute $W$ and the $p$-value. Repeat 20 times and report the range of $W$ values.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    W_values = []
    for trial in range(20):
        x = rng.normal(0, 1, size=100)
        W, p = stats.shapiro(x)
        W_values.append(W)

    print(f"W range: [{min(W_values):.4f}, {max(W_values):.4f}]")
    print(f"Mean W: {np.mean(W_values):.4f}")
    ```

    Under normality with $n = 100$, $W$ is typically between 0.97 and 0.99. None of the 20 realisations should be flagged as significant (at $\alpha = 0.05$) in the majority of cases, with at most one or two false rejections expected. $\square$

---

**Exercise 2.** Compare the Shapiro-Wilk $p$-value for $n = 50$ observations from (a) $\mathcal{N}(0,1)$, (b) $t_5$, (c) $\text{Lognormal}(0, 0.5)$, and (d) $\text{Uniform}(0,1)$.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(42)
    distributions = {
        "N(0,1)": rng.normal(0, 1, 50),
        "t(5)": rng.standard_t(5, 50),
        "Lognormal(0,0.5)": rng.lognormal(0, 0.5, 50),
        "Uniform(0,1)": rng.uniform(0, 1, 50),
    }

    for name, x in distributions.items():
        W, p = stats.shapiro(x)
        print(f"{name:>20}: W = {W:.4f}, p = {p:.4g}")
    ```

    The normal sample should have $p > 0.05$. The $t_5$ data may or may not reject depending on the specific draw (moderate departures). The lognormal data should reject decisively (skewness). The uniform data should also reject (platykurtosis), demonstrating the test's sensitivity to varied departure types. $\square$

---

**Exercise 3.** Explain why $W \leq 1$ always holds. Under what conditions does $W = 1$ exactly?

??? success "Solution to Exercise 3"

    The Shapiro-Wilk statistic can be written as the squared correlation between the ordered sample and the expected normal order statistics (up to a constant). By the Cauchy-Schwarz inequality,

    $$
    \left(\sum a_i X_{(i)}\right)^2 \leq \left(\sum a_i^2\right)\left(\sum X_{(i)}^2\right).
    $$

    Since the denominator $\sum(X_i - \bar{X})^2 = \sum X_{(i)}^2 - n\bar{X}^2 \leq \sum X_{(i)}^2$ and the coefficients $a_i$ are normalised so that $\sum a_i^2 = 1$, we have $W \leq 1$.

    Equality $W = 1$ holds when $X_{(i)} = c_1 + c_2\, m_i$ for all $i$ (the ordered sample is a perfect affine function of the expected normal order statistics), which can only happen if the data lie exactly on the theoretical normal quantiles. For continuous data this occurs with probability zero; $W = 1$ is only achieved in degenerate cases. $\square$

---

**Exercise 4.** Run a Monte Carlo study to compare the power of the Shapiro-Wilk test and the KS test (against $\mathcal{N}(0,1)$) for $n = 50$ observations from a $t_5$ distribution.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n, reps, alpha = 50, 5000, 0.05
    rej_sw, rej_ks = 0, 0

    for _ in range(reps):
        x = rng.standard_t(df=5, size=n)
        _, p_sw = stats.shapiro(x)
        _, p_ks = stats.kstest(x, 'norm', args=(0, 1))
        if p_sw < alpha:
            rej_sw += 1
        if p_ks < alpha:
            rej_ks += 1

    print(f"Shapiro-Wilk power: {rej_sw / reps:.4f}")
    print(f"KS power:           {rej_ks / reps:.4f}")
    ```

    The Shapiro-Wilk test should have substantially higher power (often 2--3 times that of the KS test) for this heavy-tailed alternative, confirming its reputation as the most powerful general-purpose normality test. $\square$

---

**Exercise 5.** For large samples, a non-significant Shapiro-Wilk test is nearly impossible when the data deviate even slightly from normality. Argue that in this regime, reporting the $W$ statistic itself (as an effect size) is more informative than the $p$-value.

??? success "Solution to Exercise 5"

    As $n$ grows, the Shapiro-Wilk test becomes increasingly powerful: the variance of $W$ under $H_0$ shrinks, and even a tiny departure from normality (e.g., $\gamma_2 = 0.1$) produces a $W$ that is reliably below the critical value. The $p$-value approaches zero for any non-normal distribution, regardless of how close it is to normal. This makes the $p$-value uninformative for practical purposes: all it says is "the sample is large enough to detect the departure."

    The $W$ statistic itself, however, conveys the *magnitude* of the departure. A value of $W = 0.998$ with $p < 0.001$ indicates that the data are extremely close to normal in shape, even though the departure is statistically significant. Conversely, $W = 0.92$ indicates a more substantial departure. Reporting $W$ alongside $g_1$ and $g_2$ allows the analyst to judge whether the deviation is practically meaningful for their application (e.g., whether a $t$-test will still be approximately valid). $\square$
