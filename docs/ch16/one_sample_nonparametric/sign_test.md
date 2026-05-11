# Sign Test

The **sign test** is one of the oldest and simplest non-parametric procedures. It tests whether the median of a population equals a hypothesized value $m_0$ by counting how many observations fall above versus below $m_0$. Because it uses only the *signs* of the deviations $X_i - m_0$ and ignores their magnitudes, the sign test requires almost no assumptions -- making it applicable even to ordinal data or situations where only the direction of change is reliably measured.

## Intuition

If the true median is $m_0$, then by definition half the population lies above $m_0$ and half lies below. Each observation is therefore equally likely to be above or below, so the number of observations exceeding $m_0$ follows a binomial distribution with success probability $p = 0.5$. A large excess of positive or negative signs provides evidence that the true median differs from $m_0$.

## Hypotheses

Let $p = P(X_i > m_0)$. Under $H_0$, the median equals $m_0$, which implies $p = 0.5$.

| Test type | $H_0$ | $H_a$ |
|:----------|:------|:------|
| Two-sided | $p = 0.5$ | $p \ne 0.5$ |
| Left-tailed | $p = 0.5$ | $p < 0.5$ |
| Right-tailed | $p = 0.5$ | $p > 0.5$ |

## Handling Ties

Observations exactly equal to $m_0$ (i.e., $X_i = m_0$) are **excluded** from the analysis. Only the remaining $n = n_+ + n_-$ observations are used, where $n_+$ counts the values above $m_0$ and $n_-$ counts those below.

## Test Statistic

### Exact Test (Small Samples)

The test statistic is

$$
S = n_+
$$

the number of positive signs. Under $H_0$,

$$
S \sim \text{Binomial}(n, 0.5)
$$

The exact two-sided $p$-value is

$$
p = 2 \min\!\bigl(P(S \le s_{\text{obs}}), \; P(S \ge s_{\text{obs}})\bigr)
$$

where $s_{\text{obs}}$ is the observed value of $S$.

### Normal Approximation (Large Samples)

For large $n$ (typically $n \ge 20$), the standardized statistic

$$
Z = \frac{S - n/2}{\sqrt{n/4}} = \frac{n_+ - n_-}{\sqrt{n}}
$$

is approximately $\mathcal{N}(0, 1)$ under $H_0$.

!!! note "Continuity correction"
    A continuity correction can improve the normal approximation for moderate sample sizes. The corrected statistic is $Z = (S - n/2 \pm 0.5) / \sqrt{n/4}$, where the sign of the correction is chosen to bring $S$ closer to $n/2$.

## Worked Example

A nutritionist hypothesizes that the median daily caloric intake of a certain population is 2000 calories. A random sample of $n = 14$ individuals yields:

$$
1850, \; 2100, \; 1950, \; 2200, \; 1900, \; 2050, \; 2000, \; 1800, \; 2150, \; 1975, \; 2300, \; 1920, \; 2080, \; 2010
$$

**Step 1.** Compute signs relative to $m_0 = 2000$:

| Value | Sign |
|:-----:|:----:|
| 1850 | $-$ |
| 2100 | $+$ |
| 1950 | $-$ |
| 2200 | $+$ |
| 1900 | $-$ |
| 2050 | $+$ |
| 2000 | (tie) |
| 1800 | $-$ |
| 2150 | $+$ |
| 1975 | $-$ |
| 2300 | $+$ |
| 1920 | $-$ |
| 2080 | $+$ |
| 2010 | $+$ |

**Step 2.** Exclude the tie: $n = 13$, $n_+ = 7$, $n_- = 6$.

**Step 3.** Compute the exact two-sided $p$-value. Under $H_0$, $S \sim \text{Binomial}(13, 0.5)$.

$$
P(S \ge 7) = \sum_{k=7}^{13} \binom{13}{k} \left(\frac{1}{2}\right)^{13} \approx 0.500
$$

$$
p = 2 \times 0.500 = 1.000
$$

Since $p = 1.000 \gg 0.05$, we fail to reject $H_0$. The data are consistent with a median caloric intake of 2000.

??? example "A significant result"
    Suppose instead we observed $n_+ = 12$ and $n_- = 1$ out of $n = 13$. Then $P(S \ge 12) = \binom{13}{12}(0.5)^{13} + \binom{13}{13}(0.5)^{13} = 14/8192 \approx 0.0017$. The two-sided $p$-value is $2 \times 0.0017 = 0.0034$, providing strong evidence against $H_0$.

## Relationship to the Binomial Test

The sign test is a special case of the **binomial test** (covered in [Binomial Test](binomial_test.md)) with $p_0 = 0.5$. More generally, the binomial test can test any hypothesized proportion, while the sign test is restricted to the median hypothesis $p_0 = 0.5$.

## Power and Limitations

The sign test has the lowest power among the common one-sample non-parametric tests because it discards all magnitude information. Its ARE relative to the one-sample $t$-test under normality is

$$
\text{ARE}(\text{sign test}, \; t\text{-test}) = \frac{2}{\pi} \approx 0.637
$$

This means the sign test requires roughly $\pi/2 \approx 1.57$ times as many observations as the $t$-test to achieve the same power under normal data. However, the sign test remains valuable when:

- Data are ordinal and magnitudes are not meaningful.
- The distribution of differences is highly asymmetric, violating the symmetry assumption of the Wilcoxon signed-rank test.
- Only the direction of change (better/worse) can be reliably determined.

## Summary

The sign test converts a median hypothesis into a binomial proportion test by encoding each observation as above ($+$) or below ($-$) the hypothesized value. Its minimal assumptions make it the most broadly applicable one-sample non-parametric test, at the cost of lower power compared to the [Wilcoxon signed-rank test](wilcoxon_signed_rank.md), which additionally exploits the magnitudes of deviations.

## Exercises

**Exercise 1.**
A nutrition study claims that a new diet reduces cholesterol. The cholesterol levels (mg/dL) of 10 patients are measured before and after the diet:

| Patient | Before | After | Difference (Before $-$ After) |
|:---:|:---:|:---:|:---:|
| 1 | 220 | 210 | 10 |
| 2 | 240 | 235 | 5 |
| 3 | 195 | 200 | $-5$ |
| 4 | 260 | 245 | 15 |
| 5 | 230 | 228 | 2 |
| 6 | 215 | 210 | 5 |
| 7 | 250 | 240 | 10 |
| 8 | 205 | 208 | $-3$ |
| 9 | 235 | 220 | 15 |
| 10 | 245 | 235 | 10 |

**(a)** State the null and alternative hypotheses for a one-sided sign test.

**(b)** Count the number of positive and negative signs (ignoring zeros). Compute the test statistic.

**(c)** Under $H_0$, what distribution does the number of positive signs follow? Compute the p-value.

**(d)** At $\alpha = 0.05$, what is your conclusion?

??? success "Solution to Exercise 1"

    **(a)** Let $\tilde{\mu}_d$ denote the population median of the paired differences.

    - $H_0$: $\tilde{\mu}_d = 0$ (the diet has no effect)
    - $H_1$: $\tilde{\mu}_d > 0$ (the diet reduces cholesterol)

    **(b)** Counting signs of the differences: 8 positive ($+$), 2 negative ($-$), 0 zeros. The test statistic for the sign test is the number of positive signs: $S^+ = 8$.

    **(c)** Under $H_0$, each difference is equally likely to be positive or negative, so $S^+ \sim \text{Binomial}(n = 10, p = 0.5)$.

    For a one-sided test ($H_1$: median $> 0$), the p-value is:

    $$
    p = P(S^+ \ge 8) = P(S^+ = 8) + P(S^+ = 9) + P(S^+ = 10)
    $$

    $$
    = \binom{10}{8}(0.5)^{10} + \binom{10}{9}(0.5)^{10} + \binom{10}{10}(0.5)^{10}
    $$

    $$
    = \frac{45 + 10 + 1}{1024} = \frac{56}{1024} \approx 0.0547
    $$

    **(d)** Since $p \approx 0.055 > 0.05$, we fail to reject $H_0$ at the 5% significance level. The evidence for a cholesterol reduction is suggestive but not statistically significant by the sign test. Note that the sign test is conservative because it discards magnitude information — the Wilcoxon signed-rank test may yield a different conclusion.
