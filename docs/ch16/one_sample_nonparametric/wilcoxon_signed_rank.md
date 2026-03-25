# Wilcoxon Signed-Rank Test

The [sign test](sign_test.md) uses only the direction of each deviation from the hypothesized median, discarding all information about how far each observation falls from $m_0$. The **Wilcoxon signed-rank test** improves on this by incorporating both the sign *and* the rank of the absolute deviation, giving greater weight to observations that differ substantially from $m_0$. This additional information yields a more powerful test whenever the underlying distribution of deviations is symmetric.

!!! note "Not to be confused with the rank-sum test"
    The **Wilcoxon signed-rank test** is for one sample or paired data. The **Wilcoxon rank-sum test** (covered in [Rank-Sum Test](../two_sample_nonparametric/rank_sum.md)) is for comparing two independent samples.

## Assumptions

1. The observations $X_1, X_2, \ldots, X_n$ are independent.
2. Each $X_i$ comes from a continuous distribution.
3. The distribution of $X_i - m_0$ is **symmetric** about zero under $H_0$.

The symmetry assumption is the key additional requirement compared to the sign test. When symmetry is violated, the sign test is the safer choice.

## Hypotheses

For a one-sample test with hypothesized median $m_0$:

$$
H_0 \colon \text{The median of } X \text{ equals } m_0
$$

$$
H_a \colon \text{The median of } X \text{ does not equal } m_0 \quad \text{(two-sided)}
$$

One-sided alternatives $H_a \colon \text{median} > m_0$ or $H_a \colon \text{median} < m_0$ follow analogously.

## Test Statistic

**Step 1.** Compute the deviations $D_i = X_i - m_0$.

**Step 2.** Discard any $D_i = 0$ (observations equal to $m_0$). Let $n$ be the number of remaining observations.

**Step 3.** Rank the absolute deviations $|D_1|, |D_2|, \ldots, |D_n|$ from smallest to largest. Assign midranks to ties.

**Step 4.** Define the signed ranks:

$$
W_i = \operatorname{sign}(D_i) \cdot R_i
$$

where $R_i$ is the rank of $|D_i|$.

**Step 5.** Compute the positive and negative rank sums:

$$
W^+ = \sum_{\{i : D_i > 0\}} R_i, \qquad W^- = \sum_{\{i : D_i < 0\}} R_i
$$

Note that $W^+ + W^- = n(n+1)/2$.

The test statistic for the two-sided test is

$$
T = \min(W^+, W^-)
$$

Small values of $T$ provide evidence against $H_0$.

## Null Distribution

Under $H_0$ with symmetric deviations, each of the $2^n$ sign assignments is equally likely. The exact null distribution of $T$ (or equivalently $W^+$) can be tabulated by enumeration for small $n$.

**Mean and variance of $W^+$ under $H_0$:**

$$
\mu_{W^+} = \frac{n(n+1)}{4}
$$

$$
\sigma_{W^+}^2 = \frac{n(n+1)(2n+1)}{24}
$$

## Normal Approximation

For large $n$ (typically $n \ge 20$), the standardized statistic

$$
Z = \frac{W^+ - \mu_{W^+}}{\sigma_{W^+}}
$$

is approximately $\mathcal{N}(0, 1)$.

When ties are present among the $|D_i|$ values, the variance requires a correction. If there are $g$ groups of ties with sizes $t_1, t_2, \ldots, t_g$, the corrected variance is

$$
\sigma_{W^+}^2 = \frac{n(n+1)(2n+1)}{24} - \frac{1}{48}\sum_{j=1}^{g}(t_j^3 - t_j)
$$

## Worked Example

A psychologist measures reaction times (in ms) for 10 subjects under a new protocol and wants to test whether the median reaction time differs from $m_0 = 250$ ms.

| Subject | $X_i$ | $D_i = X_i - 250$ | $|D_i|$ | Rank | Signed rank |
|:-------:|:------:|:------------------:|:--------:|:----:|:-----------:|
| 1 | 268 | 18 | 18 | 5 | $+5$ |
| 2 | 241 | $-9$ | 9 | 2 | $-2$ |
| 3 | 273 | 23 | 23 | 7 | $+7$ |
| 4 | 230 | $-20$ | 20 | 6 | $-6$ |
| 5 | 255 | 5 | 5 | 1 | $+1$ |
| 6 | 262 | 12 | 12 | 3 | $+3$ |
| 7 | 289 | 39 | 39 | 9 | $+9$ |
| 8 | 237 | $-13$ | 13 | 4 | $-4$ |
| 9 | 295 | 45 | 45 | 10 | $+10$ |
| 10 | 275 | 25 | 25 | 8 | $+8$ |

**Compute rank sums:**

$$
W^+ = 5 + 7 + 1 + 3 + 9 + 10 + 8 = 43
$$

$$
W^- = 2 + 6 + 4 = 12
$$

**Check:** $W^+ + W^- = 43 + 12 = 55 = 10(11)/2$. $\checkmark$

**Test statistic:** $T = \min(43, 12) = 12$.

**Normal approximation:**

$$
\mu_{W^+} = \frac{10(11)}{4} = 27.5
$$

$$
\sigma_{W^+} = \sqrt{\frac{10(11)(21)}{24}} = \sqrt{96.25} \approx 9.811
$$

$$
Z = \frac{43 - 27.5}{9.811} \approx 1.580
$$

$$
p = 2\,\Phi(-1.580) \approx 0.114
$$

At $\alpha = 0.05$, we fail to reject $H_0$. There is insufficient evidence that the median reaction time differs from 250 ms.

## Comparison with the Sign Test

| Feature | Sign Test | Wilcoxon Signed-Rank |
|:--------|:----------|:---------------------|
| Uses signs | Yes | Yes |
| Uses magnitudes | No | Yes (via ranks) |
| Symmetry assumption | No | Yes |
| ARE vs $t$-test (normal data) | $2/\pi \approx 0.637$ | $3/\pi \approx 0.955$ |
| Robustness to outliers | Very high | High |

The signed-rank test achieves nearly the same power as the $t$-test under normality (ARE $\approx 0.955$) and can exceed the $t$-test's power under heavy-tailed distributions. The sign test is preferred only when the symmetry assumption is untenable or when data are purely ordinal.

## Summary

The Wilcoxon signed-rank test extends the sign test by ranking the absolute deviations from the hypothesized median and weighting each sign by its rank. This produces a more powerful test whenever the distribution of deviations is symmetric, at the cost of an additional assumption. The test statistic $T = \min(W^+, W^-)$ can be evaluated exactly for small samples or via a normal approximation for larger samples.
