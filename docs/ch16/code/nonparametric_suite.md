# Non-Parametric Test Suite

## Overview

This page collects the core nonparametric tests into a unified reference, highlighting when
to use each test and how they relate to one another. Nonparametric methods make minimal
distributional assumptions---typically only that observations are independent and drawn from
continuous distributions---making them indispensable when normality cannot be justified, when
data are ordinal, or when outliers are a concern.

## Taxonomy of Nonparametric Tests

The table below organizes the main tests by the experimental design they address.

| Design | Test | What It Uses |
|---|---|---|
| One sample / paired | Sign test | Signs of differences |
| One sample / paired | Wilcoxon signed-rank | Signed ranks of differences |
| Two independent samples | Wilcoxon rank-sum | Ranks in pooled sample |
| Two independent samples | Mann--Whitney $U$ | Pairwise comparisons (equivalent to rank-sum) |
| $k$ independent samples | Kruskal--Wallis $H$ | Ranks in pooled sample |
| $k$ independent samples | Mood's median test | Counts above/below grand median |
| Randomness | Wald--Wolfowitz runs test | Runs in a binary sequence |

## Power and Assumptions

Each test sits on a spectrum between generality and power.

**Sign test.** The weakest assumptions: only requires the differences to be independent and
continuous. It uses only the signs of the differences:

$$
Z_{\text{sign}} = \frac{2\,n_+ - n}{\sqrt{n}},
$$

where $n_+$ counts positive differences and $n$ is the number of nonzero differences. This
is a binomial test in disguise.

**Wilcoxon signed-rank test.** Adds the assumption that the distribution of differences is
**symmetric** about the median. By using rank magnitudes in addition to signs, it recovers
more information:

$$
W^{+} = \sum_{i:\,D_i > 0} \operatorname{rank}(|D_i|),
$$

with null mean and variance

$$
\operatorname{E}[W^+] = \frac{n'(n'+1)}{4}, \qquad
\operatorname{Var}(W^+) = \frac{n'(n'+1)(2n'+1)}{24}.
$$

**Rank-sum / Mann--Whitney.** For two independent samples, all $N$ observations are ranked
together. The statistic $U$ counts wins:

$$
U = \sum_{i=1}^{m}\sum_{j=1}^{n} \mathbf{1}[X_i > Y_j].
$$

Under the null, $\operatorname{E}[U] = mn/2$.

**Kruskal--Wallis $H$.** Generalizes the rank-sum to $k$ groups:

$$
H = \frac{12}{N(N+1)}\sum_{j=1}^{k}\frac{R_j^2}{n_j} - 3(N+1) \;\sim\; \chi^2_{k-1}.
$$

**Mood's median test.** The most robust multi-group test, using only the binary indicator
"above or below the grand median" for each observation.

## Decision Guide

The following flowchart summarizes the decision process:

1. **Are the samples paired or independent?**
      - Paired $\to$ go to step 2.
      - Independent $\to$ go to step 3.
2. **Paired: Is the difference distribution symmetric?**
      - Yes $\to$ Wilcoxon signed-rank (more powerful).
      - No or unknown $\to$ Sign test (safer).
3. **Independent: How many groups?**
      - Two groups $\to$ Mann--Whitney $U$ (handles ties) or Wilcoxon rank-sum.
      - Three or more $\to$ Kruskal--Wallis $H$ (or Mood's median test for extra robustness).

## Example: Applying Multiple Tests

The code below applies several tests to the same paired dataset, allowing a direct
comparison of $p$-values.

```python
import numpy as np
from scipy import stats

paired_data = np.array([
    [93, 76], [70, 72], [81, 75], [65, 68], [79, 65],
    [54, 54], [94, 88], [91, 81], [77, 65], [65, 57],
    [95, 86], [89, 87], [78, 78], [80, 77], [76, 76]
])

post, pre = paired_data[:, 0], paired_data[:, 1]

# Sign test (normal approximation)
diffs = post - pre
nonzero = diffs[diffs != 0]
n_plus = (nonzero > 0).sum()
n = len(nonzero)
z_sign = (2 * n_plus - n) / np.sqrt(n)
p_sign = 2 * stats.norm.cdf(-abs(z_sign))
print(f"Sign test:          Z = {z_sign:.4f}, p = {p_sign:.4f}")

# Wilcoxon signed-rank test
stat_sr, p_sr = stats.wilcoxon(post, pre, alternative="two-sided",
                                mode="approx", zero_method="pratt")
print(f"Signed-rank test:   W+ = {stat_sr}, p = {p_sr:.4f}")

# Wilcoxon rank-sum test (treating as independent for illustration)
stat_rs, p_rs = stats.ranksums(post, pre)
print(f"Rank-sum test:      Z = {stat_rs:.4f}, p = {p_rs:.4f}")
```

Notice that the signed-rank test typically produces the smallest $p$-value (highest power)
because it uses the most information, while the sign test produces the largest $p$-value.

## Interpretation

- **Nonparametric $\neq$ assumption-free.** Every test still requires independence. The
  signed-rank test additionally requires symmetry. Kruskal--Wallis tests for any
  distributional difference, not just a location shift, unless the group distributions have
  the same shape.
- **Asymptotic relative efficiency (ARE).** Under normality, the Wilcoxon signed-rank test
  has ARE $= 3/\pi \approx 0.955$ relative to the paired $t$-test---it loses very little
  power. Under heavy-tailed distributions its ARE exceeds $1$.
- **Multiple comparisons.** If Kruskal--Wallis rejects $H_0$, post-hoc pairwise tests
  (e.g., Dunn's test) with a Bonferroni or Holm correction can identify which groups
  differ.

## Exercises

**Exercise 1.** A researcher has 12 paired observations and is unsure whether the
differences are symmetric. Which test should they use---sign test or Wilcoxon signed-rank?
Justify your answer and explain the trade-off involved.

??? success "Solution to Exercise 1"

    The researcher should use the **sign test**. The Wilcoxon signed-rank test requires
    the distribution of differences to be symmetric about the median under $H_0$. If this
    assumption is violated, the null distribution of $W^+$ is incorrect and the test may
    have inflated Type I error.

    The trade-off is **power**: the sign test uses only the signs of the differences and
    discards magnitude information, so it is less powerful when the symmetry assumption
    actually holds. In this case, however, safety (controlling Type I error) outweighs
    power, since the researcher cannot verify symmetry. If the sample were larger, the
    researcher could examine a histogram of differences to assess symmetry and potentially
    switch to the signed-rank test. $\square$

---

**Exercise 2.** Show that the asymptotic relative efficiency of the sign test relative to
the one-sample $t$-test under normality is $2/\pi \approx 0.637$.

??? success "Solution to Exercise 2"

    Under $H_0$ with $D_i \sim N(0, \sigma^2)$, the sign test statistic is based on
    $n_+ \sim \text{Binomial}(n, 1/2)$. The variance of $\hat{p} = n_+/n$ is
    $1/(4n)$, giving $\operatorname{Var}(Z_{\text{sign}}) \approx 1$.

    The $t$-test uses $\bar{D}/(\hat{\sigma}/\sqrt{n})$, which detects a shift $\delta$
    with power determined by the noncentrality parameter $\delta\sqrt{n}/\sigma$.

    For the sign test, under a local alternative $D_i \sim N(\delta, \sigma^2)$,
    $P(D_i > 0) = \mathcal{N}(\delta/\sigma)$. By a Taylor expansion around $\delta = 0$:

    $$
    P(D_i > 0) \approx \frac{1}{2} + \frac{\delta}{\sigma\sqrt{2\pi}}.
    $$

    The noncentrality parameter of the sign test is proportional to
    $\delta\sqrt{n}/(\sigma\sqrt{2\pi} \cdot 1/2) = 2\delta\sqrt{n}/(\sigma\sqrt{2\pi})$.

    The ARE is the square of the ratio of noncentrality parameters:

    $$
    \text{ARE} = \left(\frac{2/\sqrt{2\pi}}{1}\right)^2 \cdot \frac{1}{1}
               = \frac{4}{2\pi} = \frac{2}{\pi} \approx 0.637.
    $$

    This means the sign test needs roughly $\pi/2 \approx 1.57$ times as many
    observations to achieve the same power as the $t$-test under normality. $\square$

---

**Exercise 3.** Three independent groups produce the following data:

- Group A: $5, 8, 12, 15$
- Group B: $7, 11, 14, 18, 20$
- Group C: $3, 6, 9$

Perform both Kruskal--Wallis and Mood's median test in Python and compare the $p$-values.

??? success "Solution to Exercise 3"

    ```python
    from scipy import stats

    a = [5, 8, 12, 15]
    b = [7, 11, 14, 18, 20]
    c = [3, 6, 9]

    # Kruskal-Wallis
    h_stat, h_p = stats.kruskal(a, b, c)
    print(f"Kruskal-Wallis: H = {h_stat:.4f}, p = {h_p:.4f}")

    # Mood's median test
    result = stats.median_test(a, b, c)
    print(f"Mood's median:  chi2 = {result.statistic:.4f}, p = {result.pvalue:.4f}")
    print(f"Grand median = {result.median}")
    ```

    The Kruskal--Wallis $p$-value will generally be smaller (more powerful) because it
    uses full rank information, while Mood's test reduces each observation to a binary
    indicator. Both tests assess the same null hypothesis---that all groups come from the
    same population---but through different lenses. $\square$

---

**Exercise 4.** Explain why Kruskal--Wallis is not simply a test of medians. What does it
actually test, and under what additional assumption does it become a test of location shift?

??? success "Solution to Exercise 4"

    The Kruskal--Wallis test has null hypothesis $H_0{:}\; F_1 = F_2 = \cdots = F_k$,
    i.e., all groups come from the same distribution. This is a test of **distributional
    equality**, not merely of equal medians. Rejecting $H_0$ could mean the groups differ
    in location, scale, shape, or any combination.

    Under the additional **location-shift assumption**---that the distributions differ
    only by a shift in location ($F_j(x) = F(x - \mu_j)$ for a common shape $F$)---the
    only way the distributions can differ is through their means/medians, and in that
    case Kruskal--Wallis becomes a test of equal medians (or equivalently, equal means).

    Without the location-shift assumption, a significant Kruskal--Wallis result might
    reflect differences in spread rather than center, which can mislead researchers who
    interpret it as a test of medians. $\square$

---

**Exercise 5.** After a significant Kruskal--Wallis result, a researcher wants to determine
which specific pairs of groups differ. Describe Dunn's test and explain why a multiple
comparisons correction is necessary.

??? success "Solution to Exercise 5"

    **Dunn's test** performs pairwise comparisons between all $\binom{k}{2}$ pairs of
    groups using the rank sums from the original Kruskal--Wallis pooled ranking. For
    groups $i$ and $j$ with rank-sum means $\bar{R}_i$ and $\bar{R}_j$:

    $$
    Z_{ij} = \frac{\bar{R}_i - \bar{R}_j}{\sqrt{\frac{N(N+1)}{12}\left(\frac{1}{n_i} + \frac{1}{n_j}\right)}},
    $$

    where $N$ is the total sample size. Each $Z_{ij}$ is compared to the standard normal.

    A **multiple comparisons correction** (e.g., Bonferroni, Holm, or Benjamini--Hochberg)
    is necessary because performing $\binom{k}{2}$ tests inflates the family-wise error
    rate. Without correction, if we test $m$ pairs at level $\alpha$, the probability of
    at least one false rejection can be as high as $1 - (1 - \alpha)^m$, which grows
    rapidly with $k$. The Bonferroni correction uses $\alpha / m$ for each pairwise test,
    ensuring the overall family-wise error rate remains at most $\alpha$. $\square$
