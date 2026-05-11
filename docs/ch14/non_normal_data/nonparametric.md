# Non-Parametric Methods


Another strategy for dealing with non-normal data is to use **non-parametric methods**. These methods do not assume a specific distribution for the data and are often used when data is ordinal or when normality assumptions are violated.

## Common Non-Parametric Tests

- **Mann-Whitney U Test**: A non-parametric alternative to the $t$-test for comparing two independent groups.
- **Kruskal-Wallis Test**: A non-parametric alternative to ANOVA for comparing more than two groups.
- **Wilcoxon Signed-Rank Test**: A non-parametric test for comparing two related samples.

## Python Implementation

```python
import numpy as np
from scipy.stats import mannwhitneyu

# Generate two non-normal datasets
group1 = np.random.exponential(scale=2, size=100)
group2 = np.random.exponential(scale=3, size=100)

# Perform Mann-Whitney U test
stat, p_value = mannwhitneyu(group1, group2)
print(f"Mann-Whitney U Test: Statistic={stat}, p-value={p_value}")

# Interpretation
alpha = 0.05
if p_value > alpha:
    print("Fail to reject H_0: No significant difference between the groups.")
else:
    print("Reject H_0: Significant difference between the groups.")
```

Non-parametric tests offer robust alternatives when normality assumptions are violated or when dealing with ordinal data. They are widely used in situations where data distributions are unknown or non-normal.

## Summary: Dealing with Non-Normal Data

When faced with non-normal data, several strategies can be employed:

- **Transformations** can help bring the data closer to normality.
- **Bootstrapping** provides an alternative approach that does not rely on parametric assumptions.
- **Non-parametric methods** offer powerful alternatives to traditional parametric tests when assumptions are violated.

The choice of method depends on the degree of non-normality, the sample size, and the specific research questions being addressed. In practice, combining these approaches with graphical and formal normality assessments can lead to more reliable statistical analysis.


## Exercises

**Exercise 1.**
Name three nonparametric tests and state the parametric test each replaces. What assumption is relaxed?

??? success "Solution to Exercise 1"
    | Nonparametric test | Replaces | Assumption relaxed |
    |---|---|---|
    | Wilcoxon signed-rank test | One-sample t-test | Normality of data |
    | Mann-Whitney U test | Two-sample t-test | Normality of data in both groups |
    | Kruskal-Wallis test | One-way ANOVA | Normality within groups |

    In each case, the nonparametric test does not require the data to follow a specific distribution. It works with the ranks of the data rather than the actual values, making it robust to outliers and non-normality.

---

**Exercise 2.**
Explain the concept of asymptotic relative efficiency (ARE). If the Wilcoxon test has ARE $= 3/\pi \approx 0.955$ relative to the t-test under normality, what does this mean?

??? success "Solution to Exercise 2"
    ARE compares the sample sizes needed by two tests to achieve the same power. An ARE of 0.955 means: under normality, the Wilcoxon test needs approximately $n/0.955 \approx 1.047n$ observations to match the t-test's power.

    In other words, the Wilcoxon test loses only about 5% efficiency relative to the t-test when data are truly normal. This is a small price to pay for the robustness gained: when data are non-normal (heavy-tailed or skewed), the Wilcoxon test can be much more powerful than the t-test. For some distributions, the ARE of Wilcoxon relative to the t-test exceeds 1 (Wilcoxon is more efficient).

---

**Exercise 3.**
A dataset has values $\{1, 2, 3, 100\}$. Explain why a nonparametric test based on ranks is more appropriate than a t-test.

??? success "Solution to Exercise 3"
    The value 100 is a severe outlier relative to the other values. The t-test statistic uses the mean (heavily influenced by 100) and the standard deviation (inflated by 100), both distorted by this single observation. The resulting t-statistic and p-value are unreliable.

    A rank-based test converts the data to ranks $\{1, 2, 3, 4\}$, where the outlier is simply the largest observation with rank 4. The extreme magnitude of 100 is irrelevant. This makes the test robust: replacing 100 with 10 or 10,000 would not change the ranks or the test result.

---

**Exercise 4.**
When are nonparametric tests not recommended, even if normality is questionable?

??? success "Solution to Exercise 4"
    Nonparametric tests are not always preferred because:

    1. **Large sample sizes:** With large $n$, the CLT makes parametric tests approximately valid, and they are more powerful (use the actual values, not just ranks). The efficiency loss from ranks is unnecessary.
    2. **Inference on means specifically:** Nonparametric tests (e.g., Mann-Whitney) test for stochastic dominance or median differences, not mean differences. If the research question is specifically about means, a t-test (possibly with bootstrap) is more appropriate.
    3. **Complex designs:** For factorial ANOVA, regression, or mixed models, nonparametric alternatives are limited or less well-developed. Robust parametric methods (Welch ANOVA, sandwich standard errors) are often preferable.
    4. **Tied data:** Rank-based tests lose power with many ties (e.g., Likert scale data with few categories).
