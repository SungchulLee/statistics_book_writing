# Checking Normality of Residuals


## Why Normality Matters

The residuals (differences between observed and predicted values) should be normally distributed for each group. This assumption ensures that the F-statistic follows the correct $F$-distribution under the null hypothesis. When residuals deviate substantially from normality, the p-values produced by ANOVA may be inaccurate, potentially leading to incorrect conclusions.

The normality assumption is particularly important for small sample sizes. For larger samples ($n \geq 30$ per group), the Central Limit Theorem provides robustness against moderate departures from normality, meaning the sampling distribution of the group means will be approximately normal regardless of the underlying distribution.

## How to Check

### Q-Q Plot (Quantile-Quantile Plot)

A Q-Q plot compares the quantiles of the observed residuals against the theoretical quantiles of a normal distribution. If the residuals are normally distributed, the points fall approximately along a straight 45-degree reference line.

- Points deviating from the line in the **tails** suggest heavy-tailed or light-tailed distributions.
- A systematic **S-shaped** curve suggests skewness.
- A few points deviating at the extremes may simply reflect natural sampling variability.

```python
import statsmodels.api as sm
import matplotlib.pyplot as plt

sm.qqplot(model.resid, line='s')
plt.title("Q-Q Plot of Residuals")
plt.show()
```

### Shapiro-Wilk Test

The Shapiro-Wilk test evaluates the null hypothesis that the data is drawn from a normal distribution. A significant result ($p < 0.05$) suggests a departure from normality.

$$
W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}
$$

where $x_{(i)}$ are the ordered sample values and $a_i$ are constants generated from the means, variances, and covariances of the order statistics of a sample of size $n$ from a normal distribution.

```python
from scipy.stats import shapiro

stat, p_value = shapiro(model.resid)
print(f"Shapiro-Wilk Test: W = {stat:.4f}, p-value = {p_value:.4f}")
```

!!! warning "Sensitivity to Sample Size"
    The Shapiro-Wilk test can be overly sensitive with large samples, flagging trivial departures from normality as statistically significant. Conversely, with small samples, the test may lack power to detect meaningful departures. Always combine formal tests with visual inspection (Q-Q plots, histograms).

### Histogram of Residuals

Plotting the residuals in a histogram provides a quick visual assessment of their distribution shape.

```python
import matplotlib.pyplot as plt

plt.hist(model.resid, bins=20, density=True, alpha=0.7, edgecolor='black')
plt.xlabel("Residuals")
plt.ylabel("Density")
plt.title("Histogram of Residuals")
plt.show()
```

Look for:

- **Skewness:** The distribution is not symmetric around zero.
- **Heavy tails (kurtosis):** More extreme values than expected under normality.
- **Bimodality:** Two peaks may indicate a missing grouping variable.

## What to Do If Normality Is Violated

- **Data transformations:** Log, square root, or Box-Cox transformations can reduce skewness and make residuals more normal (see [Transformations to Achieve Normality](../../ch14/non_normal_data/transformations.md)).
- **Non-parametric alternatives:** The Kruskal-Wallis test compares medians instead of means and does not require normality (see [Kruskal-Wallis Test](../../ch16/multi_group_nonparametric/kruskal_wallis.md)).
- **Bootstrapping:** Resampling methods can provide valid inference without distributional assumptions (see [The Bootstrap Principle](../../ch17/bootstrap/principle.md)).
## Exercises

**Exercise 1.**
A one-way ANOVA with $k = 3$ groups has $n = 8$ observations per group. The Shapiro-Wilk test on the residuals gives $p = 0.02$. Should the researcher abandon ANOVA? Explain your reasoning.

??? success "Solution to Exercise 1"
    Not necessarily. The researcher should combine the formal test result with visual diagnostics (Q-Q plot, histogram). With only $n = 8$ per group, the Shapiro-Wilk test may be detecting a moderate departure from normality that does not severely affect the F-test. ANOVA is reasonably robust to mild non-normality, especially when group sizes are equal.

    However, if the Q-Q plot reveals heavy tails, strong skewness, or outliers, the researcher should consider alternatives: applying a variance-stabilizing transformation (log, square root), using Welch's ANOVA (which is more robust), or switching to a non-parametric alternative like the Kruskal-Wallis test.

---

**Exercise 2.**
A Q-Q plot of ANOVA residuals shows points that follow the reference line in the center but curve upward at both tails. What does this pattern indicate about the residual distribution, and how might it affect ANOVA inference?

??? success "Solution to Exercise 2"
    Points curving upward at both tails indicate a **heavy-tailed (leptokurtic) distribution** -- the residuals have more extreme values than a normal distribution would predict. This means there is excess kurtosis.

    Heavy tails can inflate the within-group variance estimate (MSW), making the F-statistic smaller and the test more conservative (reduced power). Additionally, extreme values may be influential observations that disproportionately affect group means. The researcher should check for outliers, consider robust methods, or use a non-parametric test if the departure is severe.

---

**Exercise 3.**
Explain why the normality assumption is less critical for large sample sizes (e.g., $n \geq 30$ per group) than for small ones. Which specific results in ANOVA remain valid without normality, and which do not?

??? success "Solution to Exercise 3"
    For large samples, the **Central Limit Theorem** ensures that the sampling distribution of the group means is approximately normal regardless of the underlying population distribution. Since the F-test is based on comparing group means, its reference distribution is approximately correct.

    **Results that remain valid:** The OLS estimates of group means are unbiased regardless of normality. The F-test p-values are approximately valid for large samples.

    **Results that may not remain valid:** Prediction intervals for individual observations still require normality. With small samples, the exact distribution of the F-statistic depends on normality, so p-values may be inaccurate if normality is violated.
