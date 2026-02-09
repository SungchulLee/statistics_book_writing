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

- **Data transformations:** Log, square root, or Box-Cox transformations can reduce skewness and make residuals more normal (see [Transformations to Achieve Normality](../../ch15/non_normal_data/transformations.md)).
- **Non-parametric alternatives:** The Kruskal-Wallis test compares medians instead of means and does not require normality (see [Kruskal-Wallis Test](../../ch19/two_sample_nonparametric/two_sample.md)).
- **Bootstrapping:** Resampling methods can provide valid inference without distributional assumptions (see [Bootstrapping](../../ch20/bootstrap/bootstrap.md)).
