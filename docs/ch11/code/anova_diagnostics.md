# Analysis of Variance Diagnostics

## Overview

Before trusting the results of an ANOVA, several key assumptions must be verified: normality of residuals, homoscedasticity (equal variances across groups), independence of observations, and the absence of influential points. This page walks through a complete diagnostic workflow, illustrating each check with formal tests and diagnostic plots, and discusses remedies when assumptions are violated.

## The Diagnostic Workflow

A typical ANOVA diagnostic pipeline consists of four stages, applied to the residuals of the fitted model $y_{ij} = \mu + \alpha_i + \varepsilon_{ij}$.

| Stage | Question | Primary Tool | Formal Test |
|---|---|---|---|
| 1 | Are residuals normal? | Q-Q plot, histogram | Shapiro-Wilk |
| 2 | Are group variances equal? | Side-by-side spreads | Levene, Bartlett |
| 3 | Are residuals independent? | Residuals vs. fitted | Durbin-Watson |
| 4 | Are any points overly influential? | Cook's distance plot | Cook's $D$ threshold |

## Step 1: Checking Normality

The Shapiro-Wilk test evaluates

$$
H_0: \text{the residuals come from a normal distribution}
$$

against the alternative that they do not. A Q-Q plot supplements the test by showing whether the sample quantiles align with theoretical normal quantiles.

```python
from scipy.stats import shapiro
import statsmodels.api as sm

resid = model.resid
stat, p_value = shapiro(resid)
sm.qqplot(resid, line='s')
```

If the Shapiro-Wilk $p$-value is small (e.g., $p < 0.05$), or the Q-Q plot shows systematic curvature, normality is suspect. Remedies include data transformations (log, square root) or switching to a non-parametric test such as Kruskal-Wallis.

## Step 2: Checking Homoscedasticity

ANOVA assumes that every group shares a common variance $\sigma^2$. Levene's test is robust to non-normality, whereas Bartlett's test is optimal when data are truly normal but is sensitive to departures from normality.

For $k$ groups, Levene's test statistic is

$$
W = \frac{(N - k)}{(k - 1)} \cdot \frac{\sum_{i=1}^{k} n_i (\bar{Z}_{i\cdot} - \bar{Z}_{\cdot\cdot})^2}{\sum_{i=1}^{k} \sum_{j=1}^{n_i} (Z_{ij} - \bar{Z}_{i\cdot})^2}
$$

where $Z_{ij} = |y_{ij} - \tilde{y}_{i}|$ and $\tilde{y}_i$ is the group median.

```python
from scipy.stats import levene, bartlett

groups = [data[data['group'] == g]['response'].values for g in data['group'].unique()]
stat_lev, p_lev = levene(*groups)
stat_bart, p_bart = bartlett(*groups)
```

When homoscedasticity is rejected, Welch's ANOVA or a heteroscedasticity-robust approach (HC3 covariance) should be used.

## Step 3: Checking Independence

The Durbin-Watson statistic detects first-order autocorrelation in the residuals:

$$
d = \frac{\sum_{t=2}^{n}(e_t - e_{t-1})^2}{\sum_{t=1}^{n} e_t^2}
$$

Values near 2 indicate no autocorrelation; values near 0 suggest positive autocorrelation, and values near 4 suggest negative autocorrelation. A common rule of thumb is that $d \in (1.5, 2.5)$ is acceptable.

```python
from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(model.resid)
```

A residuals-versus-fitted-values scatter plot should show no discernible pattern.

## Step 4: Identifying Influential Points

Cook's distance measures the influence of each observation on the fitted model. A common threshold is

$$
D_i > \frac{4}{n}
$$

where $n$ is the total number of observations. Observations exceeding this threshold should be investigated for data-entry errors or genuinely unusual conditions.

```python
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]
threshold = 4 / len(cooks_d)
```

## Putting It All Together

The following function runs the full pipeline on any one-way ANOVA design and produces a 2-by-2 diagnostic panel (Q-Q plot, histogram of residuals, residuals vs. fitted, Cook's distance).

```python
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

def run_full_diagnostics(data, response_col, group_col):
    formula = f'{response_col} ~ {group_col}'
    model = ols(formula, data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # Q-Q plot
    sm.qqplot(model.resid, line='s', ax=axes[0, 0])
    # Histogram
    axes[0, 1].hist(model.resid, bins=15, density=True, alpha=0.7, edgecolor='black')
    # Residuals vs Fitted
    axes[1, 0].scatter(model.fittedvalues, model.resid, alpha=0.6)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    # Cook's distance
    cooks_d = model.get_influence().cooks_distance[0]
    axes[1, 1].stem(range(len(cooks_d)), cooks_d, markerfmt=",")
    axes[1, 1].axhline(y=4 / len(cooks_d), color='r', linestyle='--')
    plt.tight_layout()
    plt.show()
```

## Interpretation

- **Normality:** If the Q-Q plot follows the reference line and Shapiro-Wilk $p > 0.05$, normality holds. Mild departures matter less with large, balanced groups due to the Central Limit Theorem.
- **Homoscedasticity:** If Levene's $p > 0.05$, equal-variance assumption is reasonable. Otherwise, use Welch's ANOVA.
- **Independence:** A Durbin-Watson value near 2 with no pattern in the residual plot supports independence.
- **Influential points:** Any observation with Cook's $D > 4/n$ should be inspected. Removing a single influential point and re-running the analysis can reveal whether conclusions are sensitive to that observation.

## Exercises

**Exercise 1.**
A researcher fits a one-way ANOVA with $k = 4$ groups and $n = 50$ total observations. The Durbin-Watson statistic is $d = 0.85$. What does this indicate, and what should the researcher do?

??? success "Solution to Exercise 1"
    A Durbin-Watson statistic of $d = 0.85$ is well below the lower bound of the acceptable range $(1.5, 2.5)$, indicating strong positive autocorrelation among the residuals. This means successive residuals tend to have the same sign, violating the independence assumption of ANOVA.

    The researcher should investigate the data collection process. If the data were collected over time, a time-series model or repeated-measures ANOVA may be more appropriate. If the ordering is not meaningful, re-randomizing observation order and re-checking may clarify whether the autocorrelation is an artifact of sorting.

---

**Exercise 2.**
Explain why Levene's test uses absolute deviations from the group median rather than the group mean. Under what circumstances would using the mean instead produce misleading results?

??? success "Solution to Exercise 2"
    Using the group median makes Levene's test robust to skewed distributions and outliers. The median is resistant to extreme values, so the transformed variable $Z_{ij} = |y_{ij} - \tilde{y}_i|$ is less influenced by non-normality than $|y_{ij} - \bar{y}_i|$.

    When the underlying distributions are heavily skewed or contain outliers, the group mean is pulled toward extreme values, inflating the absolute deviations for the affected group. This can cause the test to falsely reject homoscedasticity (inflated Type I error) or, conversely, to mask genuine variance differences. The median-based version (Brown-Forsythe variant) maintains the nominal Type I error rate under a wider range of distributional shapes.

---

**Exercise 3.**
Derive the expectation of Cook's distance threshold $4/n$. Specifically, if $D_i \sim \text{Beta}\!\bigl(\tfrac{p}{2},\, \tfrac{n-p}{2}\bigr)$ approximately, show that $E[D_i] \approx p/(n-p)$ and explain why $4/n$ is a practical simplification.

??? success "Solution to Exercise 3"
    Cook's distance for observation $i$ can be related to the Beta distribution. Under the approximation $D_i \sim \text{Beta}(p/2,\, (n-p)/2)$, the expectation of a $\text{Beta}(\alpha, \beta)$ random variable is

    $$
    E[D_i] = \frac{\alpha}{\alpha + \beta} = \frac{p/2}{p/2 + (n-p)/2} = \frac{p}{n}
    $$

    For one-way ANOVA with $k$ groups, $p = k$ (including the intercept), so $E[D_i] = k/n$. The threshold $4/n$ corresponds roughly to choosing observations whose Cook's distance is about 4 times the average, which is a standard rule of thumb for flagging influential points. When $k$ is small relative to $n$, $4/n$ and $4k/n$ are of similar magnitude, making $4/n$ a convenient simplification. $\square$

---

**Exercise 4.**
You run ANOVA diagnostics on a dataset and find that normality holds but Levene's test rejects equal variances with $p = 0.003$. Bartlett's test gives $p = 0.001$. The sample sizes are $n_1 = 50$, $n_2 = 12$, $n_3 = 45$. Describe the appropriate next steps and the specific alternative analysis you would use.

??? success "Solution to Exercise 4"
    Since both Levene's and Bartlett's tests reject homoscedasticity, the classical ANOVA F-test is unreliable because it assumes equal variances. The unbalanced design ($n_2 = 12$ is much smaller than the other groups) exacerbates the problem: when smaller groups have larger variance, the F-test becomes liberal (inflated Type I error), and when smaller groups have smaller variance, it becomes conservative.

    The appropriate alternative is Welch's ANOVA, which does not assume equal variances and adjusts the degrees of freedom using a Satterthwaite-type approximation. For post-hoc pairwise comparisons, Games-Howell is the natural complement to Welch's ANOVA, as it also accounts for unequal variances and unequal sample sizes.

---

**Exercise 5.**
Prove that the Durbin-Watson statistic satisfies $0 \le d \le 4$ and that $d = 2$ corresponds to zero first-order autocorrelation in the residuals.

??? success "Solution to Exercise 5"
    The Durbin-Watson statistic is

    $$
    d = \frac{\sum_{t=2}^{n}(e_t - e_{t-1})^2}{\sum_{t=1}^{n} e_t^2}
    $$

    **Lower bound:** Since $(e_t - e_{t-1})^2 \ge 0$ for all $t$, the numerator is non-negative. The denominator is a sum of squares, so it is positive (assuming not all residuals are zero). Therefore $d \ge 0$.

    **Upper bound:** Expand the numerator:

    $$
    \sum_{t=2}^{n}(e_t - e_{t-1})^2 = \sum_{t=2}^{n} e_t^2 - 2\sum_{t=2}^{n} e_t e_{t-1} + \sum_{t=2}^{n} e_{t-1}^2
    $$

    The first and third sums are each at most $\sum_{t=1}^{n} e_t^2$, and by the Cauchy-Schwarz inequality $|\sum e_t e_{t-1}| \le \sum e_t^2$, so the numerator is at most $4 \sum e_t^2$, giving $d \le 4$.

    **Zero autocorrelation:** When the first-order autocorrelation $\hat{\rho}_1 = \sum_{t=2}^{n} e_t e_{t-1} / \sum_{t=1}^{n} e_t^2 \approx 0$, the cross term vanishes and the two remaining sums in the expansion are each approximately $\sum e_t^2$, giving $d \approx 2(1 - \hat{\rho}_1) \approx 2$. $\square$
