# One-Way Analysis of Variance End-to-End Pipeline

## Overview

This page demonstrates a complete one-way ANOVA pipeline from model fitting through post-hoc testing to visualization. The workflow covers fitting the ANOVA model with statsmodels, running Tukey's HSD for pairwise comparisons, performing pairwise Welch $t$-tests with Bonferroni correction, and producing a boxplot summary. The PlantGrowth dataset serves as a running example throughout.

## Step 1: Fit the One-Way ANOVA Model

The one-way ANOVA tests

$$
H_0: \mu_1 = \mu_2 = \cdots = \mu_k
$$

against $H_A$: at least one $\mu_i$ differs. Using the formula interface in statsmodels, the factor is wrapped in `C()` to indicate a categorical variable.

```python
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

url = ('https://raw.githubusercontent.com/vincentarelbundock/'
       'Rdatasets/master/csv/datasets/PlantGrowth.csv')
df = pd.read_csv(url, usecols=[1, 2])

model = ols('weight ~ C(group)', data=df).fit()
aov = anova_lm(model)
print(aov)
```

The ANOVA table reports the between-group sum of squares ($SSB$), the within-group sum of squares ($SSW$), the $F$-statistic, and the $p$-value. Reject $H_0$ when $p < \alpha$.

## Step 2: Tukey HSD Post-Hoc Test

When ANOVA rejects, Tukey's Honest Significant Difference identifies which specific pairs differ while controlling the family-wise error rate. For a balanced design with $n$ observations per group,

$$
\text{HSD} = q_{\alpha,\, k,\, N-k}\; \sqrt{\frac{MSW}{n}}
$$

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(endog=df['weight'], groups=df['group'], alpha=0.05)
print(tukey)
```

The output shows the mean difference, confidence interval, and whether each pair is significantly different.

## Step 3: Pairwise Welch t-Tests with Bonferroni Correction

When variances may differ across groups, Welch's $t$-test does not assume equal variances. The Bonferroni correction multiplies each raw $p$-value by $m = \binom{k}{2}$:

$$
p_{\text{adj}} = \min\!\bigl(m \cdot p_{\text{raw}},\; 1\bigr)
$$

```python
from itertools import combinations
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

groups = df['group'].unique()
p_raw, labels = [], []
for g1, g2 in combinations(groups, 2):
    x = df.loc[df['group'] == g1, 'weight'].values
    y = df.loc[df['group'] == g2, 'weight'].values
    stat, p = ttest_ind(x, y, equal_var=False)
    p_raw.append(p)
    labels.append(f"{g1} vs {g2}")

_, p_bonf, _, _ = multipletests(p_raw, alpha=0.05, method='bonferroni')
for lbl, p, pb in zip(labels, p_raw, p_bonf):
    print(f"{lbl:<12}  p = {p:.4f}   p_bonf = {pb:.4f}")
```

## Step 4: Visualization

A boxplot provides a quick visual comparison of group distributions.

```python
import matplotlib.pyplot as plt

order = ['ctrl', 'trt1', 'trt2']
data = [df.loc[df['group'] == g, 'weight'].values for g in order]
plt.boxplot(data, labels=order)
plt.xlabel('Group')
plt.ylabel('Weight')
plt.title('PlantGrowth weights by group')
plt.tight_layout()
plt.show()
```

## Interpretation

- **ANOVA $F$-test:** A significant $p$-value indicates that at least one treatment group differs from the control or from another treatment.
- **Tukey HSD:** Provides simultaneous confidence intervals. Pairs whose intervals exclude zero are significantly different.
- **Bonferroni-corrected Welch tests:** More conservative than Tukey for the same number of comparisons but do not require equal variances.
- **Boxplot:** Visualizes the median, interquartile range, and potential outliers for each group, supporting the numerical findings.

## Exercises

**Exercise 1.**
The PlantGrowth dataset has three groups: ctrl, trt1, trt2 with 10 observations each. The ANOVA yields $F = 4.85$ with $p = 0.016$. How many pairwise comparisons are needed, and what is the Bonferroni-adjusted significance level for each test?

??? success "Solution to Exercise 1"
    With $k = 3$ groups there are $\binom{3}{2} = 3$ pairwise comparisons. The Bonferroni-adjusted significance level for each individual test is

    $$
    \alpha_{\text{adj}} = \frac{\alpha}{m} = \frac{0.05}{3} \approx 0.0167
    $$

    Each pairwise test must have a raw $p$-value below $0.0167$ to be declared significant.

---

**Exercise 2.**
Explain why the Welch $t$-test is preferred over the pooled (Student) $t$-test for pairwise comparisons after ANOVA when group variances may differ. What happens to the Welch test when variances are actually equal?

??? success "Solution to Exercise 2"
    The pooled $t$-test assumes $\sigma_1^2 = \sigma_2^2$ and estimates a common variance by pooling both samples. When this assumption fails, the pooled test can have inflated Type I error (if the smaller group has larger variance) or reduced power (if the larger group has larger variance).

    The Welch $t$-test uses separate variance estimates and adjusts the degrees of freedom via the Satterthwaite approximation:

    $$
    \nu = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}
    $$

    When variances are actually equal ($s_1^2 \approx s_2^2$), the Welch degrees of freedom approach $n_1 + n_2 - 2$, and the Welch test becomes nearly identical to the pooled test. The cost is a slight loss of power due to fewer effective degrees of freedom, but this loss is negligible for moderate sample sizes.

---

**Exercise 3.**
A one-way ANOVA on four groups gives $MSW = 8.5$ with $N - k = 76$ degrees of freedom. The Tukey critical value is $q_{0.05,4,76} = 3.70$ and all groups have $n = 20$. Compute the minimum mean difference required for significance.

??? success "Solution to Exercise 3"
    The Tukey HSD threshold is

    $$
    \text{HSD} = q_{\alpha,k,N-k} \sqrt{\frac{MSW}{n}} = 3.70 \sqrt{\frac{8.5}{20}} = 3.70 \sqrt{0.425} = 3.70 \times 0.6519 \approx 2.41
    $$

    Any pair of group means with $|\bar{y}_i - \bar{y}_j| > 2.41$ is significantly different at the $\alpha = 0.05$ level.

---

**Exercise 4.**
In the pipeline above, Tukey HSD and Bonferroni-corrected Welch $t$-tests may give different conclusions for the same pair of groups. Under what conditions would you trust one over the other? Discuss assumptions and power.

??? success "Solution to Exercise 4"
    **Trust Tukey HSD when:** (1) the equal-variance assumption holds (Levene's test is not significant), (2) group sizes are equal or nearly so, and (3) all pairwise comparisons are of interest. Tukey is designed specifically for the all-pairs problem and has higher power than Bonferroni in this setting.

    **Trust Bonferroni-corrected Welch tests when:** (1) group variances are unequal, (2) sample sizes are unbalanced, or (3) only a subset of comparisons are planned. The Welch test does not assume equal variances, making it more reliable when homoscedasticity is violated.

    In general, when both methods agree, the conclusion is robust. When they disagree, the disagreement usually involves a borderline comparison. In such cases, checking the diagnostic plots (boxplots, variance ratio) helps decide which set of assumptions is more defensible.

---

**Exercise 5.**
Prove that for a balanced one-way ANOVA ($n_1 = n_2 = \cdots = n_k = n$), the $F$-statistic can be written as

$$
F = \frac{n \sum_{i=1}^{k}(\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot})^2 / (k-1)}{\sum_{i=1}^{k}\sum_{j=1}^{n}(y_{ij} - \bar{y}_{i\cdot})^2 / (kn - k)}
$$

and explain why larger $n$ increases power even if the group means do not change.

??? success "Solution to Exercise 5"
    **Derivation.** For a balanced design with $n$ observations per group, $N = kn$. The between-group sum of squares is

    $$
    SSB = \sum_{i=1}^{k} n(\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot})^2 = n \sum_{i=1}^{k}(\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot})^2
    $$

    The within-group sum of squares is $SSW = \sum_{i=1}^{k}\sum_{j=1}^{n}(y_{ij} - \bar{y}_{i\cdot})^2$. The mean squares are $MSB = SSB/(k-1)$ and $MSW = SSW/(kn - k)$. The $F$-statistic is $F = MSB/MSW$, giving the stated expression.

    **Why larger $n$ increases power:** As $n$ grows, each group mean $\bar{y}_{i\cdot}$ converges to the population mean $\mu_i$ by the law of large numbers, so $\sum(\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot})^2$ stabilizes near $\sum(\mu_i - \bar{\mu})^2$. The numerator $MSB$ thus scales linearly with $n$. Meanwhile, $MSW$ converges to $\sigma^2$ regardless of $n$. Therefore $F \approx n \sum(\mu_i - \bar{\mu})^2 / [(k-1)\sigma^2]$ grows with $n$, making rejection of $H_0$ increasingly likely when the alternative is true. $\square$
