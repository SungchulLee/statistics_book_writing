# Robust Tests for Equality of Variances

## Levene's Test and Brown–Forsythe Test

Levene's test (using the group mean) and the Brown–Forsythe test (using the group median) are robust statistical tests for assessing the equality of variances across multiple groups. Unlike the F-test, which is highly sensitive to departures from normality, these tests are more robust to non-normal data. The Brown–Forsythe test, which uses the median instead of the mean, is especially effective when the data contain outliers.

### Hypotheses

**Null Hypothesis ($H_0$):** The population variances are equal across the groups:

$$
H_0: \sigma_1^2 = \sigma_2^2 = \dots = \sigma_k^2
$$

**Alternative Hypothesis ($H_1$):** At least one group has a variance that differs from the others:

$$
H_1: \sigma_i^2 \neq \sigma_j^2 \quad \text{for at least one pair} \quad i \neq j
$$

These tests essentially evaluate whether the variances across different groups are homogeneous. If the null hypothesis is rejected, it indicates that at least one of the group variances is significantly different.

### Assumptions

1. **Independence:** The observations must be independent both within and between groups.
2. **Random Sampling:** The data should be drawn from random samples.

These tests do **not** require the assumption of normality, making them more robust than the F-test for variance comparison. They can be applied to data that deviate from normality, as long as the other assumptions are satisfied. The Brown–Forsythe test is particularly useful when the data contain outliers, as it is more robust than the F-test or Bartlett's test in such cases.

### Test Statistic

Both tests transform the data by computing the absolute deviations of each observation from the group center — the **mean** for Levene's test, or the **median** for the Brown–Forsythe test. The absolute deviations are then used to construct an ANOVA-like test statistic:

$$
W = \frac{(N - k)}{(k - 1)} \cdot \frac{\sum_{i=1}^{k} n_i (\bar{Z}_i - \bar{Z})^2}{\sum_{i=1}^{k} \sum_{j=1}^{n_i} (Z_{ij} - \bar{Z}_i)^2} \sim F_{k-1, \, N-k}
$$

where:

- $N$ is the total number of observations,
- $k$ is the number of groups,
- $n_i$ is the number of observations in group $i$,
- $Z_{ij}$ is the absolute deviation of observation $j$ in group $i$ from the group center,
- $\bar{Z}_i$ is the mean of the absolute deviations in group $i$,
- $\bar{Z}$ is the overall mean of the absolute deviations across all groups.

The absolute deviations are computed as:

$$
Z_{ij} = |X_{ij} - \tilde{X}_i|
$$

where $X_{ij}$ is the original data point, and $\tilde{X}_i$ is the group **mean** (Levene's test) or the group **median** (Brown–Forsythe test).

### Decision Rule

$$
W > F_{\text{critical}} \quad \Rightarrow \quad \text{Reject } H_0
$$

### Example Problem and Solution

**Example:** A researcher wants to test whether the variances in exam scores differ between three groups of students taught by different instructors. The exam scores are:

- Group 1: $65, 70, 75, 80, 85$
- Group 2: $60, 65, 70, 75, 90$
- Group 3: $55, 60, 65, 70, 95$

Test whether the variances are equal at a 5% significance level using the Brown–Forsythe test.

**Step 1 — Formulate Hypotheses:**

- $H_0$: The variances of the exam scores are equal across the three groups.
- $H_1$: At least one group has a significantly different variance.

**Step 2 — Compute Absolute Deviations from Group Medians:**

- **Group 1** (median = 75): $|65-75|=10$, $|70-75|=5$, $|75-75|=0$, $|80-75|=5$, $|85-75|=10$
- **Group 2** (median = 70): $|60-70|=10$, $|65-70|=5$, $|70-70|=0$, $|75-70|=5$, $|90-70|=20$
- **Group 3** (median = 65): $|55-65|=10$, $|60-65|=5$, $|65-65|=0$, $|70-65|=5$, $|95-65|=30$

**Step 3 — Compute Group Means of Absolute Deviations:**

- Group 1: $(10 + 5 + 0 + 5 + 10) / 5 = 6$
- Group 2: $(10 + 5 + 0 + 5 + 20) / 5 = 8$
- Group 3: $(10 + 5 + 0 + 5 + 30) / 5 = 10$

**Step 4 — Compute the Test Statistic:**

Using the formula for $W$ with $N = 15$ and $k = 3$, the test statistic is computed and compared to the F-distribution with $k - 1 = 2$ and $N - k = 12$ degrees of freedom.

**Step 5 — Decision Rule:**

If the computed $W$ exceeds the critical value from the F-distribution for $\alpha = 0.05$, reject the null hypothesis. Otherwise, fail to reject.

**Step 6 — Conclusion:**

Based on the test result, we either conclude that the variances are equal or that there is significant evidence that at least one group's variance is different.

---

## Fligner–Killeen Test

The Fligner–Killeen test is a non-parametric test for comparing variances across multiple groups. Like the Brown–Forsythe test, it uses the absolute deviations from the group median as a measure of dispersion. However, instead of operating on the raw deviations, the Fligner–Killeen test transforms these deviations into **ranks** and bases its analysis on the ranked values.

### Hypotheses

**Null Hypothesis ($H_0$):** The population variances are equal across the groups:

$$
H_0: \sigma_1^2 = \sigma_2^2 = \dots = \sigma_k^2
$$

**Alternative Hypothesis ($H_1$):** At least one group has a variance that differs from the others:

$$
H_1: \sigma_i^2 \neq \sigma_j^2 \quad \text{for at least one pair} \quad i \neq j
$$

### Test Statistic

**Step 1 — Compute Absolute Deviations:**

For each observation $X_{ij}$ in group $i$, compute the absolute deviation from the group median $\tilde{X}_i$:

$$
Z_{ij} = |X_{ij} - \tilde{X}_i|
$$

**Step 2 — Rank the Absolute Deviations:**

Rank the absolute deviations $Z_{ij}$ across all groups (pooled ranks):

$$
R_{ij} = \text{rank}(|X_{ij} - \tilde{X}_i|)
$$

where $R_{ij}$ represents the rank of the absolute deviation $Z_{ij}$ in the combined data.

**Step 3 — Calculate the Test Statistic:**

The test statistic is computed based on the ranks $R_{ij}$, weighted by group size $n_i$. A popular version of the Fligner–Killeen test statistic is:

$$
H = \sum_{i=1}^k n_i \left( \bar{R}_i - \bar{R} \right)^2 \sim \chi^2_{k-1}
$$

where:

- $k$ is the number of groups,
- $n_i$ is the number of observations in group $i$,
- $\bar{R}_i$ is the mean rank of the absolute deviations within group $i$:

$$
\bar{R}_i = \frac{1}{n_i} \sum_{j=1}^{n_i} R_{ij}
$$

- $\bar{R}$ is the overall mean rank of the absolute deviations across all groups:

$$
\bar{R} = \frac{1}{N} \sum_{i=1}^k \sum_{j=1}^{n_i} R_{ij}
$$

where $N = \sum_{i=1}^k n_i$ is the total number of observations.

This formula measures how much the group mean ranks $\bar{R}_i$ deviate from the overall mean rank $\bar{R}$, weighted by the group sizes $n_i$.

**Step 4 — Determine the Distribution:**

The test statistic follows an asymptotic chi-square distribution with $k - 1$ degrees of freedom.

### Assumptions

1. **Independence:** Observations must be independent within and between groups.
2. **Random Sampling:** The data should be drawn from random samples.

### Decision Rule

$$
H > \chi^2_{\text{critical}} \quad \Rightarrow \quad \text{Reject } H_0
$$
