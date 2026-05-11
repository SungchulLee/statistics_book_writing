# Derivation and Chi-Square Approximation

Bartlett's test statistic can be motivated through the likelihood ratio principle. Under the null hypothesis that all $k$ groups share a common variance, the likelihood function simplifies, and the ratio of the restricted to unrestricted likelihoods leads directly to the familiar Bartlett formula. This section presents the derivation step by step and explains the correction factor that ensures an accurate chi-square approximation.

## Setup and Notation

Suppose $k$ independent samples are drawn from normal populations:

$$
X_{ij} \sim N(\mu_i, \sigma_i^2), \quad i = 1, \ldots, k, \quad j = 1, \ldots, n_i
$$

Let $N = \sum_{i=1}^{k} n_i$ denote the total sample size, $S_i^2$ the sample variance of group $i$, and define the pooled variance

$$
S_p^2 = \frac{\sum_{i=1}^{k} (n_i - 1) S_i^2}{N - k}
$$

which is the weighted average of the group variances, with weights proportional to degrees of freedom $\nu_i = n_i - 1$.

## Likelihood Ratio

Under the null hypothesis $H_0\colon \sigma_1^2 = \cdots = \sigma_k^2 = \sigma^2$, the maximum likelihood estimator of the common variance is $\hat{\sigma}^2 = S_p^2$. Under the alternative, each group has its own MLE $\hat{\sigma}_i^2 = S_i^2$.

The log-likelihood ratio statistic is

$$
-2 \ln \Lambda = \sum_{i=1}^{k} \nu_i \ln\!\left(\frac{S_p^2}{S_i^2}\right) = (N-k)\ln S_p^2 - \sum_{i=1}^{k} \nu_i \ln S_i^2
$$

where $\nu_i = n_i - 1$. This quantity is always non-negative (by the concavity of the logarithm and Jensen's inequality) and equals zero if and only if all sample variances are identical.

## The Correction Factor

The statistic $-2\ln\Lambda$ converges to $\chi^2_{k-1}$ as the sample sizes grow, but the convergence is slow for moderate sample sizes. Bartlett (1937) introduced a correction factor to improve the chi-square approximation:

$$
C = 1 + \frac{1}{3(k-1)}\left(\sum_{i=1}^{k} \frac{1}{\nu_i} - \frac{1}{N - k}\right)
$$

The corrected test statistic is

$$
T = \frac{-2\ln\Lambda}{C} = \frac{(N-k)\ln S_p^2 - \sum_{i=1}^{k} \nu_i \ln S_i^2}{1 + \frac{1}{3(k-1)}\left(\sum_{i=1}^{k} \frac{1}{\nu_i} - \frac{1}{N-k}\right)}
$$

Under $H_0$ and normality, $T \stackrel{\text{approx}}{\sim} \chi^2_{k-1}$.

!!! note "Purpose of the Correction"
    The correction factor $C$ is always greater than 1, so dividing by $C$ shrinks the test statistic. Without the correction, the uncorrected statistic $-2\ln\Lambda$ tends to be too large in small samples, leading to an inflated Type I error rate. The correction brings the actual rejection rate closer to the nominal level $\alpha$.

## Intuition Behind the Formula

The numerator of $T$ can be interpreted as follows:

- $(N - k)\ln S_p^2$ is the log of the pooled variance, weighted by the total degrees of freedom.
- $\sum \nu_i \ln S_i^2$ is the sum of the log group variances, each weighted by its own degrees of freedom.

If all group variances are equal, then $S_i^2 \approx S_p^2$ for every $i$, and the numerator is near zero. If one or more groups have substantially different variances, the individual $\ln S_i^2$ values diverge from $\ln S_p^2$, and the numerator becomes large.

The test statistic effectively measures how much the individual log-variances deviate from the pooled log-variance, adjusting for sample sizes.

## Decision Rule

Reject $H_0$ at significance level $\alpha$ if

$$
T > \chi^2_{1-\alpha,\, k-1}
$$

where $\chi^2_{1-\alpha,\, k-1}$ is the $(1-\alpha)$ quantile of the chi-square distribution with $k - 1$ degrees of freedom. The test is always one-sided (right-tailed) because any departure from equal variances increases $T$.

## Example

Three groups have the following sample sizes and variances:

| Group | $n_i$ | $S_i^2$ | $\nu_i = n_i - 1$ |
|---|---|---|---|
| 1 | 10 | 5.2 | 9 |
| 2 | 12 | 8.1 | 11 |
| 3 | 8 | 4.7 | 7 |

**Step 1.** Compute totals: $N = 30$, $N - k = 27$.

**Step 2.** Pooled variance:

$$
S_p^2 = \frac{9(5.2) + 11(8.1) + 7(4.7)}{27} = \frac{46.8 + 89.1 + 32.9}{27} = \frac{168.8}{27} = 6.252
$$

**Step 3.** Numerator:

$$
27 \ln(6.252) - [9\ln(5.2) + 11\ln(8.1) + 7\ln(4.7)]
$$

$$
= 27(1.8331) - [9(1.6487) + 11(2.0919) + 7(1.5476)]
$$

$$
= 49.494 - [14.838 + 23.011 + 10.833] = 49.494 - 48.682 = 0.812
$$

**Step 4.** Correction factor:

$$
C = 1 + \frac{1}{3(2)}\left(\frac{1}{9} + \frac{1}{11} + \frac{1}{7} - \frac{1}{27}\right) = 1 + \frac{1}{6}(0.1111 + 0.0909 + 0.1429 - 0.0370) = 1 + \frac{0.3079}{6} = 1.0513
$$

**Step 5.** Test statistic:

$$
T = \frac{0.812}{1.0513} = 0.772
$$

**Step 6.** Compare with $\chi^2_{0.95,\, 2} = 5.991$. Since $0.772 < 5.991$, we fail to reject $H_0$. There is insufficient evidence to conclude that the group variances differ.

## Python Verification

```python
import numpy as np
from scipy import stats

# Group data
n = np.array([10, 12, 8])
s2 = np.array([5.2, 8.1, 4.7])
k = len(n)
nu = n - 1
N = n.sum()

# Pooled variance
s2_pooled = np.sum(nu * s2) / np.sum(nu)

# Numerator
numerator = np.sum(nu) * np.log(s2_pooled) - np.sum(nu * np.log(s2))

# Correction factor
C = 1 + (1 / (3 * (k - 1))) * (np.sum(1 / nu) - 1 / np.sum(nu))

# Test statistic
T = numerator / C

# p-value
p_value = stats.chi2.sf(T, k - 1)

print(f"Pooled variance: {s2_pooled:.3f}")
print(f"Test statistic T: {T:.3f}")
print(f"Correction factor C: {C:.4f}")
print(f"P-value: {p_value:.4f}")
```


## Exercises

**Exercise 1.**
Describe the main concept of Derivation and Chi-Square Approximation and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Derivation and Chi-Square Approximation is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

---

**Exercise 2.**
State the key assumptions required by the method discussed here. How can each assumption be checked?

??? success "Solution to Exercise 2"
    The main assumptions typically include: (1) independence of observations -- verified by understanding the data collection process and checking for serial correlation; (2) distributional requirements (e.g., normality) -- checked with Q-Q plots and formal tests like Shapiro-Wilk; (3) equal variances (if applicable) -- assessed with boxplots and Levene's test. When assumptions are violated, consider robust alternatives, transformations, or nonparametric methods.

---

**Exercise 3.**
Work through a small numerical example illustrating the application of the technique from this section.

??? success "Solution to Exercise 3"
    A structured approach to applying this technique involves: (1) clearly stating the hypotheses or estimation goal; (2) verifying that the data meet the required assumptions; (3) computing the relevant test statistic, estimate, or model fit; (4) obtaining the p-value, confidence interval, or posterior distribution; (5) interpreting the result in the context of the original question. Following these steps systematically ensures a rigorous and reproducible analysis.

---

**Exercise 4.**
Compare the approach from this section with an alternative method. When would you choose each?

??? success "Solution to Exercise 4"
    The method discussed here is appropriate when its assumptions hold and the sample size is sufficient for the asymptotic approximations to be accurate. Alternative approaches include: (1) nonparametric methods -- preferred when distributional assumptions are suspect; (2) bootstrap methods -- useful when analytical reference distributions are unavailable; (3) Bayesian methods -- valuable when incorporating prior information or when direct probability statements about parameters are desired. Running multiple approaches and comparing results provides a useful robustness check.
