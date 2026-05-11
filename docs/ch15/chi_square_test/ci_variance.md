# Confidence Interval for the Population Variance

A hypothesis test tells us whether to reject a specific value of $\sigma^2$, but a confidence interval provides a range of plausible values. The confidence interval for the population variance follows directly from the pivotal quantity derived in the previous section: $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$ under normality. Because the chi-square distribution is right-skewed, the resulting interval is asymmetric around $S^2$.

## Derivation from the Pivotal Quantity

Starting from the distributional result

$$
\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}
$$

we can write a probability statement for any $0 < \alpha < 1$:

$$
P\!\left(\chi^2_{\alpha/2,\, n-1} \le \frac{(n-1)S^2}{\sigma^2} \le \chi^2_{1-\alpha/2,\, n-1}\right) = 1 - \alpha
$$

where $\chi^2_{\alpha/2,\, n-1}$ denotes the lower $\alpha/2$ quantile and $\chi^2_{1-\alpha/2,\, n-1}$ denotes the upper $\alpha/2$ quantile of the chi-square distribution with $n - 1$ degrees of freedom.

Inverting the inequality by dividing all three parts into $(n-1)S^2$ and flipping the direction gives

$$
P\!\left(\frac{(n-1)S^2}{\chi^2_{1-\alpha/2,\, n-1}} \le \sigma^2 \le \frac{(n-1)S^2}{\chi^2_{\alpha/2,\, n-1}}\right) = 1 - \alpha
$$

## Confidence Interval Formula

The $100(1-\alpha)\%$ confidence interval for the population variance $\sigma^2$ is

$$
\left(\frac{(n-1)S^2}{\chi^2_{1-\alpha/2,\, n-1}},\; \frac{(n-1)S^2}{\chi^2_{\alpha/2,\, n-1}}\right)
$$

The corresponding confidence interval for the population standard deviation $\sigma$ is obtained by taking square roots:

$$
\left(\sqrt{\frac{(n-1)S^2}{\chi^2_{1-\alpha/2,\, n-1}}},\; \sqrt{\frac{(n-1)S^2}{\chi^2_{\alpha/2,\, n-1}}}\right)
$$

!!! note "Asymmetry of the Interval"
    Unlike confidence intervals for the mean (which are symmetric around $\bar{X}$), confidence intervals for the variance are asymmetric around $S^2$. The upper bound extends further from $S^2$ than the lower bound does. This asymmetry reflects the right-skewed shape of the chi-square distribution.

## One-Sided Confidence Bounds

In some applications, only an upper or lower bound on $\sigma^2$ is needed.

**Upper confidence bound** (at level $1 - \alpha$):

$$
\sigma^2 \le \frac{(n-1)S^2}{\chi^2_{\alpha,\, n-1}}
$$

**Lower confidence bound** (at level $1 - \alpha$):

$$
\sigma^2 \ge \frac{(n-1)S^2}{\chi^2_{1-\alpha,\, n-1}}
$$

The upper bound is useful when the concern is that the variance is too large (e.g., quality control), while the lower bound applies when the concern is that the variance is too small (e.g., ensuring sufficient variability for a sampling plan).

## Example

A random sample of $n = 25$ light bulbs from a production line has a sample variance of $S^2 = 120$ (hours$^2$). Construct a 95% confidence interval for the population variance $\sigma^2$, assuming the lifetimes are normally distributed.

**Step 1.** Identify the relevant quantities:

- $n = 25$, so $n - 1 = 24$
- $S^2 = 120$
- $\alpha = 0.05$

**Step 2.** Find the chi-square critical values for $\nu = 24$ degrees of freedom:

- $\chi^2_{0.025,\, 24} = 12.401$
- $\chi^2_{0.975,\, 24} = 39.364$

**Step 3.** Compute the confidence interval:

$$
\left(\frac{24 \times 120}{39.364},\; \frac{24 \times 120}{12.401}\right) = (73.22,\; 232.24)
$$

We are 95% confident that the population variance lies between 73.22 and 232.24 hours$^2$.

**Step 4.** For the standard deviation:

$$
\left(\sqrt{73.22},\; \sqrt{232.24}\right) = (8.56,\; 15.24)
$$

The 95% confidence interval for $\sigma$ is approximately $(8.56, 15.24)$ hours.

## Width of the Confidence Interval

The width of the confidence interval for $\sigma^2$ depends on two factors:

1. **Sample size.** Larger $n$ means more degrees of freedom, which narrows the gap between the chi-square quantiles and produces a tighter interval.
2. **Confidence level.** Higher confidence (smaller $\alpha$) widens the interval because the quantiles move further into the tails.

For small samples, the interval can be very wide, reflecting substantial uncertainty about $\sigma^2$. The ratio of the upper to lower endpoint provides a useful measure:

$$
\text{Width ratio} = \frac{\chi^2_{1-\alpha/2,\, n-1}}{\chi^2_{\alpha/2,\, n-1}}
$$

As $n \to \infty$, this ratio approaches 1, and the interval shrinks toward the point estimate $S^2$.

## Duality with Hypothesis Testing

The confidence interval and the chi-square test for $\sigma^2$ are dual procedures. A value $\sigma_0^2$ lies outside the $100(1-\alpha)\%$ confidence interval if and only if the chi-square test rejects $H_0\colon \sigma^2 = \sigma_0^2$ at significance level $\alpha$.

## Python Implementation

```python
import numpy as np
from scipy import stats

# Sample data
n = 25
s_squared = 120
alpha = 0.05
df = n - 1

# Chi-square critical values
chi2_lower = stats.chi2.ppf(alpha / 2, df)
chi2_upper = stats.chi2.ppf(1 - alpha / 2, df)

# Confidence interval for variance
ci_lower = df * s_squared / chi2_upper
ci_upper = df * s_squared / chi2_lower

# Confidence interval for standard deviation
ci_sd_lower = np.sqrt(ci_lower)
ci_sd_upper = np.sqrt(ci_upper)

print(f"95% CI for variance: ({ci_lower:.2f}, {ci_upper:.2f})")
print(f"95% CI for std dev:  ({ci_sd_lower:.2f}, {ci_sd_upper:.2f})")
```


## Exercises

**Exercise 1.**
Describe the main concept of Confidence Interval for the Population Variance and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Confidence Interval for the Population Variance is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
