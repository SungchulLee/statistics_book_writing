# Two-Sample t-Test (Weed Prices)

## Overview

This page applies a complete hypothesis-testing workflow to California high-quality weed prices in January 2014 versus January 2015. The four steps are: check normality with the Shapiro-Wilk test, construct confidence intervals, run an independent two-sample $t$-test, and perform a chi-square goodness-of-fit test on quality-tier proportions. The example emphasizes the importance of verifying assumptions before applying parametric tests.

## Step 1 -- Normality Check (Shapiro-Wilk)

The two-sample $t$-test assumes each sample comes from a normal population. The Shapiro-Wilk test checks this:

$$
H_0\colon \text{the data are normally distributed}, \qquad H_1\colon \text{the data are not normally distributed}.
$$

A large p-value ($> 0.05$) means we have no evidence against normality.

### Code

```python
import numpy as np
from scipy import stats

np.random.seed(42)

CA_JAN2014 = np.array([
    248.75, 248.59, 248.63, 248.37, 248.02, 247.68, 247.36,
    246.85, 246.44, 246.06, 245.81, 245.48, 245.18, 244.87,
    244.55, 244.23, 243.89, 243.60, 243.34, 243.08, 242.85,
    242.64, 242.36, 242.15, 241.88, 241.64, 241.40, 241.14,
    240.91, 240.65, 240.42,
])

CA_JAN2015 = np.array([
    245.02, 244.88, 244.76, 244.65, 244.53, 244.42, 244.30,
    244.18, 244.08, 243.97, 243.85, 243.74, 243.63, 243.52,
    243.40, 243.28, 243.17, 243.06, 242.95, 242.83, 242.72,
    242.61, 242.49, 242.38, 242.27, 242.15, 242.04, 241.93,
    241.81, 241.70, 241.59,
])

for label, data in [("Jan 2014", CA_JAN2014), ("Jan 2015", CA_JAN2015)]:
    stat, p = stats.shapiro(data)
    print(f"Shapiro-Wilk ({label}): W={stat:.4f}, p={p:.4f}")
```

## Step 2 -- Confidence Interval for the Mean

A $100(1-\alpha)\%$ confidence interval for the mean, assuming normality, is

$$
\bar{x} \pm t_{\alpha/2,\, n-1} \cdot \frac{s}{\sqrt{n}}.
$$

### Code

```python
def confidence_interval(data, confidence=0.95):
    n = len(data)
    m = data.mean()
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return m - h, m + h

ci14 = confidence_interval(CA_JAN2014)
ci15 = confidence_interval(CA_JAN2015)
print(f"Jan 2014 95% CI: [{ci14[0]:.2f}, {ci14[1]:.2f}]")
print(f"Jan 2015 95% CI: [{ci15[0]:.2f}, {ci15[1]:.2f}]")
```

If the two intervals do not overlap, this is informal evidence that the means differ (though non-overlapping CIs is a more conservative criterion than the $t$-test).

## Step 3 -- Independent Two-Sample t-Test

We test whether the mean prices in the two years are equal:

$$
H_0\colon \mu_{2014} = \mu_{2015}, \qquad H_1\colon \mu_{2014} \neq \mu_{2015}.
$$

The pooled two-sample $t$-statistic (assuming equal variances) is

$$
t = \frac{\bar{x} - \bar{y}}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}, \qquad s_p^2 = \frac{(n_1 - 1)s_x^2 + (n_2 - 1)s_y^2}{n_1 + n_2 - 2}.
$$

### Code

```python
t_stat, p_val = stats.ttest_ind(CA_JAN2014, CA_JAN2015, equal_var=True)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {p_val:.6f}")
print("Reject H0" if p_val < 0.05 else "Fail to reject H0")
```

## Step 4 -- Chi-Square Goodness-of-Fit

To check whether the quality-tier distribution (High, Medium, Low) changed between years, we use the chi-square goodness-of-fit test:

$$
\chi^2 = \sum_{i=1}^k \frac{(O_i - E_i)^2}{E_i},
$$

where $O_i$ are observed counts and $E_i$ are expected counts based on the 2014 distribution. Under $H_0$ (no change), $\chi^2 \sim \chi^2_{k-1}$.

### Code

```python
expected_2014 = np.array([453020, 688699, 271937])
observed_2015 = np.array([461900, 695432, 267120])

chi2, p_chi = stats.chisquare(observed_2015, f_exp=expected_2014)
print(f"Chi-square stat: {chi2:.2f}, p-value: {p_chi:.6f}")
```

## Interpretation

- **Normality**: Both January 2014 and January 2015 price samples pass the Shapiro-Wilk test, confirming that the $t$-test assumptions are reasonable.
- **Confidence intervals**: The 2014 mean is higher (around \$244) than the 2015 mean (around \$243), and if the intervals are separated, this suggests a real price decline.
- **Two-sample $t$-test**: If the p-value is below 0.05, we conclude that high-quality weed prices in California decreased significantly from January 2014 to January 2015.
- **Chi-square test**: The goodness-of-fit test checks whether the proportions of customers buying high, medium, and low quality changed between years. A significant result indicates a shift in purchasing patterns.

## Exercises

**Exercise 1.** The Shapiro-Wilk test has low power for small samples. If $n = 10$ and the data are slightly non-normal, what is likely to happen? How does this affect the reliability of a subsequent $t$-test?

??? success "Solution to Exercise 1"

    With $n = 10$, the Shapiro-Wilk test has low power: it is unlikely to reject normality even if the data come from a moderately non-normal distribution. This means the test may "pass" normality even when the assumption is violated.

    However, the $t$-test is fairly robust to mild non-normality, especially for symmetric distributions. For small $n$, a better strategy is to combine the Shapiro-Wilk test with a Q-Q plot for visual assessment. If serious non-normality is suspected, a nonparametric alternative (e.g., the Mann-Whitney $U$ test) should be used instead. $\square$

---

**Exercise 2.** Compute the 99% confidence interval for the mean of the January 2014 prices. How does it compare to the 95% interval?

??? success "Solution to Exercise 2"

    ```python
    ci99 = confidence_interval(CA_JAN2014, confidence=0.99)
    print(f"99% CI: [{ci99[0]:.2f}, {ci99[1]:.2f}]")
    ```

    The 99% CI uses $t_{0.005,\,30}$ instead of $t_{0.025,\,30}$. Since $t_{0.005} > t_{0.025}$, the margin of error is larger and the interval is wider. The general formula is

    $$
    \text{width} = 2 \cdot t_{\alpha/2,\,n-1} \cdot \frac{s}{\sqrt{n}}.
    $$

    Higher confidence requires a wider interval to maintain the stated coverage probability. $\square$

---

**Exercise 3.** Suppose the equal-variance assumption is questionable. Run a Welch $t$-test and compare the result. When should you prefer Welch over the pooled test?

??? success "Solution to Exercise 3"

    ```python
    t_w, p_w = stats.ttest_ind(CA_JAN2014, CA_JAN2015, equal_var=False)
    print(f"Welch t = {t_w:.4f}, p = {p_w:.6f}")
    ```

    The Welch test does not assume equal variances and uses the Welch-Satterthwaite degrees of freedom:

    $$
    \text{df} = \frac{\left(\frac{s_x^2}{n_1} + \frac{s_y^2}{n_2}\right)^2}{\frac{(s_x^2/n_1)^2}{n_1-1} + \frac{(s_y^2/n_2)^2}{n_2-1}}.
    $$

    In practice, the Welch test is almost as powerful as the pooled test when variances are equal and much more reliable when they are not. Many statisticians recommend using Welch's test by default. $\square$

---

**Exercise 4.** In the chi-square goodness-of-fit test, why do we use the 2014 proportions as expected values? What would change if we rescaled the expected counts to match the 2015 total?

??? success "Solution to Exercise 4"

    The null hypothesis is that the 2015 quality distribution is the same as 2014. The expected counts should therefore reflect 2014 proportions applied to the 2015 total sample size:

    $$
    E_i = n_{2015} \cdot \frac{O_{i,2014}}{n_{2014}}.
    $$

    If we use raw 2014 counts directly as expected values (as in the code), `scipy.stats.chisquare` internally rescales them so $\sum E_i = \sum O_i$. This rescaling is essential because the chi-square statistic requires $\sum O_i = \sum E_i$. Without rescaling, the test would conflate differences in sample size with differences in proportions. $\square$

---

**Exercise 5.** If the $t$-test yields $p = 0.03$ and the effect size (difference of means) is \$1.20, discuss whether this result is practically significant. What additional information would help?

??? success "Solution to Exercise 5"

    Statistical significance ($p = 0.03 < 0.05$) means the observed difference is unlikely under $H_0$, but it does not address whether the difference matters in practice. A price difference of \$1.20 on a base price of roughly \$244 is about 0.5%, which may be economically negligible depending on context.

    Additional useful information includes:

    - **Cohen's $d$**: $d = (\bar{x} - \bar{y}) / s_p$ measures the effect in standard-deviation units. A $d$ below 0.2 is conventionally considered "small."
    - **Confidence interval for the difference**: e.g., [\$0.15, \$2.25] tells us the range of plausible differences.
    - **Domain context**: Is \$1.20 meaningful to buyers or sellers? Does it exceed transaction costs or measurement error?

    A statistically significant but practically negligible result is common with large samples, where even tiny effects produce small p-values. $\square$
