# Chi-Square Goodness-of-Fit (Manual)

## Overview

This page demonstrates how to compute the chi-square goodness-of-fit test statistic and p-value **manually** using NumPy and SciPy, without relying on the convenience function `scipy.stats.chisquare`. The example uses a rock-paper-scissors dataset with three equally likely categories. A visualization of the chi-square distribution is included, highlighting the tail probability that corresponds to the p-value.

## Hypotheses

- **Null Hypothesis** ($H_0$): The outcomes follow an equal (uniform) distribution across all categories.
- **Alternative Hypothesis** ($H_A$): The outcomes do not follow an equal distribution.

## Test Statistic

Given $k$ categories with observed counts $O_i$ and expected counts $E_i$, the chi-square statistic is

$$
\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}
$$

Under $H_0$ this statistic approximately follows a $\chi^2$ distribution with

$$
\text{df} = k - 1
$$

degrees of freedom, provided every expected count is at least 5.

## Code

The script below computes the statistic from scratch and plots the $\chi^2$ density with the rejection region shaded.

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Observed data and expected mean-based values
observed_counts = np.array([4, 13, 7])
expected_counts = np.ones(3) * observed_counts.mean()
degrees_of_freedom = observed_counts.shape[0] - 1

# Chi-square test statistic and p-value calculation
chi_square_statistic = np.sum(
    (observed_counts - expected_counts) ** 2 / expected_counts
)
p_value = stats.chi2(degrees_of_freedom).sf(chi_square_statistic)

print(f"Chi-square Statistic = {chi_square_statistic:.4f}")
print(f"p-value = {p_value:.4f}")
```

**Step-by-step breakdown:**

1. **Observed counts** are stored as a NumPy array: $[4,\;13,\;7]$.
2. **Expected counts** under $H_0$ are each $24 / 3 = 8$.
3. The **test statistic** is computed element-wise:

$$
\chi^2 = \frac{(4-8)^2}{8} + \frac{(13-8)^2}{8} + \frac{(7-8)^2}{8} = 2 + 3.125 + 0.125 = 5.25
$$

4. The **p-value** is the survival function (upper-tail probability) of the $\chi^2(2)$ distribution evaluated at $5.25$.

### Visualization

```python
fig, ax = plt.subplots(figsize=(12, 4))

# Left portion of the chi-square pdf (non-rejection region)
x_left = np.linspace(0, chi_square_statistic, 100)
y_left = stats.chi2(degrees_of_freedom).pdf(x_left)
ax.plot(x_left, y_left, linewidth=3)
x_fill_left = np.concatenate([[0], x_left, [chi_square_statistic], [0]])
y_fill_left = np.concatenate([[0], y_left, [0], [0]])
ax.fill(x_fill_left, y_fill_left, alpha=0.1)

# Right tail (rejection region)
x_right = np.linspace(chi_square_statistic, 20, 100)
y_right = stats.chi2(degrees_of_freedom).pdf(x_right)
ax.plot(x_right, y_right, linewidth=3)
x_fill_right = np.concatenate(
    [[chi_square_statistic], x_right, [20], [chi_square_statistic]]
)
y_fill_right = np.concatenate([[0], y_right, [0], [0]])
ax.fill(x_fill_right, y_fill_right, alpha=0.1)

# Annotate p-value
ax.annotate(
    f"p-value = {p_value:.02%}",
    xy=((12.5 + 15.0) / 2, 0.01),
    xytext=(16.5, 0.10),
    fontsize=15,
    arrowprops=dict(width=0.2, headwidth=8),
)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_position("zero")
ax.spines["left"].set_position("zero")
plt.tight_layout()
plt.show()
```

The shaded right tail represents the probability of observing a $\chi^2$ value at least as extreme as $5.25$ under $H_0$.

## Interpretation

With $\chi^2 = 5.25$ and $\text{df} = 2$, the p-value is approximately $0.0724$. At the conventional significance level $\alpha = 0.05$, we **fail to reject** $H_0$. There is not enough evidence to conclude that the outcomes deviate from a uniform distribution.

Key points to remember:

- The manual computation makes explicit what `scipy.stats.chisquare` does internally.
- The visualization reinforces the connection between the test statistic, the $\chi^2$ distribution, and the p-value.
- Always verify the expected-count condition ($E_i \ge 5$) before trusting the chi-square approximation.

## Exercises

**1.** A six-sided die is rolled 120 times. The observed counts for faces 1 through 6 are $[15, 22, 18, 25, 20, 20]$. Compute the chi-square statistic by hand and determine the degrees of freedom.

??? success "Solution to Exercise 1"

    Under $H_0$ each face has expected count $E_i = 120/6 = 20$. The statistic is

    $$
    \chi^2 = \frac{(15-20)^2}{20} + \frac{(22-20)^2}{20} + \frac{(18-20)^2}{20} + \frac{(25-20)^2}{20} + \frac{(20-20)^2}{20} + \frac{(20-20)^2}{20}
    $$

    $$
    = \frac{25}{20} + \frac{4}{20} + \frac{4}{20} + \frac{25}{20} + 0 + 0 = 1.25 + 0.20 + 0.20 + 1.25 + 0 + 0 = 2.90
    $$

    Degrees of freedom: $\text{df} = 6 - 1 = 5$. $\square$

---

**2.** Explain why `stats.chi2(df).sf(statistic)` is used to compute the p-value rather than `stats.chi2(df).cdf(statistic)`.

??? success "Solution to Exercise 2"

    The chi-square goodness-of-fit test is always a **right-tailed** test. Large values of the statistic indicate departure from $H_0$. The p-value is therefore $P(\chi^2 \ge \text{observed statistic})$, which is the upper-tail probability. The survival function `sf` computes $1 - F(x)$ where $F$ is the CDF, giving exactly this upper-tail area. Using `cdf` directly would give the left-tail probability, which is the complement of what we need. $\square$

---

**3.** Suppose the expected proportions are not uniform but instead $(p_1, p_2, p_3) = (0.5, 0.3, 0.2)$ with a total of 24 observations. Recompute the expected counts and the chi-square statistic using the same observed counts $[4, 13, 7]$.

??? success "Solution to Exercise 3"

    Expected counts: $E_1 = 0.5 \times 24 = 12$, $E_2 = 0.3 \times 24 = 7.2$, $E_3 = 0.2 \times 24 = 4.8$.

    $$
    \chi^2 = \frac{(4-12)^2}{12} + \frac{(13-7.2)^2}{7.2} + \frac{(7-4.8)^2}{4.8}
    $$

    $$
    = \frac{64}{12} + \frac{33.64}{7.2} + \frac{4.84}{4.8} = 5.333 + 4.672 + 1.008 = 11.014
    $$

    With $\text{df} = 2$, the p-value is approximately $0.0041$. At $\alpha = 0.05$ we would reject $H_0$. $\square$

---

**4.** Why must expected counts (not observed counts) appear in the denominator of each term of the chi-square statistic? What would go wrong if we used observed counts instead?

??? success "Solution to Exercise 4"

    The chi-square statistic standardizes each squared deviation $(O_i - E_i)^2$ by the **expected** count $E_i$ because, under $H_0$, the variance of the count in category $i$ is approximately $E_i$ (for a multinomial distribution each cell count is approximately Poisson with mean $E_i$, and the variance of a Poisson random variable equals its mean). Dividing by $E_i$ therefore produces a quantity whose sampling distribution is approximately $\chi^2$.

    If observed counts were used instead, categories that happen to have very small observed counts could produce artificially inflated contributions (division by a near-zero number), while the resulting statistic would no longer follow a $\chi^2$ distribution. The theoretical justification relies specifically on the expected counts under $H_0$. $\square$

---

**5.** Prove that for $k = 2$ categories, the chi-square goodness-of-fit statistic reduces to the square of the one-proportion z-test statistic.

??? success "Solution to Exercise 5"

    Let the two categories have observed counts $O_1$ and $O_2 = n - O_1$ and expected counts $E_1 = np_0$ and $E_2 = n(1 - p_0)$. Then

    $$
    \chi^2 = \frac{(O_1 - np_0)^2}{np_0} + \frac{(O_2 - n(1-p_0))^2}{n(1-p_0)}
    $$

    Since $O_2 = n - O_1$, the second numerator is $(n - O_1 - n + np_0)^2 = (np_0 - O_1)^2 = (O_1 - np_0)^2$. Factor out $(O_1 - np_0)^2$:

    $$
    \chi^2 = (O_1 - np_0)^2 \left[\frac{1}{np_0} + \frac{1}{n(1-p_0)}\right] = (O_1 - np_0)^2 \cdot \frac{1}{np_0(1-p_0)}
    $$

    Writing $\hat{p} = O_1/n$, we get $O_1 - np_0 = n(\hat{p} - p_0)$, so

    $$
    \chi^2 = \frac{n^2(\hat{p} - p_0)^2}{np_0(1-p_0)} = \left(\frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}}\right)^2 = z^2
    $$

    which is exactly the square of the standard one-proportion z-test statistic. $\square$
