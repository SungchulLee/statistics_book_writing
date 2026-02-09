# Goodness-of-Fit Test

## Overview

The **Goodness-of-Fit Test** is a statistical procedure used to evaluate the degree to which an observed frequency distribution matches an expected distribution, usually for discrete or categorical data. This test assesses whether the observed frequencies of events across different categories align with the expected frequencies predicted by a specific theoretical model or hypothesis. By comparing observed and expected counts, the test determines if any significant differences exist, which might suggest that the observed data does not fit the expected pattern.

The most widely used Goodness-of-Fit Test is the **Chi-Square Goodness-of-Fit Test**, which is particularly effective for categorical data. This test calculates the difference between observed and expected frequencies across categories and uses this information to evaluate the likelihood that any observed variation is due to random chance rather than a deviation from the expected distribution.

## Example Scenario

Suppose a company claims that its market share across three age groups—18-25, 26-40, and 41+—is evenly distributed, with one-third of its customers falling into each age category. To verify this claim, a researcher randomly surveys 300 customers and records their age groups. The Goodness-of-Fit Test can be used to determine if the observed distribution of customers across these age groups significantly differs from the company's claimed distribution, or if the actual customer age distribution fits the company's stated proportions.

## Hypotheses

- **Null Hypothesis ($H_0$)**: The observed data follows the expected distribution.
- **Alternative Hypothesis ($H_A$)**: The observed data does not follow the expected distribution.

## Frequency Table

In a Goodness-of-Fit Test, we use a **frequency table** that shows:

1. **Categories or Groups**: The different categories for a single variable (e.g., age groups, product preferences, etc.).
2. **Observed Frequencies**: The actual counts observed in each category from the sample data.
3. **Expected Frequencies**: The counts expected in each category based on a hypothesized distribution (often given as a proportion or percentage for each category).

For example, if we are testing whether customer age distribution fits a company's claim that customers are evenly distributed across three age groups (18-25, 26-40, and 41+), the frequency table might look like this:

| Age Group | Observed Frequency | Expected Frequency |
|-----------|--------------------|--------------------|
| 18-25     | 90                 | 100                |
| 26-40     | 120                | 100                |
| 41+       | 90                 | 100                |

The Goodness-of-Fit Test compares the observed frequencies to the expected frequencies to see if there is a significant difference, indicating a possible deviation from the claimed distribution.

## Expected Frequencies

The **expected frequencies** are the theoretical counts we would expect in each category if the null hypothesis were true.

### Uniform Distribution

If we expect the counts to be evenly distributed across all categories, then each category's expected frequency is simply:

$$
E_i = \frac{\text{Total Sample Size}}{\text{Number of Categories}}
$$

where $E_i$ is the expected frequency for each category.

For example, if we have 300 customers distributed equally across three age groups, each age group's expected frequency would be:

$$
E_i = \frac{300}{3} = 100
$$

### Proportional Distribution

Often, the expected distribution is not uniform but rather follows specific proportions. In this case, the expected frequency for each category is calculated by:

$$
E_i = p_i \times \text{Total Sample Size}
$$

where $p_i$ is the expected proportion for category $i$.

For instance, if a company believes 40% of its customers are aged 18-25, 30% are aged 26-40, and 30% are aged 41+, the expected frequencies for a sample of 300 customers would be:

- Age 18-25: $E = 0.4 \times 300 = 120$
- Age 26-40: $E = 0.3 \times 300 = 90$
- Age 41+: $E = 0.3 \times 300 = 90$

## Test Statistic

The test statistic for the Chi-Square Goodness-of-Fit Test is calculated using the formula

$$
\chi^2 = \sum_{i=1}^k \frac{(O_i - E_i)^2}{E_i}
$$

where:

- $O_i$ = Observed frequency in category $i$
- $E_i$ = Expected frequency in category $i$

## Degrees of Freedom

The degrees of freedom ($\text{df}$) for the Chi-Square Goodness-of-Fit Test is calculated as

$$
\text{df} = k - 1
$$

where $k$ is the number of categories.

## Critical Region

$$
\begin{array}{lll}
\text{Null} & \text{The observed data follows the expected distribution} \\
& \text{Observed frequencies are close to expected frequencies} \\
& O_{i} \approx E_{i} \quad \Rightarrow \quad \text{statistic} \approx 0 \\
\\
\text{Alternative} & \text{The observed data does not follow the expected distribution} \\
& \text{Observed frequencies are quite different from expected frequencies} \\
& O_{i} \not\approx E_{i} \quad \Rightarrow \quad \text{statistic} \approx \text{large positive number}
\end{array}
$$

## Critical Value and p-Value

- **Critical Value**: Based on the degrees of freedom and the chosen significance level (e.g., 0.05), the critical value is obtained from the Chi-Square distribution table.
- **p-Value**: The p-value is calculated from the Chi-Square distribution using the test statistic and degrees of freedom. It represents the probability of observing a test statistic as extreme as, or more extreme than, the one calculated under the null hypothesis.

## Decision Rule

- If the test statistic exceeds the critical value, or if the p-value is less than the significance level, reject the null hypothesis. This indicates that the observed frequencies significantly differ from the expected frequencies.
- If the test statistic does not exceed the critical value, or if the p-value is greater than the significance level, fail to reject the null hypothesis. This suggests that the observed data fits the expected distribution well.

## Assumptions and Limitations

### Assumptions

1. **Random Sampling**: The observations should be randomly sampled from the population to ensure that the sample is representative.
2. **Expected Frequency Threshold**: Each category should ideally have an expected frequency of at least 5. If expected frequencies are too low, the chi-square approximation may not be accurate.
3. **Mutually Exclusive Categories**: The categories must be mutually exclusive, meaning each observation falls into only one category.
4. **Independence of Observations**: Each observation should be independent of others. In other words, the presence of one observation in a category should not influence others.

### Limitations

1. **Sensitivity to Small Sample Sizes**: The Chi-Square Goodness-of-Fit Test relies on the large sample approximation. If expected frequencies in some categories are low, the test may not be valid. In cases with small sample sizes or low expected frequencies, an alternative method like an **exact test** may be preferred.
2. **Number of Categories**: When there are too many categories (particularly with low expected counts), the chi-square approximation may lose accuracy, potentially leading to unreliable results.
3. **One-Dimensional Analysis**: The Goodness-of-Fit Test is designed to test a single variable against a hypothesized distribution. It cannot assess relationships between two or more variables, as is done in the Chi-Square Test of Independence.
4. **Approximation Limitation**: The Chi-Square Test statistic is an approximation that may be less accurate for distributions with very skewed or non-normal expected frequencies.

When these assumptions are met, the Goodness-of-Fit Test can provide valid and useful insights. However, if they are violated, the results should be interpreted cautiously, and alternative methods may be more appropriate.

---

## Example A: Rock-Paper-Scissors

> **Source**: [Khan Academy — Goodness of Fit Example](https://www.khanacademy.org/math/ap-statistics/chi-square-tests/chi-square-goodness-fit/v/goodness-of-fit-example)

Kenny frequently engages in rock-paper-scissors games with the expectation that he will win, tie, and lose with equal frequency. However, he begins to suspect that his games do not conform to this anticipated pattern. To investigate his suspicion, Kenny collects a random sample of 24 games and records their outcomes:

|                 | Win | Loss | Tie |
|:---------------:|:---:|:----:|:---:|
| Number of Games | 4   | 13   | 7   |

Kenny wants to use the recorded results to conduct a $\chi^2$ goodness-of-fit test to assess whether the distribution of his outcomes deviates from an even distribution. What are the test statistic and p-value?

### Solution

**Step 1: Define Hypotheses**

- **Null Hypothesis** $H_0$: The outcomes (win, loss, tie) are equally likely, meaning they follow an even distribution.
- **Alternative Hypothesis** $H_1$: The outcomes do not follow an even distribution.

**Step 2: Observed and Expected Frequencies**

| Outcome  | Observed | Expected | $(O_i - E_i)^2 / E_i$       |
|:--------:|:--------:|:--------:|:----------------------------:|
| Win      | 4        | 8        | $(4 - 8)^2 / 8 = 2$         |
| Loss     | 13       | 8        | $(13 - 8)^2 / 8 = 3.125$    |
| Tie      | 7        | 8        | $(7 - 8)^2 / 8 = 0.125$     |
| $\chi^2$ |          |          | 5.25                         |

Total games: 24. Expected frequency for each outcome (if outcomes are evenly distributed): $24 / 3 = 8$.

**Step 3: Calculate the $\chi^2$ Test Statistic**

$$
\chi^2 = \frac{(4 - 8)^2}{8} + \frac{(13 - 8)^2}{8} + \frac{(7 - 8)^2}{8} = 5.25
$$

**Step 4: Degrees of Freedom**

$$
\text{df} = k - 1 = 3 - 1 = 2
$$

**Step 5: Find the p-value**

Using a chi-square distribution with 2 degrees of freedom, the p-value for $\chi^2 = 5.25$ is approximately:

$$
p \approx 0.0725
$$

**Conclusion**: With a p-value of approximately 0.0725, if Kenny is testing at a significance level of 0.05, he would fail to reject the null hypothesis. This suggests there is not strong evidence that the distribution of his outcomes significantly deviates from an even distribution.

### Python Implementation (Without `scipy.stats.chisquare`)

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Observed data and expected mean-based values
observed_counts = np.array([4, 13, 7])
expected_counts = np.ones(3) * observed_counts.mean()
degrees_of_freedom = observed_counts.shape[0] - 1

# Chi-square test statistic and p-value calculation
chi_square_statistic = np.sum((observed_counts - expected_counts) ** 2 / expected_counts)
p_value = stats.chi2(degrees_of_freedom).sf(chi_square_statistic)

# Display the statistic and p-value
print(f"Chi-square Statistic = {chi_square_statistic:.4f}")
print(f"p-value = {p_value:.4f}\n")

# Plotting setup
fig, ax = plt.subplots(figsize=(12, 4))

# Chi-square distribution plot up to observed statistic
x_left = np.linspace(0, chi_square_statistic, 100)
y_left = stats.chi2(degrees_of_freedom).pdf(x_left)
ax.plot(x_left, y_left, color='b', linewidth=3)

# Fill left area under the curve (non-significant region)
x_fill_left = np.concatenate([[0], x_left, [chi_square_statistic], [0]])
y_fill_left = np.concatenate([[0], y_left, [0], [0]])
ax.fill(x_fill_left, y_fill_left, color='b', alpha=0.1)

# Chi-square distribution plot for tail area (significant region)
x_right = np.linspace(chi_square_statistic, 20, 100)
y_right = stats.chi2(degrees_of_freedom).pdf(x_right)
ax.plot(x_right, y_right, color='r', linewidth=3)

# Fill right area under the curve (significant region)
x_fill_right = np.concatenate([[chi_square_statistic], x_right, [20], [chi_square_statistic]])
y_fill_right = np.concatenate([[0], y_right, [0], [0]])
ax.fill(x_fill_right, y_fill_right, color='r', alpha=0.1)

# Annotate p-value with an arrow
annotation_xy = ((12.5 + 15.0) / 2, 0.01)
annotation_xytext = (16.5, 0.10)
arrow_properties = dict(color='k', width=0.2, headwidth=8)
ax.annotate(f'p-value = {p_value:.02%}', annotation_xy, xytext=annotation_xytext,
            fontsize=15, arrowprops=arrow_properties)

# Customize plot aesthetics
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_position("zero")
ax.spines['left'].set_position("zero")

plt.show()
```

### Python Implementation (With `scipy.stats.chisquare`)

```python
from scipy import stats

# Observed frequencies for each outcome: Win, Loss, Tie
observed_frequencies = [4, 13, 7]

# Expected frequencies assuming an even distribution
total_games = sum(observed_frequencies)
expected_frequencies = [total_games / 3] * 3

# Perform the chi-square goodness-of-fit test
chi_square_statistic, p_value = stats.chisquare(f_obs=observed_frequencies, f_exp=expected_frequencies)

# Output results
print(f"{chi_square_statistic = }")
print(f"{p_value = }")
```

---

## Example B: Loaded Dice?

### Question

We have a die. To test whether it is loaded, we roll it 60 times, and here is the outcome. Decide whether this die is loaded.

$$
\begin{array}{crr}
 & \text{observed} & \text{expected} \\
\text{value} & \text{frequency} & \text{frequency} \\ \hline
1 & 5 & 10 \\
2 & 7 & 10 \\
3 & 17 & 10 \\
4 & 14 & 10 \\
5 & 8 & 10 \\
6 & 9 & 10 \\ \hline
\text{sum} & 60 & 60
\end{array}
$$

### Why Not Test Each Category Individually?

One might notice that the value 3 appeared 17 times out of 60 and attempt a one-proportion z-test:

```python
import numpy as np
import scipy.stats as stats

def main():
    p_0 = 1/6
    p_hat = 17 / 60
    n = 60

    # z test statistic
    statistic = (p_hat - p_0) / np.sqrt(p_0 * (1 - p_0) / n)

    # two-sided test
    p_value = stats.norm().sf(abs(statistic)) * 2

    print(f"{statistic = :.02f}")
    print(f"{p_value   = :.02%}")

if __name__ == "__main__":
    main()
```

Is this good enough evidence that the die is loaded?

Not too fast. We can run a similar test for each row. If we have many rows, we will eventually see a very small p-value. It is like playing Russian roulette — if we keep on going, sooner or later, we will hit a very small p-value even if the die is fair. So, we cannot conclude that the die is loaded from a single-category test. We need a test that considers **all categories simultaneously**.

### Hypotheses

$$
\begin{array}{lll}
\text{Null} & \text{Die is not loaded} \\
\\
\text{Alternative} & \text{Die is loaded}
\end{array}
$$

### Test Statistic

```python
import numpy as np

def main():
    observed = np.array([5, 7, 17, 14, 8, 9])
    expected = np.array([10] * 6)
    statistic = np.sum((observed - expected)**2 / expected)
    print(f'{statistic = }')

if __name__ == "__main__":
    main()
```

### Critical Region

$$
\begin{array}{lll}
\text{Null} & \text{Die is not loaded} \\
& \text{Observed frequencies are close to expected frequencies} \\
& O_i \approx E_i \quad \Rightarrow \quad \text{statistic} \approx 0 \\
\\
\text{Alternative} & \text{Die is loaded} \\
& \text{Observed frequencies are quite different from expected frequencies} \\
& O_i \not\approx E_i \quad \Rightarrow \quad \text{statistic} \approx \text{large positive number}
\end{array}
$$

### Sampling Distribution

$$
\sum_{i=1}^k \frac{(O_i - E_i)^2}{E_i}
= \sum_{i=1}^k \left(\frac{\left(\sum_{j=1}^{n} X_j\right) - np_i}{\sqrt{np_i}}\right)^2
\approx \sum_{i=1}^k Z_i^2
= \chi^2_{k-1}
$$

In this die case:

$$
\sum_{i=1}^6 \frac{(O_i - E_i)^2}{E_i} \approx \chi^2_5
$$

### Conditions for Test (Rule of Thumb)

The $\chi^2$ approximation is good when all the expected frequencies are five or more. In this die case, all the expected frequencies are 10, which satisfies the condition. Hence, the $\chi^2$ approximation is reasonable.

### p-value

$$
\text{p-value} = P\left(\sum_{i=1}^k \frac{(O_i - E_i)^2}{E_i} \ge \text{statistic} \;\middle|\; H_0\right)
$$

### Conclusion

$$\text{The die is not loaded.}$$

### Python Implementation (Without `scipy.stats.chisquare`)

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def main():
    observed = np.array([5, 7, 17, 14, 8, 9])
    expected = np.array([10] * 6)
    df = observed.shape[0] - 1

    statistic = np.sum((observed - expected)**2 / expected)
    p_value = stats.chi2(df).sf(statistic)
    print(f"{statistic = :.02f}")
    print(f"{p_value    = :.02%}")

    _, ax = plt.subplots(figsize=(12, 4))

    x = np.linspace(0, statistic)
    y = stats.chi2(df).pdf(x)
    ax.plot(x, y, color='b', linewidth=3)

    x = np.concatenate([[0], x, [statistic], [0]])
    y = np.concatenate([[0], y, [0], [0]])
    ax.fill(x, y, color='b', alpha=0.1)

    x = np.linspace(statistic, 20, 100)
    y = stats.chi2(df).pdf(x)
    ax.plot(x, y, color='r', linewidth=3)

    x = np.concatenate([[statistic], x, [20], [statistic]])
    y = np.concatenate([[0], y, [0], [0]])
    ax.fill(x, y, color='r', alpha=0.1)

    xy = ((12.5 + 15.0) / 2, 0.01)
    xytext = (16.5, 0.10)
    arrowprops = dict(color='k', width=0.2, headwidth=8)
    ax.annotate(f'{p_value = :.02%}', xy, xytext=xytext, fontsize=15, arrowprops=arrowprops)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position("zero")
    ax.spines['left'].set_position("zero")

    plt.show()

if __name__ == "__main__":
    main()
```

### Python Implementation (With `scipy.stats.chisquare`)

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def main():
    observed = np.array([5, 7, 17, 14, 8, 9])
    expected = np.array([10] * 6)
    df = observed.shape[0] - 1

    statistic, p_value = stats.chisquare(observed, f_exp=expected)
    print(f"{statistic = :.02f}")
    print(f"{p_value    = :.02%}")

    _, ax = plt.subplots(figsize=(12, 4))

    x = np.linspace(0, statistic)
    y = stats.chi2(df).pdf(x)
    ax.plot(x, y, color='b', linewidth=3)

    x = np.concatenate([[0], x, [statistic], [0]])
    y = np.concatenate([[0], y, [0], [0]])
    ax.fill(x, y, color='b', alpha=0.1)

    x = np.linspace(statistic, 20, 100)
    y = stats.chi2(df).pdf(x)
    ax.plot(x, y, color='r', linewidth=3)

    x = np.concatenate([[statistic], x, [20], [statistic]])
    y = np.concatenate([[0], y, [0], [0]])
    ax.fill(x, y, color='r', alpha=0.1)

    xy = ((12.5 + 15.0) / 2, 0.01)
    xytext = (16.5, 0.10)
    arrowprops = dict(color='k', width=0.2, headwidth=8)
    ax.annotate(f'{p_value = :.02%}', xy, xytext=xytext, fontsize=15, arrowprops=arrowprops)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position("zero")
    ax.spines['left'].set_position("zero")

    plt.show()

if __name__ == "__main__":
    main()
```
