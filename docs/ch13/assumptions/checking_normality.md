# Checking Normality in Linear Regression

Normality of the residuals is one of the key assumptions in linear regression. This assumption posits that the residuals (errors) of the model should follow a normal distribution. Although the linearity assumption is more critical for the validity of the regression coefficients, checking the normality of residuals is important for ensuring the validity of confidence intervals, hypothesis tests, and prediction intervals. This section outlines methods to check for normality in linear regression, including visual inspections and statistical tests.

## 1. Understanding Normality in Regression

**Definition:**
Normality in linear regression refers to the residuals being normally distributed, meaning that when plotted, they should form a bell-shaped curve centered around zero:

$$
\epsilon_i \sim N(0, \sigma^2)
$$

**Why It Matters:**
While normality of residuals is not required for unbiased estimation of regression coefficients, it is crucial for:

- **Confidence Intervals and Hypothesis Tests:** These rely on the assumption of normally distributed errors to be accurate. The t-distribution used for coefficient tests is derived under the normality assumption.
- **Prediction Intervals:** Normality ensures that prediction intervals for new observations are valid.
- **Small samples:** For large samples, the Central Limit Theorem provides asymptotic normality of the test statistics even without normal errors. For small samples ($n < 30$), the normality assumption becomes critical.

## 2. Histogram of Residuals

One of the simplest methods to assess normality is by plotting a **histogram of the residuals**. This visual inspection helps determine whether the residuals are approximately normally distributed.

**Steps:**

1. **Fit the Linear Regression Model:** Obtain the residuals from your model.
2. **Create the Histogram:** Plot the residuals using a histogram.
3. **Assess the Plot:** Compare the shape of the histogram to that of a normal distribution.

**Example:**

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Assuming 'model' is your fitted OLS model
residuals = model.resid

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(residuals, bins=30, density=True, edgecolor='black', alpha=0.7, label='Residuals')

# Overlay a normal distribution curve
x = np.linspace(residuals.min(), residuals.max(), 100)
ax.plot(x, norm.pdf(x, loc=residuals.mean(), scale=residuals.std()),
        'r-', linewidth=2, label='Normal PDF')

ax.set_xlabel('Residuals')
ax.set_ylabel('Density')
ax.set_title('Histogram of Residuals with Normal Overlay')
ax.legend()
plt.show()
```

**Interpretation:**

- **Normality:** The histogram should resemble a bell-shaped curve, symmetric around zero.
- **Non-Normality:** If the histogram is skewed, has multiple peaks (multimodal), or is too flat (platykurtic) or too peaked (leptokurtic), the residuals may not be normally distributed.

## 3. Q-Q Plot (Quantile-Quantile Plot)

The **Q-Q plot** is a more precise visual tool for assessing normality. It compares the quantiles of the residuals to the quantiles of a standard normal distribution.

**How it works:**

1. Sort the residuals and compute their empirical quantiles.
2. Compute the corresponding theoretical quantiles from a standard normal distribution.
3. Plot empirical quantiles (y-axis) against theoretical quantiles (x-axis).
4. If the residuals are normally distributed, the points will fall along the 45-degree reference line.

**Steps:**

1. **Fit the Linear Regression Model:** Obtain the residuals.
2. **Create the Q-Q Plot:** Use the residuals to create the plot.
3. **Assess the Plot:** Check whether the residuals follow the 45-degree reference line.

**Example:**

```python
import scipy.stats as stats
import matplotlib.pyplot as plt

# Assuming 'model' is your fitted OLS model
residuals = model.resid

fig, ax = plt.subplots(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot of Residuals')
plt.show()
```

**Interpretation:**

- **Normality:** If the points closely follow the 45-degree line, the residuals are likely normally distributed.
- **Non-Normality:** Deviations from the line, especially at the tails, indicate departures from normality.

**Common Q-Q plot patterns:**

| Pattern | Indicates |
|---------|-----------|
| Points follow the line | Normal distribution |
| S-shaped curve | Heavy tails (leptokurtic) |
| Inverted S-shape | Light tails (platykurtic) |
| Curve upward at both ends | Right-skewed distribution |
| Curve downward at both ends | Left-skewed distribution |

## 4. Shapiro-Wilk Test

The **Shapiro-Wilk test** is a statistical test specifically designed to test the normality of the residuals. It is one of the most powerful tests for detecting departures from normality.

**Hypotheses:**

- $H_0$: The residuals are normally distributed
- $H_1$: The residuals are not normally distributed

**Test Statistic:**

$$
W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}
$$

where $x_{(i)}$ are the ordered residuals and $a_i$ are constants generated from the expected values of order statistics of a standard normal distribution.

**Steps:**

1. **Fit the Linear Regression Model:** Obtain the residuals.
2. **Perform the Shapiro-Wilk Test:** Apply the test to your residuals.
3. **Interpret the Results:** A significant p-value (typically < 0.05) suggests that the residuals are not normally distributed.

**Example:**

```python
from scipy.stats import shapiro

# Assuming 'model' is your fitted OLS model
residuals = model.resid

stat, p_value = shapiro(residuals)
print(f'Shapiro-Wilk statistic: {stat:.4f}')
print(f'Shapiro-Wilk p-value: {p_value:.4f}')

if p_value > 0.05:
    print('No significant evidence of non-normality (fail to reject H0)')
else:
    print('Significant evidence of non-normality (reject H0)')
```

**Interpretation:**

- **p-value > 0.05:** No significant evidence of non-normality; the residuals can be considered normally distributed.
- **p-value < 0.05:** Significant evidence of non-normality, indicating a potential violation of the assumption.

**Note:** The Shapiro-Wilk test is limited to sample sizes of $n \leq 5000$ in most implementations.

## 5. Anderson-Darling Test

The **Anderson-Darling test** is another statistical test that is particularly sensitive to deviations in the tails of the distribution, making it a good choice for checking normality in residuals.

**Key Feature:** The Anderson-Darling test gives more weight to observations in the tails of the distribution compared to other tests like the Kolmogorov-Smirnov test, making it better at detecting departures from normality in the tails.

**Test Statistic:**

$$
A^2 = -n - \sum_{i=1}^{n} \frac{2i-1}{n} \left[\ln F(x_{(i)}) + \ln(1-F(x_{(n+1-i)}))\right]
$$

where $F$ is the CDF of the hypothesized normal distribution and $x_{(i)}$ are the ordered residuals.

**Steps:**

1. **Fit the Linear Regression Model:** Obtain the residuals.
2. **Perform the Anderson-Darling Test:** Apply the test to your residuals.
3. **Interpret the Results:** The test provides a statistic and critical values at various significance levels.

**Example:**

```python
from scipy.stats import anderson

# Assuming 'model' is your fitted OLS model
residuals = model.resid

result = anderson(residuals, dist='norm')
print(f'Anderson-Darling statistic: {result.statistic:.4f}')
print()
for i in range(len(result.critical_values)):
    sig_level = result.significance_level[i]
    crit_value = result.critical_values[i]
    status = 'REJECT' if result.statistic > crit_value else 'Fail to reject'
    print(f'At {sig_level}% significance: Critical value = {crit_value:.4f} → {status}')
```

**Interpretation:**

- If the test statistic is **less than** the critical value at a given significance level, fail to reject $H_0$ — the residuals are consistent with normality.
- If the test statistic is **greater than** the critical value, reject $H_0$ — the residuals are not normally distributed at that significance level.

## 6. Jarque-Bera Test

The **Jarque-Bera test** is a goodness-of-fit test that specifically tests whether the skewness and kurtosis of the residuals match that of a normal distribution.

**Test Statistic:**

$$
JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)
$$

where:

- $n$ is the sample size
- $S$ is the sample skewness (should be 0 for a normal distribution)
- $K$ is the sample kurtosis (should be 3 for a normal distribution)

Under $H_0$ (normality), $JB \sim \chi^2(2)$.

**Steps:**

1. **Fit the Linear Regression Model:** Obtain the residuals.
2. **Perform the Jarque-Bera Test:** Apply the test to your residuals.
3. **Interpret the Results:** A significant p-value indicates non-normality in terms of skewness and/or kurtosis.

**Example:**

```python
from statsmodels.stats.stattools import jarque_bera

# Assuming 'model' is your fitted OLS model
residuals = model.resid

jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)
print(f'Jarque-Bera statistic: {jb_stat:.4f}')
print(f'Jarque-Bera p-value: {jb_pvalue:.4f}')
print(f'Skewness: {skew:.4f}')
print(f'Kurtosis: {kurtosis:.4f}')
```

**Interpretation:**

- **p-value > 0.05:** No significant evidence of non-normality in skewness or kurtosis.
- **p-value < 0.05:** Significant evidence of non-normality, suggesting a potential violation of the assumption.

## Comparison of Normality Tests

| Test | Type | Sensitive To | Sample Size | Key Advantage |
|------|------|-------------|-------------|---------------|
| Histogram | Visual | Overall shape | Any | Intuitive, easy to interpret |
| Q-Q Plot | Visual | Tail behavior | Any | Precise, reveals type of non-normality |
| Shapiro-Wilk | Formal | General departures | $n \leq 5000$ | Most powerful for small samples |
| Anderson-Darling | Formal | Tail departures | Any | Good for detecting heavy tails |
| Jarque-Bera | Formal | Skewness and kurtosis | Large $n$ | Based on asymptotic theory |

## Practical Recommendations

1. **Start with visual methods** — Histograms and Q-Q plots provide immediate insight into the type and severity of non-normality.
2. **Confirm with formal tests** — Use Shapiro-Wilk for small samples and Jarque-Bera or Anderson-Darling for larger samples.
3. **Consider the sample size** — With large samples, even minor deviations from normality will be statistically significant but may be practically unimportant.
4. **Focus on the tails** — Departures in the tails of the distribution are more problematic than minor deviations near the center.

If residuals are found to be non-normal, various remedies including transforming the dependent variable (log, Box-Cox), using robust regression techniques, or applying bootstrap methods can help address the issue. Properly assessing and addressing the normality assumption ensures more reliable inference from your linear regression models.
