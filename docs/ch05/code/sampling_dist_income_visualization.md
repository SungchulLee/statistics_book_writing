# Sampling Distribution Income Visualization

## Overview

This page demonstrates the Central Limit Theorem in action using simulated income data. Income distributions are typically right-skewed, making them an excellent real-world example for showing how the sampling distribution of $\bar{X}$ becomes more concentrated and more normal as the sample size increases. By comparing individual incomes, means of 5, and means of 20, we can visually see the CLT at work and verify the standard error formula.

## Setup: Simulated Income Data

Real income data tends to be right-skewed with a long upper tail. We simulate this using a shifted Exponential distribution:

$$
\text{Income} = 20{,}000 + Y, \qquad Y \sim \text{Exp}(\text{scale} = 50{,}000)
$$

This produces a population with:

- **Mean**: approximately \$70,000
- **Standard deviation**: approximately \$50,000
- **Right skew**: the long tail captures the fact that a few individuals earn much more than the average

## Theoretical Background

For any population with mean $\mu$ and standard deviation $\sigma$, the sampling distribution of the sample mean $\bar{X}$ based on $n$ observations has:

$$
E[\bar{X}] = \mu, \qquad \text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}}
$$

The standard error decreases with sample size. Crucially, the ratio of standard errors for two different sample sizes is:

$$
\frac{\text{SE}(n_1)}{\text{SE}(n_2)} = \sqrt{\frac{n_2}{n_1}}
$$

For $n_1 = 5$ and $n_2 = 20$:

$$
\frac{\text{SE}(5)}{\text{SE}(20)} = \sqrt{\frac{20}{5}} = 2
$$

!!! tip "Key Insight"
    To halve the standard error, you must quadruple the sample size. This is the "square root law" of sampling.

## Simulation Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(1)

# Generate a right-skewed income population
n_population = 10_000
income = np.random.exponential(scale=50_000, size=n_population) + 20_000
loans_income = pd.Series(income)

# 1. Sample 1000 individual incomes
sample_data = pd.DataFrame({
    "income": loans_income.sample(1000),
    "type": "Population Sample (n=1000)"
})

# 2. Distribution of means with n=5
sample_mean_05 = pd.DataFrame({
    "income": [loans_income.sample(5).mean() for _ in range(1000)],
    "type": "Sampling Distribution (Mean of 5)"
})

# 3. Distribution of means with n=20
sample_mean_20 = pd.DataFrame({
    "income": [loans_income.sample(20).mean() for _ in range(1000)],
    "type": "Sampling Distribution (Mean of 20)"
})

results = pd.concat([sample_data, sample_mean_05, sample_mean_20],
                     ignore_index=True)

# Visualize side by side
g = sns.FacetGrid(results, col="type", col_wrap=1, height=2.5, aspect=2.5)
g.map(plt.hist, "income", bins=40, range=[0, 200_000],
      color="steelblue", edgecolor="black", alpha=0.8)
g.set_axis_labels("Income ($)", "Frequency")
g.set_titles("{col_name}")
plt.tight_layout()
plt.show()
```

## Standard Error Verification

We can verify the theoretical formula $\text{SE} = \sigma / \sqrt{n}$ against the empirical standard deviations of the simulated sampling distributions.

```python
pop_std = loans_income.std()
se_5_theory = pop_std / np.sqrt(5)
se_20_theory = pop_std / np.sqrt(20)

print(f"Population std:          ${pop_std:,.0f}")
print(f"Theoretical SE (n=5):    ${se_5_theory:,.0f}")
print(f"Theoretical SE (n=20):   ${se_20_theory:,.0f}")
print(f"Ratio SE(5)/SE(20):      {se_5_theory / se_20_theory:.2f}")
```

The ratio should be close to 2.0, confirming the square root law.

## Interpretation

!!! note "Key Observations"

    1. **Population sample** (top panel): The distribution of individual incomes is strongly right-skewed, with most incomes clustered near the lower end and a long tail extending past \$200,000.
    2. **Mean of 5** (middle panel): The sampling distribution is already much more concentrated than the population and slightly less skewed, but still visibly non-normal.
    3. **Mean of 20** (bottom panel): The sampling distribution is even more concentrated and closely approximates a normal distribution. The standard error is half that of the $n = 5$ case.

### Summary Statistics

| Distribution | Mean | Std Dev |
|---|---|---|
| Population sample | $\approx \$70{,}000$ | $\approx \$50{,}000$ |
| Sampling dist. ($n = 5$) | $\approx \$70{,}000$ | $\approx \$22{,}000$ |
| Sampling dist. ($n = 20$) | $\approx \$70{,}000$ | $\approx \$11{,}000$ |

All three distributions have the same mean (the population mean), but the spread decreases as $1/\sqrt{n}$.

## Exercises

**Exercise 1.** Explain why the mean of the sampling distribution equals the population mean regardless of sample size. Does this property depend on the population being normally distributed?

??? success "Solution to Exercise 1"
    By linearity of expectation:

    $$
    E[\bar{X}] = E\!\left[\frac{1}{n}\sum_{i=1}^n X_i\right] = \frac{1}{n}\sum_{i=1}^n E[X_i] = \frac{1}{n} \cdot n\mu = \mu
    $$

    This holds for **any** population with a finite mean, not just normal populations. The only requirements are that the observations are identically distributed with mean $\mu$ and that the expectation exists. The result does not depend on independence either (though independence is needed for the variance formula). $\square$

---

**Exercise 2.** The simulation uses \$50,000 as the Exponential scale parameter and adds \$20,000. Derive the exact population mean and standard deviation of this shifted Exponential distribution.

??? success "Solution to Exercise 2"
    If $Y \sim \text{Exp}(\text{scale} = \beta)$ where $\beta = 50{,}000$, then:

    $$
    E[Y] = \beta = 50{,}000, \qquad \text{Var}(Y) = \beta^2 = 2.5 \times 10^9
    $$

    For the shifted variable $X = 20{,}000 + Y$:

    $$
    E[X] = 20{,}000 + 50{,}000 = 70{,}000
    $$

    $$
    \text{Var}(X) = \text{Var}(Y) = 2.5 \times 10^9 \quad (\text{shift does not affect variance})
    $$

    $$
    \sigma_X = \sqrt{2.5 \times 10^9} = 50{,}000
    $$

    So the population has mean \$70,000 and standard deviation \$50,000. $\square$

---

**Exercise 3.** How large must $n$ be so that the standard error of $\bar{X}$ is at most \$5,000, given $\sigma = 50{,}000$?

??? success "Solution to Exercise 3"
    We need:

    $$
    \frac{50{,}000}{\sqrt{n}} \le 5{,}000 \implies \sqrt{n} \ge 10 \implies n \ge 100
    $$

    A sample size of at least $n = 100$ is needed. $\square$

---

**Exercise 4.** The code computes empirical standard errors from 1,000 simulated means. Explain why the empirical SE might differ slightly from the theoretical value. How would increasing the number of simulations from 1,000 to 100,000 affect this discrepancy?

??? success "Solution to Exercise 4"
    The empirical SE is itself a statistic computed from a finite number of simulations. It has its own sampling variability. Specifically, if $\hat{\text{SE}}$ is the standard deviation of $B$ simulated means, then the approximate standard error of $\hat{\text{SE}}$ itself is:

    $$
    \text{SE}(\hat{\text{SE}}) \approx \frac{\hat{\text{SE}}}{\sqrt{2B}}
    $$

    With $B = 1{,}000$: the precision of the estimated SE is about $\hat{\text{SE}} / \sqrt{2000} \approx \hat{\text{SE}} / 44.7$.

    With $B = 100{,}000$: the precision improves to $\hat{\text{SE}} / \sqrt{200{,}000} \approx \hat{\text{SE}} / 447$.

    Increasing simulations by a factor of 100 improves the precision of the SE estimate by a factor of 10. With 100,000 simulations, the empirical SE will be very close to the theoretical value. $\square$

---

**Exercise 5.** Suppose you are designing a survey to estimate mean household income in a city. Budget constraints limit you to $n = 50$ respondents. Using $\sigma \approx \$50{,}000$, compute the standard error and the approximate 95% margin of error for $\bar{X}$. If the budget doubles (allowing $n = 100$), how does the margin of error change?

??? success "Solution to Exercise 5"
    For $n = 50$:

    $$
    \text{SE} = \frac{50{,}000}{\sqrt{50}} = \frac{50{,}000}{7.071} \approx \$7{,}071
    $$

    The 95% margin of error is:

    $$
    \text{MOE} = 1.96 \times 7{,}071 \approx \$13{,}859
    $$

    For $n = 100$:

    $$
    \text{SE} = \frac{50{,}000}{\sqrt{100}} = \$5{,}000
    $$

    $$
    \text{MOE} = 1.96 \times 5{,}000 = \$9{,}800
    $$

    Doubling the sample size reduces the margin of error by a factor of $\sqrt{2} \approx 1.41$, from about \$13,859 to about \$9,800. This is a 29% reduction in margin of error for a 100% increase in cost, illustrating the diminishing returns of increasing sample size. $\square$
