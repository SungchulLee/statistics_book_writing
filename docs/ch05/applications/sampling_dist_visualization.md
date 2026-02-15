# Sampling Distribution Visualization: Effect of Sample Size

## Overview

This section demonstrates how the **sampling distribution** of the sample mean becomes more concentrated as sample size increases. Using realistic income data, we visualize the three-way distinction between the population distribution, a sample distribution, and the sampling distribution.

## The Three Distributions

When we repeatedly draw samples and compute statistics, we encounter three distinct distributions:

1. **Population Distribution**: The distribution of all values in the entire population
2. **Sample Distribution**: The distribution of values in a single, specific sample
3. **Sampling Distribution**: The distribution of a statistic (e.g., sample mean) computed from many different samples

This is central to the **Central Limit Theorem**: as sample size increases, the sampling distribution of the mean approaches a normal distribution, regardless of the shape of the population distribution.

## Empirical Demonstration with Loan Income Data

The following code uses real income data to show how the sampling distribution concentrates with larger sample sizes:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(seed=1)

# Load income data (or use simulated data with similar properties)
# loans_income is a Series of income values
# For demonstration, we'll create synthetic data with similar characteristics
np.random.seed(1)
# Simulate left-skewed income distribution (like real loan data)
loans_income = np.random.exponential(scale=50000, size=10000) + 20000
loans_income = pd.Series(loans_income)

# Create three datasets:
# 1. A sample of 1000 individual income values from the population
sample_data = pd.DataFrame({
    'income': loans_income.sample(1000),
    'type': 'Population Sample\n(n=1000)',
})

# 2. Sampling distribution when drawing samples of size 5
# (Draw 1000 samples, compute mean of each)
sample_mean_05 = pd.DataFrame({
    'income': [loans_income.sample(5).mean() for _ in range(1000)],
    'type': 'Sampling Distribution\n(Mean of 5)',
})

# 3. Sampling distribution when drawing samples of size 20
sample_mean_20 = pd.DataFrame({
    'income': [loans_income.sample(20).mean() for _ in range(1000)],
    'type': 'Sampling Distribution\n(Mean of 20)',
})

# Combine all three
results = pd.concat([sample_data, sample_mean_05, sample_mean_20], ignore_index=True)

print("Summary of the three distributions:")
print(results.groupby('type')['income'].agg(['count', 'mean', 'std', 'min', 'max']))
print()

# Visualize all three distributions
g = sns.FacetGrid(results, col='type', col_wrap=1, height=2.5, aspect=2.5)
g.map(plt.hist, 'income', bins=40, range=[0, 200000], color='steelblue', edgecolor='black')
g.set_axis_labels('Income ($)', 'Frequency')
g.set_titles('{col_name}')

# Adjust layout
for ax in g.axes.flat:
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()
```

## Interpreting the Visualization

### Population Sample (Top Panel)
Shows the actual distribution of income values from the population. This distribution is **right-skewed** with a long tail of high earners—typical of real income data.

### Sampling Distribution with n=5 (Middle Panel)
When we draw samples of just 5 people and compute their mean income, the distribution of these 1000 sample means is:
- More **concentrated** (narrower) than the population
- More **symmetric** (approaching normal shape)
- Still retains some of the right skew of the population

This is because with small sample sizes, individual extreme values heavily influence the mean.

### Sampling Distribution with n=20 (Bottom Panel)
With larger samples of 20 people:
- Even more **concentrated** around the true population mean
- Much more **bell-shaped** (approaching normal)
- The relationship is quantified by the standard error: $SE = \frac{\sigma}{\sqrt{n}}$

## Key Observations

### Standard Error Decreases with Sample Size

The **standard error** (standard deviation of the sampling distribution) is inversely proportional to $\sqrt{n}$:

$$SE(\bar{X}) = \frac{\sigma}{\sqrt{n}}$$

Comparing our simulations:
- For $n = 5$: $SE \approx \frac{\sigma}{\sqrt{5}} \approx 0.447\sigma$
- For $n = 20$: $SE \approx \frac{\sigma}{\sqrt{20}} \approx 0.224\sigma$

The standard error for $n=20$ is roughly half that of $n=5$, making estimates more precise.

```python
# Verify standard error relationship
pop_std = loans_income.std()
se_5 = pop_std / np.sqrt(5)
se_20 = pop_std / np.sqrt(20)

print(f"Population standard deviation: ${pop_std:,.0f}")
print(f"SE for n=5:  ${se_5:,.0f}")
print(f"SE for n=20: ${se_20:,.0f}")
print(f"Ratio SE(5)/SE(20): {se_5/se_20:.2f}")
```

### Convergence to Normality

The Central Limit Theorem states that regardless of the shape of the population distribution, the sampling distribution of the mean approaches normality as $n$ increases. Even though income is right-skewed, the sampling distributions become increasingly normal.

### Practical Implications

1. **Sample Size Planning**: To reduce uncertainty by half, we need to increase sample size by a factor of 4 (since $\sqrt{4} = 2$)
2. **Confidence Intervals**: Narrower sampling distributions lead to narrower confidence intervals
3. **Hypothesis Testing**: Larger samples provide more statistical power to detect true effects

## Quantitative Comparison

```python
import numpy as np
import pandas as pd

# Quantify the effect
np.random.seed(1)
loans_income = np.random.exponential(scale=50000, size=10000) + 20000

sample_means_5 = np.array([np.mean(np.random.choice(loans_income, 5)) for _ in range(1000)])
sample_means_20 = np.array([np.mean(np.random.choice(loans_income, 20)) for _ in range(1000)])

print("Sampling Distribution Comparison:")
print(f"{'Statistic':<20} {'n=5':<20} {'n=20':<20}")
print("-" * 60)
print(f"{'Mean':<20} ${sample_means_5.mean():>18,.0f} ${sample_means_20.mean():>18,.0f}")
print(f"{'Std Dev':<20} ${sample_means_5.std():>18,.0f} ${sample_means_20.std():>18,.0f}")
print(f"{'25th percentile':<20} ${np.percentile(sample_means_5, 25):>18,.0f} ${np.percentile(sample_means_20, 25):>18,.0f}")
print(f"{'75th percentile':<20} ${np.percentile(sample_means_5, 75):>18,.0f} ${np.percentile(sample_means_20, 75):>18,.0f}")
print(f"{'IQR':<20} ${np.percentile(sample_means_5, 75) - np.percentile(sample_means_5, 25):>18,.0f} ${np.percentile(sample_means_20, 75) - np.percentile(sample_means_20, 25):>18,.0f}")
```

## Summary

The sampling distribution demonstrates:
- **Statistical precision** improves as $1/\sqrt{n}$
- **Concentration** around the true population parameter increases with sample size
- **Normality** emerges even when the population is non-normal (CLT)
- **Practical trade-offs** between sample size and estimation accuracy

This fundamental concept underlies confidence intervals, hypothesis testing, and all statistical inference based on sample means.
