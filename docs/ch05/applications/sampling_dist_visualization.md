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

## Exercises

**Exercise 1.**
Population is skewed with $\mu = 50$, $\sigma = 10$. Sample $n = 100$. (a) What is the approximate shape of the sampling distribution of $\bar X$? (b) Its mean and SE?

??? success "Solution to Exercise 1"
    (a) By CLT (large $n$), $\bar X$ is approximately **normal** despite population skew.

    (b) Mean: $\mu = 50$. SE: $\sigma/\sqrt n = 10/10 = 1$.

    So $\bar X \approx N(50, 1)$. The skew of the underlying population transfers to a residual (small) skew in the sampling distribution but is dominated by CLT-induced normality at $n = 100$.

---

**Exercise 2.**
**Sample size and convergence rate.** For an exponential population (skewness 2), at what $n$ does the sampling distribution of $\bar X$ become "approximately normal"? Justify using Berry-Esseen.

??? success "Solution to Exercise 2"
    Berry-Esseen bound: $\sup_x |F_{\bar X_n}(x) - \Phi((x - \mu)/(\sigma/\sqrt n))| \le C \cdot \rho/(\sigma^3 \sqrt n)$, where $\rho = \mathbb{E}|X - \mu|^3$.

    For $\mathrm{Exp}(1)$: $\rho \approx 2.0$, $\sigma = 1$. Bound: $0.5 \cdot 2 / \sqrt n = 1/\sqrt n$.

    For "approximately normal" with max KS distance $\le 0.05$: $\sqrt n \ge 1/0.05 = 20$, so $n \ge 400$.

    For $\le 0.10$: $n \ge 100$.

    **In practice:** $n = 30$ is sufficient for mild skew; $n = 100$ for moderate skew (e.g., exponential); $n = 1000+$ for heavy skew or heavy tails. Always plot to verify.

---

**Exercise 3.**
**Bootstrap as alternative.** When the population shape is unknown and $n$ is moderate, the bootstrap gives a non-parametric estimate of the sampling distribution. Outline the procedure.

??? success "Solution to Exercise 3"
    1. Given sample $X_1, \ldots, X_n$ from unknown population.
    2. Draw $B$ bootstrap samples, each of size $n$ with replacement.
    3. Compute $\bar X^*_b$ for each bootstrap sample.
    4. The collection $\{\bar X^*_1, \ldots, \bar X^*_B\}$ approximates the sampling distribution of $\bar X$.

    Use this distribution to:

    - Estimate SE: sample SD of $\{\bar X^*_b\}$.
    - Construct CI: 2.5th and 97.5th percentiles of $\{\bar X^*_b\}$ for a 95% **percentile interval**.

    Bootstrap captures skewness, heavy tails, and other features that the CLT-based normal approximation misses. Especially valuable when $n$ is too small for CLT but too large for exact small-sample inference.

---

**Exercise 4.**
**Visualizing the CLT.** Describe a sequence of plots demonstrating the CLT for exponential samples at $n = 1, 5, 30, 100$.

??? success "Solution to Exercise 4"
    For each $n$, generate many (say $B = 10000$) samples of size $n$ from $\mathrm{Exp}(1)$, compute $\bar X_n$ for each, plot a histogram.

    Expected pattern:

    - $n = 1$: histogram looks exponential (right-skewed, peaked at 0).
    - $n = 5$: still visibly skewed but less so; appears unimodal with longer right tail.
    - $n = 30$: approximately normal with mean 1 and SD $1/\sqrt{30} \approx 0.18$. Slight residual right skew.
    - $n = 100$: clearly normal-shaped, SD $\approx 0.1$.

    Overlaying $N(1, 1/n)$ density on each histogram makes the CLT convergence visible. The narrowing of the bell and the disappearance of skew tell the story.

    A second useful plot: Q-Q plot of $\bar X_n$ vs. normal at each $n$. Points fall increasingly on the diagonal line.

---

**Exercise 5.**
**Effect of population variance.** For a population with $\mu = 50$, compare the sampling distributions of $\bar X$ at $n = 100$ for $\sigma = 5, 10, 50$.

??? success "Solution to Exercise 5"
    All three sampling distributions are approximately $N(\mu, \sigma^2/n) = N(50, \sigma^2/100)$:

    - $\sigma = 5$: $\bar X \approx N(50, 0.25)$, SD = 0.5.
    - $\sigma = 10$: $\bar X \approx N(50, 1.0)$, SD = 1.
    - $\sigma = 50$: $\bar X \approx N(50, 25)$, SD = 5.

    All centered at 50; the SE scales linearly with $\sigma$. Higher population variability spreads the sampling distribution proportionally.

    **Implication for sample-size planning:** to achieve a target precision $\mathrm{SE} = \sigma_{\text{target}}$, need $n = (\sigma/\sigma_{\text{target}})^2$. High-variability populations require much larger samples to achieve the same precision.

---

**Exercise 6.**
**Sampling distribution of the median.** Briefly contrast with the sampling distribution of the mean: shape, SE formula, robustness.

??? success "Solution to Exercise 6"
    For a sample of size $n$ from population with density $f$ and median $m$:

    - **Shape:** asymptotically normal (median has its own CLT under regularity conditions).
    - **SE formula:** $\mathrm{SE}(\tilde X) \approx 1/(2 f(m) \sqrt n)$. Depends on the density at the median — large $f(m)$ gives small SE.

    **Comparison:**

    | Statistic | Bias | Variance | Robustness |
    |---|---|---|---|
    | Mean | 0 | $\sigma^2/n$ | Sensitive to outliers |
    | Median | 0 | $1/(4 n f(m)^2)$ | Robust |

    For normal data: $\mathrm{Var}(\tilde X)/\mathrm{Var}(\bar X) = \pi/2 \approx 1.57$ — the median is less efficient (about 64% efficiency).

    For heavy-tailed data (e.g., $t_3$ or Laplace): the median is *more* efficient than the mean. The mean's variance balloons because of outliers.

    The choice between mean and median should match the data: mean for clean symmetric data, median for outlier-prone or heavy-tailed data.
