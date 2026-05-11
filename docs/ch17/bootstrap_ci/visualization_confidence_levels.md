# Bootstrap Confidence Intervals: Visual Interpretation of Confidence Levels


## Overview

This section demonstrates how to visualize bootstrap confidence intervals at different confidence levels (e.g., 90% vs 95%). Visual comparison clarifies what "confidence level" truly means: the procedure's long-run coverage property, not a probability statement about a particular interval.

## The Bootstrap Confidence Interval Procedure

The bootstrap method for constructing confidence intervals follows these steps:

1. **Sample from the sample**: Repeatedly resample (with replacement) from the original sample
2. **Compute bootstrap statistics**: For each resample, compute the statistic of interest (e.g., mean, median)
3. **Extract quantiles**: Use the percentiles of bootstrap statistics to form the interval

For an $\alpha$ significance level, the confidence interval is:

$$[\hat{F}_{\alpha/2}^*, \, \hat{F}_{1-\alpha/2}^*]$$

where $\hat{F}_q^*$ denotes the $q$-th quantile of the bootstrap distribution.

## Example: Bootstrap Confidence Intervals for Mean Income

This example constructs 90% and 95% confidence intervals for mean income using real loan data:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Set random seed
np.random.seed(seed=3)

# Simulate income data (or load real loan income data)
np.random.seed(seed=3)
loans_income = np.random.exponential(scale=50000, size=5000) + 20000

# Draw a single sample of n=20 from the population
original_sample = resample(loans_income, n_samples=20, replace=False)
original_mean = original_sample.mean()

print(f"Original sample size: {len(original_sample)}")
print(f"Original sample mean: ${original_mean:,.0f}")
print()

# Bootstrap procedure: resample from the sample 500 times
bootstrap_means = []
for _ in range(500):
    bootstrap_sample = resample(original_sample)  # with replacement
    bootstrap_means.append(bootstrap_sample.mean())

bootstrap_means = pd.Series(bootstrap_means)

# Compute confidence intervals
ci_90_lower, ci_90_upper = bootstrap_means.quantile([0.05, 0.95])
ci_95_lower, ci_95_upper = bootstrap_means.quantile([0.025, 0.975])

print("90% Confidence Interval: [${:,.0f}, ${:,.0f}]".format(ci_90_lower, ci_90_upper))
print("95% Confidence Interval: [${:,.0f}, ${:,.0f}]".format(ci_95_lower, ci_95_upper))
print(f"Mean of bootstrap means: ${bootstrap_means.mean():,.0f}")
print()

# Visualize both confidence levels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: 90% Confidence Interval
ax1.hist(bootstrap_means, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(ci_90_lower, color='darkred', linestyle='--', linewidth=2.5, label='90% CI limits')
ax1.axvline(ci_90_upper, color='darkred', linestyle='--', linewidth=2.5)

# Shade the confidence interval region
ax1.axvspan(ci_90_lower, ci_90_upper, alpha=0.2, color='green', label='90% CI')

# Add annotation
ci_90_mid = (ci_90_lower + ci_90_upper) / 2
ax1.text(ci_90_mid, 35, f'90% CI\n[${ci_90_lower:,.0f}, ${ci_90_upper:,.0f}]',
         ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='darkred', linewidth=1.5))

# Mark the sample mean
ax1.axvline(original_mean, color='black', linestyle='-', linewidth=2, label=f'Sample mean: ${original_mean:,.0f}')

ax1.set_xlabel('Bootstrap Sample Mean ($)', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('90% Bootstrap Confidence Interval', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.spines[['top', 'right']].set_visible(False)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: 95% Confidence Interval
ax2.hist(bootstrap_means, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(ci_95_lower, color='darkblue', linestyle='--', linewidth=2.5, label='95% CI limits')
ax2.axvline(ci_95_upper, color='darkblue', linestyle='--', linewidth=2.5)

# Shade the confidence interval region
ax2.axvspan(ci_95_lower, ci_95_upper, alpha=0.2, color='orange', label='95% CI')

# Add annotation
ci_95_mid = (ci_95_lower + ci_95_upper) / 2
ax2.text(ci_95_mid, 35, f'95% CI\n[${ci_95_lower:,.0f}, ${ci_95_upper:,.0f}]',
         ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='darkblue', linewidth=1.5))

# Mark the sample mean
ax2.axvline(original_mean, color='black', linestyle='-', linewidth=2, label=f'Sample mean: ${original_mean:,.0f}')

ax2.set_xlabel('Bootstrap Sample Mean ($)', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('95% Bootstrap Confidence Interval', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.spines[['top', 'right']].set_visible(False)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

## Key Insights from the Visualization

### 1. Confidence Level vs Interval Width

- **90% CI**: Narrower interval, excludes more extreme 5% in each tail
- **95% CI**: Wider interval, only excludes 2.5% in each tail

The trade-off is fundamental:

- Higher confidence → wider interval (less precise)
- Lower confidence → narrower interval (more precise)

### 2. What "95% Confidence" Really Means

A common misconception: "The true mean has a 95% probability of lying in this interval."

**Correct interpretation**: If we repeated the sampling and bootstrap procedure many times, **95% of the computed intervals would contain the true population parameter**.

For a given sample, the true parameter either is or isn't in the interval—there's no probability involved. The probability is in the **procedure**, not in any particular interval.

```python
# Simulation to demonstrate long-run coverage
np.random.seed(42)

# True population
true_pop = np.random.exponential(scale=50000, size=10000) + 20000
true_mean = true_pop.mean()

# Repeat the procedure many times
n_simulations = 100
ci_covers = []

for sim in range(n_simulations):
    # Draw a sample of size 20
    sample = np.random.choice(true_pop, size=20, replace=False)

    # Bootstrap
    boot_means = np.array([np.mean(np.random.choice(sample, size=len(sample))) for _ in range(500)])

    # 95% CI
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])

    # Check if true mean is covered
    covered = ci_lower <= true_mean <= ci_upper
    ci_covers.append(covered)

coverage_pct = 100 * np.mean(ci_covers)
print(f"Coverage across {n_simulations} simulations: {coverage_pct:.1f}%")
print(f"Expected coverage: 95.0%")
```

## Bootstrap Method Advantages

1. **Distribution-free**: Makes no assumptions about the underlying distribution
2. **Flexibility**: Works for any statistic (mean, median, correlation, etc.)
3. **Intuitive**: The bootstrap distribution reflects actual sampling variability
4. **Simple implementation**: No need to know theoretical formulas

## Percentile vs Other Bootstrap CI Methods

The **percentile method** (using quantiles directly) is simple but can be biased for skewed distributions. For better performance:

```python
# Percentile method (simplest, shown above)
ci_percentile = (bootstrap_means.quantile(0.025), bootstrap_means.quantile(0.975))

# BCa (Bias-Corrected and Accelerated) method - more advanced
from scipy.stats import bootstrap

def statistic(x):
    return np.mean(x)

result = bootstrap((original_sample,), statistic, n_resamples=500, method='bca')
ci_bca = result.confidence_interval
```

## Practical Recommendations

1. **Choose confidence level based on purpose**:
   - **90%**: When precision matters more (e.g., manufacturing)
   - **95%**: Standard in most applications
   - **99%**: For high-stakes decisions (e.g., pharmaceutical trials)

2. **Report confidence intervals, not p-values**: CIs communicate both point estimate and uncertainty

3. **Use at least 1000 bootstrap replications**: Ensures stable quantile estimates

4. **Check assumptions**: While bootstrap is distribution-free, ensure the sample is representative of the population

## Summary

Bootstrap confidence intervals:

- Provide intuitive visual representation of sampling uncertainty
- Make the trade-off between confidence and precision explicit
- Enable inference for any statistic without theoretical formulas
- Depend on the sample representing the population well

The width of the interval reflects both the variability in the data and the chosen confidence level—a key principle of statistical inference.


## Exercises

**Exercise 1.**
Describe the main concept of Bootstrap Confidence Intervals: Visual Interpretation of Confidence Levels and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Bootstrap Confidence Intervals: Visual Interpretation of Confidence Levels is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
