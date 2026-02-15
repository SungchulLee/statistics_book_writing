# Power Analysis

## Definition of Power

The **power** of a hypothesis test is the probability that the test correctly rejects a false null hypothesis. It is the complement of the Type II error rate:

$$\text{Power} = 1 - \beta = P(\text{Reject } H_0 \mid H_a \text{ is true})$$

A test with high power is more likely to detect a true effect when one exists. Researchers typically aim for a power of at least 0.80 (80%), meaning there is an 80% chance of detecting a true effect.

## Factors Affecting Power

Four key factors determine the power of a test:

### 1. Significance Level ($\alpha$)

Increasing $\alpha$ (e.g., from 0.01 to 0.05) makes it easier to reject $H_0$, thus increasing power. However, this also increases the risk of a Type I error.

### 2. Sample Size ($n$)

Larger sample sizes increase power because they reduce the standard error of the test statistic, making it easier to detect true differences. This is often the most practical lever for increasing power.

### 3. Effect Size

The effect size measures the magnitude of the true difference or effect. Larger effect sizes are easier to detect, leading to higher power. Common measures include:

- **Cohen's $d$** for comparing means: $d = \frac{\mu_1 - \mu_0}{\sigma}$
- **Proportion difference** for comparing proportions

### 4. Population Variability ($\sigma$)

Lower variability in the population makes it easier to detect true effects, increasing power. While researchers cannot usually control population variability, they can reduce measurement error through better study design.

## Power Analysis in Practice

Power analysis is used in two main ways:

### A Priori Power Analysis (Sample Size Determination)

Before conducting a study, researchers use power analysis to determine the minimum sample size needed to detect an expected effect size with desired power and significance level.

```python
from scipy import stats
import numpy as np

def sample_size_z_test(effect_size, alpha=0.05, power=0.80, alternative='two-sided'):
    """Calculate required sample size for a one-sample z-test."""
    if alternative == 'two-sided':
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(n))

# Example: detect effect size of 0.5 with 80% power
n = sample_size_z_test(effect_size=0.5)
print(f"Required sample size: {n}")
```

### Post Hoc Power Analysis

After conducting a study, researchers can compute the achieved power given the observed effect size, sample size, and significance level. However, post hoc power analysis on non-significant results is generally discouraged as it provides little additional information beyond the p-value.

## Visualizing Power

Power can be understood by visualizing the distributions under both $H_0$ and $H_a$:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_power(mu_0, mu_a, sigma, n, alpha=0.05):
    """Visualize the power of a one-sided z-test."""
    se = sigma / np.sqrt(n)
    z_crit = stats.norm.ppf(1 - alpha)
    x_crit = mu_0 + z_crit * se

    x = np.linspace(mu_0 - 4*se, mu_a + 4*se, 300)

    # Distribution under H0
    y_h0 = stats.norm(mu_0, se).pdf(x)
    # Distribution under Ha
    y_ha = stats.norm(mu_a, se).pdf(x)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, y_h0, 'b-', label=f'$H_0$: $\\mu = {mu_0}$')
    ax.plot(x, y_ha, 'r-', label=f'$H_a$: $\\mu = {mu_a}$')

    # Shade rejection region under H0 (alpha)
    x_reject = x[x >= x_crit]
    ax.fill_between(x_reject, stats.norm(mu_0, se).pdf(x_reject), alpha=0.3, color='blue', label=f'$\\alpha$ = {alpha}')

    # Shade power under Ha
    ax.fill_between(x_reject, stats.norm(mu_a, se).pdf(x_reject), alpha=0.3, color='red', label=f'Power = {1 - stats.norm(mu_a, se).cdf(x_crit):.3f}')

    ax.axvline(x_crit, color='k', linestyle='--', label=f'Critical value = {x_crit:.2f}')
    ax.legend()
    ax.set_xlabel('Sample Mean')
    ax.set_ylabel('Density')
    ax.set_title('Power of a Hypothesis Test')
    plt.show()

plot_power(mu_0=50, mu_a=52, sigma=10, n=25)
```

## Relationship Between Power, Sample Size, and Effect Size

| Effect Size | Required $n$ (Power = 0.80, $\alpha$ = 0.05) |
|---|---|
| Small ($d = 0.2$) | ~393 |
| Medium ($d = 0.5$) | ~64 |
| Large ($d = 0.8$) | ~26 |

These values illustrate why detecting small effects requires substantially larger sample sizes.

## Power Analysis Using statsmodels

Statsmodels provides comprehensive power analysis functions for various test types:

### One-Sample t-Test

```python
from statsmodels.stats.power import TTestPower
import numpy as np

# Power analysis for one-sample t-test
analysis = TTestPower()

# Scenario: How many subjects needed to detect a 5-point difference?
# Assume μ₀ = 0, μ_a = 5, σ = 15
effect_size = 5 / 15  # Cohen's d = 0.333

n_needed = analysis.solve_power(effect_size=effect_size, alpha=0.05,
                                power=0.80, alternative='two-sided')
print(f"One-sample t-test:")
print(f"  Effect size (Cohen's d): {effect_size:.3f}")
print(f"  Sample size needed for 80% power: {int(np.ceil(n_needed))}")

# Power for a given sample size
power = analysis.power(effect_size=effect_size, nobs=50, alpha=0.05,
                      alternative='two-sided')
print(f"  Power with n=50: {power:.3f}")
```

### Two-Sample t-Test (Independent Samples)

```python
from statsmodels.stats.power import TTestIndPower

# Power analysis for two-sample t-test
analysis = TTestIndPower()

# Scenario: Two-group comparison (equal sample sizes)
# Effect size: Cohen's d = 0.5 (medium effect)
effect_size = 0.5

n_per_group = analysis.solve_power(effect_size=effect_size, alpha=0.05,
                                   power=0.80, ratio=1.0,
                                   alternative='two-sided')
print(f"\nTwo-sample t-test (equal n):")
print(f"  Effect size (Cohen's d): {effect_size:.3f}")
print(f"  Sample size per group for 80% power: {int(np.ceil(n_per_group))}")
print(f"  Total sample size: {2 * int(np.ceil(n_per_group))}")

# Unequal sample sizes (e.g., 2:1 ratio)
n_treatment = analysis.solve_power(effect_size=effect_size, alpha=0.05,
                                   power=0.80, ratio=2.0,
                                   alternative='two-sided')
n_control = n_treatment / 2
print(f"\nTwo-sample t-test (2:1 ratio):")
print(f"  Treatment n: {int(np.ceil(n_treatment))}")
print(f"  Control n: {int(np.ceil(n_control))}")
```

### Test for Proportions (A/B Testing)

```python
from statsmodels.stats.power import proportions_ztest
import statsmodels.stats.api as sms

# Example: A/B test for conversion rates
# Control: 1.1% conversion rate
# Treatment: 1.65% conversion rate
p0 = 0.011  # Control baseline
p1 = 0.0165  # Treatment goal

# Calculate effect size (h = 2 * arcsin(√p1) - 2 * arcsin(√p0))
effect_size = sms.proportion_effectsize(p1, p0)

# Required sample size
analysis = sms.FTestAnovaPower()  # or use proportions_ztest
n_needed = sms.tt_solve_power(effect_size=effect_size, alpha=0.05,
                              power=0.80, alternative='larger')

# Alternative: Use proportion_effectsize with proportions_ztest
from statsmodels.stats.proportion import proportions_ztest
n_ab = proportions_ztest(effect_size=effect_size, alpha=0.05,
                         power=0.80, alternative='larger')

print(f"\nA/B Test (Proportions):")
print(f"  Control rate: {p0:.2%}")
print(f"  Treatment goal: {p1:.2%}")
print(f"  Effect size: {effect_size:.4f}")
print(f"  Sample size per group for 80% power: {int(np.ceil(n_ab))}")
```

### One-Way ANOVA

```python
from statsmodels.stats.power import FTestAnovaPower

# Power analysis for one-way ANOVA
analysis = FTestAnovaPower()

# Scenario: 4 groups, medium effect size (f = 0.25)
effect_size = 0.25  # Medium effect in ANOVA
k_groups = 4

n_per_group = analysis.solve_power(effect_size=effect_size, alpha=0.05,
                                   power=0.80, k_groups=k_groups)
print(f"\nOne-way ANOVA (4 groups):")
print(f"  Effect size (Cohen's f): {effect_size:.3f}")
print(f"  Sample size per group for 80% power: {int(np.ceil(n_per_group))}")
print(f"  Total sample size: {k_groups * int(np.ceil(n_per_group))}")
```

### Power Curves: Visualizing Sample Size vs. Power

```python
import matplotlib.pyplot as plt

# Create power curves for different effect sizes
fig, ax = plt.subplots(figsize=(10, 6))

analysis = TTestPower()
sample_sizes = np.arange(10, 200, 5)

for d in [0.2, 0.5, 0.8]:
    power_values = [analysis.power(effect_size=d, nobs=n, alpha=0.05)
                    for n in sample_sizes]
    ax.plot(sample_sizes, power_values, linewidth=2, label=f"d = {d:.1f}")

# Add reference lines
ax.axhline(0.80, color='red', linestyle='--', linewidth=1, label='Power = 0.80')
ax.axhline(0.90, color='orange', linestyle='--', linewidth=1, label='Power = 0.90')

ax.set_xlabel('Sample Size (n)', fontsize=12)
ax.set_ylabel('Power', fontsize=12)
ax.set_title('Power Curves: Sample Size vs. Power\n(One-Sample t-Test, α = 0.05)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1])

plt.tight_layout()
plt.show()
```

### Power Analysis Workflow

```python
def design_study(test_type, effect_size, alpha=0.05, power=0.80,
                 **kwargs):
    """
    Comprehensive power analysis for study design.

    Parameters:
    -----------
    test_type : str
        Type of test ('one-sample', 'two-sample', 'anova', 'proportions')
    effect_size : float
        Standardized effect size
    alpha : float
        Significance level
    power : float
        Desired statistical power
    **kwargs : dict
        Additional parameters (e.g., k_groups for ANOVA)

    Returns:
    --------
    dict : Sample size requirements and recommendations
    """
    results = {'test_type': test_type, 'alpha': alpha, 'power': power}

    if test_type == 'one-sample':
        analysis = TTestPower()
        n = analysis.solve_power(effect_size=effect_size, alpha=alpha,
                                power=power, alternative='two-sided')
        results['sample_size'] = int(np.ceil(n))

    elif test_type == 'two-sample':
        analysis = TTestIndPower()
        n = analysis.solve_power(effect_size=effect_size, alpha=alpha,
                                power=power, ratio=1.0, alternative='two-sided')
        results['sample_size_per_group'] = int(np.ceil(n))
        results['total_sample_size'] = 2 * int(np.ceil(n))

    elif test_type == 'anova':
        analysis = FTestAnovaPower()
        k = kwargs.get('k_groups', 3)
        n = analysis.solve_power(effect_size=effect_size, alpha=alpha,
                                power=power, k_groups=k)
        results['k_groups'] = k
        results['sample_size_per_group'] = int(np.ceil(n))
        results['total_sample_size'] = k * int(np.ceil(n))

    return results

# Example usage
print("\n" + "="*60)
print("STUDY DESIGN: Two-Sample Comparison")
print("="*60)
design = design_study('two-sample', effect_size=0.5)
for key, value in design.items():
    print(f"{key:.<30} {value}")
```

## Key Takeaways

- Power is the probability of correctly detecting a true effect.
- Always conduct a power analysis before a study to ensure adequate sample size.
- Increasing sample size is the most practical way to increase power.
- There is a direct tradeoff between $\alpha$, $\beta$, sample size, and effect size.
- Statsmodels provides convenient functions for power analysis across many test types.
- Use power curves to visualize the relationship between sample size and power.
