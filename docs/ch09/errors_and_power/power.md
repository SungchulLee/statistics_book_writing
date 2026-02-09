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

## Key Takeaways

- Power is the probability of correctly detecting a true effect.
- Always conduct a power analysis before a study to ensure adequate sample size.
- Increasing sample size is the most practical way to increase power.
- There is a direct tradeoff between $\alpha$, $\beta$, sample size, and effect size.
