# F Distribution

## Overview

The **F distribution** arises as the ratio of two independent chi-square random variables, each divided by their degrees of freedom. It is fundamental for comparing variances between two populations and for Analysis of Variance (ANOVA).

---

## Definition

Let $X_1^2 \sim \chi^2_{d_1}$ and $X_2^2 \sim \chi^2_{d_2}$ be independent. Then:

$$
F = \frac{X_1^2 / d_1}{X_2^2 / d_2} \sim F_{d_1, d_2}
$$

where $d_1$ is the numerator degrees of freedom and $d_2$ is the denominator degrees of freedom.

Since each $\chi^2$ is a sum of squared standard normals:

$$
X_1^2 = \sum_{i=1}^{d_1} Z_i^2, \qquad X_2^2 = \sum_{i=1}^{d_2} Z_i'^2
$$

---

## Degrees of Freedom

The F distribution depends on **two** sets of degrees of freedom, which distinguishes it from the chi-square distribution:

### Numerator ($d_1$) and Denominator ($d_2$)

- **Low $d_1$ and $d_2$:** Highly right-skewed.
- **Increasing $d_1$ and $d_2$:** Distribution becomes more symmetric.
- **Special case:** $F(1, d_2) = \frac{\chi^2(1)/1}{\chi^2(d_2)/d_2}$ relates directly to a squared $t$ variable.

### Comparison with Chi-Square

The chi-square distribution has a single degrees-of-freedom parameter controlling its shape. The F distribution's dual dependency creates a richer family of shapes due to the **ratio** of two independent variance-like quantities.

---

## Properties

- **Non-negativity:** $F \geq 0$ (ratio of non-negative quantities).
- **Asymmetry:** Positively skewed, especially for small degrees of freedom.
- **Mean:** $\frac{d_2}{d_2 - 2}$ for $d_2 > 2$.
- **Mode:** For $d_1 > 2$:

$$
\text{Mode} = \frac{d_2(d_1 - 2)}{d_1(d_2 + 2)}
$$

---

## Random Samples

### Direct Sampling

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)
d1, d2 = 5, 10
data = stats.f(d1, d2).rvs(10_000)

fig, ax = plt.subplots(figsize=(12, 3))
bins = np.linspace(0, 5, 100)
ax.hist(data, bins=bins, density=True, alpha=0.7, label='F Samples')
ax.plot(bins, stats.f(d1, d2).pdf(bins), '--r', lw=3, label=f'F({d1},{d2}) PDF')
ax.legend()
plt.show()
```

### Sampling from Definition (Ratio of Chi-Squares)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)
d1, d2, n = 5, 10, 10_000

f_data = (stats.chi2(d1).rvs(n) / d1) / (stats.chi2(d2).rvs(n) / d2)

fig, ax = plt.subplots(figsize=(12, 3))
bins = np.linspace(0, 5, 100)
ax.hist(f_data, bins=bins, density=True, alpha=0.7, label='χ²-Ratio Samples')
ax.plot(bins, stats.f(d1, d2).pdf(bins), '--r', lw=3, label=f'F({d1},{d2}) PDF')
ax.legend()
plt.show()
```

---

## Why F?

The F distribution arises naturally when comparing the variances of two independent normal populations.

### Step 1: Distribution of Scaled Variances

For samples from normal populations:

$$
\frac{(n_1-1)S_1^2}{\sigma_1^2} \sim \chi^2_{n_1-1}, \qquad \frac{(n_2-1)S_2^2}{\sigma_2^2} \sim \chi^2_{n_2-1}
$$

These are independent because the two samples are independent.

### Step 2: Ratio of Chi-Squares

Dividing each by its degrees of freedom and taking the ratio:

$$
\frac{S_1^2 / \sigma_1^2}{S_2^2 / \sigma_2^2} \sim F_{n_1-1, \, n_2-1}
$$

### Step 3: Under the Null Hypothesis

If we test $H_0: \sigma_1^2 = \sigma_2^2$, the population variances cancel:

$$
\frac{S_1^2}{S_2^2} \sim F_{n_1-1, \, n_2-1}
$$

This is the **F-test for equality of two variances**.

### Why Not Something Else?

The F distribution is the unique distribution that arises from the ratio of independent chi-square variables divided by their degrees of freedom. This same logic extends to ANOVA, where F ratios measure whether between-group variability is significantly larger than within-group variability.

---

## Limitations

The F distribution result is **exact only under normality**:

- **Normality required:** The chi-square results for each $S^2$ depend on the population being normal. Without normality, the distribution of $S_1^2/S_2^2$ deviates from $F$.
- **Small samples, non-normal:** The F-test is unreliable. For skewed or heavy-tailed populations, the true Type I error rate can be much higher than the nominal level.
- **Robust alternatives:** Levene's test, Brown–Forsythe test, and Fligner–Killeen test maintain validity under broader distributional conditions and are widely preferred in practice.

| Scenario | F-Test Validity |
|:---|:---|
| Normal populations | Exact |
| Large samples, mild non-normality | Approximately valid |
| Small samples, skewed/heavy-tailed | Unreliable; use robust alternatives |

---

## PPF Example

```python
from scipy import stats

# 95th percentile of F(5, 20)
f_95 = stats.f(5, 20).ppf(0.95)
print(f"F_0.95(5, 20) = {f_95:.4f}")
```

---

## Key Takeaways

- The F distribution is the ratio of two independent chi-square variables, each divided by their degrees of freedom.
- It governs comparisons of variances and is the foundation of ANOVA.
- Both numerator and denominator degrees of freedom affect the shape of the distribution.
- The exactness of the F-test depends critically on normality; robust alternatives are often preferred in practice.
