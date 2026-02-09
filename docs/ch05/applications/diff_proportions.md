# Sampling Distribution of the Difference of Two Sample Proportions

## Overview

When comparing proportions from two independent populations (e.g., treatment vs control, brand A vs brand B), the relevant statistic is $\hat{p}_1 - \hat{p}_2$. Its sampling distribution enables confidence intervals and hypothesis tests for the difference $p_1 - p_2$.

## Mathematical Formulation

Let $\hat{p}_1$ and $\hat{p}_2$ be sample proportions from two independent samples of sizes $n_1$ and $n_2$, from populations with true proportions $p_1$ and $p_2$.

### Properties

**Expected value:**

$$
E[\hat{p}_1 - \hat{p}_2] = p_1 - p_2
$$

**Variance:**

$$
\text{Var}(\hat{p}_1 - \hat{p}_2) = \frac{p_1(1-p_1)}{n_1} + \frac{p_2(1-p_2)}{n_2}
$$

**Standard error:**

$$
\text{SE}(\hat{p}_1 - \hat{p}_2) = \sqrt{\frac{p_1(1-p_1)}{n_1} + \frac{p_2(1-p_2)}{n_2}}
$$

### Normal Approximation

For sufficiently large $n_1$ and $n_2$ (with $n_i p_i \geq 5$ and $n_i(1-p_i) \geq 5$ for both $i$):

$$
Z = \frac{(\hat{p}_1 - \hat{p}_2) - (p_1 - p_2)}{\sqrt{\frac{p_1(1-p_1)}{n_1} + \frac{p_2(1-p_2)}{n_2}}} \approx N(0, 1)
$$

In practice, since $p_1$ and $p_2$ are unknown, we substitute the sample proportions:

$$
Z \approx \frac{(\hat{p}_1 - \hat{p}_2) - (p_1 - p_2)}{\sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}}
$$

## Confidence Interval

A $(1 - \alpha)$ confidence interval for $p_1 - p_2$:

$$
(\hat{p}_1 - \hat{p}_2) \pm z_{\alpha/2} \sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}
$$

## Hypothesis Testing

### For General $H_0: p_1 - p_2 = d_0$

Use the estimated standard error:

$$
Z = \frac{(\hat{p}_1 - \hat{p}_2) - d_0}{\sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}}
$$

### For $H_0: p_1 = p_2$ (Special Case)

Under the null, $p_1 = p_2 = p$. Use the **pooled proportion**:

$$
\hat{p}_{\text{pool}} = \frac{X_1 + X_2}{n_1 + n_2}
$$

$$
Z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}_{\text{pool}}(1-\hat{p}_{\text{pool}})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}
$$

## Conditions for Validity

The normal approximation requires all four of the following:

$$
n_1 \hat{p}_1 \geq 5, \quad n_1(1 - \hat{p}_1) \geq 5, \quad n_2 \hat{p}_2 \geq 5, \quad n_2(1 - \hat{p}_2) \geq 5.
$$

When these conditions are not met, exact methods (Fisher's exact test) or simulation-based approaches should be used.

## Example

**Problem.** In a study, 120 out of 200 patients in group 1 responded to treatment ($\hat{p}_1 = 0.60$), while 90 out of 200 patients in group 2 responded ($\hat{p}_2 = 0.45$). Find a 95% confidence interval for $p_1 - p_2$.

**Solution.**

$$
\hat{p}_1 - \hat{p}_2 = 0.60 - 0.45 = 0.15
$$

$$
\text{SE} = \sqrt{\frac{0.60 \times 0.40}{200} + \frac{0.45 \times 0.55}{200}}
= \sqrt{\frac{0.24}{200} + \frac{0.2475}{200}}
= \sqrt{0.0012 + 0.001238}
\approx 0.0494
$$

$$
\text{CI} = 0.15 \pm 1.96 \times 0.0494 = 0.15 \pm 0.097 = (0.053, \; 0.247)
$$

```python
import numpy as np
from scipy import stats

p1_hat, p2_hat = 0.60, 0.45
n1, n2 = 200, 200

diff = p1_hat - p2_hat
se = np.sqrt(p1_hat * (1 - p1_hat) / n1 + p2_hat * (1 - p2_hat) / n2)
z_star = stats.norm.ppf(0.975)

ci_lower = diff - z_star * se
ci_upper = diff + z_star * se
print(f"Difference: {diff:.2f}")
print(f"SE: {se:.4f}")
print(f"95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")
```

Since the confidence interval does not contain 0, there is statistically significant evidence that the treatment response rates differ between the two groups.

## Summary

| Property | Result |
|----------|--------|
| $E[\hat{p}_1 - \hat{p}_2]$ | $p_1 - p_2$ |
| $\text{SE}(\hat{p}_1 - \hat{p}_2)$ | $\sqrt{\frac{p_1(1-p_1)}{n_1} + \frac{p_2(1-p_2)}{n_2}}$ |
| Distribution (large $n$) | Approximately $N(0, 1)$ after standardization |
| CI formula | $(\hat{p}_1 - \hat{p}_2) \pm z^* \cdot \widehat{\text{SE}}$ |
| Validity condition | $n_i p_i \geq 5$ and $n_i(1-p_i) \geq 5$ for both groups |
