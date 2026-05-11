# Comparing Two Correlations

In many applications, we need to determine not just whether a single correlation is significant, but whether two correlations differ from each other. For example: is the correlation between study hours and GPA stronger for men than for women? Is the association between two biomarkers stronger in the treatment group than in the control group? This section covers the standard methods for testing whether two correlation coefficients are equal.

---

## Two Settings

There are two distinct settings for comparing correlations:

1. **Independent samples**: the two correlations come from different, unrelated groups. For example, $r_1$ is the height-weight correlation among men and $r_2$ is the height-weight correlation among women.

2. **Dependent (overlapping) samples**: the two correlations come from the same sample and share a variable. For example, $r_{XY}$ and $r_{XZ}$ both involve the same variable $X$.

The methods differ substantially between these two cases.

---

## Fisher's z-Transformation

Both methods rely on **Fisher's z-transformation**, which stabilizes the variance of the sample correlation and makes its distribution approximately normal:

$$
z = \frac{1}{2} \ln\!\left(\frac{1 + r}{1 - r}\right) = \text{arctanh}(r)
$$

For a sample of size $n$ from a bivariate normal distribution, the transformed correlation $z$ is approximately normally distributed:

$$
z \;\dot\sim\; N\!\left(\frac{1}{2}\ln\!\left(\frac{1+\rho}{1-\rho}\right),\; \frac{1}{n-3}\right)
$$

The key property is that the variance $1/(n-3)$ does **not depend on** $\rho$, unlike the variance of $r$ itself. This makes $z$ much more amenable to inference.

---

## Comparing Two Independent Correlations

Given two independent samples of sizes $n_1$ and $n_2$ with sample correlations $r_1$ and $r_2$, we test

$$
H_0\!: \rho_1 = \rho_2 \quad \text{vs} \quad H_1\!: \rho_1 \neq \rho_2
$$

Apply the Fisher z-transformation to each:

$$
z_1 = \text{arctanh}(r_1), \quad z_2 = \text{arctanh}(r_2)
$$

Under $H_0$, the difference $z_1 - z_2$ has approximate variance $\frac{1}{n_1 - 3} + \frac{1}{n_2 - 3}$. The test statistic is

$$
Z = \frac{z_1 - z_2}{\sqrt{\frac{1}{n_1 - 3} + \frac{1}{n_2 - 3}}}
$$

Under $H_0$, $Z$ follows approximately a standard normal distribution. Reject $H_0$ at level $\alpha$ if $|Z| > z_{\alpha/2}$.

### Example

A researcher finds $r_1 = 0.65$ ($n_1 = 50$) for men and $r_2 = 0.40$ ($n_2 = 60$) for women.

$$
z_1 = \text{arctanh}(0.65) = 0.7753, \quad z_2 = \text{arctanh}(0.40) = 0.4236
$$

$$
Z = \frac{0.7753 - 0.4236}{\sqrt{\frac{1}{47} + \frac{1}{57}}} = \frac{0.3517}{\sqrt{0.02128 + 0.01754}} = \frac{0.3517}{0.1970} = 1.785
$$

The two-sided p-value is $2 \times P(Z > 1.785) \approx 0.074$. At $\alpha = 0.05$, we do not reject $H_0$; there is insufficient evidence that the correlations differ between men and women.

---

## Comparing Two Dependent Correlations

When two correlations share a common variable (e.g., $r_{XY}$ and $r_{XZ}$ from the same sample of size $n$), the Fisher z-test for independent samples does not apply because $r_{XY}$ and $r_{XZ}$ are correlated.

### Steiger's Test (Williams' Modification)

To test $H_0\!: \rho_{XY} = \rho_{XZ}$, the test statistic proposed by Williams (1959), building on Steiger (1980), is

$$
t = (r_{XY} - r_{XZ}) \sqrt{\frac{(n-1)(1 + r_{YZ})}{2\left(\frac{n-1}{n-3}\right)|R| + \bar{r}^2(1 - r_{YZ})^3}}
$$

where $|R|$ is the determinant of the $3 \times 3$ correlation matrix of $(X, Y, Z)$ and $\bar{r} = (r_{XY} + r_{XZ})/2$.

Under $H_0$, this statistic follows approximately a $t$-distribution with $n - 3$ degrees of freedom.

A simpler (but less accurate) approximation replaces the denominator with a formula involving only $r_{YZ}$:

$$
t \approx (r_{XY} - r_{XZ}) \sqrt{\frac{(n-3)(1 + r_{YZ})}{2(1 - r_{XY}^2 - r_{XZ}^2 - r_{YZ}^2 + 2 r_{XY} r_{XZ} r_{YZ})}}
$$

This is the version most commonly implemented in software.

---

## Computation in Python

```python
import numpy as np
from scipy import stats

# Comparing two independent correlations
r1, n1 = 0.65, 50
r2, n2 = 0.40, 60

z1 = np.arctanh(r1)
z2 = np.arctanh(r2)
se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
Z_stat = (z1 - z2) / se
p_value = 2 * (1 - stats.norm.cdf(abs(Z_stat)))

print(f"z1 = {z1:.4f}, z2 = {z2:.4f}")
print(f"Z statistic = {Z_stat:.4f}")
print(f"Two-sided p-value = {p_value:.4f}")
```

For comparing dependent correlations, the `pingouin` library provides `pingouin.corr` with options for comparing overlapping correlations, and the `cocor` R package offers a comprehensive suite of comparison tests.

---

## Confidence Interval for the Difference

For two independent correlations, a $(1 - \alpha)$ confidence interval for $\rho_1 - \rho_2$ can be constructed by back-transforming:

$$
(z_1 - z_2) \pm z_{\alpha/2} \sqrt{\frac{1}{n_1 - 3} + \frac{1}{n_2 - 3}}
$$

gives a confidence interval for $\zeta_1 - \zeta_2$ (where $\zeta = \text{arctanh}(\rho)$). To convert back to the correlation scale, apply $\tanh$ to each endpoint. Note that this gives a confidence interval for $\zeta_1 - \zeta_2$, not directly for $\rho_1 - \rho_2$, because the tanh transformation is nonlinear.

---

## Assumptions

Both comparison methods assume:

1. **Bivariate normality** within each sample.
2. **Random sampling** from the respective populations.
3. **Sufficient sample size** (typically $n \ge 25$ for each group for the normal approximation to be adequate).

When normality is violated, bootstrap methods provide a nonparametric alternative for comparing correlations.

---

## Summary

Comparing two correlations requires different methods depending on whether the samples are independent or overlapping. For independent samples, Fisher's z-transformation converts each correlation to a normally distributed variable, and a simple $Z$-test compares the transformed values. For dependent samples with a shared variable, Steiger's test (Williams' modification) accounts for the correlation between the two coefficients. In both cases, the Fisher z-transformation is the key tool that stabilizes the variance and enables standard normal-theory inference.

## Exercises

**Exercise 1.**
Two independent samples yield $r_1 = 0.65$ ($n_1 = 50$) and $r_2 = 0.40$ ($n_2 = 60$). Test whether the two population correlations are equal using Fisher's z-transformation at $\alpha = 0.05$.

??? success "Solution to Exercise 1"
    Apply Fisher's z-transformation: $z_r = \frac{1}{2}\ln\frac{1+r}{1-r}$.

    $$
    z_1 = \frac{1}{2}\ln\frac{1.65}{0.35} = \frac{1}{2}\ln(4.714) = \frac{1}{2}(1.5506) = 0.7753
    $$

    $$
    z_2 = \frac{1}{2}\ln\frac{1.40}{0.60} = \frac{1}{2}\ln(2.333) = \frac{1}{2}(0.8473) = 0.4236
    $$

    The test statistic is:

    $$
    Z = \frac{z_1 - z_2}{\sqrt{\frac{1}{n_1-3} + \frac{1}{n_2-3}}} = \frac{0.7753 - 0.4236}{\sqrt{\frac{1}{47} + \frac{1}{57}}} = \frac{0.3517}{\sqrt{0.02128 + 0.01754}} = \frac{0.3517}{0.1970} = 1.785
    $$

    Since $|Z| = 1.785 < 1.96$, we fail to reject $H_0: \rho_1 = \rho_2$ at $\alpha = 0.05$.

---

**Exercise 2.**
Explain why comparing correlations directly (without transformation) is problematic when correlations are far from zero.

??? success "Solution to Exercise 2"
    The sampling distribution of $r$ is skewed when $\rho \neq 0$: it is compressed toward the boundary ($\pm 1$) on the side closer to $\rho$ and stretched on the other side. As $|\rho| \to 1$, the distribution becomes increasingly skewed and its variance decreases.

    Fisher's z-transformation $z_r = \frac{1}{2}\ln\frac{1+r}{1-r}$ stabilizes the variance and symmetrizes the distribution. After transformation, $z_r$ is approximately $N(z_\rho, 1/(n-3))$ regardless of $\rho$. Without the transformation, the standard error of $r$ depends on $\rho$, making comparisons unreliable.

---

**Exercise 3.**
Two correlations are computed from the **same** sample: $r_{XY} = 0.70$ and $r_{XZ} = 0.50$ with $n = 100$. Why can't you use the independent-samples test to compare them?

??? success "Solution to Exercise 3"
    The independent-samples test assumes $r_1$ and $r_2$ come from separate, unrelated samples. When both correlations come from the same sample, they share the variable $X$ and the same observations, making them dependent. Their covariance depends on $r_{YZ}$ (the correlation between $Y$ and $Z$ in the sample).

    Ignoring the dependence and using the independent-samples test would overestimate the standard error of the difference (treating the two correlations as more variable than they actually are), reducing power.

    The correct test uses **Steiger's (1980) method** or **Hotelling's (1940) test**, which accounts for the correlation between the two correlation coefficients through the formula involving $r_{XY}$, $r_{XZ}$, and $r_{YZ}$.

---

**Exercise 4.**
A researcher reports that the correlation between height and income is $r = 0.15$ in a sample of $n = 2000$. Is this "small" correlation statistically significant? Is it practically important?

??? success "Solution to Exercise 4"
    Testing $H_0: \rho = 0$: $t = r\sqrt{n-2}/\sqrt{1-r^2} = 0.15\sqrt{1998}/\sqrt{0.9775} = 0.15 \times 44.7/0.9887 = 6.78$.

    With $df = 1998$, $t = 6.78$ is highly significant ($p < 0.0001$). The correlation is statistically distinguishable from zero.

    However, $r^2 = 0.0225$: height explains only 2.25% of the variance in income. Practically, this is a weak association. With $n = 2000$, even tiny correlations become significant. This illustrates the importance of distinguishing statistical significance (is the effect nonzero?) from practical significance (is the effect large enough to matter?). Effect size measures like $r^2$ are essential for interpretation.
