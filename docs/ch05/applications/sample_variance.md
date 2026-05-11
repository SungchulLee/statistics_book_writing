# Sampling Distribution of S-squared

## Overview

The **sampling distribution of the sample variance** $S^2$ describes how the variance computed from a random sample behaves across repeated samples drawn from a population. This concept is critical for understanding how precisely we can estimate the true population variance $\sigma^2$.

## Mathematical Definition

Let $X_1, X_2, \dots, X_n$ be i.i.d. from a population with mean $\mu$ and variance $\sigma^2$. The sample variance is:

$$
S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2
$$

## Properties

### Expected Value (Unbiasedness)

$$
E[S^2] = \sigma^2
$$

This unbiasedness is why we divide by $n-1$ (degrees of freedom) rather than $n$ — Bessel's correction compensates for the loss of one degree of freedom when estimating $\mu$ from the sample.

### Chi-Square Connection (Normal Populations)

If the population is **normal**, the scaled sample variance follows a chi-square distribution:

$$
\frac{(n-1)S^2}{\sigma^2} = \sum_{i=1}^n \left(\frac{X_i - \bar{X}}{\sigma}\right)^2 \sim \chi^2_{n-1}
$$

Equivalently:

$$
S^2 \sim \frac{\sigma^2}{n-1} \cdot \chi^2_{n-1}
$$

### Variance of S-squared
Under normality:

$$
\text{Var}(S^2) = \frac{2\sigma^4}{n-1}
$$

**Derivation.** Since $\text{Var}(\chi^2_{n-1}) = 2(n-1)$:

$$
\text{Var}\!\left(\frac{(n-1)S^2}{\sigma^2}\right) = 2(n-1) \;\;\Longrightarrow\;\;
\text{Var}(S^2) = \frac{2\sigma^4}{n-1}
$$

### Standard Error of S-squared

$$
\text{SE}(S^2) = \sigma^2 \sqrt{\frac{2}{n-1}} \sim O\!\left(\frac{1}{\sqrt{n}}\right)
$$

## Speed of Convergence

Both $\bar{X}$ and $S^2$ converge at the same asymptotic rate:

| Estimator | Standard Error | Rate |
|-----------|---------------|------|
| $\bar{X}$ | $\sigma / \sqrt{n}$ | $O(1/\sqrt{n})$ |
| $S^2$ | $\sigma^2 \sqrt{2/(n-1)}$ | $O(1/\sqrt{n})$ |

There is **no** faster or slower rate between them in terms of convergence speed.

## The Major Limitation of S-squared
The critical distinction between $\bar{X}$ and $S^2$ lies not in speed but in **distributional robustness**:

✅ **Sample mean $\bar{X}$** benefits from the Central Limit Theorem (CLT), which guarantees approximate normality of $\bar{X}$ regardless of the population shape, as long as $n$ is large enough.

❌ **Sample variance $S^2$** relies on:

$$
\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}
$$

which **only holds under normality**. For skewed, heavy-tailed, or otherwise non-normal populations, this chi-squared result no longer applies, and $S^2$ can behave unpredictably even with large samples.

## Worked Examples

### Example 1: Expected Value and Variance of S-squared
**Problem.** A sample of $n = 10$ from $N(\mu, 25)$. Find $E[S^2]$ and $\text{Var}(S^2)$.

**Solution.** For $Y \sim \chi^2_{n-1}$, $EY = n-1$ and $\text{Var}(Y) = 2(n-1)$.

$$
E\!\left[\frac{(n-1)S^2}{\sigma^2}\right] = n - 1
\;\;\Longrightarrow\;\;
E[S^2] = \sigma^2 = 25
$$

$$
\text{Var}\!\left(\frac{(n-1)S^2}{\sigma^2}\right) = 2(n-1)
\;\;\Longrightarrow\;\;
\text{Var}(S^2) = \frac{2\sigma^4}{n-1} = \frac{2(25^2)}{9} = \frac{1250}{9} \approx 138.89
$$

### Example 2: Probability Involving S-squared (Normal Population)
**Problem.** A sample of $n = 10$ from $N(\mu, 25)$. Find $P(S^2 > 30)$.

**Solution.**

$$
\frac{(n-1)S^2}{\sigma^2} = \frac{9 \times 30}{25} = 10.8
$$

$$
P(S^2 > 30) = P(\chi^2_9 > 10.8) \approx 0.2897
$$

```python
from scipy import stats

chi2_stat = 9 * 30 / 25
p_value = stats.chi2(df=9).sf(chi2_stat)
print(f"P(S^2 > 30) = {p_value:.4f}")
```

### Example 3: Without Normality Assumption

**Problem.** A sample of $n = 10$ from a population with variance 25 (no normality assumed). What can be said about $P(S^2 > 30)$?

**Solution.** Without normality, $\frac{(n-1)S^2}{\sigma^2}$ does **not** follow a chi-square distribution. We know $E[S^2] = 25$, but we cannot determine $P(S^2 > 30)$ without additional information about the population's shape.

Chebyshev's inequality could provide a bound if $\text{Var}(S^2)$ were known, but that quantity depends on higher moments of the non-normal population, which are unavailable.

## Confidence Interval for sigma-squared
Using the chi-square pivot (under normality):

$$
P\!\left(\chi^2_{\alpha/2, \, n-1} \leq \frac{(n-1)S^2}{\sigma^2} \leq \chi^2_{1-\alpha/2, \, n-1}\right) = 1 - \alpha
$$

Rearranging:

$$
\left[\frac{(n-1)S^2}{\chi^2_{1-\alpha/2, \, n-1}}, \;\; \frac{(n-1)S^2}{\chi^2_{\alpha/2, \, n-1}}\right]
$$

!!! note
    Because the chi-square distribution is asymmetric, this confidence interval is **not** symmetric around $S^2$.

## Simulation: Sampling Distribution of S-squared
### Normal Population

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(1)

population = stats.norm().rvs(100_000)
sample_size = 10
n_samples = 10_000

sample_vars = [
    np.var(np.random.choice(population, size=sample_size, replace=False), ddof=1)
    for _ in range(n_samples)
]

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6))

ax0.hist(population, bins=100, density=True, alpha=0.5)
ax0.set_title('Population Distribution (Normal)', fontsize=16)

ax1.hist(sample_vars, bins=100, density=True, alpha=0.5)
ax1.set_title(rf'Sampling Distribution of $S^2$ (n = {sample_size})', fontsize=16)

for ax in (ax0, ax1):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
```

### Income (Skewed) Population

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(1)

url = 'https://raw.githubusercontent.com/gedeck/practical-statistics-for-data-scientists/master/data/loans_income.csv'
df = pd.read_csv(url)
population = df['x'].values
sample_size = 10
n_samples = 10_000

sample_vars = [
    np.var(np.random.choice(population, size=sample_size, replace=False), ddof=1)
    for _ in range(n_samples)
]

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6))

ax0.hist(population, bins=100, density=True, alpha=0.5)
ax0.set_title('Population Distribution (Income — Skewed)', fontsize=16)

ax1.hist(sample_vars, bins=100, density=True, alpha=0.5)
ax1.set_title(rf'Sampling Distribution of $S^2$ (n = {sample_size})', fontsize=16)

for ax in (ax0, ax1):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
```

## Summary

| Property | Result |
|----------|--------|
| $E[S^2]$ | $\sigma^2$ (unbiased, always) |
| $\text{Var}(S^2)$ | $2\sigma^4/(n-1)$ (under normality) |
| Distribution | $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}$ (under normality only) |
| Robustness | ❌ No CLT-like guarantee — sensitive to non-normality |
| CI for $\sigma^2$ | Asymmetric, based on chi-square quantiles |

## Exercises

**Exercise 1.**
**Mean and variance of $S^2$.** Sample of $n = 10$ from $N(\mu, 25)$. Compute (a) $\mathbb{E}[S^2]$, (b) $\mathrm{Var}(S^2)$.

??? success "Solution to Exercise 1"
    (a) $\mathbb{E}[S^2] = \sigma^2 = 25$ (Bessel's correction makes $S^2$ unbiased).

    (b) Under normality, $(n-1) S^2/\sigma^2 \sim \chi^2_{n-1}$. Variance of $\chi^2_{n-1}$ is $2(n-1)$. Therefore:

    $$
    \mathrm{Var}(S^2) = \frac{\sigma^4}{(n-1)^2} \cdot 2(n-1) = \frac{2\sigma^4}{n-1} = \frac{2 \cdot 625}{9} \approx 138.9
    $$

    SD of $S^2 \approx 11.8$ — substantial variability. With $n = 10$, the variance estimate is very noisy.

---

**Exercise 2.**
**$P(S^2 > 30)$.** Same setup: $n = 10$, $\sigma^2 = 25$, normal population.

??? success "Solution to Exercise 2"
    $\chi^2 = (n-1)s^2/\sigma^2 = 9 \cdot 30/25 = 10.8$.

    $P(S^2 > 30) = P(\chi^2_9 > 10.8) \approx 0.290$. About 29%.

    Even though the true variance is 25, the sample variance can easily exceed 30 due to sampling variability — a routine occurrence with small samples.

---

**Exercise 3.**
**Without normality.** What can be said about $P(S^2 > 30)$ if the population is not assumed normal?

??? success "Solution to Exercise 3"
    Without normality, $(n-1)S^2/\sigma^2$ does *not* follow $\chi^2_{n-1}$. The chi-squared result is normal-specific.

    $\mathbb{E}[S^2] = \sigma^2$ remains valid (no normality needed for unbiasedness), but the distribution of $S^2$ can be quite different.

    **Chebyshev's bound** can be used if $\mathrm{Var}(S^2)$ is known, but in general $\mathrm{Var}(S^2)$ depends on the fourth moment of the population (the kurtosis), which is sensitive to the distribution shape.

    **Practical:** for skewed or heavy-tailed data, $S^2$ has *more* variance than the chi-squared formula suggests. Bootstrap is the recommended tool for inference about $\sigma^2$ in non-normal settings.

---

**Exercise 4.**
**Prove the chi-square distribution of $S^2$** under normality. Specifically: if $X_1, \ldots, X_n$ are i.i.d. $N(\mu, \sigma^2)$, then $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$.

??? success "Solution to Exercise 4"
    Decompose:

    $$
    \frac{1}{\sigma^2}\sum_{i=1}^n (X_i - \mu)^2 = \frac{1}{\sigma^2}\sum_{i=1}^n (X_i - \bar X)^2 + \frac{n(\bar X - \mu)^2}{\sigma^2}
    $$

    The left side $\sim \chi^2_n$ (sum of $n$ squared standard normals). The second term on the right $\sim \chi^2_1$ (square of $\sqrt n(\bar X - \mu)/\sigma \sim N(0, 1)$).

    By **Cochran's theorem** (applied to orthogonal projection of $X$ onto $\mathrm{span}\{\mathbf 1\}$ and its orthogonal complement), the two terms on the right are independent. So:

    $$
    \chi^2_n = \frac{(n-1) S^2}{\sigma^2} + \chi^2_1
    $$

    with two independent chi-squareds on the right. By moment-generating function properties, $(n-1) S^2/\sigma^2 \sim \chi^2_{n-1}$.

    $\square$

    This derivation underpins all of normal-theory inference about $\sigma^2$ and is the reason $t$ and $F$ distributions arise naturally.

---

**Exercise 5.**
**Confidence interval for $\sigma^2$.** Given $n = 10$, $s^2 = 16$ from a normal population, construct a 95% CI for $\sigma^2$.

??? success "Solution to Exercise 5"
    Pivot: $(n-1)s^2/\sigma^2 \sim \chi^2_9$.

    95% CI uses $\chi^2_{0.025, 9} = 2.700$ and $\chi^2_{0.975, 9} = 19.023$:

    $$
    P(2.700 \le 9 s^2/\sigma^2 \le 19.023) = 0.95
    $$

    Inverting:

    $$
    \frac{9 s^2}{19.023} \le \sigma^2 \le \frac{9 s^2}{2.700}
    $$

    With $s^2 = 16$: CI $= (9 \cdot 16/19.023, 9 \cdot 16/2.700) = (7.57, 53.33)$.

    Wide and asymmetric — chi-squared is skewed for small df. With $n = 10$, the variance is barely constrained. The CI shrinks as $n$ grows.

---

**Exercise 6.**
**$S^2$ vs. $\sigma^2_{\text{MLE}}$.** The MLE of $\sigma^2$ uses denominator $n$, not $n - 1$. Compare bias, variance, and MSE.

??? success "Solution to Exercise 6"
    $S^2 = \frac{1}{n-1}\sum(X_i - \bar X)^2$ (unbiased): $\mathbb{E}[S^2] = \sigma^2$, $\mathrm{Var}(S^2) = 2\sigma^4/(n-1)$.

    $\hat\sigma^2_{\text{MLE}} = \frac{1}{n}\sum(X_i - \bar X)^2 = \frac{n-1}{n} S^2$ (biased): $\mathbb{E}[\hat\sigma^2_{\text{MLE}}] = (n-1)\sigma^2/n$.

    $\mathrm{Var}(\hat\sigma^2_{\text{MLE}}) = ((n-1)/n)^2 \cdot 2\sigma^4/(n-1) = 2(n-1)\sigma^4/n^2$.

    $\mathrm{MSE}(\hat\sigma^2_{\text{MLE}}) = \mathrm{Var} + \mathrm{bias}^2 = 2(n-1)\sigma^4/n^2 + \sigma^4/n^2 = (2n-1)\sigma^4/n^2$.

    $\mathrm{MSE}(S^2) = \mathrm{Var}(S^2) = 2\sigma^4/(n-1)$.

    Compare ratios: $\mathrm{MSE}(\hat\sigma^2_{\text{MLE}})/\mathrm{MSE}(S^2) = (2n-1)(n-1)/(2n^2)$, less than 1 for all $n \ge 2$.

    **MLE has smaller MSE** despite being biased — the lower variance more than compensates for the bias. This is a classic example of **bias-variance trade-off**: accepting some bias to reduce overall error.

    Most software still uses $S^2$ (Bessel-corrected) because unbiasedness is a clean property and the MSE difference vanishes as $n \to \infty$.
