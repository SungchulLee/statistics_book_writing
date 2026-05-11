# Advanced Descriptive Measures

## Overview

The arithmetic mean and sample variance are the most common summary statistics, but they are not always the most appropriate. This page covers three topics that extend the basic toolkit:

1. **Geometric mean** — the correct average for multiplicative processes such as investment returns.
2. **Chebyshev's inequality** — a distribution-free bound on how much data can lie far from the mean.
3. **Population vs sample variance** — a simulation showing why dividing by $n - 1$ produces an unbiased estimator.

---

## 1. Geometric vs Arithmetic Mean

### The Problem

For a sequence of returns $r_1, r_2, \ldots, r_T$, the arithmetic mean

$$
\bar{r} = \frac{1}{T} \sum_{t=1}^{T} r_t
$$

overestimates the compound growth rate whenever returns vary. The correct measure is the **geometric mean**:

$$
r_g = \left(\prod_{t=1}^{T} (1 + r_t)\right)^{1/T} - 1
$$

### Code

```python
import numpy as np
from scipy import stats

returns = np.array([0.36, 0.23, -0.48, -0.30, 0.15, 0.31])

arith_mean = np.mean(returns)
geo_mean = stats.mstats.gmean(1 + returns) - 1

print(f"Arithmetic mean: {arith_mean:.4f}  ({arith_mean*100:.2f}%)")
print(f"Geometric  mean: {geo_mean:.4f}  ({geo_mean*100:.2f}%)")
print(f"Compound value of \$1: \${np.prod(1 + returns):.4f}")
print(f"Using geo mean:       \${(1 + geo_mean)**len(returns):.4f}")
```

### Output

```
Arithmetic mean: 0.0450  (4.50%)
Geometric  mean: -0.0139  (-1.39%)
Compound value of $1: $0.9196
Using geo mean:       $0.9196
```

### Interpretation

The arithmetic mean suggests a positive average return of 4.50%, but \$1 invested actually shrinks to \$0.92. The geometric mean correctly reports a negative compound rate of $-1.39\%$. This discrepancy arises because the arithmetic mean ignores the asymmetry of compounding: a 50% loss requires a 100% gain to recover, not a 50% gain.

!!! warning "Common Mistake"
    Never use the arithmetic mean to summarize compound growth. The geometric mean is the only correct summary for multiplicative processes.

---

## 2. Chebyshev's Inequality

### Statement

For **any** distribution with finite mean $\mu$ and standard deviation $\sigma$, the proportion of data within $k$ standard deviations of the mean satisfies:

$$
P(|X - \mu| < k\sigma) \ge 1 - \frac{1}{k^2} \qquad \text{for } k > 1
$$

This bound makes no assumptions about the shape of the distribution.

### Key Values

| $k$ | Minimum fraction within $k\sigma$ |
|---|---|
| 2 | $\ge 75\%$ |
| 3 | $\ge 88.9\%$ |
| 4 | $\ge 93.75\%$ |
| 5 | $\ge 96\%$ |

### Example: Heights

Suppose heights have mean $\mu = 174$ cm and standard deviation $\sigma = 4$ cm. What fraction of the population lies between 166 cm and 182 cm?

$$
k = \frac{182 - 174}{4} = 2
$$

By Chebyshev's inequality, at least $1 - 1/4 = 75\%$ of the population falls in this range—regardless of the shape of the height distribution.

### Code

```python
import numpy as np
import matplotlib.pyplot as plt

def chebyshev(k):
    return 1 - 1 / k**2

z_vals = np.arange(1.1, 10, 0.1)
cheb_vals = [chebyshev(z) for z in z_vals]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(z_vals, cheb_vals, lw=2, color="seagreen")
ax.set_xlabel("k (standard deviations)")
ax.set_ylabel("Minimum fraction")
ax.set_title("Chebyshev's Inequality: 1 - 1/k²")
ax.axhline(0.75, color="grey", linestyle=":", alpha=0.5)
ax.annotate("k=2: ≥ 75%", (2, 0.75), fontsize=9,
            xytext=(4, 0.6), arrowprops=dict(arrowstyle="->"))
plt.tight_layout()
plt.show()
```

### Interpretation

Chebyshev's bound is **conservative**: for a normal distribution, 95% of data lies within 2 standard deviations, far exceeding the 75% Chebyshev guarantee. The power of the bound is its universality—it applies to any distribution with finite variance, including heavily skewed or multimodal ones.

---

## 3. Population vs Sample Variance

### The Bias Problem

The naive variance estimator divides by $n$:

$$
\hat{\sigma}^2_{\text{biased}} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

This systematically underestimates the true population variance $\sigma^2$ because $\bar{x}$ is closer to the sample points than $\mu$ is. Bessel's correction uses $n - 1$:

$$
s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

### Simulation

We verify unbiasedness by drawing 10,000 samples of size 100 from a population and comparing the average of each estimator to the true variance:

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
population = np.random.normal(170, 10, 1000)
pop_var = np.var(population)

n_samples = 10000
sample_size = 100

biased_vars = np.empty(n_samples)
unbiased_vars = np.empty(n_samples)

for i in range(n_samples):
    sample = np.random.choice(population, size=sample_size, replace=False)
    biased_vars[i] = np.var(sample, ddof=0)
    unbiased_vars[i] = np.var(sample, ddof=1)

print(f"Population variance:          {pop_var:.4f}")
print(f"Mean of biased (ddof=0):      {biased_vars.mean():.4f}")
print(f"Mean of unbiased (ddof=1):    {unbiased_vars.mean():.4f}")
```

### Visualization

```python
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(biased_vars, bins=40, alpha=0.5, label="Biased (ddof=0)", density=True)
ax.hist(unbiased_vars, bins=40, alpha=0.5, label="Unbiased (ddof=1)", density=True)
ax.axvline(pop_var, color="red", linestyle="--", lw=2,
           label=f"True σ² = {pop_var:.1f}")
ax.set_xlabel("Variance estimate")
ax.set_title("Population vs Sample Variance")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
```

### Interpretation

The histogram of the biased estimator (ddof=0) is shifted slightly to the left of the true variance, confirming the systematic underestimation. The unbiased estimator (ddof=1) centers on the true value. The difference is small for $n = 100$ (a factor of $100/99 \approx 1.01$) but becomes substantial for small samples.

---

## Exercises

**Exercise 1.**
An investment has annual returns of $+20\%$, $-20\%$, $+20\%$, $-20\%$. Compute the arithmetic mean and geometric mean. What is the final value of \$1000 invested?

??? success "Solution to Exercise 1"
    Arithmetic mean: $(0.20 - 0.20 + 0.20 - 0.20)/4 = 0$. The arithmetic mean suggests no growth.

    Geometric mean:

    $$
    r_g = (1.20 \times 0.80 \times 1.20 \times 0.80)^{1/4} - 1 = (0.9216)^{1/4} - 1 \approx -0.0202
    $$

    Final value: $1000 \times 1.20 \times 0.80 \times 1.20 \times 0.80 = 1000 \times 0.9216 = \$921.60$.

    The geometric mean correctly indicates a loss of about 2% per year, while the arithmetic mean incorrectly suggests break-even.

---

**Exercise 2.**
Prove Chebyshev's inequality. Start from Markov's inequality: for a nonneg random variable $Y$ and $a > 0$, $P(Y \ge a) \le E[Y]/a$.

??? success "Solution to Exercise 2"
    Apply Markov's inequality to $Y = (X - \mu)^2$ with $a = (k\sigma)^2 = k^2 \sigma^2$:

    $$
    P\bigl((X - \mu)^2 \ge k^2 \sigma^2\bigr) \le \frac{E[(X - \mu)^2]}{k^2 \sigma^2} = \frac{\sigma^2}{k^2 \sigma^2} = \frac{1}{k^2}
    $$

    Since $(X - \mu)^2 \ge k^2 \sigma^2$ is equivalent to $|X - \mu| \ge k\sigma$:

    $$
    P(|X - \mu| \ge k\sigma) \le \frac{1}{k^2}
    $$

    Taking complements:

    $$
    P(|X - \mu| < k\sigma) \ge 1 - \frac{1}{k^2}
    $$

    $\square$

---

**Exercise 3.**
For the population variance simulation above, what happens to the bias of the ddof=0 estimator as the sample size $n$ increases? Express the bias as a function of $n$ and $\sigma^2$.

??? success "Solution to Exercise 3"
    The expected value of the biased estimator is:

    $$
    E\left[\frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2\right] = \frac{n-1}{n} \sigma^2
    $$

    So the bias is:

    $$
    \text{Bias} = \frac{n-1}{n}\sigma^2 - \sigma^2 = -\frac{\sigma^2}{n}
    $$

    As $n \to \infty$, the bias approaches 0. For $n = 100$, the bias is $-\sigma^2/100$, which is only 1% of the true variance. For $n = 5$, the bias is $-\sigma^2/5 = -20\%$—much more substantial.

---

**Exercise 4.**
A dataset has mean 50 and standard deviation 5. Using Chebyshev's inequality, find the minimum proportion of data in the interval $[35, 65]$. Then compare with the proportion you would get if the data were normally distributed.

??? success "Solution to Exercise 4"
    The interval $[35, 65]$ is $[50 - 15, 50 + 15]$, so $k = 15/5 = 3$.

    By Chebyshev: at least $1 - 1/9 \approx 88.9\%$.

    For a normal distribution: $P(|Z| < 3) = P(-3 < Z < 3) \approx 0.9974$, so about $99.7\%$.

    The normal distribution concentrates much more of its mass near the mean than Chebyshev's universal lower bound requires. The gap ($99.7\%$ vs $88.9\%$) reflects the cost of making no distributional assumptions.

---

**Exercise 5.**
Show that the geometric mean is always less than or equal to the arithmetic mean for nonneg values. When does equality hold?

??? success "Solution to Exercise 5"
    This is the **AM-GM inequality**. For nonneg values $a_1, \ldots, a_n$:

    $$
    \frac{a_1 + a_2 + \cdots + a_n}{n} \ge (a_1 \cdot a_2 \cdots a_n)^{1/n}
    $$

    **Proof via Jensen's inequality:** The logarithm is a concave function. By Jensen's inequality:

    $$
    \log\left(\frac{1}{n}\sum_{i=1}^n a_i\right) \ge \frac{1}{n}\sum_{i=1}^n \log(a_i) = \log\left(\prod_{i=1}^n a_i\right)^{1/n}
    $$

    Since $\log$ is strictly increasing, exponentiating both sides gives AM $\ge$ GM.

    Equality holds if and only if $a_1 = a_2 = \cdots = a_n$, because Jensen's inequality is strict for strictly concave functions unless all values are equal. $\square$

---

**Exercise 6.**
The **harmonic mean** of positive values $a_1, \ldots, a_n$ is $H = n / \sum_i (1/a_i)$. Prove the inequality $H \le G \le A$ (harmonic $\le$ geometric $\le$ arithmetic) and explain when the harmonic mean is the *correct* mean to use.

??? success "Solution to Exercise 6"
    **Proof:** Apply the AM-GM inequality to $1/a_1, \ldots, 1/a_n$:

    $$
    \frac{1}{n}\sum_i \frac{1}{a_i} \ge \left(\prod_i \frac{1}{a_i}\right)^{1/n} = \frac{1}{G}
    $$

    Inverting (both sides positive): $H = n / \sum_i (1/a_i) \le G$. Combined with $G \le A$ from Exercise 5: $H \le G \le A$.

    **When harmonic mean is correct:** harmonic mean averages **rates** properly. Examples:

    - Average speed for equal *distances* traveled at different speeds. Travel 60 mph for 50 miles and 30 mph for 50 miles: average speed is $H(60, 30) = 40$ mph, not $A(60, 30) = 45$. The arithmetic mean overstates because more time is spent at the slower speed.
    - Price-earnings ratios across a portfolio when weighted by *earnings* (the harmonic-mean P/E) vs. weighted by *market cap* (the weighted-arithmetic-mean P/E).
    - Combining $F_1$ scores: $F_1 = H(\text{precision}, \text{recall}) = 2 PR / (P + R)$.

    The general rule: when averaging quantities whose **reciprocals** add naturally (rates of change, frequencies per unit), use the harmonic mean. Confusing it with the arithmetic mean produces systematic bias.

---

**Exercise 7.**
The sample mean and median both estimate "the center" of a symmetric distribution. **Compare their relative efficiency** under (a) the normal distribution and (b) the Laplace (double-exponential) distribution.

??? success "Solution to Exercise 7"
    Relative efficiency of the median compared to the mean is $\mathrm{ARE}(\tilde X, \bar X) = \mathrm{Var}(\bar X)/\mathrm{Var}(\tilde X)$ — smaller means median is less efficient.

    **(a) Normal distribution:** $\mathrm{ARE} = 2/\pi \approx 0.637$. The mean is more efficient by a factor of $\pi/2$. For a sample of size 100, the median has variance equivalent to a mean computed from only 63 observations. This is the classical Gaussian-efficiency cost of using the median.

    **(b) Laplace distribution** (density $\propto e^{-|x|}$, heavy-tailed compared to normal): $\mathrm{ARE} = 2 > 1$. The median is twice as efficient as the mean. With Laplace tails, the mean is "wasteful" because extreme observations have high influence relative to the information they carry.

    **Lesson:** the choice between mean and median is not just about robustness — it's about matching the estimator to the tail behavior of the distribution. Under normality, the mean is optimal (BLUE); under heavier-tailed distributions, the median (or other robust estimators) can be far more efficient. This is the theoretical underpinning of M-estimators (Huber 1964), which interpolate between the mean (efficient for light tails) and the median (efficient for heavy tails).
