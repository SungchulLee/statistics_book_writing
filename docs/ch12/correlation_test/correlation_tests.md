# 18.6 Correlation Tests


This section covers three widely used statistical tests for assessing the significance of the relationship between two variables: Pearson's correlation, Spearman's rank correlation, and Kendall's tau.

---

## Setup: Shared Data Generation

The following modules generate three types of data to illustrate different correlation scenarios.

### `global_name_space.py`

```python
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Correlation Test Examples')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)
ARGS.size = 1000
```

### `load_data.py`

```python
import numpy as np
from global_name_space import ARGS

def load_data(data_type=0):
    data_dict = {}

    x = np.random.rand(ARGS.size) * 20
    eps = np.random.rand(ARGS.size) * 10

    # Dataset 0: No relationship (random scatter)
    y = np.random.rand(ARGS.size) * 20
    data_dict[0] = (x, y)

    # Dataset 1: Monotonic nonlinear relationship (cubic)
    y = (x + eps) ** 3
    data_dict[1] = (x, y)

    # Dataset 2: Non-monotonic relationship (sine)
    y = np.sin(x + eps)
    data_dict[2] = (x, y)

    return data_dict
```

The three datasets represent:

- **Dataset 0**: No relationship — random scatter, both Pearson and rank-based correlations should be near zero.
- **Dataset 1**: Monotonic nonlinear — Pearson may understate the relationship since it measures linearity, but Spearman and Kendall (which measure monotonicity) should detect it.
- **Dataset 2**: Non-monotonic (sinusoidal) — all correlation measures should be weak since the relationship is periodic, not monotonic or linear.

---

## Pearson's Correlation Test

[Documentation: `scipy.stats.pearsonr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)

Pearson's $r$ measures the **linear** relationship between two variables. The test statistic under the null hypothesis $H_0: \rho = 0$ follows a $t$-distribution with $n-2$ degrees of freedom.

```python
import matplotlib.pyplot as plt
import scipy.stats as stats
from load_data import load_data

def main():
    data_dict = load_data()
    _, axes = plt.subplots(1, len(data_dict), figsize=(12, 3))

    for ax, (x, y) in zip(axes, data_dict.values()):
        ax.plot(x, y, ".k")
        coef, p_val = stats.pearsonr(x, y)
        ax.set_title(f"Pearson's r: {coef:.4f}\np-value: {p_val:.4f}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
```

**When to use**: When both variables are continuous and you expect a **linear** relationship. Pearson's $r$ is sensitive to outliers and assumes bivariate normality for the test's p-value to be exact.

---

## Spearman's Rank Correlation Test

[Video: Spearman's Rank Correlation](https://www.youtube.com/watch?v=YpG2MlulP_o) |
[Documentation: `scipy.stats.spearmanr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)

Spearman's $\rho_s$ measures the **monotonic** relationship between two variables. It is computed as Pearson's $r$ applied to the ranks of the data rather than the raw values. This makes it robust to outliers and applicable to non-linear but monotonic relationships.

```python
import matplotlib.pyplot as plt
import scipy.stats as stats
from load_data import load_data

def main():
    data_dict = load_data()
    _, axes = plt.subplots(1, len(data_dict), figsize=(12, 3))

    for ax, (x, y) in zip(axes, data_dict.values()):
        ax.plot(x, y, ".k")
        coef, p_val = stats.spearmanr(x, y)
        ax.set_title(f"Spearman's ρ: {coef:.4f}\np-value: {p_val:.4f}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
```

**When to use**: When the relationship may be monotonic but not necessarily linear, or when the data contains outliers or is ordinal.

---

## Kendall's Tau

[Video 1: Kendall's Tau Explained](https://www.youtube.com/watch?v=oXVxaSoY94k) |
[Video 2: Kendall's Tau Calculation](https://www.youtube.com/watch?v=V4MgE43SrgM) |
[Documentation: `scipy.stats.kendalltau`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html)

Kendall's $\tau$ also measures the strength of a **monotonic** relationship, but it is based on the number of concordant and discordant pairs rather than ranks. It tends to be more robust for small sample sizes and has better statistical properties for hypothesis testing.

$$
\tau = \frac{(\text{number of concordant pairs}) - (\text{number of discordant pairs})}{\binom{n}{2}}
$$

```python
import matplotlib.pyplot as plt
import scipy.stats as stats
from load_data import load_data

def main():
    data_dict = load_data()
    _, axes = plt.subplots(1, len(data_dict), figsize=(12, 3))

    for ax, (x, y) in zip(axes, data_dict.values()):
        ax.plot(x, y, ".k")
        coef, p_val = stats.kendalltau(x, y)
        ax.set_title(f"Kendall's τ: {coef:.4f}\np-value: {p_val:.4f}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
```

**When to use**: Similar situations to Spearman's $\rho_s$, but preferred when sample sizes are small or when you want a more interpretable measure based on pairwise concordance.

---

## Comparison of the Three Tests

| Feature | Pearson's $r$ | Spearman's $\rho_s$ | Kendall's $\tau$ |
|---------|--------------|-------------------|----------------|
| Measures | Linear association | Monotonic association | Monotonic association |
| Data type | Continuous | Continuous or ordinal | Continuous or ordinal |
| Sensitivity to outliers | High | Low | Low |
| Assumption | Bivariate normality (for exact p-value) | None (rank-based) | None (rank-based) |
| Range | $[-1, 1]$ | $[-1, 1]$ | $[-1, 1]$ |
| Small samples | Less reliable | Moderate | More reliable |

---

## Worked Example: Age and Income

**Problem**: Test whether age and income are related using $\alpha = 0.05$.

```
age    = [18, 25, 57, 45, 26, 64, 37, 40, 24, 33]
income = [15000, 29000, 68000, 52000, 32000, 80000, 41000, 45000, 26000, 33000]
```

### Solution

All three tests yield $p \approx 0.0000$, indicating a strong, statistically significant relationship between age and income.

```python
import matplotlib.pyplot as plt
import scipy.stats as stats

def main():
    x = [18, 25, 57, 45, 26, 64, 37, 40, 24, 33]
    y = [15_000, 29_000, 68_000, 52_000, 32_000, 80_000, 41_000, 45_000, 26_000, 33_000]

    coef, p_val = stats.pearsonr(x, y)
    print(f"Pearson's r:   coef = {coef:.4f},  p-value = {p_val:.4f}")

    coef, p_val = stats.spearmanr(x, y)
    print(f"Spearman's ρ:  coef = {coef:.4f},  p-value = {p_val:.4f}")

    coef, p_val = stats.kendalltau(x, y)
    print(f"Kendall's τ:   coef = {coef:.4f},  p-value = {p_val:.4f}")

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, y, "ok")
    ax.set_xlabel("Age")
    ax.set_ylabel("Income")
    plt.show()

if __name__ == "__main__":
    main()
```

**Interpretation**: Since all p-values are well below $\alpha = 0.05$, we reject $H_0: \rho = 0$ and conclude that there is a statistically significant positive relationship between age and income in this sample. Note that this does not establish a causal relationship—confounders such as experience, education, and industry could influence both variables.
## Exercises

**Exercise 1.**
For a sample of $n = 25$ observations, the Pearson correlation is $r = 0.42$. Test the null hypothesis $H_0: \rho = 0$ at $\alpha = 0.05$ using the $t$-test for correlation.

??? success "Solution to Exercise 1"
    The test statistic is:

    $$
    t = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}} = \frac{0.42\sqrt{23}}{\sqrt{1-0.1764}} = \frac{0.42 \times 4.796}{\sqrt{0.8236}} = \frac{2.014}{0.9075} = 2.219
    $$

    With $df = n - 2 = 23$, the critical value for a two-tailed test at $\alpha = 0.05$ is $t_{0.025, 23} \approx 2.069$.

    Since $|t| = 2.219 > 2.069$, we reject $H_0$ and conclude that there is a statistically significant linear relationship between the two variables at the 5% level.

---

**Exercise 2.**
Explain the difference between testing $H_0: \rho = 0$ and testing $H_0: \rho = \rho_0$ for some $\rho_0 \neq 0$. Why does the second test require Fisher's $z$-transformation?

??? success "Solution to Exercise 2"
    When $\rho = 0$, the sampling distribution of $r$ is symmetric and the $t$-statistic $r\sqrt{(n-2)/(1-r^2)}$ follows an exact $t$-distribution with $n-2$ degrees of freedom.

    When $\rho \neq 0$, the sampling distribution of $r$ is **skewed** (especially for $|\rho|$ close to 1), so the $t$-test is no longer valid. Fisher's $z$-transformation:

    $$
    z = \frac{1}{2}\ln\frac{1+r}{1-r} = \text{arctanh}(r)
    $$

    transforms $r$ to an approximately normal distribution with mean $\text{arctanh}(\rho)$ and standard error $1/\sqrt{n-3}$, enabling valid hypothesis tests and confidence intervals for any $\rho_0$.

---

**Exercise 3.**
Two independent samples of sizes $n_1 = 40$ and $n_2 = 35$ yield Pearson correlations $r_1 = 0.55$ and $r_2 = 0.30$. Test whether the two population correlations are equal at $\alpha = 0.05$.

??? success "Solution to Exercise 3"
    Apply Fisher's $z$-transformation to each:

    $$
    z_1 = \text{arctanh}(0.55) = 0.6184, \quad z_2 = \text{arctanh}(0.30) = 0.3095
    $$

    The test statistic for comparing two independent correlations:

    $$
    Z = \frac{z_1 - z_2}{\sqrt{\frac{1}{n_1-3} + \frac{1}{n_2-3}}} = \frac{0.6184 - 0.3095}{\sqrt{\frac{1}{37} + \frac{1}{32}}} = \frac{0.3089}{\sqrt{0.02703 + 0.03125}} = \frac{0.3089}{\sqrt{0.05828}} = \frac{0.3089}{0.2414} = 1.28
    $$

    The critical value for a two-tailed test at $\alpha = 0.05$ is $z_{0.025} = 1.96$. Since $|Z| = 1.28 < 1.96$, we fail to reject $H_0$. There is insufficient evidence to conclude that the two population correlations differ.
