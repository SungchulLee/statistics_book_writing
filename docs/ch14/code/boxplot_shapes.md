# Distribution Shapes via Boxplots

## Overview

Boxplots summarise a distribution through its quartiles, median, and potential outliers, making them effective tools for detecting skewness and heavy tails at a glance. A symmetric, normal-like distribution produces a roughly symmetric boxplot with few outliers, while skewed or heavy-tailed distributions leave characteristic visual signatures. This page demonstrates how boxplots reveal departures from normality using lognormal (skewed) and Student-$t$ (heavy-tailed) examples.

## Anatomy of a Boxplot

A standard boxplot displays five summary statistics and marks outliers:

| Component | Definition |
|---|---|
| Median line | $Q_2$ (50th percentile) |
| Box edges | $Q_1$ (25th percentile) to $Q_3$ (75th percentile) |
| Interquartile range | $\text{IQR} = Q_3 - Q_1$ |
| Lower whisker | Smallest observation $\geq Q_1 - 1.5\,\text{IQR}$ |
| Upper whisker | Largest observation $\leq Q_3 + 1.5\,\text{IQR}$ |
| Outliers | Points beyond the whiskers |

For a normal distribution $\mathcal{N}(\mu, \sigma^2)$, the theoretical quartiles are

$$
Q_1 = \mu - 0.6745\,\sigma, \qquad Q_3 = \mu + 0.6745\,\sigma,
$$

giving $\text{IQR} = 1.349\,\sigma$. The whisker boundaries extend to approximately $\mu \pm 2.698\,\sigma$. The probability of an observation falling beyond the whiskers under normality is approximately

$$
P(|X - \mu| > 2.698\,\sigma) \approx 0.007,
$$

so about 0.7% of observations are expected to appear as outliers.

## Skewed Distribution: Lognormal

When data come from a right-skewed distribution such as $\text{Lognormal}(0, 0.7)$, the boxplot shows:

- the median is closer to the lower edge of the box,
- the upper whisker is much longer than the lower whisker, and
- many outliers appear above the upper whisker.

### Code

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)
x_skew = rng.lognormal(0.0, 0.7, size=400)

fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot(x_skew, showmeans=True)
ax.set_title("Boxplot: skewed distribution (lognormal)")
ax.set_ylabel("Values")
plt.tight_layout()
plt.show()
```

## Heavy-Tailed Distribution: Student-t

A Student-$t$ distribution with low degrees of freedom (e.g., $\nu = 3$) is symmetric but has much heavier tails than the normal. The boxplot shows:

- a roughly symmetric box (the median is centred), but
- outliers on *both* sides, far more than the $\approx 0.7\%$ expected under normality.

### Code

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)
x_t = rng.standard_t(df=3, size=400)

fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot(x_t, showmeans=True)
ax.set_title("Boxplot: heavy tails (t, df=3)")
ax.set_ylabel("Values")
plt.tight_layout()
plt.show()
```

## Interpretation

Boxplots provide a quick diagnostic for normality:

- **Symmetric box + few outliers:** consistent with normality.
- **Asymmetric box or unequal whiskers:** suggests skewness.
- **Symmetric box + many outliers:** suggests heavy tails (leptokurtic).

While boxplots alone cannot confirm normality, they are a valuable first screening tool, especially when comparing multiple groups side by side.

## Exercises

**Exercise 1.** Generate $n = 500$ standard normal observations and create a boxplot. Count the number of outliers and compare with the theoretical expectation of $0.007 \times 500 \approx 3.5$.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=500)

    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    outliers = np.sum((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr))
    print(f"Outliers: {outliers} (expected ~3-4)")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(x, showmeans=True)
    ax.set_title("Boxplot: N(0,1), n=500")
    plt.tight_layout()
    plt.show()
    ```

    The observed number of outliers should be in the range of 1--7, close to the theoretical expectation of about 3.5 for $n = 500$. $\square$

---

**Exercise 2.** Create side-by-side boxplots for samples of size 400 drawn from (a) $\mathcal{N}(0,1)$, (b) $\text{Lognormal}(0, 0.5)$, and (c) $t_5$. Describe the visual differences.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    normal = rng.normal(0, 1, size=400)
    lognorm = rng.lognormal(0, 0.5, size=400)
    t5 = rng.standard_t(df=5, size=400)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([normal, lognorm, t5], labels=["N(0,1)", "Lognormal", "t(5)"],
               showmeans=True)
    ax.set_title("Side-by-Side Boxplots")
    ax.set_ylabel("Values")
    plt.tight_layout()
    plt.show()
    ```

    The normal boxplot is symmetric with very few outliers. The lognormal boxplot has a long upper whisker and many upper outliers (right skew). The $t_5$ boxplot is symmetric but has outliers on both sides, reflecting heavier tails. $\square$

---

**Exercise 3.** Derive the theoretical probability that a single observation from $\mathcal{N}(0,1)$ falls outside the whiskers of a boxplot (i.e., beyond $Q_1 - 1.5\,\text{IQR}$ or $Q_3 + 1.5\,\text{IQR}$).

??? success "Solution to Exercise 3"

    For $X \sim \mathcal{N}(0,1)$, the theoretical quartiles are $Q_1 = \mathcal{N}^{-1}(0.25) = -0.6745$ and $Q_3 = \mathcal{N}^{-1}(0.75) = 0.6745$. Thus $\text{IQR} = 1.3490$ and the whisker limits are

    $$
    Q_1 - 1.5 \times \text{IQR} = -0.6745 - 2.0235 = -2.6980,
    $$

    $$
    Q_3 + 1.5 \times \text{IQR} = 0.6745 + 2.0235 = 2.6980.
    $$

    The probability of falling outside is

    $$
    P(|X| > 2.6980) = 2\,\mathcal{N}(-2.6980) = 2 \times 0.003488 = 0.006977 \approx 0.7\%.
    $$

    $\square$

---

**Exercise 4.** For a $t_\nu$ distribution, the kurtosis is $3 + 6/(\nu - 4)$ for $\nu > 4$. Use this to explain why a $t_3$ boxplot shows far more outliers than a normal boxplot.

??? success "Solution to Exercise 4"

    The kurtosis formula $\kappa = 3 + 6/(\nu - 4)$ requires $\nu > 4$; for $\nu = 3$ the kurtosis is actually infinite (the fourth moment does not exist). This means the tails of the $t_3$ distribution decay as $|x|^{-4}$ (a power law), far more slowly than the exponential decay $e^{-x^2/2}$ of the normal distribution. Consequently, the probability of extreme values is much higher. The whisker limits computed from the sample IQR are similar to the normal case (the box is roughly the same width since the central 50% of $t_3$ is comparable to a normal), but the slow tail decay means many observations lie far beyond the whiskers, appearing as outliers. $\square$

---

**Exercise 5.** Write a function that takes a data array, creates a boxplot, and annotates it with the sample skewness and excess kurtosis. Test it on $\text{Lognormal}(0, 0.7)$ data.

??? success "Solution to Exercise 5"

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    def annotated_boxplot(data, title="Boxplot"):
        g1 = stats.skew(data, bias=False)
        g2 = stats.kurtosis(data, fisher=True, bias=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(data, showmeans=True)
        ax.set_title(title)
        ax.set_ylabel("Values")
        ax.text(0.02, 0.95,
                f"Skewness: {g1:.3f}\nExcess kurtosis: {g2:.3f}",
                transform=ax.transAxes, verticalalignment='top',
                fontsize=10, bbox=dict(boxstyle='round', alpha=0.1))
        plt.tight_layout()
        plt.show()

    rng = np.random.default_rng(42)
    x = rng.lognormal(0, 0.7, size=400)
    annotated_boxplot(x, title="Lognormal(0, 0.7)")
    ```

    The annotation will show substantial positive skewness (typically $> 1$) and positive excess kurtosis, both consistent with the asymmetric boxplot and its many upper outliers. $\square$
