# Graphical Normality Checks

## Overview

Graphical methods provide an intuitive first step in assessing whether data follow a normal distribution. Before running any formal hypothesis test, visual inspection of histograms, density overlays, and quantile-quantile (Q-Q) plots can reveal skewness, heavy tails, multimodality, and other departures from normality. These methods complement formal tests by showing *how* the data deviate, not just *whether* they deviate.

## Histogram with Normal Overlay

The simplest graphical check plots a histogram of the observed data and superimposes the probability density function (pdf) of a normal distribution whose mean and variance match the sample estimates.

Let $X_1, X_2, \ldots, X_n$ be an independent random sample. The sample mean and sample standard deviation are

$$
\bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i, \qquad S = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X})^2}.
$$

The fitted normal density is then

$$
\hat{f}(x) = \frac{1}{S\sqrt{2\pi}} \exp\!\Bigl(-\frac{(x - \bar{X})^2}{2S^2}\Bigr).
$$

If the histogram bars align closely with $\hat{f}$, the data are consistent with normality.

### Code

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

n = 100
data = np.random.normal(loc=0, scale=1, size=n)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(data, bins=15, density=True, alpha=0.6, edgecolor="black")
x_grid = np.linspace(data.min() - 0.5, data.max() + 0.5, 200)
ax.plot(x_grid, stats.norm.pdf(x_grid, data.mean(), data.std(ddof=1)),
        linewidth=2, label="Fitted Normal PDF")
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.set_title("Histogram with Normal Overlay")
ax.legend()
plt.tight_layout()
plt.show()
```

## Kernel Density Estimate

A kernel density estimate (KDE) smooths the histogram and is useful for spotting multimodality or asymmetry. With a Gaussian kernel and bandwidth $h$, the KDE is

$$
\hat{f}_h(x) = \frac{1}{nh}\sum_{i=1}^{n} \phi\!\Bigl(\frac{x - X_i}{h}\Bigr),
$$

where $\phi$ denotes the standard normal density. Overlay this with the fitted normal curve; systematic differences indicate departures from normality.

## Q-Q Plot

The quantile-quantile (Q-Q) plot is the single most informative graphical normality check. For each order statistic $X_{(i)}$, compute the corresponding theoretical quantile

$$
q_i = \mathcal{N}^{-1}\!\Bigl(\frac{i - 0.5}{n}\Bigr),
$$

and plot the pairs $(q_i,\, X_{(i)})$. Under normality the points should fall approximately on a straight line. Common departures have recognisable signatures:

| Pattern | Departure |
|---|---|
| S-shaped curve | Heavy tails (leptokurtic) |
| Concave arc | Right skew |
| Convex arc | Left skew |
| Staircase steps | Discreteness or rounding |

## Interpretation

Graphical checks are subjective but invaluable. A histogram that is clearly bimodal, a KDE that reveals pronounced asymmetry, or a Q-Q plot that curves sharply at the tails all signal that formal normality tests will likely reject. Conversely, if the graphical checks look clean, a borderline $p$-value from a formal test may be less concerning. Always combine graphical and formal methods for a well-rounded assessment.

## Exercises

**Exercise 1.** Generate $n = 200$ observations from a standard normal distribution. Plot a histogram with 20 bins and overlay the fitted normal density. Comment on the fit.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, size=200)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data, bins=20, density=True, alpha=0.6, edgecolor="black")
    x_grid = np.linspace(-4, 4, 200)
    ax.plot(x_grid, stats.norm.pdf(x_grid, data.mean(), data.std(ddof=1)),
            linewidth=2)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Histogram with Normal Overlay")
    plt.tight_layout()
    plt.show()
    ```

    With $n = 200$ standard normal draws the histogram bars should closely track the bell-shaped fitted curve. Minor deviations are expected due to sampling variability. $\square$

---

**Exercise 2.** Generate $n = 300$ observations from a $\text{Lognormal}(0, 0.6)$ distribution. Create a Q-Q plot against the normal distribution and describe the shape you observe.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    data = rng.lognormal(0, 0.6, size=300)

    stats.probplot(data, dist="norm", plot=plt)
    plt.title("Q-Q Plot: Lognormal Data vs Normal")
    plt.tight_layout()
    plt.show()
    ```

    The Q-Q plot shows a concave (upward-bending) curve: the upper quantiles of the data exceed the theoretical normal quantiles. This is the hallmark of right skewness. The lower tail may approximately follow the line, but the upper tail departs markedly. $\square$

---

**Exercise 3.** Explain why a Q-Q plot can reveal the *type* of departure from normality (e.g., skewness versus heavy tails), whereas a single $p$-value from a formal test cannot.

??? success "Solution to Exercise 3"

    A formal normality test produces a single test statistic and $p$-value that measure overall departure from normality. The $p$-value says nothing about *how* the distribution deviates. In contrast, a Q-Q plot displays every order statistic against its theoretical counterpart, so the analyst can see whether the deviation occurs in the tails (heavy tails produce an S-shape), in one tail only (skewness produces a concave or convex arc), or in the centre (multimodality produces steps or flat regions). This diagnostic richness is why graphical checks are recommended alongside formal tests. $\square$

---

**Exercise 4.** For a random sample of size $n$ from a continuous distribution $F$, show that the expected value of the $i$-th order statistic's plotting position in a Q-Q plot is approximately $\mathcal{N}^{-1}\!\bigl(\frac{i - 0.5}{n}\bigr)$ under the null hypothesis that $F = \mathcal{N}$.

??? success "Solution to Exercise 4"

    Under $F = \mathcal{N}$ the probability integral transform gives $U_i = \mathcal{N}(X_i) \sim \text{Uniform}(0,1)$. The $i$-th order statistic of the uniform sample, $U_{(i)}$, has $\mathbb{E}[U_{(i)}] = \frac{i}{n+1}$. For large $n$ the Blom approximation replaces this with $p_i = \frac{i - 0.375}{n + 0.25} \approx \frac{i - 0.5}{n}$, which corrects for the bias of the uniform order statistics near the boundaries. Applying $\mathcal{N}^{-1}$ gives the theoretical quantile $q_i = \mathcal{N}^{-1}(p_i)$. When the data truly come from $\mathcal{N}$, the ordered sample values $X_{(i)}$ satisfy $\mathbb{E}[X_{(i)}] \approx q_i$, so the Q-Q plot is expected to lie on the identity line. $\square$

---

**Exercise 5.** Write a Python function that takes a data array and produces a side-by-side figure with (a) a histogram with normal overlay and (b) a Q-Q plot. Test it on data from a $t$-distribution with 4 degrees of freedom.

??? success "Solution to Exercise 5"

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    def normality_panel(data):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram with normal overlay
        ax = axes[0]
        ax.hist(data, bins=30, density=True, alpha=0.6, edgecolor="black")
        x_grid = np.linspace(data.min() - 1, data.max() + 1, 300)
        ax.plot(x_grid,
                stats.norm.pdf(x_grid, data.mean(), data.std(ddof=1)),
                linewidth=2, label="Fitted Normal")
        ax.set_title("Histogram + Normal Overlay")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()

        # Q-Q plot
        ax = axes[1]
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot vs Normal")

        plt.tight_layout()
        plt.show()

    rng = np.random.default_rng(42)
    t_data = rng.standard_t(df=4, size=500)
    normality_panel(t_data)
    ```

    For $t_4$ data the histogram shows heavier tails than the normal overlay (more mass beyond $\pm 3$). The Q-Q plot exhibits a characteristic S-shape: the lower-left points bend below the line and the upper-right points bend above it, confirming excess kurtosis. $\square$
