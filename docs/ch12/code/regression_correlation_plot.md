# Regression Correlation Plot

## Overview

This page explores how the Pearson correlation coefficient $r$ depends on two factors in a simple linear regression model: the slope of the true regression line and the scale of the error term. Through eight configurations with varying slopes and noise levels, we build intuition for when $r$ is large and when it is small.

---

## The Linear Model

Consider the simple linear regression:

$$
Y = \beta_1 + \beta_2 X + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2)
$$

The Pearson correlation between $X$ and $Y$ depends on both the slope $\beta_2$ and the error scale $\sigma$. In fact, for a fixed distribution of $X$, the population correlation is:

$$
\rho_{XY} = \frac{\beta_2 \, \sigma_X}{\sqrt{\beta_2^2 \, \sigma_X^2 + \sigma^2}}
$$

This formula reveals two important relationships:

1. **Larger $|\beta_2|$ increases $|\rho|$**: a steeper true slope means the signal is stronger relative to the noise.
2. **Larger $\sigma$ decreases $|\rho|$**: more noise drowns out the linear signal.

---

## Simulation Setup

We generate data from eight configurations that vary the slope and error scale:

```python
import numpy as np

np.random.seed(42)
DATA_SIZE = 100

CONFIGS = [
    # (beta1, beta2, error_scale)
    (2, 0.05, 1),    # tiny slope, low noise
    (2, -0.6, 1),    # moderate negative slope
    (2, 1.0,  1),    # unit slope
    (2, 3.0,  1),    # steep slope, low noise
    (2, 3.0,  3),    # steep slope, moderate noise
    (2, 3.0, 10),    # steep slope, high noise
    (2, 3.0, 20),    # steep slope, very high noise
    (2, 3.0, 50),    # steep slope, extreme noise
]


def generate(beta1, beta2, error_scale, n=DATA_SIZE):
    x = np.random.randint(1, n, n).astype(float)
    y = beta1 + beta2 * x + error_scale * np.random.randn(n)
    r = np.corrcoef(x, y)[0, 1]
    return x, y, r
```

The first four configurations fix $\sigma = 1$ and vary the slope; the last four fix $\beta_2 = 3$ and increase the noise.

---

## Generating the Panel Plot

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(20, 9))

for idx, (b1, b2, es) in enumerate(CONFIGS):
    row, col = divmod(idx, 4)
    ax = axes[row, col]
    x, y, r = generate(b1, b2, es)

    ax.scatter(x, y, alpha=0.5, s=15, edgecolors="grey")
    xs = np.sort(x)
    ax.plot(xs, b1 + b2 * xs, color="#FA954D", lw=2, alpha=0.8)
    ax.set_title(f"Y = {b1} + {b2}X + {es}u", fontsize=10)
    ax.annotate(f"r = {r:.3f}", xy=(0.05, 0.9),
                xycoords="axes fraction", fontsize=11,
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

fig.suptitle("How Slope and Noise Affect Pearson Correlation",
             fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
```

---

## Varying the Slope (Fixed Noise)

With $\sigma = 1$ held constant:

| $\beta_2$ | Expected $|r|$ | Explanation |
|:---:|:---:|:---|
| 0.05 | Very low | The regression line is nearly flat; the signal is barely distinguishable from noise |
| $-0.6$ | Moderate | A moderate negative slope produces a clear downward trend |
| 1.0 | High | The signal-to-noise ratio is approximately 1:1, favoring signal |
| 3.0 | Very high | The steep slope dominates; points cluster tightly around the line |

---

## Varying the Noise (Fixed Slope)

With $\beta_2 = 3$ held constant:

| $\sigma$ | Expected $|r|$ | Explanation |
|:---:|:---:|:---|
| 1 | Near 1 | Very low noise; the linear pattern is unmistakable |
| 3 | High | Moderate noise; points spread but the trend is clear |
| 10 | Moderate | High noise; the cloud of points is wider but the trend persists |
| 50 | Near 0 | Extreme noise overwhelms the signal; the scatter plot looks random |

---

## Interpretation

The correlation coefficient $r$ reflects the **signal-to-noise ratio** in a linear model. Specifically, the coefficient of determination $r^2$ gives the fraction of variance in $Y$ explained by $X$:

$$
r^2 = \frac{\beta_2^2 \, \sigma_X^2}{\beta_2^2 \, \sigma_X^2 + \sigma^2}
$$

This can be rewritten in terms of the signal-to-noise ratio $\text{SNR} = \beta_2^2 \sigma_X^2 / \sigma^2$:

$$
r^2 = \frac{\text{SNR}}{1 + \text{SNR}}
$$

As $\text{SNR} \to \infty$, $r^2 \to 1$. As $\text{SNR} \to 0$, $r^2 \to 0$. This formula unifies both the slope and noise effects into a single quantity.

---

## Exercises

**Exercise 1.**
Using the population correlation formula, compute the theoretical $\rho_{XY}$ for $\beta_2 = 3$, $\sigma = 10$, and $X \sim \text{Uniform}(1, 100)$. Compare with the simulated value. (Hint: $\text{Var}(X) = (b-a)^2/12$ for $X \sim \text{Uniform}(a,b)$.)

??? success "Solution to Exercise 1"

    For $X \sim \text{Uniform}(1, 100)$, the variance is:

    $$
    \sigma_X^2 = \frac{(100 - 1)^2}{12} = \frac{9801}{12} = 816.75, \quad \sigma_X \approx 28.58
    $$

    The theoretical correlation is:

    $$
    \rho = \frac{\beta_2 \sigma_X}{\sqrt{\beta_2^2 \sigma_X^2 + \sigma^2}} = \frac{3 \times 28.58}{\sqrt{9 \times 816.75 + 100}} = \frac{85.73}{\sqrt{7450.75 + 100}} = \frac{85.73}{\sqrt{7550.75}} \approx \frac{85.73}{86.90} \approx 0.987
    $$

    ```python
    import numpy as np
    np.random.seed(42)
    x = np.random.uniform(1, 100, 1000)
    y = 2 + 3 * x + 10 * np.random.randn(1000)
    print(f"Simulated r = {np.corrcoef(x, y)[0, 1]:.4f}")
    print(f"Theoretical rho = 0.987")
    ```

    The simulated value should be close to the theoretical value of approximately 0.987. $\square$

---

**Exercise 2.**
Derive the formula $\rho_{XY} = \frac{\beta_2 \sigma_X}{\sqrt{\beta_2^2 \sigma_X^2 + \sigma^2}}$ starting from the definitions of covariance and variance for $Y = \beta_1 + \beta_2 X + \varepsilon$ where $X \perp \varepsilon$.

??? success "Solution to Exercise 2"

    Since $Y = \beta_1 + \beta_2 X + \varepsilon$ with $X \perp \varepsilon$:

    $$
    \text{Cov}(X, Y) = \text{Cov}(X, \beta_1 + \beta_2 X + \varepsilon) = \beta_2 \text{Var}(X) = \beta_2 \sigma_X^2
    $$

    using bilinearity of covariance and $\text{Cov}(X, \varepsilon) = 0$ by independence.

    $$
    \text{Var}(Y) = \text{Var}(\beta_2 X + \varepsilon) = \beta_2^2 \sigma_X^2 + \sigma^2
    $$

    Therefore:

    $$
    \rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sqrt{\text{Var}(Y)}} = \frac{\beta_2 \sigma_X^2}{\sigma_X \sqrt{\beta_2^2 \sigma_X^2 + \sigma^2}} = \frac{\beta_2 \sigma_X}{\sqrt{\beta_2^2 \sigma_X^2 + \sigma^2}}
    $$

    $\square$

---

**Exercise 3.**
For fixed $\sigma_X$ and $\sigma$, find the value of $\beta_2$ such that $r^2 = 0.5$ (i.e., the signal explains exactly half the variance). Express your answer in terms of $\sigma_X$ and $\sigma$.

??? success "Solution to Exercise 3"

    Setting $r^2 = 0.5$ in the formula:

    $$
    \frac{\beta_2^2 \sigma_X^2}{\beta_2^2 \sigma_X^2 + \sigma^2} = 0.5
    $$

    Cross-multiplying:

    $$
    2\beta_2^2 \sigma_X^2 = \beta_2^2 \sigma_X^2 + \sigma^2
    $$

    $$
    \beta_2^2 \sigma_X^2 = \sigma^2
    $$

    $$
    \beta_2 = \pm\frac{\sigma}{\sigma_X}
    $$

    When the slope equals the ratio of error standard deviation to predictor standard deviation, exactly half the variance in $Y$ is explained by $X$. This is the "break-even" point where signal and noise contribute equally. $\square$

---

**Exercise 4.**
Create a $3 \times 3$ panel plot where rows correspond to different slopes ($\beta_2 \in \{0.5, 2, 5\}$) and columns correspond to different noise levels ($\sigma \in \{1, 5, 20\}$). Annotate each panel with the sample $r$. Discuss the pattern.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    slopes = [0.5, 2, 5]
    noises = [1, 5, 20]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, b2 in enumerate(slopes):
        for j, sigma in enumerate(noises):
            ax = axes[i, j]
            x = np.random.uniform(1, 50, 100)
            y = 1 + b2 * x + sigma * np.random.randn(100)
            r = np.corrcoef(x, y)[0, 1]

            ax.scatter(x, y, s=10, alpha=0.5)
            xs = np.sort(x)
            ax.plot(xs, 1 + b2 * xs, 'r-', lw=2)
            ax.set_title(f"b2={b2}, sigma={sigma}\nr={r:.3f}", fontsize=9)

    plt.suptitle("Slope vs Noise: Effect on r", fontsize=14)
    plt.tight_layout()
    plt.show()
    ```

    The pattern follows the signal-to-noise ratio: $|r|$ increases moving down the rows (larger slope) and decreases moving across columns (larger noise). The top-right panel (small slope, large noise) has $|r|$ near zero, while the bottom-left (large slope, small noise) has $|r|$ near one. $\square$

---

**Exercise 5.**
Prove that $r^2 = \frac{\text{SNR}}{1 + \text{SNR}}$ where $\text{SNR} = \beta_2^2 \sigma_X^2 / \sigma^2$. Then show that $\text{SNR} = \frac{r^2}{1 - r^2}$, and interpret this inverse relationship.

??? success "Solution to Exercise 5"

    From the formula for $\rho_{XY}$:

    $$
    r^2 = \rho^2 = \frac{\beta_2^2 \sigma_X^2}{\beta_2^2 \sigma_X^2 + \sigma^2}
    $$

    Dividing numerator and denominator by $\sigma^2$:

    $$
    r^2 = \frac{\beta_2^2 \sigma_X^2 / \sigma^2}{\beta_2^2 \sigma_X^2 / \sigma^2 + 1} = \frac{\text{SNR}}{\text{SNR} + 1}
    $$

    To invert, solve for SNR:

    $$
    r^2 (\text{SNR} + 1) = \text{SNR}
    $$

    $$
    r^2 = \text{SNR}(1 - r^2)
    $$

    $$
    \text{SNR} = \frac{r^2}{1 - r^2}
    $$

    Interpretation: $r^2 / (1 - r^2)$ is the ratio of explained variance to unexplained variance. When $r^2 = 0.5$, $\text{SNR} = 1$ (equal signal and noise). When $r^2 = 0.9$, $\text{SNR} = 9$ (the signal is nine times stronger than the noise). This inverse formula is useful for converting observed correlations into signal-to-noise ratios. $\square$
