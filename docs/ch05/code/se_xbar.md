# Standard Error of X-bar

## Overview

The **standard error** of a statistic is the standard deviation of its sampling distribution. For the sample mean $\bar{X}$, the standard error quantifies how much $\bar{X}$ varies from sample to sample. It is the single most important quantity for understanding the precision of the sample mean as an estimator of the population mean $\mu$. This page derives the standard error formula, estimates it via simulation, and shows how to visualize it.

## Definition

For a random sample $X_1, \ldots, X_n$ from a population with mean $\mu$ and standard deviation $\sigma$, the standard error of $\bar{X}$ is:

$$
\text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}}
$$

!!! info "Standard Error vs. Standard Deviation"

    - The **standard deviation** $\sigma$ measures the spread of individual observations in the population.
    - The **standard error** $\sigma / \sqrt{n}$ measures the spread of the sample mean $\bar{X}$ across repeated samples.
    - The standard error is always smaller than $\sigma$ (for $n > 1$) and decreases as $n$ increases.

In practice, $\sigma$ is usually unknown and is estimated by the sample standard deviation $s$, giving the **estimated standard error**:

$$
\widehat{\text{SE}}(\bar{X}) = \frac{s}{\sqrt{n}}
$$

## Derivation

Starting from the definition of $\bar{X}$ with i.i.d. observations:

$$
\text{Var}(\bar{X}) = \text{Var}\!\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n}
$$

The standard error is the square root of the variance:

$$
\text{SE}(\bar{X}) = \sqrt{\text{Var}(\bar{X})} = \frac{\sigma}{\sqrt{n}}
$$

## Simulation

The following code simulates 10,000 sample means from a Uniform(0, 1) population with $n = 5$, computes the empirical standard error, and visualizes the result.

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

# Simulate 10,000 sample means
X_bar = []
for _ in range(10_000):
    x = np.random.uniform(size=(5,))
    X_bar.append(x.mean())

# Compute empirical statistics
average = np.array(X_bar).mean()
standard_error = np.array(X_bar).std()

print(f"Estimated Mean of X_bar:  {average:.4f}")
print(f"Standard Error of X_bar:  {standard_error:.4f}")

# Visualize
fig, ax = plt.subplots(figsize=(12, 3))
ax.set_title("Sampling Distribution of X-bar")
ax.hist(X_bar, bins=100, density=True, alpha=0.3)
ax.vlines(average, ymin=0, ymax=5, color="k", lw=5, label="Mean")
ax.vlines(average + standard_error, ymin=0, ymax=5,
          color="k", ls="--", label="Mean +/- SE")
ax.vlines(average - standard_error, ymin=0, ymax=5,
          color="k", ls="--")
ax.legend()
plt.show()
```

### Expected Output

For Uniform(0, 1) with $n = 5$:

- **Theoretical mean**: $\mu = 0.5$
- **Theoretical SE**: $\sigma / \sqrt{n} = (1/\sqrt{12}) / \sqrt{5} \approx 0.1291$
- **Empirical values** should be close to these theoretical targets.

## How Standard Error Decreases with n

The standard error decreases as $1/\sqrt{n}$, which means:

- Doubling $n$ reduces SE by a factor of $\sqrt{2} \approx 1.41$.
- Quadrupling $n$ halves the SE.
- To reduce SE by a factor of 10, you need 100 times more observations.

| $n$ | $\text{SE}(\bar{X}) = \sigma/\sqrt{n}$ | Relative to $n=1$ |
|---|---|---|
| 1 | $\sigma$ | 100% |
| 4 | $\sigma/2$ | 50% |
| 25 | $\sigma/5$ | 20% |
| 100 | $\sigma/10$ | 10% |
| 10,000 | $\sigma/100$ | 1% |

!!! warning "Diminishing Returns"
    The $1/\sqrt{n}$ relationship means that each additional observation contributes less and less to precision. Going from $n = 100$ to $n = 400$ (a 4x increase in cost) only halves the standard error.

## Exercises

**Exercise 1.** A population has $\sigma = 10$. Compute the standard error of $\bar{X}$ for $n = 25$, $n = 100$, and $n = 400$. Verify the "quadrupling" rule.

??? success "Solution to Exercise 1"
    $$
    \text{SE}(n=25) = \frac{10}{\sqrt{25}} = \frac{10}{5} = 2.0
    $$

    $$
    \text{SE}(n=100) = \frac{10}{\sqrt{100}} = \frac{10}{10} = 1.0
    $$

    $$
    \text{SE}(n=400) = \frac{10}{\sqrt{400}} = \frac{10}{20} = 0.5
    $$

    Each time $n$ is multiplied by 4, the SE is halved: $2.0 \to 1.0 \to 0.5$. This confirms the quadrupling rule. $\square$

---

**Exercise 2.** A researcher wants the standard error of $\bar{X}$ to be at most 0.5. The population standard deviation is estimated to be $\sigma \approx 8$. What minimum sample size is needed?

??? success "Solution to Exercise 2"
    We need:

    $$
    \frac{\sigma}{\sqrt{n}} \le 0.5 \implies \sqrt{n} \ge \frac{8}{0.5} = 16 \implies n \ge 256
    $$

    A minimum sample size of $n = 256$ is required. $\square$

---

**Exercise 3.** Prove that $\text{SE}(\bar{X})$ is a decreasing and convex function of $n$ for $n \ge 1$. What does convexity imply about the marginal benefit of increasing sample size?

??? success "Solution to Exercise 3"
    Let $f(n) = \sigma / \sqrt{n} = \sigma \cdot n^{-1/2}$ for $n > 0$.

    First derivative:

    $$
    f'(n) = -\frac{\sigma}{2} n^{-3/2} < 0
    $$

    So $f$ is strictly decreasing: larger samples always give smaller standard errors.

    Second derivative:

    $$
    f''(n) = \frac{3\sigma}{4} n^{-5/2} > 0
    $$

    So $f$ is strictly convex. Convexity means the rate of decrease of the standard error slows down as $n$ grows. In practical terms, each additional observation reduces the standard error by less than the previous one -- there are diminishing marginal returns to increasing sample size. $\square$

---

**Exercise 4.** Show that when using the estimated standard error $\widehat{\text{SE}} = s / \sqrt{n}$, the quantity $(\bar{X} - \mu) / \widehat{\text{SE}}$ follows a $t$-distribution with $n - 1$ degrees of freedom when the population is normal.

??? success "Solution to Exercise 4"
    Recall that for $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$:

    - $Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0, 1)$
    - $Q = \frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)$
    - $Z$ and $Q$ are independent.

    By definition, the $t$-distribution is the ratio $T = Z / \sqrt{Q/k}$ where $Z \sim N(0,1)$, $Q \sim \chi^2(k)$, and $Z \perp Q$.

    $$
    T = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \cdot \frac{1}{\sqrt{(n-1)S^2 / (\sigma^2(n-1))}} = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \cdot \frac{\sigma}{S} = \frac{\bar{X} - \mu}{S/\sqrt{n}} \sim t(n-1)
    $$

    $\square$

---

**Exercise 5.** Modify the simulation to use an Exponential(1) population instead of Uniform(0, 1). Compare the theoretical SE $(\sigma/\sqrt{n} = 1/\sqrt{5})$ with the empirical SE. Is the formula $\text{SE} = \sigma/\sqrt{n}$ still valid for non-normal populations?

??? success "Solution to Exercise 5"
    ```python
    import numpy as np
    np.random.seed(0)

    X_bar = [np.random.exponential(size=5).mean() for _ in range(10_000)]

    empirical_se = np.std(X_bar)
    theoretical_se = 1 / np.sqrt(5)

    print(f"Theoretical SE: {theoretical_se:.4f}")
    print(f"Empirical SE:   {empirical_se:.4f}")
    ```

    The theoretical SE is $1/\sqrt{5} \approx 0.4472$ and the empirical SE should be very close to this value.

    Yes, the formula $\text{SE}(\bar{X}) = \sigma/\sqrt{n}$ is valid for **any** population with finite variance, regardless of the population shape. This is because $\text{Var}(\bar{X}) = \sigma^2/n$ follows directly from the independence of the observations and the properties of variance. The formula does not depend on normality. What does depend on the population shape is the **distribution** of $\bar{X}$ (normal vs. non-normal), but the standard error formula itself is universal. $\square$
