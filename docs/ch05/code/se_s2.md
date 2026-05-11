# Standard Error of S-squared

## Overview

Just as the sample mean $\bar{X}$ has a standard error that measures its variability across samples, the sample variance $S^2$ also has a standard error. Understanding the precision of $S^2$ is important when we need to estimate or make inferences about the population variance $\sigma^2$. This page derives the standard error of $S^2$, estimates it via simulation from a Uniform population, and illustrates it graphically.

## Definition

The **standard error of $S^2$** is the standard deviation of its sampling distribution:

$$
\text{SE}(S^2) = \sqrt{\text{Var}(S^2)}
$$

For a normal population, this has the closed-form expression:

$$
\text{SE}(S^2) = \sigma^2 \sqrt{\frac{2}{n-1}}
$$

More generally, for any population with finite fourth moment:

$$
\text{Var}(S^2) = \frac{1}{n}\left(\mu_4 - \frac{n-3}{n-1}\sigma^4\right)
$$

where $\mu_4 = E[(X - \mu)^4]$ is the fourth central moment.

## Example: Uniform(0, 1) Population

For $X \sim \text{Uniform}(0, 1)$:

$$
\sigma^2 = \frac{1}{12}, \qquad \mu_4 = \frac{1}{80}
$$

With $n = 5$:

$$
\text{Var}(S^2) = \frac{1}{5}\left(\frac{1}{80} - \frac{2}{4} \cdot \frac{1}{144}\right) = \frac{1}{5}\left(\frac{1}{80} - \frac{1}{288}\right)
$$

$$
= \frac{1}{5} \cdot \frac{288 - 80}{80 \times 288} = \frac{1}{5} \cdot \frac{208}{23040} = \frac{208}{115200} \approx 0.001806
$$

$$
\text{SE}(S^2) \approx \sqrt{0.001806} \approx 0.0425
$$

## Simulation

The following code simulates 10,000 values of $S^2$ from a Uniform(0, 1) population with $n = 5$ and visualizes the result with the estimated mean and standard error.

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

S_square = []
for _ in range(10_000):
    x = np.random.uniform(size=(5,))
    sigma = x.std(ddof=1)
    S_square.append(sigma ** 2)

average = np.array(S_square).mean()
standard_error = np.array(S_square).std()

print(f"Estimated Mean of S^2:   {average:.4f}")
print(f"Standard Error of S^2:   {standard_error:.4f}")

# Visualize
fig, ax = plt.subplots(figsize=(12, 3))
ax.set_title("Sampling Distribution of S^2")
ax.hist(S_square, bins=100, density=True, alpha=0.3)
ax.vlines(average, ymin=0, ymax=12, color="k", lw=5, label="Mean")
ax.vlines(average + standard_error, ymin=0, ymax=12,
          color="k", ls="--", label="Mean +/- SE")
ax.vlines(average - standard_error, ymin=0, ymax=12,
          color="k", ls="--")
ax.legend()
plt.show()
```

### Expected Output

For Uniform(0, 1) with $n = 5$:

- **Theoretical mean**: $E[S^2] = \sigma^2 = 1/12 \approx 0.0833$
- **Theoretical SE**: approximately $0.0425$
- The histogram is right-skewed (since $S^2 \ge 0$), which is typical for the sampling distribution of a variance.

## Interpretation

!!! note "Key Observations"

    1. The sampling distribution of $S^2$ is **right-skewed**, unlike the approximately symmetric distribution of $\bar{X}$.
    2. The mean of the simulated $S^2$ values is close to $\sigma^2 = 1/12$, confirming that $S^2$ is unbiased.
    3. The standard error of $S^2$ is much smaller than its mean, indicating reasonable precision even with $n = 5$.
    4. The dashed lines at mean $\pm$ SE show the typical range of $S^2$ values. Because the distribution is skewed, values above mean $+$ SE are more common than values below mean $-$ SE.

!!! warning "Standard Error of S-squared Depends on Population Shape"
    Unlike $\text{SE}(\bar{X}) = \sigma/\sqrt{n}$, which depends only on $\sigma$ and $n$, the standard error of $S^2$ depends on the population's fourth moment (kurtosis). Heavy-tailed populations produce more variable $S^2$ values.

## Exercises

**Exercise 1.** For a $N(0, 1)$ population with $n = 10$, compute the theoretical $E[S^2]$ and $\text{SE}(S^2)$.

??? success "Solution to Exercise 1"
    For $N(0, 1)$: $\sigma^2 = 1$.

    $$
    E[S^2] = \sigma^2 = 1
    $$

    $$
    \text{SE}(S^2) = \sigma^2 \sqrt{\frac{2}{n-1}} = 1 \cdot \sqrt{\frac{2}{9}} = \sqrt{\frac{2}{9}} = \frac{\sqrt{2}}{3} \approx 0.4714
    $$

    $\square$

---

**Exercise 2.** Derive $\text{Var}(S^2) = 2\sigma^4 / (n-1)$ for a normal population by using the chi-squared distribution result $(n-1)S^2/\sigma^2 \sim \chi^2(n-1)$.

??? success "Solution to Exercise 2"
    Let $Q = (n-1)S^2/\sigma^2$. Then $Q \sim \chi^2(n-1)$ and $\text{Var}(Q) = 2(n-1)$.

    Since $S^2 = \sigma^2 Q / (n-1)$:

    $$
    \text{Var}(S^2) = \left(\frac{\sigma^2}{n-1}\right)^2 \text{Var}(Q) = \frac{\sigma^4}{(n-1)^2} \cdot 2(n-1) = \frac{2\sigma^4}{n-1}
    $$

    Therefore:

    $$
    \text{SE}(S^2) = \sqrt{\frac{2\sigma^4}{n-1}} = \sigma^2 \sqrt{\frac{2}{n-1}}
    $$

    $\square$

---

**Exercise 3.** Show that for Uniform(0, 1), the fourth central moment is $\mu_4 = 1/80$.

??? success "Solution to Exercise 3"
    For $X \sim \text{Uniform}(0, 1)$ with $\mu = 1/2$:

    $$
    \mu_4 = E[(X - \mu)^4] = \int_0^1 \left(x - \frac{1}{2}\right)^4 dx
    $$

    Substituting $u = x - 1/2$, so $du = dx$ and the limits become $-1/2$ to $1/2$:

    $$
    \mu_4 = \int_{-1/2}^{1/2} u^4 \, du = \left[\frac{u^5}{5}\right]_{-1/2}^{1/2} = \frac{(1/2)^5}{5} - \frac{(-1/2)^5}{5} = \frac{2 \cdot (1/32)}{5} = \frac{1}{80}
    $$

    $\square$

---

**Exercise 4.** Explain why the sampling distribution of $S^2$ is right-skewed when $n$ is small, and why it becomes more symmetric as $n$ increases.

??? success "Solution to Exercise 4"
    The sample variance $S^2$ is bounded below by 0 but has no finite upper bound (in principle). For small $n$, the constraint $S^2 \ge 0$ creates a "floor" that truncates the left tail, while occasional extreme observations can push $S^2$ to large values, creating a long right tail.

    As $n$ increases, two effects symmetrise the distribution:

    1. **Central Limit Theorem for $S^2$**: For large $n$, $S^2$ is approximately a sum of many weakly dependent terms (the squared deviations), so its distribution converges to normal by a CLT-type argument.
    2. **Chi-squared convergence**: For normal populations, $(n-1)S^2/\sigma^2 \sim \chi^2(n-1)$. The skewness of $\chi^2(k)$ is $2\sqrt{2/k}$, which goes to 0 as $k = n - 1$ grows.

    Both effects mean that for large $n$, the distribution of $S^2$ becomes approximately symmetric (normal), and the standard error provides a good summary of the spread. $\square$

---

**Exercise 5.** Repeat the simulation using an Exponential(1) population. Compare the empirical SE of $S^2$ with the theoretical value, noting that for $\text{Exp}(1)$, $\sigma^2 = 1$ and $\mu_4 = 9$.

??? success "Solution to Exercise 5"
    ```python
    import numpy as np
    np.random.seed(0)

    S_square = [np.random.exponential(size=5).var(ddof=1) for _ in range(10_000)]
    empirical_se = np.std(S_square)
    ```

    Theoretical calculation with $\sigma^2 = 1$, $\mu_4 = 9$, $n = 5$:

    $$
    \text{Var}(S^2) = \frac{1}{n}\left(\mu_4 - \frac{n-3}{n-1}\sigma^4\right) = \frac{1}{5}\left(9 - \frac{2}{4}\right) = \frac{1}{5} \cdot 8.5 = 1.7
    $$

    $$
    \text{SE}(S^2) = \sqrt{1.7} \approx 1.304
    $$

    For comparison, the normal-theory SE would be $\sigma^2 \sqrt{2/(n-1)} = \sqrt{2/4} = \sqrt{0.5} \approx 0.707$.

    The Exponential population's SE is about 1.84 times larger than the normal-theory value, reflecting the heavier tails (excess kurtosis $= 6$) of the Exponential distribution. The empirical SE from simulation should be close to 1.304. $\square$
