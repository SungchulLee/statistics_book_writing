# Skewness Test

## Overview

The D'Agostino skewness test assesses whether the skewness of a dataset differs significantly from zero, the value expected under a normal distribution. It transforms the sample skewness into a $Z$-statistic that is approximately standard normal under the null hypothesis, making it a targeted test for asymmetry. This page derives the key formulas, demonstrates the test in Python, and discusses its practical use.

## Sample Skewness

The Fisher-Pearson coefficient of skewness (bias-corrected) for a sample $X_1, \ldots, X_n$ is

$$
g_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n} \left(\frac{X_i - \bar{X}}{S}\right)^3,
$$

where $\bar{X}$ is the sample mean and $S$ is the sample standard deviation (with Bessel correction). Under normality, $\mathbb{E}[g_1] = 0$ and

$$
\text{Var}(g_1) \approx \frac{6(n-2)}{(n+1)(n+3)}.
$$

## The D'Agostino Transformation

Because the distribution of $g_1$ is not exactly normal even under $H_0$, D'Agostino and Pearson (1973) proposed a nonlinear transformation that maps $g_1$ to a statistic $Z_1$ that is much closer to $\mathcal{N}(0,1)$. The transformation involves computing

$$
Y = g_1 \sqrt{\frac{(n+1)(n+3)}{6(n-2)}},
$$

followed by additional adjustments for higher-order cumulants. The final $Z_1$ is approximately standard normal under $H_0: \text{skewness} = 0$.

## Hypotheses

$$
H_0: \gamma_1 = 0 \quad (\text{population skewness is zero}), \qquad H_1: \gamma_1 \neq 0.
$$

The two-sided $p$-value is $p = 2\,\mathcal{N}(-|Z_1|)$, where $\mathcal{N}$ is the standard normal CDF.

### Code

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
x = rng.lognormal(mean=0.0, sigma=0.6, size=300)

g1 = stats.skew(x, bias=False)
z, p = stats.skewtest(x)

print(f"Sample size n = {x.size}")
print(f"Sample skewness (Fisher's g1) = {g1:.4f}")
print(f"D'Agostino skewness test: Z = {z:.4f}, p-value = {p:.4g}")
if p < 0.05:
    print("=> Evidence of non-zero skewness (departing from normality).")
else:
    print("=> No strong evidence of non-zero skewness.")
```

## Interpretation

For the lognormal data in the example, $g_1$ will be substantially positive (right skew), and the $p$-value will be very small, strongly rejecting the hypothesis of zero skewness. The skewness test is particularly useful when you suspect asymmetry but not necessarily other departures. It is one component of the D'Agostino $K^2$ omnibus test, which combines skewness and kurtosis tests.

**Minimum sample size.** SciPy requires $n \geq 8$ for `skewtest`. The approximation improves with larger $n$; for $n < 20$ the $p$-value should be interpreted cautiously.

## Exercises

**Exercise 1.** Generate $n = 300$ observations from a standard normal distribution. Compute $g_1$ and the skewness test $p$-value. Do you expect to reject at $\alpha = 0.05$?

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, size=300)

    g1 = stats.skew(x, bias=False)
    z, p = stats.skewtest(x)

    print(f"g1 = {g1:.4f}")
    print(f"Z = {z:.4f}, p = {p:.4g}")
    ```

    Since the data are truly normal, $g_1$ should be close to 0 and $p > 0.05$ in most realisations. We do not expect to reject, as the probability of a Type I error is only 5%. $\square$

---

**Exercise 2.** For $n = 200$ observations from a $\text{Uniform}(0, 1)$ distribution, compute the sample skewness and run the skewness test. The uniform distribution is symmetric but non-normal. Does the test reject?

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=200)

    g1 = stats.skew(x, bias=False)
    z, p = stats.skewtest(x)
    print(f"g1 = {g1:.4f}, p = {p:.4g}")
    ```

    The uniform distribution has population skewness $\gamma_1 = 0$, so the skewness test should *not* reject (large $p$-value). This illustrates a limitation: the skewness test can miss non-normal distributions that happen to be symmetric (the uniform is platykurtic but symmetric). A kurtosis test or omnibus test would detect the departure. $\square$

---

**Exercise 3.** Show that for a $\text{Lognormal}(\mu, \sigma^2)$ distribution the population skewness is $(e^{\sigma^2} + 2)\sqrt{e^{\sigma^2} - 1}$. Compute the value for $\sigma = 0.6$.

??? success "Solution to Exercise 3"

    The moment-generating properties of the lognormal give $\mathbb{E}[X^k] = e^{k\mu + k^2\sigma^2/2}$. After computing the first three central moments, the skewness simplifies to

    $$
    \gamma_1 = (e^{\sigma^2} + 2)\sqrt{e^{\sigma^2} - 1}.
    $$

    For $\sigma = 0.6$: $e^{0.36} = 1.4333$, so $e^{\sigma^2} - 1 = 0.4333$ and $\sqrt{0.4333} = 0.6583$. Then

    $$
    \gamma_1 = (1.4333 + 2)(0.6583) = 3.4333 \times 0.6583 \approx 2.261.
    $$

    This large positive skewness explains why the skewness test rejects decisively for lognormal samples. $\square$

---

**Exercise 4.** Run a simulation with 5,000 replicates to estimate the power of the skewness test at $\alpha = 0.05$ for $n = 100$ observations from $\text{Lognormal}(0, 0.4)$.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n, reps, alpha = 100, 5000, 0.05
    rejections = 0

    for _ in range(reps):
        x = rng.lognormal(0, 0.4, size=n)
        _, p = stats.skewtest(x)
        if p < alpha:
            rejections += 1

    power = rejections / reps
    print(f"Empirical power: {power:.4f}")
    ```

    The power will typically be between 0.7 and 0.9 for this combination of sample size and alternative distribution. The lognormal with $\sigma = 0.4$ has moderate skewness ($\gamma_1 \approx 1.32$), and $n = 100$ provides reasonable power to detect it. $\square$

---

**Exercise 5.** Prove that $\text{Var}(g_1) \approx 6/n$ for large $n$, starting from the exact formula $\text{Var}(g_1) = 6(n-2)/[(n+1)(n+3)]$.

??? success "Solution to Exercise 5"

    Starting from

    $$
    \text{Var}(g_1) = \frac{6(n-2)}{(n+1)(n+3)},
    $$

    divide numerator and denominator by $n^2$:

    $$
    \text{Var}(g_1) = \frac{6(1 - 2/n)}{(1 + 1/n)(1 + 3/n) \cdot n} \cdot \frac{n}{n} = \frac{6\,(1 - 2/n)}{n\,(1 + 1/n)(1 + 3/n)}.
    $$

    As $n \to \infty$, $(1 - 2/n) \to 1$, $(1 + 1/n) \to 1$, and $(1 + 3/n) \to 1$, giving

    $$
    \text{Var}(g_1) \to \frac{6}{n}.
    $$

    This shows that the standard error of $g_1$ is of order $1/\sqrt{n}$, and larger samples make it easier to detect departures from zero skewness. $\square$
