# D'Agostino K-Squared Test

## Overview

The D'Agostino $K^2$ test is an omnibus normality test that combines the skewness test and the kurtosis test into a single statistic. By jointly testing whether both the skewness and excess kurtosis are consistent with zero, it detects a broader range of departures from normality than either test alone. It is implemented in SciPy as `stats.normaltest`.

## The Test Statistic

The $K^2$ statistic is the sum of the squared $Z$-scores from the skewness and kurtosis tests:

$$
K^2 = Z_1^2 + Z_2^2,
$$

where $Z_1$ is the D'Agostino skewness statistic and $Z_2$ is the D'Agostino kurtosis statistic (see the individual test pages for details). Under $H_0$ (normality), $Z_1$ and $Z_2$ are approximately independent standard normals, so

$$
K^2 \;\underset{H_0}{\sim}\; \chi^2_2 \quad \text{(asymptotically)}.
$$

## Hypotheses

$$
H_0: \text{the data are normally distributed}, \qquad H_1: \text{the data are not normally distributed}.
$$

The $p$-value is

$$
p = P(\chi^2_2 \geq K^2_{\text{obs}}).
$$

We reject $H_0$ when $p < \alpha$.

### Code

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(0, 1, size=240),
                    rng.lognormal(0, 0.6, size=60)])

K2, p = stats.normaltest(x)
print(f"Sample size n = {x.size}")
print(f"D'Agostino's K^2 statistic = {K2:.4f}")
print(f"p-value = {p:.4g}")
if p < 0.05:
    print("=> Reject normality at alpha = 0.05.")
else:
    print("=> Fail to reject normality at alpha = 0.05.")
```

## Why an Omnibus Test

Consider two scenarios:

1. **Right-skewed data** with normal kurtosis: the skewness test rejects, but the kurtosis test may not. The $K^2$ test still rejects because $Z_1^2$ is large.
2. **Symmetric heavy-tailed data**: the skewness test does not reject ($Z_1 \approx 0$), but the kurtosis test does. Again, $K^2$ rejects.

By combining both moments, $K^2$ is sensitive to departures in either direction. However, if a dataset has a very specific pattern (e.g., slightly skewed *and* slightly leptokurtic), the combined evidence from both components may push $K^2$ past the critical value even when neither individual test would reject.

## Interpretation

For the mixture example (normal + lognormal), the data inherit both nonzero skewness and nonzero excess kurtosis from the lognormal component. The $K^2$ test is expected to reject strongly. In practice, if $K^2$ rejects, it is informative to examine $Z_1$ and $Z_2$ separately (via `stats.skewtest` and `stats.kurtosistest`) to understand *which* moment is driving the rejection.

**Sample-size requirement.** SciPy requires $n \geq 20$ for `stats.normaltest`, since both the skewness and kurtosis transformations need a minimum sample size.

## Exercises

**Exercise 1.** Generate $n = 500$ standard normal observations. Run the D'Agostino $K^2$ test and verify that $p > 0.05$.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=500)

    K2, p = stats.normaltest(x)
    print(f"K^2 = {K2:.4f}, p = {p:.4g}")
    ```

    Under the null, the rejection rate should be approximately 5%. For a single draw from $\mathcal{N}(0,1)$, we expect $p > 0.05$ in 95% of cases. $\square$

---

**Exercise 2.** Generate $n = 300$ observations from a $\text{Uniform}(0,1)$ distribution. Run the $K^2$ test. Which component ($Z_1$ or $Z_2$) is mainly responsible for the rejection?

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, size=300)

    K2, p = stats.normaltest(x)
    z1, p1 = stats.skewtest(x)
    z2, p2 = stats.kurtosistest(x)

    print(f"K^2 = {K2:.4f}, p = {p:.4g}")
    print(f"Skewness:  Z1 = {z1:.4f}, p = {p1:.4g}")
    print(f"Kurtosis:  Z2 = {z2:.4f}, p = {p2:.4g}")
    ```

    The uniform distribution is symmetric ($\gamma_1 = 0$) but platykurtic ($\gamma_2 = -1.2$). Therefore $Z_1$ will be small and non-significant, while $Z_2$ will be large and highly significant. The rejection of $K^2$ is driven entirely by the kurtosis component. $\square$

---

**Exercise 3.** Explain why $K^2$ follows a $\chi^2_2$ distribution under $H_0$ and state the conditions required for this approximation to hold.

??? success "Solution to Exercise 3"

    Under $H_0$, the transformations of D'Agostino map $g_1$ and $g_2$ to $Z_1$ and $Z_2$, each of which is approximately $\mathcal{N}(0,1)$. These transformations are designed so that $Z_1$ and $Z_2$ are approximately independent (one depends on odd central moments, the other on even ones). The sum of squares of two independent standard normals is by definition $\chi^2_2$. The approximation requires (1) $n$ large enough for the normal approximation to $Z_1$ and $Z_2$ (SciPy requires $n \geq 8$ for $Z_1$ and $n \geq 20$ for $Z_2$), and (2) the data are i.i.d. If the data exhibit serial dependence, the variance of $g_1$ and $g_2$ changes and the $\chi^2_2$ approximation breaks down. $\square$

---

**Exercise 4.** Run a Monte Carlo simulation with 10,000 replicates to verify that the empirical size of the $K^2$ test at $\alpha = 0.05$ is approximately 0.05 for $n = 50$.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n, reps, alpha = 50, 10000, 0.05
    rejections = 0

    for _ in range(reps):
        x = rng.normal(0, 1, size=n)
        _, p = stats.normaltest(x)
        if p < alpha:
            rejections += 1

    empirical_size = rejections / reps
    print(f"Empirical size: {empirical_size:.4f}")
    ```

    The result should be close to 0.05 (typically 0.045--0.055), confirming that the test is correctly sized for $n = 50$. The standard error of the estimate is $\sqrt{0.05 \times 0.95 / 10000} \approx 0.0022$. $\square$

---

**Exercise 5.** Prove that if $Z_1$ and $Z_2$ are independent $\mathcal{N}(0,1)$ random variables, then $K^2 = Z_1^2 + Z_2^2$ has CDF $F_{K^2}(k) = 1 - e^{-k/2}$ for $k \geq 0$.

??? success "Solution to Exercise 5"

    If $Z_1, Z_2 \overset{\text{iid}}{\sim} \mathcal{N}(0,1)$, then by definition $Z_1^2 \sim \chi^2_1$ and $Z_2^2 \sim \chi^2_1$. Since they are independent, $K^2 = Z_1^2 + Z_2^2 \sim \chi^2_2$. The $\chi^2_2$ distribution has density

    $$
    f_{K^2}(k) = \frac{1}{2} e^{-k/2}, \qquad k \geq 0,
    $$

    which is the $\text{Exponential}(1/2)$ density. Integrating:

    $$
    F_{K^2}(k) = \int_0^k \frac{1}{2} e^{-t/2}\, dt = \bigl[-e^{-t/2}\bigr]_0^k = 1 - e^{-k/2}.
    $$

    This confirms the claim. A consequence is that the $p$-value of the $K^2$ test can be computed as $p = e^{-K^2_{\text{obs}}/2}$, providing a simple closed-form expression. $\square$
