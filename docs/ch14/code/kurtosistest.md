# Kurtosis Test

## Overview

The D'Agostino kurtosis test evaluates whether the excess kurtosis of a dataset differs significantly from zero, the value expected under normality. Excess kurtosis measures the heaviness of tails relative to the normal distribution. This test transforms the sample excess kurtosis into a $Z$-statistic that is approximately standard normal under the null hypothesis, providing a targeted check for tail behaviour.

## Sample Excess Kurtosis

The bias-corrected Fisher excess kurtosis for a sample $X_1, \ldots, X_n$ is

$$
g_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum_{i=1}^{n} \left(\frac{X_i - \bar{X}}{S}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}.
$$

Under normality, $\mathbb{E}[g_2] = 0$. Distributions with $g_2 > 0$ are called *leptokurtic* (heavier tails than normal), and those with $g_2 < 0$ are called *platykurtic* (lighter tails).

## The D'Agostino Kurtosis Transformation

Similar to the skewness test, D'Agostino and Pearson transform $g_2$ through a nonlinear mapping to produce a statistic $Z_2$ that is approximately $\mathcal{N}(0,1)$ under $H_0$. The variance of $g_2$ under normality is approximately

$$
\text{Var}(g_2) \approx \frac{24n(n-2)(n-3)}{(n+1)^2(n+3)(n+5)}.
$$

After standardisation and further corrections, the resulting $Z_2$ is used for inference.

## Hypotheses

$$
H_0: \gamma_2 = 0 \quad (\text{population excess kurtosis is zero}), \qquad H_1: \gamma_2 \neq 0.
$$

The two-sided $p$-value is $p = 2\,\mathcal{N}(-|Z_2|)$.

### Code

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(0, 1, size=220),
                    rng.standard_t(df=4, size=80)])

g2 = stats.kurtosis(x, fisher=True, bias=False)
z, p = stats.kurtosistest(x)

print(f"Sample size n = {x.size}")
print(f"Sample excess kurtosis (Fisher) g2 = {g2:.4f}")
print(f"D'Agostino kurtosis test: Z = {z:.4f}, p-value = {p:.4g}")
if p < 0.05:
    print("=> Evidence of non-normal kurtosis (departing from normality).")
else:
    print("=> No strong evidence of non-normal kurtosis.")
```

## Interpretation

The example above uses a mixture of normal and $t_4$ draws, producing heavier tails than a pure normal sample. The excess kurtosis $g_2$ should be notably positive, and the test should reject at $\alpha = 0.05$. The kurtosis test is especially useful for detecting heavy-tailed contamination, outlier-prone distributions, and light-tailed distributions such as the uniform.

**Minimum sample size.** SciPy requires $n \geq 20$ for `kurtosistest`. For smaller samples the normal approximation for $Z_2$ is unreliable.

## Exercises

**Exercise 1.** Generate $n = 300$ standard normal observations. Compute $g_2$ and the kurtosis test $p$-value. Verify that $g_2$ is close to zero and the test does not reject.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(10)
    x = rng.normal(0, 1, size=300)

    g2 = stats.kurtosis(x, fisher=True, bias=False)
    z, p = stats.kurtosistest(x)

    print(f"g2 = {g2:.4f}")
    print(f"Z = {z:.4f}, p = {p:.4g}")
    ```

    For truly normal data, $g_2$ should be near zero (typically $|g_2| < 0.5$) and $p > 0.05$. The test correctly retains the null hypothesis. $\square$

---

**Exercise 2.** Generate $n = 300$ observations from a $\text{Uniform}(0,1)$ distribution. Compute $g_2$ and run the kurtosis test. The uniform is symmetric but platykurtic ($\gamma_2 = -1.2$). Does the test detect this?

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=300)

    g2 = stats.kurtosis(x, fisher=True, bias=False)
    z, p = stats.kurtosistest(x)

    print(f"g2 = {g2:.4f}, p = {p:.4g}")
    ```

    The sample $g_2$ should be near $-1.2$ and the $p$-value should be very small, rejecting the null of zero excess kurtosis. This demonstrates that the kurtosis test can detect non-normal distributions even when they are symmetric, complementing the skewness test which would fail to detect the uniform's non-normality. $\square$

---

**Exercise 3.** Derive $\text{Var}(g_2) \to 24/n$ as $n \to \infty$ from the exact formula.

??? success "Solution to Exercise 3"

    Starting from

    $$
    \text{Var}(g_2) = \frac{24n(n-2)(n-3)}{(n+1)^2(n+3)(n+5)},
    $$

    factor out $n^5$ from the denominator and $n^3$ from the numerator:

    $$
    \text{Var}(g_2) = \frac{24\, n^3\bigl(1 - \frac{2}{n}\bigr)\bigl(1 - \frac{3}{n}\bigr)}{n^5\bigl(1 + \frac{1}{n}\bigr)^2\bigl(1 + \frac{3}{n}\bigr)\bigl(1 + \frac{5}{n}\bigr)} = \frac{24}{n} \cdot \frac{(1 - 2/n)(1 - 3/n)}{(1 + 1/n)^2(1 + 3/n)(1 + 5/n)}.
    $$

    As $n \to \infty$, every correction factor tends to 1, giving $\text{Var}(g_2) \to 24/n$. Thus the standard error of $g_2$ is approximately $\sqrt{24/n}$. $\square$

---

**Exercise 4.** Explain why a dataset with many outliers would produce $g_2 > 0$ even if the underlying distribution is normal, and discuss the implications for the kurtosis test.

??? success "Solution to Exercise 4"

    The fourth power in the kurtosis formula $(X_i - \bar{X})^4$ gives extreme observations disproportionate influence. A few outliers with large $|X_i - \bar{X}|$ will inflate the numerator of $g_2$ far more than they inflate $S^4$ in the denominator, pushing $g_2$ above zero. If the outliers are genuine (i.e., the true distribution has heavy tails), then $g_2 > 0$ correctly detects leptokurtosis. However, if the outliers are data-entry errors or measurement artefacts, then the kurtosis test may reject normality for spurious reasons. Before interpreting a significant kurtosis test, it is important to inspect the data for anomalous observations and assess whether they are genuine. $\square$

---

**Exercise 5.** Using a Monte Carlo simulation with 5,000 replicates, estimate the power of the kurtosis test at $\alpha = 0.05$ for $n = 100$ observations from a $t_5$ distribution. Compare this with the power against $t_{10}$.

??? success "Solution to Exercise 5"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n, reps, alpha = 100, 5000, 0.05

    for df in [5, 10]:
        rejections = 0
        for _ in range(reps):
            x = rng.standard_t(df=df, size=n)
            _, p = stats.kurtosistest(x)
            if p < alpha:
                rejections += 1
        print(f"t({df}): power = {rejections / reps:.4f}")
    ```

    The $t_5$ distribution has excess kurtosis $\gamma_2 = 6/(5-4) = 6$, while $t_{10}$ has $\gamma_2 = 6/(10-4) = 1$. Since $t_5$ is further from normal, the power against $t_5$ should be substantially higher (often $> 0.9$) than against $t_{10}$ (typically $0.3$--$0.6$). This illustrates that the kurtosis test's power depends on the magnitude of the excess kurtosis. $\square$
