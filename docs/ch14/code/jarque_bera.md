# Jarque-Bera Test

## Overview

The Jarque-Bera (JB) test is a widely used normality test in econometrics and finance that, like the D'Agostino $K^2$ test, jointly assesses skewness and kurtosis. It uses a simpler formula based directly on the sample skewness and excess kurtosis, and its null distribution is $\chi^2_2$ asymptotically. The test is best suited for large samples and is the default normality diagnostic in many econometric software packages.

## The Test Statistic

Given a sample $X_1, \ldots, X_n$ with sample skewness $g_1$ and sample excess kurtosis $g_2$, the Jarque-Bera statistic is

$$
\text{JB} = \frac{n}{6}\!\left(g_1^2 + \frac{g_2^2}{4}\right).
$$

The rationale is straightforward: under normality $\mathbb{E}[g_1] = 0$ and $\mathbb{E}[g_2] = 0$, with approximate variances $\text{Var}(g_1) \approx 6/n$ and $\text{Var}(g_2) \approx 24/n$. Standardising and summing squares gives

$$
\text{JB} \approx \left(\frac{g_1}{\sqrt{6/n}}\right)^2 + \left(\frac{g_2}{\sqrt{24/n}}\right)^2 \;\underset{H_0}{\sim}\; \chi^2_2.
$$

## Hypotheses

$$
H_0: \gamma_1 = 0 \text{ and } \gamma_2 = 0, \qquad H_1: \gamma_1 \neq 0 \text{ or } \gamma_2 \neq 0.
$$

The $p$-value is $p = P(\chi^2_2 \geq \text{JB}_{\text{obs}})$. We reject $H_0$ when $p < \alpha$.

### Code

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(0, 1, size=240),
                    rng.lognormal(0, 0.6, size=60)])

jb_stat, p = stats.jarque_bera(x)
g1 = stats.skew(x, bias=False)
g2 = stats.kurtosis(x, fisher=True, bias=False)

print(f"Sample size n = {x.size}")
print(f"Skewness g1 = {g1:.4f}")
print(f"Excess kurtosis g2 = {g2:.4f}")
print(f"Jarque-Bera: JB = {jb_stat:.4f}, p-value = {p:.4g}")
if p < 0.05:
    print("=> Reject normality at alpha = 0.05.")
else:
    print("=> Fail to reject normality at alpha = 0.05.")
```

## Comparison with D'Agostino K-Squared

Both the JB and D'Agostino $K^2$ tests combine skewness and kurtosis, but they differ in construction:

| Feature | Jarque-Bera | D'Agostino $K^2$ |
|---|---|---|
| Formula | Direct quadratic in $g_1, g_2$ | Nonlinear transformations to $Z_1, Z_2$ |
| Null distribution | $\chi^2_2$ (asymptotic) | $\chi^2_2$ (better finite-sample fit) |
| Small-sample accuracy | Poor for $n < 100$ | Better for $n \geq 20$ |
| Typical use | Econometrics, finance | General statistics |

For large $n$ the two tests give similar results. For moderate $n$, the D'Agostino $K^2$ test tends to have more accurate $p$-values.

## Interpretation

In the mixture example, the lognormal component introduces both positive skewness and positive excess kurtosis. The JB statistic will be large and the $p$-value essentially zero. The relative contributions of skewness and kurtosis can be assessed by examining $g_1^2$ versus $g_2^2/4$ in the JB formula.

## Exercises

**Exercise 1.** Generate $n = 500$ standard normal observations. Compute the JB statistic and $p$-value. Verify that the test does not reject at $\alpha = 0.05$.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, size=500)

    jb, p = stats.jarque_bera(x)
    print(f"JB = {jb:.4f}, p = {p:.4g}")
    ```

    Under the null, $p > 0.05$ in about 95% of realisations. The JB statistic should be small (typically $< 6$, the 95th percentile of $\chi^2_2$). $\square$

---

**Exercise 2.** Verify the JB formula by computing $\frac{n}{6}(g_1^2 + g_2^2/4)$ manually and comparing it to the output of `stats.jarque_bera`.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 240), rng.lognormal(0, 0.6, 60)])

    g1 = stats.skew(x, bias=True)
    g2 = stats.kurtosis(x, fisher=True, bias=True)
    n = x.size
    jb_manual = (n / 6) * (g1**2 + g2**2 / 4)

    jb_scipy, _ = stats.jarque_bera(x)

    print(f"Manual JB:  {jb_manual:.4f}")
    print(f"SciPy JB:   {jb_scipy:.4f}")
    ```

    Note that `stats.jarque_bera` uses biased estimates of skewness and kurtosis (i.e., `bias=True`), so the manual computation must do the same to match. The values should agree to numerical precision. $\square$

---

**Exercise 3.** Explain why the Jarque-Bera test has poor size control for small samples (e.g., $n = 30$).

??? success "Solution to Exercise 3"

    The JB statistic is derived from the asymptotic variances $\text{Var}(g_1) \approx 6/n$ and $\text{Var}(g_2) \approx 24/n$. For small $n$, these approximations are inaccurate: the true finite-sample variances differ, and the distributions of $g_1$ and $g_2$ are not well approximated by normals. Additionally, $g_1$ and $g_2$ may not be independent for small $n$. As a result, the $\chi^2_2$ reference distribution is a poor fit: the actual null distribution of JB has a different shape, typically shifted to the right. This means the nominal $\alpha = 0.05$ critical value ($\approx 5.99$) is too large, and the test *under-rejects* (empirical size $< 0.05$), leading to low power. For small samples, the D'Agostino $K^2$ test or the Shapiro-Wilk test are preferred. $\square$

---

**Exercise 4.** Show that the weight $1/4$ on $g_2^2$ in the JB formula arises from the ratio of asymptotic variances: $\text{Var}(g_2)/\text{Var}(g_1) \to 4$ as $n \to \infty$.

??? success "Solution to Exercise 4"

    From the asymptotic variances, $\text{Var}(g_1) \to 6/n$ and $\text{Var}(g_2) \to 24/n$. Their ratio is

    $$
    \frac{\text{Var}(g_2)}{\text{Var}(g_1)} = \frac{24/n}{6/n} = 4.
    $$

    In the JB formula, we standardise each moment: $(g_1/\sqrt{6/n})^2 + (g_2/\sqrt{24/n})^2 = \frac{n}{6} g_1^2 + \frac{n}{24} g_2^2 = \frac{n}{6}(g_1^2 + g_2^2/4)$. The factor $1/4$ on $g_2^2$ thus compensates for the fact that $g_2$ has four times the variance of $g_1$, ensuring both standardised components have unit variance and their sum follows $\chi^2_2$. $\square$

---

**Exercise 5.** Run a Monte Carlo study with 10,000 replicates to estimate the empirical size of the JB test at $\alpha = 0.05$ for $n \in \{30, 100, 500, 2000\}$. Plot the results and identify the sample size at which the empirical size stabilises near 0.05.

??? success "Solution to Exercise 5"

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    reps, alpha = 10000, 0.05
    ns = [30, 100, 500, 2000]
    sizes = []

    for n in ns:
        rej = sum(1 for _ in range(reps)
                  if stats.jarque_bera(rng.normal(0, 1, n))[1] < alpha)
        sizes.append(rej / reps)
        print(f"n = {n:>4}: empirical size = {rej/reps:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, sizes, marker="o")
    ax.axhline(0.05, color="red", linestyle="--", label="Nominal 0.05")
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("Empirical rejection rate")
    ax.set_title("JB Test: Empirical Size vs n")
    ax.legend()
    plt.tight_layout()
    plt.show()
    ```

    For $n = 30$, the empirical size is typically well below 0.05 (around 0.02--0.03), confirming the poor small-sample calibration. By $n = 500$ the empirical size should stabilise near 0.05, demonstrating that the $\chi^2_2$ approximation becomes adequate for moderate to large samples. $\square$
