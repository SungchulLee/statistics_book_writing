# Formal Normality Test Suite

## Overview

Formal normality tests provide objective, quantitative evidence for or against the hypothesis that data come from a normal distribution. Unlike graphical checks, they yield a test statistic and $p$-value, enabling a principled decision at a chosen significance level. This page surveys the most widely used tests, their null hypotheses, strengths, and sample-size considerations.

## The Hypothesis-Testing Framework

All normality tests share a common structure. The null and alternative hypotheses are

$$
H_0: X_1, \ldots, X_n \sim \mathcal{N}(\mu, \sigma^2) \quad \text{for some } \mu, \sigma^2, \qquad H_1: \text{the data are not normally distributed}.
$$

A test statistic $T$ is computed from the sample. Under $H_0$, $T$ has a known (or tabulated) distribution. The $p$-value is

$$
p = P(T \geq T_{\text{obs}} \mid H_0),
$$

where the direction of the inequality depends on the specific test. We reject $H_0$ when $p < \alpha$.

## Overview of Common Tests

| Test | Sensitive to | Sample-size guidance | SciPy function |
|---|---|---|---|
| Shapiro-Wilk | General departures | Best for $n \leq 5000$ | `stats.shapiro` |
| D'Agostino $K^2$ | Skewness and kurtosis | $n \geq 20$ | `stats.normaltest` |
| Jarque-Bera | Skewness and kurtosis | Large $n$ (asymptotic) | `stats.jarque_bera` |
| Kolmogorov-Smirnov | Distribution shape | Any $n$; needs known params | `stats.kstest` |
| Anderson-Darling | Tails | Tabulated critical values | `stats.anderson` |
| Lilliefors | Shape (estimated params) | Corrects KS when params estimated | Bootstrap or `lilliefors` |

### Code

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
data = rng.normal(0, 1, size=100)

# Shapiro-Wilk
W, p_sw = stats.shapiro(data)
print(f"Shapiro-Wilk:     W = {W:.4f}, p = {p_sw:.4g}")

# D'Agostino K^2
K2, p_k2 = stats.normaltest(data)
print(f"D'Agostino K^2:   K2 = {K2:.4f}, p = {p_k2:.4g}")

# Jarque-Bera
JB, p_jb = stats.jarque_bera(data)
print(f"Jarque-Bera:      JB = {JB:.4f}, p = {p_jb:.4g}")

# Kolmogorov-Smirnov (fully specified N(0,1))
D, p_ks = stats.kstest(data, 'norm', args=(0, 1))
print(f"KS (vs N(0,1)):   D = {D:.4f}, p = {p_ks:.4g}")

# Anderson-Darling
ad = stats.anderson(data, dist="norm")
print(f"Anderson-Darling: A^2 = {ad.statistic:.4f}")
for cv, sl in zip(ad.critical_values, ad.significance_level):
    print(f"  {sl:.0f}% critical value: {cv:.4f}")
```

## Choosing a Test

No single test is uniformly best. General guidance:

- **Small samples ($n < 50$):** Shapiro-Wilk has the best power against a wide range of alternatives.
- **Moderate samples ($50 \leq n \leq 5000$):** Shapiro-Wilk or Anderson-Darling are preferred. D'Agostino $K^2$ is a good omnibus choice when you suspect skewness or kurtosis problems.
- **Large samples ($n > 5000$):** Nearly any test will reject for tiny departures. Complement tests with effect-size measures (sample skewness, excess kurtosis) and graphical checks.

## Interpretation

A significant result (small $p$-value) means the data are unlikely to have come from a normal distribution, but it does not indicate *how* they depart. Always pair formal tests with graphical diagnostics. Conversely, a non-significant result does not prove normality; the test may simply lack power against the true alternative.

## Exercises

**Exercise 1.** Generate $n = 200$ standard normal observations and run the Shapiro-Wilk, D'Agostino $K^2$, and Jarque-Bera tests. Report the $p$-values. Do any tests reject at $\alpha = 0.05$?

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, size=200)

    W, p_sw = stats.shapiro(data)
    K2, p_k2 = stats.normaltest(data)
    JB, p_jb = stats.jarque_bera(data)

    print(f"Shapiro-Wilk:   p = {p_sw:.4g}")
    print(f"D'Agostino K^2: p = {p_k2:.4g}")
    print(f"Jarque-Bera:    p = {p_jb:.4g}")
    ```

    Since the data truly come from a normal distribution, all three $p$-values should be well above 0.05 (typically $> 0.3$). By definition, each test has only a 5% chance of a false rejection under $H_0$. $\square$

---

**Exercise 2.** Repeat Exercise 1 but draw from a $\text{Lognormal}(0, 0.5)$ distribution. Compare the $p$-values and explain which test is most sensitive to right skewness.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(42)
    data = rng.lognormal(0, 0.5, size=200)

    W, p_sw = stats.shapiro(data)
    K2, p_k2 = stats.normaltest(data)
    JB, p_jb = stats.jarque_bera(data)

    print(f"Shapiro-Wilk:   p = {p_sw:.4g}")
    print(f"D'Agostino K^2: p = {p_k2:.4g}")
    print(f"Jarque-Bera:    p = {p_jb:.4g}")
    ```

    All three tests should reject convincingly ($p \ll 0.05$). The Shapiro-Wilk test typically gives the smallest $p$-value for moderate-sized skewed samples because it compares the full ordered sample to the expected normal order statistics, making it sensitive to any shape departure. D'Agostino $K^2$ and Jarque-Bera are also powerful here because the lognormal has substantial skewness and excess kurtosis. $\square$

---

**Exercise 3.** Explain why the Kolmogorov-Smirnov test requires the null-hypothesis parameters to be fully specified. What happens to the $p$-value if you estimate $\mu$ and $\sigma$ from the data and plug them in?

??? success "Solution to Exercise 3"

    The KS test compares the empirical CDF $F_n(x)$ to a completely specified theoretical CDF $F_0(x)$. Its critical values and $p$-values are derived under the assumption that $F_0$ is fixed *before* seeing the data. If $\mu$ and $\sigma$ are estimated from the same data, the fitted CDF $\hat{F}(x)$ is by construction closer to $F_n$ than a generic $F_0$ would be. This makes the KS distance $D_n$ systematically smaller, inflating the $p$-value and reducing the test's power. The corrected procedure is the Lilliefors test, which uses simulation or special tables to account for parameter estimation. $\square$

---

**Exercise 4.** A colleague argues that if the Shapiro-Wilk test fails to reject, the data are "proven normal." Write a brief rebuttal using the concepts of Type II error and statistical power.

??? success "Solution to Exercise 4"

    Failure to reject $H_0$ is not proof of $H_0$. A non-significant $p$-value means the data are *compatible* with normality, but they may also be compatible with many non-normal distributions that the test lacks the power to distinguish. The probability of a Type II error $\beta$ depends on the sample size $n$, the significance level $\alpha$, and the true alternative distribution. For small $n$, power ($1 - \beta$) can be quite low, so a non-rejection carries little information. Proper reasoning requires either a power analysis or supplementary evidence (graphical checks, domain knowledge). $\square$

---

**Exercise 5.** Design a Monte Carlo experiment to estimate the empirical size of the Shapiro-Wilk test at $\alpha = 0.05$. Draw 10,000 samples of size $n = 50$ from $\mathcal{N}(0,1)$, apply the test, and report the rejection rate. How close is it to 0.05?

??? success "Solution to Exercise 5"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n, reps, alpha = 50, 10000, 0.05
    rejections = 0

    for _ in range(reps):
        x = rng.normal(0, 1, size=n)
        _, p = stats.shapiro(x)
        if p < alpha:
            rejections += 1

    empirical_size = rejections / reps
    print(f"Empirical size: {empirical_size:.4f}")
    ```

    The empirical rejection rate should be close to 0.05 (typically between 0.045 and 0.055). This confirms that the Shapiro-Wilk test is correctly sized: under $H_0$ it rejects approximately $\alpha \times 100\%$ of the time. Deviations from 0.05 are due to Monte Carlo sampling error, which scales as $\sqrt{\alpha(1-\alpha)/\text{reps}} \approx 0.002$. $\square$
