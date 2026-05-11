# Anderson-Darling Test

## Overview

The Anderson-Darling (AD) test is an EDF-based normality test that gives extra weight to the tails of the distribution compared to the Kolmogorov-Smirnov test. This tail sensitivity makes it particularly effective at detecting departures from normality that manifest as heavy tails, skewness in the extremes, or outlier-prone distributions. SciPy reports the $A^2$ statistic and tabulated critical values rather than an exact $p$-value.

## The Test Statistic

Given order statistics $X_{(1)} \leq \cdots \leq X_{(n)}$ and a hypothesised CDF $F_0$, the Anderson-Darling statistic is

$$
A^2 = -n - \sum_{i=1}^{n} \frac{2i - 1}{n} \Bigl[\ln F_0(X_{(i)}) + \ln\bigl(1 - F_0(X_{(n+1-i)})\bigr)\Bigr].
$$

Equivalently, writing $U_i = F_0(X_{(i)})$ (the probability-integral-transformed values):

$$
A^2 = -n - \frac{1}{n}\sum_{i=1}^{n} (2i - 1)\bigl[\ln U_i + \ln(1 - U_{n+1-i})\bigr].
$$

The weighting $(2i-1)/n$ places greater emphasis on observations in the tails (where $U_i$ is near 0 or 1), since $\ln U_i$ and $\ln(1 - U_i)$ diverge as $U_i \to 0$ or $U_i \to 1$.

## Why Tail Sensitivity Matters

The KS statistic $D_n = \sup_x |F_n(x) - F_0(x)|$ treats all parts of the distribution equally. In contrast, $A^2$ is an integral of the squared difference $(F_n - F_0)^2$ weighted by $[F_0(1 - F_0)]^{-1}$:

$$
A^2 = n \int_{-\infty}^{\infty} \frac{[F_n(x) - F_0(x)]^2}{F_0(x)\,[1 - F_0(x)]}\, dF_0(x).
$$

The denominator $F_0(1-F_0)$ is smallest in the tails, amplifying any discrepancies there.

## Hypotheses and Critical Values

$$
H_0: \text{the data follow a normal distribution}, \qquad H_1: \text{the data do not follow a normal distribution}.
$$

SciPy's `stats.anderson` returns the statistic $A^2$ along with critical values at significance levels 15%, 10%, 5%, 2.5%, and 1%. Reject $H_0$ at level $\alpha$ if $A^2$ exceeds the corresponding critical value.

### Code

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(0, 1, size=230),
                    rng.lognormal(0, 0.6, size=70)])

res = stats.anderson(x, dist="norm")
print(f"Sample size n = {x.size}")
print(f"Anderson-Darling A^2 = {res.statistic:.4f}")
print("Critical values vs significance levels:")
for cv, sl in zip(res.critical_values, res.significance_level):
    print(f"  {sl:.1f}% -> {cv:.4f} (reject if A^2 > {cv:.4f})")

reject_5 = res.statistic > res.critical_values[
    list(res.significance_level).index(5.0)]
print(f"Decision at 5%: {'Reject' if reject_5 else 'Fail to reject'} normality")
```

## Interpretation

For the mixture data (normal + lognormal), the $A^2$ statistic will be large and exceed all tabulated critical values, giving strong evidence against normality. The AD test is especially well suited here because the lognormal component inflates the right tail.

When the AD test rejects but the KS test does not, it is often because the departure occurs in the tails -- precisely the region the AD test is designed to detect.

## Exercises

**Exercise 1.** Generate $n = 300$ standard normal observations. Run the Anderson-Darling test and verify that $A^2$ is below the 15% critical value.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=300)

    res = stats.anderson(x, dist="norm")
    print(f"A^2 = {res.statistic:.4f}")
    print(f"15% critical value: {res.critical_values[0]:.4f}")
    print("Below 15% critical value:", res.statistic < res.critical_values[0])
    ```

    Under normality, $A^2$ should be small and below all critical values in most realisations. $\square$

---

**Exercise 2.** Generate $n = 300$ observations from a $t_4$ distribution. Run the AD test and the KS test (against $\mathcal{N}(0,1)$). Which test rejects more strongly?

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = rng.standard_t(df=4, size=300)

    # Anderson-Darling
    ad = stats.anderson(x, dist="norm")
    print(f"A^2 = {ad.statistic:.4f}")
    for cv, sl in zip(ad.critical_values, ad.significance_level):
        flag = " *" if ad.statistic > cv else ""
        print(f"  {sl:.1f}%: {cv:.4f}{flag}")

    # KS test
    D, p_ks = stats.kstest(x, 'norm', args=(0, 1))
    print(f"\nKS: D = {D:.4f}, p = {p_ks:.4g}")
    ```

    The $t_4$ distribution has heavier tails than the normal. The AD test should reject strongly (at all significance levels) because it up-weights tail discrepancies. The KS test will also reject because $\text{Var}(t_4) = 2 \neq 1$, but the AD test provides more decisive evidence in this tail-driven scenario. $\square$

---

**Exercise 3.** Show that the AD statistic can be written as a weighted integral of the squared EDF difference. Start from the definition and derive the integral form.

??? success "Solution to Exercise 3"

    Define the weighted Cramer-von Mises functional:

    $$
    A^2 = n \int_{-\infty}^{\infty} \frac{[F_n(x) - F_0(x)]^2}{F_0(x)(1 - F_0(x))}\, dF_0(x).
    $$

    Substituting $u = F_0(x)$ (so $du = f_0(x)\,dx = dF_0(x)$):

    $$
    A^2 = n \int_0^1 \frac{[F_n(F_0^{-1}(u)) - u]^2}{u(1-u)}\, du.
    $$

    The weight $1/[u(1-u)]$ diverges as $u \to 0$ or $u \to 1$, assigning infinite weight to deviations at the extreme tails. In practice, the integral is evaluated using the order statistics, yielding the discrete sum form $A^2 = -n - \frac{1}{n}\sum (2i-1)[\ln U_i + \ln(1 - U_{n+1-i})]$, which can be verified by expanding the integral using the step-function form of $F_n$. $\square$

---

**Exercise 4.** The AD test in SciPy can also test for exponential and logistic distributions. Write code that tests the same dataset against all three distributions and compares the results.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = rng.exponential(2.0, size=200)

    for dist in ["norm", "expon", "logistic"]:
        res = stats.anderson(x, dist=dist)
        print(f"\n{dist}: A^2 = {res.statistic:.4f}")
        for cv, sl in zip(res.critical_values, res.significance_level):
            flag = " REJECT" if res.statistic > cv else ""
            print(f"  {sl}%: {cv:.4f}{flag}")
    ```

    For exponential data, the test against `"expon"` should not reject (correct null), while the test against `"norm"` and `"logistic"` should reject (wrong distributions). This demonstrates the AD test's versatility across distribution families. $\square$

---

**Exercise 5.** Explain why the Anderson-Darling test does not provide an exact $p$-value in SciPy and describe how you could obtain one via simulation.

??? success "Solution to Exercise 5"

    The null distribution of $A^2$ is non-standard: it depends on the hypothesised distribution family and whether parameters are estimated. Closed-form expressions exist only for specific cases, and SciPy implements the test using tabulated critical values rather than computing an exact $p$-value. To obtain a simulation-based $p$-value:

    1. Compute $A^2_{\text{obs}}$ from the data.
    2. For $b = 1, \ldots, B$, simulate $x^{*(b)} \sim \mathcal{N}(\hat{\mu}, \hat{\sigma}^2)$ of the same size, compute $A^{2*}_{(b)}$.
    3. The $p$-value is $\hat{p} = \frac{1}{B}\sum_b \mathbf{1}(A^{2*}_{(b)} \geq A^2_{\text{obs}})$.

    This is analogous to the Lilliefors bootstrap and correctly accounts for parameter estimation. With $B = 5000$, the resulting $p$-value has standard error $\approx \sqrt{0.05 \times 0.95/5000} \approx 0.003$ near the 5% boundary. $\square$
