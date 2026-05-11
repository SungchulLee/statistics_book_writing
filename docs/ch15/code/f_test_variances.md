# F-Test of Equality of Variances

## Overview

The F-test of equality of variances is a two-sample test that compares the variances of two normally distributed populations. The test statistic is the ratio of the two sample variances and follows an F-distribution under the null hypothesis. While elegant and optimal under exact normality, the F-test is notoriously sensitive to departures from normality, making robust alternatives such as Levene's test generally preferable in practice.

## Test Setup

Let $X_1, \ldots, X_{n_1} \overset{\text{iid}}{\sim} N(\mu_1, \sigma_1^2)$ and $Y_1, \ldots, Y_{n_2} \overset{\text{iid}}{\sim} N(\mu_2, \sigma_2^2)$ be independent samples. The hypotheses are

$$
H_0 : \sigma_1^2 = \sigma_2^2 \quad \text{versus} \quad H_1 : \sigma_1^2 \neq \sigma_2^2.
$$

## Test Statistic

The F-statistic is defined as the ratio of the two sample variances:

$$
F = \frac{S_1^2}{S_2^2},
$$

where $S_i^2 = \frac{1}{n_i - 1}\sum_{j=1}^{n_i}(X_{ij} - \bar{X}_i)^2$. Under $H_0$,

$$
F \sim F(n_1 - 1,\; n_2 - 1).
$$

## Decision Rule

For a two-sided test at level $\alpha$, the $p$-value is

$$
p = 2\min\!\bigl(F_{F\text{-dist}}(F_{\text{obs}}),\; 1 - F_{F\text{-dist}}(F_{\text{obs}})\bigr),
$$

where $F_{F\text{-dist}}$ is the CDF of $F(n_1-1, n_2-1)$. Reject $H_0$ when $p < \alpha$.

## Code

```python
import numpy as np
import scipy.stats as stats


def f_test(data_0, data_1):
    """
    Two-sample F-test for equality of variances.

    H0: sigma_1^2 = sigma_2^2
    H1: sigma_1^2 != sigma_2^2
    """
    statistic = data_0.var(ddof=1) / data_1.var(ddof=1)
    df1 = data_0.shape[0] - 1
    df2 = data_1.shape[0] - 1
    p_value = 2 * min(
        stats.f(df1, df2).cdf(statistic),
        stats.f(df1, df2).sf(statistic),
    )
    return statistic, p_value
```

The following example generates two samples where the second has a progressively larger standard deviation, then applies the F-test:

```python
size, seed = 100, 1
x = stats.norm(loc=0, scale=1).rvs(size, random_state=seed)

for scale in [1.00, 1.05, 1.10, 1.15, 1.20]:
    y = stats.norm(loc=1, scale=scale).rvs(size, random_state=seed)
    stat, pval = f_test(x, y)
    print(f"sigma_y={scale:.2f}  F={stat:.2f}  p={pval:.3f}")
```

## Interpretation

- When both populations share the same variance ($\sigma_1 = \sigma_2$), the F-statistic is close to 1 and the $p$-value is large.
- As the variance ratio departs from 1, the F-statistic moves away from 1 and the $p$-value decreases.
- The F-test is extremely sensitive to non-normality; even moderate skewness or heavy tails can cause the actual Type I error rate to far exceed $\alpha$. Levene's or Brown--Forsythe tests are preferred when normality is doubtful.

## Exercises

**Exercise 1.** Two laboratories measure the concentration of a chemical compound. Lab A reports $n_1 = 15$ measurements with $S_1^2 = 3.2$, and Lab B reports $n_2 = 20$ measurements with $S_2^2 = 1.8$. Compute the F-statistic, identify the degrees of freedom, and obtain the two-sided $p$-value using Python.

??? success "Solution to Exercise 1"

    $$
    F = \frac{S_1^2}{S_2^2} = \frac{3.2}{1.8} \approx 1.778, \quad df_1 = 14,\; df_2 = 19.
    $$

    ```python
    import scipy.stats as stats

    F = 3.2 / 1.8
    df1, df2 = 14, 19
    p = 2 * min(stats.f(df1, df2).cdf(F), stats.f(df1, df2).sf(F))
    print(f"F = {F:.3f}, p = {p:.4f}")
    ```

    The $p$-value will be moderate (around 0.17), so we fail to reject $H_0$ at $\alpha = 0.05$.

---

**Exercise 2.** Show that if $F = S_1^2/S_2^2 \sim F(d_1, d_2)$ under $H_0$, then $1/F = S_2^2/S_1^2 \sim F(d_2, d_1)$. Explain why the two-sided $p$-value is the same regardless of which sample variance is placed in the numerator.

??? success "Solution to Exercise 2"

    Under $H_0$, $(n_1-1)S_1^2/\sigma^2 \sim \chi^2(d_1)$ and $(n_2-1)S_2^2/\sigma^2 \sim \chi^2(d_2)$ independently. By definition of the F-distribution,

    $$
    F = \frac{S_1^2}{S_2^2} = \frac{\chi^2(d_1)/d_1}{\chi^2(d_2)/d_2} \cdot \frac{d_1}{d_2} \cdot \frac{d_2}{d_1}.
    $$

    More directly, if $F \sim F(d_1, d_2)$ then $1/F \sim F(d_2, d_1)$ by the reciprocal property of the F-distribution. For the two-sided test, $p = 2\min(P(F' \le F_{\text{obs}}), P(F' \ge F_{\text{obs}}))$. Swapping numerator and denominator replaces $F_{\text{obs}}$ by $1/F_{\text{obs}}$ and swaps degrees of freedom, but the two-sided formula yields the same value because $P(F(d_1,d_2) \ge F_{\text{obs}}) = P(F(d_2,d_1) \le 1/F_{\text{obs}})$. $\square$

---

**Exercise 3.** Write a Monte Carlo simulation with 10,000 replications. In each replication draw $n_1 = n_2 = 20$ from $N(0,1)$ and apply the F-test at $\alpha = 0.05$. Verify the empirical Type I error is close to 0.05. Then repeat with $t(3)$ data and comment on the result.

??? success "Solution to Exercise 3"

    ```python
    import numpy as np
    import scipy.stats as stats

    rng = np.random.default_rng(0)
    n, alpha, n_sims = 20, 0.05, 10000

    for dist_name in ["Normal", "t(3)"]:
        rej = 0
        for _ in range(n_sims):
            if dist_name == "Normal":
                x = rng.normal(0, 1, n)
                y = rng.normal(0, 1, n)
            else:
                x = stats.t(df=3).rvs(n, random_state=rng)
                y = stats.t(df=3).rvs(n, random_state=rng)
            F = np.var(x, ddof=1) / np.var(y, ddof=1)
            p = 2 * min(stats.f(n-1, n-1).cdf(F), stats.f(n-1, n-1).sf(F))
            if p < alpha:
                rej += 1
        print(f"{dist_name}: rejection rate = {rej/n_sims:.4f}")
    ```

    Under normality the rate is close to 0.05. Under $t(3)$ the rate will be substantially inflated (often 0.12--0.20), demonstrating the F-test's sensitivity to heavy tails.

---

**Exercise 4.** Suppose you want a one-sided test $H_0: \sigma_1^2 \le \sigma_2^2$ versus $H_1: \sigma_1^2 > \sigma_2^2$. State the rejection rule and modify the Python code to return the one-sided $p$-value.

??? success "Solution to Exercise 4"

    For $H_1: \sigma_1^2 > \sigma_2^2$, large values of $F = S_1^2/S_2^2$ provide evidence against $H_0$. Reject when

    $$
    F > F_{1-\alpha}(n_1-1, n_2-1).
    $$

    The one-sided $p$-value is $p = P\bigl(F(n_1-1,n_2-1) \ge F_{\text{obs}}\bigr)$.

    ```python
    import numpy as np
    import scipy.stats as stats

    def f_test_one_sided(data_0, data_1):
        F = np.var(data_0, ddof=1) / np.var(data_1, ddof=1)
        df1 = len(data_0) - 1
        df2 = len(data_1) - 1
        p = stats.f(df1, df2).sf(F)
        return F, p
    ```

    $\square$

---

**Exercise 5.** Prove that $E[F] = \frac{d_2}{d_2 - 2}$ for $d_2 > 2$ when $F \sim F(d_1, d_2)$, and explain why $E[F]$ does not depend on $d_1$.

??? success "Solution to Exercise 5"

    Write $F = (U/d_1)/(V/d_2)$ where $U \sim \chi^2(d_1)$ and $V \sim \chi^2(d_2)$ are independent. Then

    $$
    E[F] = \frac{d_2}{d_1} \cdot E[U] \cdot E[1/V].
    $$

    Since $E[U] = d_1$ and for $V \sim \chi^2(d_2)$ with $d_2 > 2$ we have $E[1/V] = 1/(d_2 - 2)$ (inverse-chi-squared moment), it follows that

    $$
    E[F] = \frac{d_2}{d_1} \cdot d_1 \cdot \frac{1}{d_2 - 2} = \frac{d_2}{d_2 - 2}.
    $$

    The $d_1$ terms cancel because $E[U/d_1] = 1$ for any $d_1$, so the numerator's degrees of freedom only affect the variance of $F$, not its mean. $\square$
