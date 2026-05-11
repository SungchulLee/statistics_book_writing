# Quantile-Quantile Plot with Normality Tests

## Overview

The quantile-quantile (Q-Q) plot is the most widely used graphical tool for assessing normality. By plotting ordered sample values against theoretical normal quantiles, it reveals departures that histograms can miss. This page pairs the Q-Q plot with three formal normality tests -- Shapiro-Wilk, D'Agostino $K^2$, and Anderson-Darling -- to show how visual and numerical diagnostics complement each other.

## Construction of the Q-Q Plot

Given an ordered sample $X_{(1)} \leq X_{(2)} \leq \cdots \leq X_{(n)}$, the Q-Q plot pairs each order statistic with the corresponding theoretical quantile

$$
q_i = \mathcal{N}^{-1}\!\Bigl(\frac{i - 0.5}{n}\Bigr), \qquad i = 1, \ldots, n,
$$

where $\mathcal{N}^{-1}$ is the standard normal quantile function. The plot displays the points $(q_i, X_{(i)})$.

Under normality, $X_{(i)} \approx \mu + \sigma\, q_i$, so the points should fall along the line

$$
y = \hat{\mu} + \hat{\sigma}\, x,
$$

where $\hat{\mu} = \bar{X}$ and $\hat{\sigma} = S$. A fitted line is typically obtained by least-squares regression of $X_{(i)}$ on $q_i$.

## Companion Normality Tests

| Test | Statistic | Null distribution |
|---|---|---|
| Shapiro-Wilk | $W = \frac{(\sum a_i X_{(i)})^2}{\sum (X_i - \bar{X})^2}$ | Tabulated / simulated |
| D'Agostino $K^2$ | $K^2 = Z_1^2 + Z_2^2$ (skewness + kurtosis) | $\chi^2_2$ (asymptotic) |
| Anderson-Darling | $A^2 = -n - \sum \frac{2i-1}{n}[\ln F_0(X_{(i)}) + \ln(1-F_0(X_{(n+1-i)}))]$ | Tabulated critical values |

### Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(0, 1, size=150),
                    rng.standard_t(df=3, size=50)])

# Q-Q plot with fitted line
osm, osr = stats.probplot(x, dist="norm", sparams=(), fit=False)
b, a = np.polyfit(osm, osr, 1)

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(osm, osr, s=15)
xx = np.linspace(osm.min(), osm.max(), 200)
ax.plot(xx, a + b * xx, linestyle="--")
ax.set_title("Q-Q Plot vs Normal with Fitted Line")
ax.set_xlabel("Theoretical quantiles (Normal)")
ax.set_ylabel("Ordered data")
plt.tight_layout()
plt.show()

# Normality tests
W, p_sw = stats.shapiro(x)
K2, p_k2 = stats.normaltest(x)
ad = stats.anderson(x, dist="norm")

print(f"Shapiro-Wilk:     W = {W:.4f}, p = {p_sw:.4g}")
print(f"D'Agostino K^2:   K2 = {K2:.4f}, p = {p_k2:.4g}")
print(f"Anderson-Darling: A^2 = {ad.statistic:.4f}")
for crit, sig in zip(ad.critical_values, ad.significance_level):
    print(f"  Critical {sig:.0f}%: {crit:.4f} -> reject if A^2 > crit")
```

## Interpretation

In the example above, the data are a mixture of normal and $t_3$ draws, introducing heavier tails. The Q-Q plot will show an S-shaped pattern: the extreme lower-left points fall below the fitted line and the extreme upper-right points rise above it. Both the Shapiro-Wilk and Anderson-Darling tests are expected to reject normality, while D'Agostino $K^2$ will flag excess kurtosis through its kurtosis component.

The key advantage of combining the Q-Q plot with formal tests is that the plot shows *where* and *how* the data depart from normality (in this case, the tails), while the tests quantify the strength of the evidence.

## Exercises

**Exercise 1.** Generate $n = 200$ standard normal observations. Produce the Q-Q plot and run all three tests. Verify that the points lie on the fitted line and that all $p$-values exceed 0.05.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(10)
    x = rng.normal(0, 1, size=200)

    fig, ax = plt.subplots(figsize=(7, 4))
    stats.probplot(x, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot: Standard Normal Sample")
    plt.tight_layout()
    plt.show()

    W, p_sw = stats.shapiro(x)
    K2, p_k2 = stats.normaltest(x)
    ad = stats.anderson(x, dist="norm")

    print(f"Shapiro-Wilk: p = {p_sw:.4g}")
    print(f"D'Agostino:   p = {p_k2:.4g}")
    print(f"Anderson-Darling: A^2 = {ad.statistic:.4f}")
    ```

    The Q-Q plot shows points closely hugging the diagonal. All $p$-values should be well above 0.05, confirming that standard normal data pass every normality check. $\square$

---

**Exercise 2.** Repeat Exercise 1 with $n = 200$ observations from $\text{Exponential}(1)$. Describe the Q-Q plot shape and compare the three $p$-values.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(20)
    x = rng.exponential(1.0, size=200)

    fig, ax = plt.subplots(figsize=(7, 4))
    stats.probplot(x, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot: Exponential(1) Data")
    plt.tight_layout()
    plt.show()

    W, p_sw = stats.shapiro(x)
    K2, p_k2 = stats.normaltest(x)
    print(f"Shapiro-Wilk: p = {p_sw:.4g}")
    print(f"D'Agostino:   p = {p_k2:.4g}")
    ```

    The Q-Q plot shows a strong concave (upward-curving) pattern because the exponential distribution is heavily right-skewed. All $p$-values are essentially zero, strongly rejecting normality. $\square$

---

**Exercise 3.** Explain the relationship between the slope and intercept of the fitted line on a Q-Q plot and the parameters $\mu$ and $\sigma$ of the normal distribution.

??? success "Solution to Exercise 3"

    Under the model $X \sim \mathcal{N}(\mu, \sigma^2)$, the $i$-th order statistic satisfies $\mathbb{E}[X_{(i)}] \approx \mu + \sigma\, q_i$ where $q_i = \mathcal{N}^{-1}((i-0.5)/n)$. The fitted line $\hat{y} = a + b\, q$ obtained by regressing $X_{(i)}$ on $q_i$ therefore has intercept $a \approx \bar{X} \approx \mu$ and slope $b \approx S \approx \sigma$. Thus the intercept estimates the mean and the slope estimates the standard deviation. If the data are truly normal, these estimates coincide with the sample mean and sample standard deviation. $\square$

---

**Exercise 4.** The Anderson-Darling test does not produce a $p$-value in SciPy; instead it returns critical values. Write code that determines the smallest significance level at which $H_0$ is rejected.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 150), rng.standard_t(3, 50)])

    ad = stats.anderson(x, dist="norm")
    rejected_levels = [sl for cv, sl in
                       zip(ad.critical_values, ad.significance_level)
                       if ad.statistic > cv]

    if rejected_levels:
        print(f"Reject at significance levels: {rejected_levels}")
        print(f"Smallest rejection level: {min(rejected_levels)}%")
    else:
        print("Fail to reject at all tabulated significance levels.")
    ```

    This iterates over the tabulated (critical value, significance level) pairs. The smallest level at which $A^2$ exceeds the critical value is the strongest statement we can make. For the heavy-tailed mixture, rejection typically occurs at all tabulated levels (including 1%). $\square$

---

**Exercise 5.** Show that when data are exactly standard normal, the Q-Q plot slope converges to 1 and the intercept converges to 0 as $n \to \infty$.

??? success "Solution to Exercise 5"

    Let $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \mathcal{N}(0, 1)$. The $i$-th order statistic satisfies

    $$
    \mathbb{E}[X_{(i)}] = \mathcal{N}^{-1}\!\Bigl(\frac{i}{n+1}\Bigr) + O(n^{-1}).
    $$

    For large $n$, $\frac{i}{n+1} \approx \frac{i - 0.5}{n}$, so $\mathbb{E}[X_{(i)}] \approx q_i$. The least-squares regression of $X_{(i)}$ on $q_i$ thus has slope $b \to \sigma = 1$ and intercept $a \to \mu = 0$ by the law of large numbers. More formally, $\bar{X} \xrightarrow{p} 0$ and $S \xrightarrow{p} 1$, and the OLS coefficients satisfy $b = S + o_p(1)$ and $a = \bar{X} + o_p(1)$, giving the result. $\square$
